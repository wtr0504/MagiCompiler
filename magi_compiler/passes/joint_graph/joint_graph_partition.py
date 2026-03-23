# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import operator
import os
from typing import Any, Optional, Sequence, Tuple
from unittest.mock import patch

import torch
import torch.fx as fx
from torch._functorch.compile_utils import get_aten_target
from torch._functorch.partitioners import NodeInfo, OpTypes, get_default_op_list, min_cut_rematerialization_partition
from torch._inductor.custom_graph_pass import CustomPartitionerFn
from torch.utils._ordered_set import OrderedSet

from magi_compiler.config import RecomputePolicy, get_compile_config
from magi_compiler.magi_backend.partition_rules import resolve_defined_ops
from magi_compiler.utils import compute_code_hash, magi_logger
from magi_compiler.utils.visualize import joint_graph_vis

SAVE_TENSOR_NODES: Optional[list[fx.Node]] = None


def is_memory_increase_by_node(node: fx.Node) -> bool:
    # Only support aten.to now
    assert get_aten_target(node) == torch.ops.prims.convert_element_type
    input_dtype = node.args[0].meta["tensor_meta"].dtype
    output_dtype = node.args[1]
    assert output_dtype is not None
    return output_dtype.itemsize > input_dtype.itemsize


def is_compute_sensitive_op(
    node: fx.Node, op_types: OpTypes, custom_compute_sensitive_ops: list[torch._ops.OpOverload]
) -> bool:
    if op_types.is_compute_intensive(node):
        return True
    if node.op != "call_function":
        return False
    if node.target in custom_compute_sensitive_ops:
        return True
    if hasattr(node.target, "default") and node.target.default in custom_compute_sensitive_ops:
        return True
    return False


def is_primal_contribute_to_bwd_directly(
    primal_node: fx.Node, node_info: NodeInfo, op_types: OpTypes, custom_compute_sensitive_ops: list[torch._ops.OpOverload]
) -> bool:
    """
    FSDP ensures that weights already reside in memory. If there exists a path from the primal to the bwd, and the path does not contain any matmul, then the primal contributes to the bwd directly.
    And we should save this primals.
    """
    if node_info.is_required_bw(primal_node):
        return True
    topology_start = set({primal_node})

    while len(topology_start) > 0:
        cur_node = topology_start.pop()
        for user in cur_node.users:
            if node_info.is_required_bw(user):
                return True
            if is_compute_sensitive_op(user, op_types, custom_compute_sensitive_ops):
                continue
            topology_start.add(user)
    return False


def is_compute_intensive_and_has_following_recomputable_ops(
    intermidiate_node: fx.Node,
    node_info: NodeInfo,
    op_types: OpTypes,
    custom_compute_sensitive_ops: list[torch._ops.OpOverload],
) -> Tuple[bool, fx.Node]:
    """
    If compute-intensive node(CIN) is not the output of fwd graph(has following memory-intensive ops in the fwd graph), then we should save this CIN node.
    NOTE: For CIN+aten.to, we should save aten.to op instead of CIN op to save more memory.
    """
    if not is_compute_sensitive_op(intermidiate_node, op_types, custom_compute_sensitive_ops):
        return False, None

    save_node = intermidiate_node
    topology_start = set({save_node})
    while len(topology_start) > 0:
        cur_node = topology_start.pop()
        fwd_user_nodes = []
        for user in cur_node.users:
            if node_info.is_required_fw(user):
                fwd_user_nodes.append(user)

        if len(fwd_user_nodes) > 1:  # multiple users, save current node
            return True, save_node
        elif len(fwd_user_nodes) == 0:  # output, return
            return False, None

        # save current node if it's user is recomputable
        next_node = fwd_user_nodes[0]
        if op_types.is_view(next_node):
            if save_node == cur_node:
                save_node = next_node
            topology_start.add(next_node)
        # Special case for aten.to, memory efficient case
        elif get_aten_target(next_node) == torch.ops.prims.convert_element_type:
            is_memory_increase = is_memory_increase_by_node(next_node)
            if not is_memory_increase:
                save_node = next_node
            topology_start.add(next_node)
        elif next_node.op == "output":
            return False, None
        else:
            return True, save_node
    assert False, f"Should not reach here: {intermidiate_node=} {save_node=}"


# TODO: We find an elegant impl to heuristically save nodes, reconstruct this later
def heuristic_choose_saved_values_set(joint_graph: fx.Graph, node_info: NodeInfo, memory_budget=1) -> list[fx.Node]:
    output: OrderedSet[fx.Node] = OrderedSet()
    op_types = get_default_op_list()
    custom_compute_sensitive_ops = get_compile_config().recompute_config.custom_compute_sensitive_ops
    custom_compute_sensitive_ops: list[torch._ops.OpOverload] = resolve_defined_ops(custom_compute_sensitive_ops)
    # Select the inputs that are required by the backward pass
    for primal_node in node_info.inputs:
        if is_primal_contribute_to_bwd_directly(primal_node, node_info, op_types, custom_compute_sensitive_ops):
            output.add(primal_node)
    magi_logger.info("MagiCompiler: saved_output forward-input = %s", output)
    # Select the compute-intensive nodes that are required by the forward pass
    for intermidiate_node in node_info.required_fw_nodes:
        is_save, save_node = is_compute_intensive_and_has_following_recomputable_ops(
            intermidiate_node, node_info, op_types, custom_compute_sensitive_ops
        )
        if not is_save or save_node is None:
            continue
        out_val = save_node.meta.get("val")
        assert out_val is not None, f"save_node {save_node} must have output, otherwise it's no need to save"
        if isinstance(out_val, tuple):
            for user in save_node.users:
                assert (
                    user.op == "call_function" and user.target == operator.getitem
                ), f"save_node {save_node} must have a getitem user"
                output.add(user)
        else:
            output.add(save_node)
    magi_logger.info("MagiCompiler: saved_output compute-intensive = %s", output)
    global SAVE_TENSOR_NODES
    SAVE_TENSOR_NODES = list(output)
    return list(output)


def custom_joint_graph_partition_fn(
    joint_module: fx.GraphModule,
    _joint_inputs,
    compiler="inductor",
    *,
    num_fwd_outputs,
    static_lifetime_input_indices: Optional[list[int]] = None,
) -> tuple[fx.GraphModule, fx.GraphModule]:
    recompute_config = get_compile_config().recompute_config
    if recompute_config.recompute_policy == RecomputePolicy.HANDCRAFT:
        magi_logger.info("MagiCompiler using handcraft recompute policy")
        # TODO: different memory budget definition from torch
        with patch("torch._functorch.config.activation_memory_budget", recompute_config.memory_budget):
            fwd_module, bwd_module = min_cut_rematerialization_partition(
                joint_module,
                _joint_inputs,
                compiler,
                num_fwd_outputs=num_fwd_outputs,
                static_lifetime_input_indices=static_lifetime_input_indices,
            )
    elif recompute_config.recompute_policy == RecomputePolicy.HEURISTIC:
        magi_logger.info("MagiCompiler using heuristic recompute policy")
        with patch("torch._functorch.partitioners.choose_saved_values_set", heuristic_choose_saved_values_set):
            fwd_module, bwd_module = min_cut_rematerialization_partition(
                joint_module,
                _joint_inputs,
                compiler,
                num_fwd_outputs=num_fwd_outputs,
                static_lifetime_input_indices=static_lifetime_input_indices,
            )
    elif recompute_config.recompute_policy == RecomputePolicy.AUTOSEARCH:
        raise ValueError(f"AutoSearch recompute policy is not supported yet")
    else:
        raise ValueError(f"Invalid recompute policy: {recompute_config.recompute_policy}")

    joint_graph_vis(joint_module, fwd_module, bwd_module, save_tensor_nodes=SAVE_TENSOR_NODES)

    return fwd_module, bwd_module


class CustomJointGraphPartitionFn(CustomPartitionerFn):
    def __call__(
        self, gm: torch.fx.GraphModule, joint_inputs: Sequence[object], **kwargs: Any
    ) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
        """
        Implementation of the custom partitioner.
        """
        return custom_joint_graph_partition_fn(gm, joint_inputs, **kwargs)

    def uuid(self) -> Optional[Any]:
        """
        Return an ID to uniquely identify your custom partitioner implementation.
        Return None to skip inductor code caching entirely.
        """
        return compute_code_hash({os.path.abspath(__file__)})
