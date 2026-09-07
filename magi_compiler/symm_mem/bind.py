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

"""Move captured-graph weights into symmetric memory.

Runs between lowering and bucketing. Downstream keys off what actually moved;
anything that cannot bind stays on NCCL.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch
import torch.distributed as dist
import torch.fx as fx

from magi_compiler.utils import magi_logger

from .symm_buffer import group_name_of, lookup_shard, open_buffer, publish, register_shard

_AGREEMENT_GROUP: tuple[Any, Any] = (None, None)
"""``(the default process group it was built for, the gloo group)``."""


@dataclass(frozen=True)
class BindCandidate:
    """One weight ready to be moved. ``local`` is ``param._local_tensor``."""

    local: torch.Tensor
    group_name: str
    gather: fx.Node | None = None
    holder: fx.Node | None = None


def bind_graph_weights(
    graph: fx.GraphModule,
    candidates: Iterable[tuple[fx.Node, fx.Node]],
    placeholder_examples: Mapping[str, Any],
    min_shard_bytes: int = 0,
) -> set[fx.Node]:
    """Move the weights behind ``candidates`` into symmetric memory.

    ``candidates`` is ``(gather, holder)``; ``placeholder_examples`` maps
    placeholder name to the live tensor. Returns the gathers now copy-engine
    backed -- the caller tags them. Failures are dropped, not raised.
    """
    plan, skipped = _plan(graph, candidates, placeholder_examples, min_shard_bytes)

    # Before the empty check, so every rank reaches the collective.
    if not _agree_across_ranks(plan):
        magi_logger.warning(
            "copy-engine binding: ranks disagree on which weights to bind; leaving every weight "
            "gather on NCCL. A gather retargeted on only some ranks would never complete."
        )
        return set()

    if not plan:
        magi_logger.info("copy-engine binding: nothing to bind (%s)", _describe(skipped))
        return set()

    allocated = _move(plan)
    publish()

    served = {c.gather for c in plan if c.gather is not None}
    magi_logger.info(
        "copy-engine binding: %d weight gather(s) bound (%d new allocation(s), %.1f MiB); %s",
        len(served),
        allocated,
        sum(c.local.numel() * c.local.element_size() for c in plan) / 2**20,
        _describe(skipped),
    )
    return served


def bind_parameters(params: Iterable[Any], min_shard_bytes: int = 0) -> int:
    """Move an explicit list of sharded parameters into symmetric memory.

    For callers with no graph (op benches, AOT compile with fake inputs).
    Returns how many parameters are now bound.
    """
    plan: list[BindCandidate] = []
    for p in params:
        if _unbindable(p, min_shard_bytes) is not None:
            continue
        try:
            plan.append(BindCandidate(local=p._local_tensor, group_name=group_name_of(p)))
        except RuntimeError:
            continue

    if not _agree_across_ranks(plan):
        magi_logger.warning("copy-engine binding: ranks disagree on which parameters to bind; binding none")
        return 0
    if not plan:
        return 0

    _move(plan)
    publish()
    return len(plan)


def _plan(
    graph: fx.GraphModule,
    candidates: Iterable[tuple[fx.Node, fx.Node]],
    placeholder_examples: Mapping[str, Any],
    min_shard_bytes: int,
) -> tuple[list[BindCandidate], Counter]:
    """Pair each candidate gather with a live shard, dropping the ones that fail."""
    plan: list[BindCandidate] = []
    skipped: Counter = Counter()

    for gather, holder in candidates:
        param = _resolve(graph, holder, placeholder_examples)
        if param is None:
            skipped["graph input has no live parameter behind it"] += 1
            continue
        why = _unbindable(param, min_shard_bytes)
        if why is not None:
            skipped[why] += 1
            continue
        try:
            group_name = group_name_of(param)
        except RuntimeError as exc:
            skipped[str(exc)] += 1
            continue
        plan.append(BindCandidate(local=param._local_tensor, group_name=group_name, gather=gather, holder=holder))

    return plan, skipped


def _resolve(graph: fx.GraphModule, holder: fx.Node, placeholder_examples: Mapping[str, Any]) -> Any:
    """The live object a weight-holding node stands for, or None."""
    if holder.op == "placeholder":
        return placeholder_examples.get(holder.name)
    if holder.op == "get_attr":
        obj: Any = graph
        for part in str(holder.target).split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj
    return None


def _unbindable(param: Any, min_shard_bytes: int) -> str | None:
    """Why ``param`` cannot back a copy-engine gather, or None if it can."""
    from torch._subclasses.fake_tensor import FakeTensor
    from torch.distributed.tensor import DTensor, Shard

    if not isinstance(param, DTensor):
        return "graph input is not a DTensor"

    local = param._local_tensor
    if isinstance(local, FakeTensor) or local.is_meta:
        return "graph input is a fake/meta tensor"

    placements = param.placements
    if len(placements) != 1 or not isinstance(placements[0], Shard) or placements[0].dim != 0:
        return f"placement {tuple(placements)} is not a single Shard(0)"
    # Fixed-stride peer reads cannot express uneven rank shards.
    if int(param.shape[0]) % int(param.device_mesh.size(0)):
        return "Shard(0) does not divide evenly across the mesh"

    if local.device.type != "cuda":
        return f"shard lives on {local.device.type}, not cuda"
    if not local.is_contiguous():
        return "shard is not contiguous"
    if local.numel() * local.element_size() < min_shard_bytes:
        return "shard is below the size floor"
    return None


_WINDOW_BYTES = 4 << 30
"""Cap on one symmetric window (4 GiB).

The driver allows 128 windows per process, but a window is opened while every
shard it will absorb is still resident, so this is also what binding adds to
the peak. A shard bigger than this still gets a window to itself.
"""


def _windows(plan: list[BindCandidate]) -> list[list[BindCandidate]]:
    """Split the plan into per-window lists, grouped by (group, dtype)."""
    fresh: list[BindCandidate] = []
    seen: set[int] = set()
    for c in plan:
        ptr = c.local.data_ptr()
        # Already symmetric: a tied weight earlier in this plan, or a previous compile.
        if ptr in seen or lookup_shard(ptr) is not None:
            continue
        seen.add(ptr)
        fresh.append(c)

    by_kind: dict[tuple[str, torch.dtype], list[BindCandidate]] = defaultdict(list)
    for c in fresh:
        by_kind[(c.group_name, c.local.dtype)].append(c)

    windows: list[list[BindCandidate]] = []
    for members in by_kind.values():
        current: list[BindCandidate] = []
        current_bytes = 0
        for c in members:
            nbytes = c.local.numel() * c.local.element_size()
            if current and current_bytes + nbytes > _WINDOW_BYTES:
                windows.append(current)
                current, current_bytes = [], 0
            current.append(c)
            current_bytes += nbytes
        if current:
            windows.append(current)
    return windows


def _move(plan: list[BindCandidate]) -> int:
    """Allocate, fill and repoint every shard. Returns the new allocation count.

    Symmetric memory cannot reuse the caching allocator's blocks, so each window
    ends with ``empty_cache``. An allocation failure is fatal: ranks have already
    issued a matching prefix of rendezvous.
    """
    windows = _windows(plan)
    allocated = 0
    for i, members in enumerate(windows):
        head = members[0]
        try:
            buffer = open_buffer(head.local.dtype, head.local.device, head.group_name, (c.local.numel() for c in members))
        except RuntimeError:
            free, total = torch.cuda.mem_get_info()
            magi_logger.error(
                "copy-engine binding: failed to open symmetric window %d of %d (%d shard(s), "
                "%.1f MiB of %s); driver has %.1f of %.1f GiB free, torch reserves %.1f GiB of "
                "which %.1f GiB is live",
                i,
                len(windows),
                len(members),
                sum(c.local.numel() * c.local.element_size() for c in members) / (1 << 20),
                head.local.dtype,
                free / (1 << 30),
                total / (1 << 30),
                torch.cuda.memory_reserved() / (1 << 30),
                torch.cuda.memory_allocated() / (1 << 30),
            )
            raise

        for c in members:
            symm = buffer.take(c.local.shape)
            symm.copy_(c.local)
            register_shard(symm, buffer)
            # In place: Dynamo already guarded these exact objects.
            c.local.data = symm
            allocated += 1

        torch.cuda.empty_cache()
    return allocated


def _agree_across_ranks(plan: list[BindCandidate]) -> bool:
    """True if every rank arrived at the same plan, in the same order.

    ``group_name`` is not in the fingerprint: expert weights sit on a subgroup of
    the world, so each rank correctly resolves a different group for the same weight.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return True

    mine = [(c.holder.name if c.holder is not None else "", tuple(c.local.shape), str(c.local.dtype)) for c in plan]
    group = _agreement_group()
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, mine, group=group)

    for rank, theirs in enumerate(gathered):
        if theirs == mine:
            continue
        i = next((i for i, (a, b) in enumerate(zip(mine, theirs)) if a != b), min(len(mine), len(theirs)))
        magi_logger.warning(
            "copy-engine binding: this rank plans %d shard(s) and rank %d plans %d; they first differ at "
            "index %d, where this rank has %s and rank %d has %s",
            len(mine),
            rank,
            len(theirs),
            i,
            mine[i] if i < len(mine) else "<end of plan>",
            rank,
            theirs[i] if i < len(theirs) else "<end of plan>",
        )
        return False
    return True


def _agreement_group():
    """A CPU group for the plan check, rebuilt if the default process group changes.

    ``new_group`` is collective, so every rank must reach here even with an empty plan.
    """
    global _AGREEMENT_GROUP
    owner, group = _AGREEMENT_GROUP
    if owner is dist.group.WORLD:
        return group
    try:
        group = dist.new_group(backend="gloo")
    except Exception as exc:  # noqa: BLE001
        magi_logger.warning("copy-engine binding: gloo group unavailable (%s); using the default group", exc)
        group = None
    _AGREEMENT_GROUP = (dist.group.WORLD, group)
    return group


def _describe(skipped: Counter) -> str:
    if not skipped:
        return "no candidate was skipped"
    return "skipped: " + ", ".join(f"{n}x {why}" for why, n in skipped.most_common())
