# Copyright (c) 2025 SandAI. All Rights Reserved.
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

import functools
import inspect
from typing import Callable, TypeVar

from torch import nn

from ._api import (
    _check_dynamic_arg_dims,
    _infer_dynamic_arg_dims,
    _magi_compile_class,
    _magi_compile_function,
    _magi_compile_instance,
)
from ._magi_register_custom_op import _magi_register_custom_op_impl
from .config import CompileConfig

_T = TypeVar("_T", bound=type[nn.Module])
_F = TypeVar("_F", bound=Callable)
_M = TypeVar("_M", bound=nn.Module)


def magi_compile(
    obj: _T | _M | _F | None = None,
    *,
    model_tag: str | None = None,
    dynamic_arg_dims: dict[str, int | list[int]] | None = None,
    enable_if: Callable[[], bool] | None = None,
    config_patch: Callable[[CompileConfig], CompileConfig] | None = None,
) -> _T | _M | _F | Callable[[_T | _M | _F], _T | _M | _F]:
    """
    Compile target objects (nn.Module classes, modules, functions, or methods).

    Supported target types
    ----------------------
    1) Class (must be an `nn.Module` subclass):
        - Affects all instances of the annotated class.
        - Compilation dispatch enters via `__call__`, while compiled execution replaces `forward` logic.
        - Example:
            @magi_compile
            class MyModel(nn.Module):
                def forward(self, x): return x

    2) Function (Standalone):
        - Wraps a callable with MagiCompiler's dispatch logic.
        - Useful for non-member functions or general callables.
        - Example:
            @magi_compile
            def my_func(x): return x

    3) Instance (nn.Module):
        - Compiles a single instance specifically.
        - Avoids affecting other instances by creating an instance-specific subclass.
        - Example:
            model = MyModel()
            model = magi_compile(model)

    4) Method (Bound/Unbound):
        - Wraps a specific function attribute (e.g., `model.forward`).
        - Enables focused compilation of specific object behaviors.
        - Example:
            model = MyModel()
            model.forward = magi_compile(model.forward)

    Usage Styles
    ------------
    The compiler supports both declarative (decorator) and imperative (function call) styles.

    A) Decorator Style:
        - Example:
            @magi_compile(dynamic_arg_dims={"x": 0})
            class MyModel(nn.Module): ...

            class MyModel(nn.Module):
                @magi_compile
                def forward(self, x): ...

    B) Imperative Style:
        - Apply directly to an existing object:
            model = magi_compile(model, dynamic_arg_dims={"x": 0})

    C) Factory Style:
        - Configure a compiler first, then apply to multiple objects:
            compiler = magi_compile(dynamic_arg_dims={"x": 0})
            model = compiler(model)
            cls = compiler(MyModel)

    Arguments
    ---------
    - dynamic_arg_dims: Dictionary mapping argument names to dynamic dimensions (int or list[int]).
    - model_tag: Optional tag for caching path (defaults to class/function name).
    - enable_if: Callable returning bool; compilation happens only if this returns True.

    Notes
    -----
    - If `dynamic_arg_dims` is omitted, it is inferred from type annotations:
      `torch.Tensor` arguments default to dynamic dimension 0.
    - Consistency: For graph stability, maintain consistent input types (e.g., avoid switching between Tensor and None).
    """
    if obj is None:
        return functools.partial(
            magi_compile,
            model_tag=model_tag,
            dynamic_arg_dims=dynamic_arg_dims,
            enable_if=enable_if,
            config_patch=config_patch,
        )

    # 1. Determine target function for dynamic dim inference
    if inspect.isclass(obj):
        assert issubclass(obj, nn.Module), f"Expected nn.Module subclass, got {obj}"
        target_func = obj.forward
        context_name = f"forward method of {obj.__name__}"
    elif isinstance(obj, nn.Module):
        target_func = obj.forward
        context_name = f"forward method of instance {obj.__class__.__name__}"
    elif callable(obj):
        target_func = obj
        context_name = f"function/method {obj.__name__}"
    else:
        raise TypeError(f"Unsupported type for magi_compile: {type(obj)}")

    # 2. Infer dynamic dims
    inferred_dims = dynamic_arg_dims or _infer_dynamic_arg_dims(target_func, context_name)
    assert (
        len(inferred_dims) > 0
    ), f"No dynamic dimensions found in {context_name}. Please provide dynamic_arg_dims explicitly."

    _check_dynamic_arg_dims(inferred_dims, target_func)

    # 3. Logic based on type
    if inspect.isclass(obj):
        return _magi_compile_class(obj, inferred_dims, enable_if, config_patch, model_tag)
    elif isinstance(obj, nn.Module):
        return _magi_compile_instance(obj, inferred_dims, enable_if, config_patch, model_tag)
    else:
        return _magi_compile_function(obj, inferred_dims, enable_if, config_patch, model_tag)


def magi_register_custom_op(
    name: str | None = None,
    mutates_args: tuple[str, ...] = (),
    infer_output_meta_fn: Callable | list[str] | None = None,
    setup_context_fn: Callable | None = None,
    backward_fn: Callable | None = None,
    is_compute_sensitive: bool = False,
    is_subgraph_boundary: bool = False,
):
    """
    A unified decorator to register a custom operator with PyTorch's library.

    This decorator combines the functionality of:
    - @torch.library.custom_op
    - @torch.library.register_fake
    - fn.register_autograd

    Arguments:
        name: The fully qualified name of the operator (e.g., "namespace::op_name").
              If None, auto-generated from the function name and source file.
        mutates_args: Tuple of argument names that are mutated by the operator.
        infer_output_meta_fn: Specifies output tensor metadata (shape, dtype, device) for tracing.
            - None (default): Assumes each output has the same metadata as the corresponding
              input tensor (1st output matches 1st tensor input, 2nd matches 2nd, etc.).
            - list[str]: Parameter names whose metadata to use for outputs.
              E.g., ["weight", "bias"] means output[0] has same shape as `weight`,
              output[1] has same shape as `bias`.
            - Callable: Custom function with same signature as the op, returns torch.empty_like()
              tensors matching the expected output shapes.
        setup_context_fn: Function to save tensors/values for backward.
            Signature: setup_context_fn(ctx, inputs, output)
        backward_fn: Function to compute gradients.
            Signature: backward_fn(ctx, *grad_outputs) -> tuple of gradients
        is_compute_sensitive: If True, marks this operator as compute-intensive (e.g., MatMul,
            Attention). During activation recomputation (rematerialization), outputs of
            compute-sensitive ops are prioritized for saving rather than recomputing,
            since recomputing them would be expensive.
        is_subgraph_boundary: If True, the FX graph will be split at this operator during
            compilation. Each sub-graph between boundary operators is compiled independently
            by Inductor, enabling piecewise compilation and more flexible scheduling
            (e.g., for CPU offloading or overlapping computation with data transfer).

    Returns:
        The registered custom operator function.

    Examples:
        1. Basic usage (forward only, auto-generated name and meta function):

        >>> @magi_register_custom_op()
        ... def my_relu(x: torch.Tensor) -> torch.Tensor:
        ...     return torch.maximum(x, torch.zeros_like(x))

        2. Multiple outputs with explicit output metadata via parameter names:

        >>> @magi_register_custom_op(
        ...     infer_output_meta_fn=["weight", "bias"],  # output shapes match weight and bias
        ... )
        ... def compute_gradients(
        ...     grad_output: torch.Tensor,
        ...     weight: torch.Tensor,
        ...     bias: torch.Tensor,
        ... ) -> tuple[torch.Tensor, torch.Tensor]:
        ...     grad_weight = grad_output.sum(dim=0).view_as(weight)
        ...     grad_bias = grad_output.sum(dim=0).view_as(bias)
        ...     return grad_weight, grad_bias

        3. Full custom op with autograd support:

        >>> def _square_meta(x: torch.Tensor) -> torch.Tensor:
        ...     return torch.empty_like(x)
        ...
        >>> def _square_setup_context(ctx, inputs, output):
        ...     (x,) = inputs
        ...     ctx.save_for_backward(x)
        ...
        >>> def _square_backward(ctx, grad_output):
        ...     (x,) = ctx.saved_tensors
        ...     return grad_output * 2 * x
        ...
        >>> @magi_register_custom_op(
        ...     name="my_ops::square",
        ...     infer_output_meta_fn=_square_meta,
        ...     setup_context_fn=_square_setup_context,
        ...     backward_fn=_square_backward,
        ... )
        ... def square(x: torch.Tensor) -> torch.Tensor:
        ...     return x * x
    """
    return _magi_register_custom_op_impl(
        name=name,
        mutates_args=mutates_args,
        infer_output_meta_fn=infer_output_meta_fn,
        setup_context_fn=setup_context_fn,
        backward_fn=backward_fn,
        is_compute_sensitive=is_compute_sensitive,
        is_subgraph_boundary=is_subgraph_boundary,
    )
