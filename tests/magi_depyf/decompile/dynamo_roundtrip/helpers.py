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

"""Shared helpers and model definitions for dynamo-roundtrip tests.

Test methodology:
  1. torch.compile(fn) and run to get expected output
  2. Extract Dynamo cache entry (transformed bytecode)
  3. Decompile -> recompile -> replace fn.__code__
  4. torch.compile again and verify output matches expected
"""

from typing import Any

import torch

from magi_compiler.magi_depyf.decompile.recompiler import CodeRecompiler
from magi_compiler.magi_depyf.inspect.introspect import Introspector

get_cache_entries = Introspector.get_cache_entries


# ---------------------------------------------------------------------------
# Library availability flags
# ---------------------------------------------------------------------------

_has_timm = False
try:
    pass

    _has_timm = True
except ImportError:
    pass

_has_transformers = False
try:
    pass

    _has_transformers = True
except ImportError:
    pass

_has_diffusers = False
try:
    pass

    _has_diffusers = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset():
    torch._dynamo.reset()


def _assert_close(actual: Any, expected: Any, atol: float = 1e-5):
    """Recursively compare outputs with tolerance."""
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), f"Expected tensor, got {type(actual)}"
        assert actual.shape == expected.shape, f"Shape mismatch: {actual.shape} vs {expected.shape}"
        diff = (actual.float().cpu() - expected.float().cpu()).abs().max().item()
        assert diff < atol, f"Tensor mismatch: max diff = {diff}"
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected), f"Length mismatch: {len(actual)} vs {len(expected)}"
        for a, e in zip(actual, expected):
            _assert_close(a, e, atol=atol)
    elif isinstance(expected, dict):
        for k in expected:
            if k in actual:
                _assert_close(actual[k], expected[k], atol=atol)
    elif expected is None:
        pass
    elif hasattr(expected, "__dict__"):
        for k, v in vars(expected).items():
            if isinstance(v, torch.Tensor) and hasattr(actual, k):
                _assert_close(getattr(actual, k), v, atol=atol)
    else:
        assert actual == expected, f"Value mismatch: {actual} vs {expected}"


GLOBAL_MODULE = None
GLOBAL_INPUT_KWARGS: dict = {}
GLOBAL_OUTPUT_FN = None


def roundtrip_and_verify(fn_or_module, inputs, input_kwargs=None, output_fn=None, backend="eager", atol=1e-5, **compile_kw):
    """Compile → decompile → recompile → replace code → re-compile and verify.

    Accepts either a plain function or an ``nn.Module``.  When given a module,
    it is automatically wrapped via the ``GLOBAL_MODULE`` pattern so that the
    wrapper function has no free variables (avoids Dynamo closure issues).

    *input_kwargs*, if given, are forwarded as keyword arguments to the module
    call (e.g. ``encoder_hidden_states``).

    *output_fn*, if given, is applied to the module output before returning
    (e.g. ``lambda out: out.last_hidden_state``).  This is stored as a global
    to avoid introducing closure variables.
    """
    global GLOBAL_MODULE, GLOBAL_INPUT_KWARGS, GLOBAL_OUTPUT_FN

    if isinstance(fn_or_module, torch.nn.Module):
        fn_or_module.eval()
        GLOBAL_MODULE = fn_or_module
        GLOBAL_INPUT_KWARGS = input_kwargs or {}
        GLOBAL_OUTPUT_FN = output_fn

        if GLOBAL_OUTPUT_FN is not None:

            def fn(*args):
                return GLOBAL_OUTPUT_FN(GLOBAL_MODULE(*args, **GLOBAL_INPUT_KWARGS))

        else:

            def fn(*args):
                return GLOBAL_MODULE(*args, **GLOBAL_INPUT_KWARGS)

    else:
        fn = fn_or_module

    _reset()
    torch.manual_seed(42)
    compiled = torch.compile(fn, backend=backend, **compile_kw)
    expected = compiled(*inputs)

    entries = get_cache_entries(fn)
    assert len(entries) >= 1, f"No cache entries for {fn.__code__.co_name}"

    tc = entries[0].code
    recompiled = CodeRecompiler.recompile(code_to_decompile=tc, reference_code=tc)

    old_code = fn.__code__
    fn.__code__ = recompiled
    try:
        _reset()
        torch.manual_seed(42)
        compiled2 = torch.compile(fn, backend=backend, **compile_kw)
        actual = compiled2(*inputs)
        _assert_close(actual, expected, atol=atol)
    finally:
        fn.__code__ = old_code
