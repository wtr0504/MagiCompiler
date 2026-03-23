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

"""Dynamo roundtrip: direct Module.forward.__code__ replacement.

Mirrors the real magi_compiler pattern: torch.compile(module) then
replace Module.forward.__code__ with decompiled+recompiled code.

This is the only test file that replaces ``klass.forward.__code__``
(the production code path in ``magi_compiler_base.py``).  All other
dynamo roundtrip tests go through a wrapper function instead.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from magi_compiler.magi_depyf.decompile.recompiler import CodeRecompiler
from tests.magi_depyf.decompile.dynamo_roundtrip.helpers import _assert_close, _reset, get_cache_entries


def _roundtrip_forward(module, inputs, backend="eager", atol=1e-5, **compile_kw):
    """Compile module directly, decompile forward, replace klass.forward.__code__."""
    _reset()
    module.eval()
    torch.manual_seed(42)
    compiled = torch.compile(module, backend=backend, **compile_kw)
    expected = compiled(*inputs)

    entries = get_cache_entries(module.forward)
    assert len(entries) >= 1, f"No cache entries for {module.__class__.__name__}.forward"

    tc = entries[0].code
    recompiled = CodeRecompiler.recompile(code_to_decompile=tc, reference_code=tc)

    klass = module.__class__
    old_code = klass.forward.__code__
    klass.forward.__code__ = recompiled
    try:
        _reset()
        torch.manual_seed(42)
        compiled2 = torch.compile(module, backend=backend, **compile_kw)
        actual = compiled2(*inputs)
        _assert_close(actual, expected, atol=atol)
    finally:
        klass.forward.__code__ = old_code


class _AdaLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.scale_shift = nn.Linear(dim, dim * 2)

    def forward(self, x, cond):
        scale, shift = self.scale_shift(cond).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class _SimpleDiTBlock(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.norm1 = _AdaLayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = _AdaLayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x, cond):
        h = self.norm1(x, cond)
        h, _ = self.attn(h, h, h)
        x = x + h
        h = self.norm2(x, cond)
        x = x + self.ff(h)
        return x


class TestForwardCodeReplacement:
    """Dynamo traces module.forward directly, producing bytecode with ``self``
    in ``co_varnames``, and we replace the class-level ``forward.__code__``
    — exactly like ``magi_compiler_base.py`` does.
    """

    def test_custom_forward(self):
        class ResBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(32, 32)
                self.linear2 = nn.Linear(32, 32)
                self.norm = nn.LayerNorm(32)

            def forward(self, x):
                residual = x
                x = F.relu(self.linear1(x))
                x = self.linear2(x)
                return self.norm(x + residual)

        _roundtrip_forward(ResBlock(), (torch.randn(2, 8, 32),))

    def test_multi_input_forward(self):
        _roundtrip_forward(_AdaLayerNorm(32), (torch.randn(2, 8, 32), torch.randn(2, 8, 32)))

    def test_dit_block_forward(self):
        _roundtrip_forward(_SimpleDiTBlock(32, heads=4), (torch.randn(2, 8, 32), torch.randn(2, 8, 32)))
