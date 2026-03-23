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

"""Dynamo roundtrip: diffusion-relevant building blocks (pure PyTorch)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tests.magi_depyf.decompile.dynamo_roundtrip.helpers import roundtrip_and_verify


class _GEGLU(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(dim, out_dim * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class _RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class _SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class _CrossAttention(nn.Module):
    def __init__(self, dim, context_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(context_dim, dim)
        self.to_v = nn.Linear(context_dim, dim)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, context):
        b, n, _ = x.shape
        h = self.heads
        q = self.to_q(x).view(b, n, h, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(b, -1, h, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(b, -1, h, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        out = attn.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


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


class _TimestepMLP(nn.Module):
    def __init__(self, time_dim, out_dim):
        super().__init__()
        self.embed = _SinusoidalEmbedding(time_dim)
        self.mlp = nn.Sequential(nn.Linear(time_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))

    def forward(self, t):
        return self.mlp(self.embed(t))


class TestDiffusionBlocks:
    """Diffusion-relevant building blocks — pure PyTorch, always available."""

    def test_geglu(self):
        roundtrip_and_verify(_GEGLU(32, 64), (torch.randn(2, 8, 32),))

    def test_rms_norm(self):
        roundtrip_and_verify(_RMSNorm(32), (torch.randn(2, 8, 32),))

    def test_sinusoidal_embedding(self):
        roundtrip_and_verify(_SinusoidalEmbedding(32), (torch.arange(4),))

    def test_cross_attention(self):
        model = _CrossAttention(32, context_dim=48, heads=4)
        roundtrip_and_verify(model, (torch.randn(2, 8, 32), torch.randn(2, 6, 48)))

    def test_ada_layer_norm(self):
        model = _AdaLayerNorm(32)
        roundtrip_and_verify(model, (torch.randn(2, 8, 32), torch.randn(2, 8, 32)))

    def test_dit_block(self):
        model = _SimpleDiTBlock(32, heads=4)
        roundtrip_and_verify(model, (torch.randn(2, 8, 32), torch.randn(2, 8, 32)))

    def test_timestep_mlp(self):
        roundtrip_and_verify(_TimestepMLP(32, 64), (torch.arange(4),))
