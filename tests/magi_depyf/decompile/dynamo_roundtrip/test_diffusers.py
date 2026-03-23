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

"""Dynamo roundtrip: diffusers attention and transformer blocks."""

import pytest
import torch

from tests.magi_depyf.decompile.dynamo_roundtrip.helpers import _has_diffusers, roundtrip_and_verify


@pytest.mark.skipif(not _has_diffusers, reason="diffusers not installed")
class TestDiffusersComponents:
    """diffusers attention and transformer blocks — directly relevant for
    Stable Diffusion / DiT debugging."""

    def test_diffusers_self_attention(self):
        from diffusers.models.attention_processor import Attention

        attn = Attention(query_dim=32, heads=4, dim_head=8)
        roundtrip_and_verify(attn, (torch.randn(2, 8, 32),), atol=1e-4)

    def test_diffusers_cross_attention(self):
        from diffusers.models.attention_processor import Attention

        attn = Attention(query_dim=32, cross_attention_dim=48, heads=4, dim_head=8)
        roundtrip_and_verify(
            attn, (torch.randn(2, 8, 32),), input_kwargs={"encoder_hidden_states": torch.randn(2, 6, 48)}, atol=1e-4
        )

    def test_diffusers_basic_transformer_block(self):
        from diffusers.models.attention import BasicTransformerBlock

        block = BasicTransformerBlock(dim=32, num_attention_heads=4, attention_head_dim=8)
        roundtrip_and_verify(block, (torch.randn(2, 8, 32),), atol=1e-4)

    def test_diffusers_cross_attn_transformer_block(self):
        from diffusers.models.attention import BasicTransformerBlock

        block = BasicTransformerBlock(dim=32, num_attention_heads=4, attention_head_dim=8, cross_attention_dim=48)
        roundtrip_and_verify(
            block, (torch.randn(2, 8, 32),), input_kwargs={"encoder_hidden_states": torch.randn(2, 6, 48)}, atol=1e-4
        )
