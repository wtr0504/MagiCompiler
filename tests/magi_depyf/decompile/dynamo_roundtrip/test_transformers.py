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

"""Dynamo roundtrip: HuggingFace transformers models."""

import pytest
import torch

from tests.magi_depyf.decompile.dynamo_roundtrip.helpers import _has_transformers, roundtrip_and_verify


def _get_last_hidden_state(out):
    return out.last_hidden_state


@pytest.mark.skipif(not _has_transformers, reason="transformers not installed")
class TestTransformersModels:
    """Tiny HuggingFace transformer models."""

    def test_bert_tiny(self):
        from transformers import BertConfig, BertModel

        config = BertConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            vocab_size=100,
            max_position_embeddings=16,
        )
        roundtrip_and_verify(BertModel(config), (torch.randint(0, 100, (1, 8)),), atol=1e-4)

    def test_gpt2_tiny(self):
        from transformers import GPT2Config, GPT2Model

        config = GPT2Config(n_embd=32, n_layer=1, n_head=4, vocab_size=100, n_positions=16)
        roundtrip_and_verify(GPT2Model(config), (torch.randint(0, 100, (1, 8)),), output_fn=_get_last_hidden_state, atol=1e-4)

    def test_t5_encoder_tiny(self):
        from transformers import T5Config, T5EncoderModel

        config = T5Config(d_model=32, d_ff=64, num_heads=4, num_layers=1, vocab_size=100)
        roundtrip_and_verify(T5EncoderModel(config), (torch.randint(0, 100, (1, 8)),), atol=1e-4)
