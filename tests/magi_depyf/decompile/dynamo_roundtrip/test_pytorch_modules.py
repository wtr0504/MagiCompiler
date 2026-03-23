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

"""Dynamo roundtrip: core PyTorch nn.Module tests."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tests.magi_depyf.decompile.dynamo_roundtrip.helpers import roundtrip_and_verify


class TestPyTorchModules:
    """Core PyTorch nn.Module tests — always available."""

    def test_mlp(self):
        """2-layer feedforward network."""
        model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 16))
        roundtrip_and_verify(model, (torch.randn(2, 32),))

    def test_conv_bn_relu(self):
        """Conv2d + BatchNorm2d + ReLU stack."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.Conv2d(16, 8, 3, padding=1), nn.BatchNorm2d(8)
        )
        roundtrip_and_verify(model, (torch.randn(1, 3, 16, 16),))

    def test_multihead_attention(self):
        """nn.MultiheadAttention self-attention."""
        attn = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        x = torch.randn(2, 8, 32)
        roundtrip_and_verify(attn, (x, x, x))

    def test_transformer_encoder_layer(self):
        """nn.TransformerEncoderLayer (self-attn + FFN)."""
        layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, batch_first=True)
        roundtrip_and_verify(layer, (torch.randn(2, 8, 32),))

    @pytest.mark.skip(reason="Dynamo cannot trace LSTM (Graph Count: 0)")
    def test_lstm(self):
        """nn.LSTM single layer."""
        lstm = nn.LSTM(input_size=16, hidden_size=32, num_layers=1, batch_first=True)
        lstm.eval()

        def fn(x):
            output, (hn, cn) = lstm(x)
            return output

        roundtrip_and_verify(fn, (torch.randn(2, 8, 16),))

    @pytest.mark.skip(reason="Dynamo cannot trace GRU (Graph Count: 0)")
    def test_gru(self):
        """nn.GRU single layer."""
        gru = nn.GRU(input_size=16, hidden_size=32, batch_first=True)
        gru.eval()

        def fn(x):
            output, _ = gru(x)
            return output

        roundtrip_and_verify(fn, (torch.randn(2, 8, 16),))

    def test_embedding_linear(self):
        """Embedding -> Linear (language model head pattern)."""
        model = nn.Sequential(nn.Embedding(100, 32), nn.Linear(32, 100))
        roundtrip_and_verify(model, (torch.randint(0, 100, (2, 8)),))

    def test_layernorm_gelu_linear(self):
        """LayerNorm -> Linear -> GELU -> Linear."""
        model = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))
        roundtrip_and_verify(model, (torch.randn(2, 8, 32),))

    def test_residual_conv_block(self):
        """Residual connection with Conv2d."""

        class ResBlock(nn.Module):
            def __init__(self, ch):
                super().__init__()
                self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(ch)
                self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(ch)

            def forward(self, x):
                residual = x
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                return F.relu(out + residual)

        roundtrip_and_verify(ResBlock(8), (torch.randn(1, 8, 16, 16),))

    def test_grouped_conv(self):
        """Depthwise separable convolution (common in MobileNet/EfficientNet)."""
        model = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1, groups=16), nn.Conv2d(16, 32, 1), nn.ReLU())
        roundtrip_and_verify(model, (torch.randn(1, 16, 8, 8),))
