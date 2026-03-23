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

"""Dynamo roundtrip: timm models (diverse architectures)."""

import pytest
import torch

from tests.magi_depyf.decompile.dynamo_roundtrip.helpers import _has_timm, roundtrip_and_verify


@pytest.mark.skipif(not _has_timm, reason="timm not installed")
class TestTimmModels:
    """Small timm models — diverse architectures."""

    def test_resnet18(self):
        import timm

        model = timm.create_model("resnet18", pretrained=False, num_classes=10)
        roundtrip_and_verify(model, (torch.randn(1, 3, 32, 32),), atol=1e-4)

    def test_mobilenetv3_small(self):
        import timm

        model = timm.create_model("mobilenetv3_small_050", pretrained=False, num_classes=10)
        roundtrip_and_verify(model, (torch.randn(1, 3, 64, 64),), atol=1e-4)

    def test_efficientnet_b0(self):
        import timm

        model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=10)
        roundtrip_and_verify(model, (torch.randn(1, 3, 64, 64),), atol=1e-4)

    def test_vit_tiny(self):
        import timm

        model = timm.create_model("vit_tiny_patch16_224", pretrained=False, img_size=64, num_classes=10)
        roundtrip_and_verify(model, (torch.randn(1, 3, 64, 64),), atol=1e-4)

    def test_convnext_tiny(self):
        import timm

        model = timm.create_model("convnext_tiny", pretrained=False, num_classes=10)
        roundtrip_and_verify(model, (torch.randn(1, 3, 32, 32),), atol=1e-4)

    def test_swin_tiny(self):
        import timm

        model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, img_size=56, num_classes=10)
        roundtrip_and_verify(model, (torch.randn(1, 3, 56, 56),), atol=1e-4)

    def test_deit_tiny(self):
        import timm

        model = timm.create_model("deit_tiny_patch16_224", pretrained=False, img_size=64, num_classes=10)
        roundtrip_and_verify(model, (torch.randn(1, 3, 64, 64),), atol=1e-4)
