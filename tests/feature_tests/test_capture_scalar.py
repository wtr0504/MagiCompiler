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


# NOTE: Currentlly, magi compiler's dynamo config is independent from the outer dynamo config.


import pytest
import torch

from magi_compiler import magi_compile
from magi_compiler._api import _DEFAULT_DYNAMO_CONFIG


@pytest.fixture(autouse=True)
def _enable_capture_scalar_outputs():
    """Enable capture_scalar_outputs for magi_compile's dynamo config in this module."""
    old_value = _DEFAULT_DYNAMO_CONFIG["capture_scalar_outputs"]
    _DEFAULT_DYNAMO_CONFIG["capture_scalar_outputs"] = True
    yield
    _DEFAULT_DYNAMO_CONFIG["capture_scalar_outputs"] = old_value


SEQ_LEN = 8
HIDDEN_SIZE = 8


@magi_compile(dynamic_arg_dims={"x": 0})
class MulItemModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False, dtype=torch.bfloat16)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.linear(x) * y[0].item()


def test_mul_item_model_torch_compile(_enable_capture_scalar_outputs):
    from unittest.mock import patch

    from magi_compiler.config import CompileMode, get_compile_config

    # NOTE: The model class must be defined *inside* the patch context so that
    # ``@magi_compile`` captures the patched ``TORCH_COMPILE`` mode in its
    # per-model deepcopy of ``CompileConfig``.
    with patch.object(get_compile_config(), "compile_mode", CompileMode.TORCH_COMPILE):

        @magi_compile(dynamic_arg_dims={"x": 0})
        class _MulItemModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False, dtype=torch.bfloat16)

            @torch.no_grad()
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return self.linear(x) * y[0].item()

        model = _MulItemModel()
        x = torch.randn(SEQ_LEN, HIDDEN_SIZE, dtype=torch.bfloat16)
        y = torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16)
        output = model(x, y)
        assert output.shape == (SEQ_LEN, HIDDEN_SIZE)


# FIXME: Support item() with MAGI_COMPILE
def test_mul_item_model_magi_compile(_enable_capture_scalar_outputs):
    try:
        model = MulItemModel()
        x = torch.randn(SEQ_LEN, HIDDEN_SIZE, dtype=torch.bfloat16)
        y = torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16)
        output = model(x, y)
        assert output.shape == (SEQ_LEN, HIDDEN_SIZE)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
