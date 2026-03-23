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

import pytest
import torch

from tests.model_definition import create_mlp_model


@pytest.fixture(scope="function")
def mlp_model(device, mlp_config):
    """MLP model fixture"""
    model = create_mlp_model(mlp_config, device)
    model.eval()
    return model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_mlp_basic_inference(device, mlp_config, mlp_model):
    """Test basic inference functionality"""
    num_tokens = 128
    input_tensor = torch.randn(num_tokens, mlp_config.hidden_size, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        output = mlp_model(input_tensor)

    # Verify output shape
    assert output.shape == (
        num_tokens,
        mlp_config.hidden_size,
    ), f"Output shape should be ({num_tokens}, {mlp_config.hidden_size}), but got {output.shape}"

    # Verify output data type
    assert output.dtype == torch.float32, f"Output data type should be torch.float32, but got {output.dtype}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_mlp_different_input_shapes(device, mlp_config, mlp_model):
    """Test different input shapes"""
    test_shapes = [
        (4, mlp_config.hidden_size),  # Small batch
        (8, mlp_config.hidden_size),  # Medium batch
        (16, mlp_config.hidden_size),  # Large batch
        # NOTE: compiler will specialize for single token, so we move it to the last
        (1, mlp_config.hidden_size),  # Single token
    ]

    with torch.no_grad():
        for num_tokens, hidden_size in test_shapes:
            input_tensor = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
            output = mlp_model(input_tensor)

            assert output.shape == (
                num_tokens,
                hidden_size,
            ), f"For input shape ({num_tokens}, {hidden_size}), output shape should be ({num_tokens}, {hidden_size}), but got {output.shape}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_mlp_inference_consistency(device, mlp_config, mlp_model):
    """Test inference consistency (multiple runs with same input should produce same output)"""
    num_tokens = 64
    torch.manual_seed(42)
    input_tensor = torch.randn(num_tokens, mlp_config.hidden_size, device=device, dtype=torch.bfloat16)

    outputs = []
    with torch.no_grad():
        for _ in range(3):
            output = mlp_model(input_tensor)
            outputs.append(output)

    # Verify all outputs are the same
    for i in range(1, len(outputs)):
        assert torch.allclose(outputs[0], outputs[i], atol=1e-5), f"Output from run {i+1} is inconsistent with the first run"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_mlp_compiled_consistency(device, mlp_config, mlp_model):
    """Test inference using magi_compiler (verify compiled code consistency)"""
    num_tokens = 128
    input_tensor = torch.randn(num_tokens, mlp_config.hidden_size, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        # First run (may trigger compilation)
        output1 = mlp_model(input_tensor)

        # Second run (should use compiled code)
        output2 = mlp_model(input_tensor)

    # Verify output shape and consistency
    assert output1.shape == (
        num_tokens,
        mlp_config.hidden_size,
    ), f"Output shape should be ({num_tokens}, {mlp_config.hidden_size}), but got {output1.shape}"

    assert torch.allclose(output1, output2, atol=1e-5), "Outputs from two inference runs should be consistent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
