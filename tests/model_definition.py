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

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from magi_compiler import magi_compile


@dataclass
class MLPConfig:
    """Configuration for the MLP module"""

    hidden_size: int
    intermediate_size: int
    params_dtype: torch.dtype = torch.bfloat16


@dataclass
class RMSNormConfig:
    """Configuration for the RMSNorm module"""

    hidden_size: int
    eps: float = 1e-6


class RMSNorm(nn.Module):
    """Simple RMSNorm implementation"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = x.to(self.weight.dtype) * self.weight
        return x.to(input_dtype)


@magi_compile(dynamic_arg_dims={"x": 0})
class MLP(torch.nn.Module):
    """MLP module with traditional architecture (up-projection, activation, and down-projection)"""

    config: MLPConfig

    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config
        self.pre_norm = RMSNorm(config.hidden_size)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, dtype=config.params_dtype)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False, dtype=config.params_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP module.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            output (torch.Tensor): Output tensor

        Shape:
            - x: (num_tokens, hidden_size)
            - output: (num_tokens, hidden_size)
        """
        # Pre-normalization
        x = self.pre_norm(x).to(torch.bfloat16)
        # Up-projection
        x = self.up_proj(x).to(torch.float32)
        # Activation (SiLU)
        x = F.silu(x).to(torch.bfloat16)
        # Down-projection
        x = self.down_proj(x).to(torch.float32)
        return x


@magi_compile(dynamic_arg_dims={"x": 0})
class RMSNormModule(torch.nn.Module):
    """Compiled RMSNorm module for testing"""

    config: RMSNormConfig

    def __init__(self, config: RMSNormConfig):
        super().__init__()
        self.config = config
        self.norm = RMSNorm(config.hidden_size, eps=config.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the RMSNorm module.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            output (torch.Tensor): Normalized output tensor

        Shape:
            - x: (num_tokens, hidden_size)
            - output: (num_tokens, hidden_size)
        """
        return self.norm(x)


def create_rms_norm_model(config: RMSNormConfig, device: torch.device) -> RMSNormModule:
    """Create RMSNorm model

    Args:
        config: RMSNorm configuration
        device: Target device

    Returns:
        model: Created RMSNorm model
    """
    model = RMSNormModule(config).to(device)
    return model


def create_mlp_model(config: MLPConfig, device: torch.device) -> MLP:
    """Create MLP model

    Args:
        config: MLP configuration
        device: Target device

    Returns:
        model: Created MLP model
    """
    model = MLP(config).to(device)
    return model


def create_mlp_model_with_initial_params(config: MLPConfig, device: torch.device) -> tuple[MLP, list[torch.Tensor]]:
    """Create MLP model and return model with initial parameter snapshot

    Args:
        config: MLP configuration
        device: Target device

    Returns:
        model: Created MLP model
        initial_params: Initial snapshot of model parameters for verifying parameter updates
    """
    model = MLP(config).to(device)
    initial_params = [p.clone().detach() for p in model.parameters()]
    return model, initial_params
