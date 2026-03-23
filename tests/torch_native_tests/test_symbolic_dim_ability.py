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

"""Basic ability test for symbolic tensors."""

import pytest
import sympy
import torch
from torch.fx.experimental.symbolic_shapes import ShapeEnv, SymNode
from torch.utils._sympy.functions import IntTrueDiv


@pytest.fixture
def shape_env():
    """Create a fresh ShapeEnv for each test."""
    return ShapeEnv()


def create_symint(env: ShapeEnv, name: str):
    """Create a symbolic integer with the given name."""
    sympy_symbol = sympy.Symbol(name, integer=True)
    sym_node = SymNode(expr=sympy_symbol, shape_env=env, pytype=int, hint=32)
    return torch.SymInt(sym_node)


class TestSymbolicDimAbility:
    """Test suite for symbolic dimension operations."""

    def test_symint_division(self, shape_env):
        """Test symbolic integer division creates IntTrueDiv expression."""
        s1 = create_symint(shape_env, "s1")
        s2 = create_symint(shape_env, "s2")

        div = s1 / s2

        assert str(div) == "IntTrueDiv(s1, s2)"

    def test_symint_addition(self, shape_env):
        """Test symbolic integer addition with constant."""
        s1 = create_symint(shape_env, "s1")

        add = s1 + 5

        assert str(add) == "s1 + 5"

    def test_symint_multiplication(self, shape_env):
        """Test symbolic integer multiplication."""
        s1 = create_symint(shape_env, "s1")
        k = create_symint(shape_env, "k")
        n = create_symint(shape_env, "n")

        mm_flops = s1 * n * k

        assert str(mm_flops) == "k*n*s1"

    def test_sympy_simplification(self, shape_env):
        """Test that sympy can simplify symbolic expressions.

        This test verifies that (k*n*s1) / (n*s1) simplifies to k.
        """
        s1 = create_symint(shape_env, "s1")
        k = create_symint(shape_env, "k")
        n = create_symint(shape_env, "n")

        mm_flops = s1 * n * k
        assert str(mm_flops) == "k*n*s1"

        mm_memory = s1 * n
        assert str(mm_memory) == "n*s1"

        # IntTrueDiv(k*n*s1, n*s1)
        mm_f_m = mm_flops / mm_memory
        assert str(mm_f_m) == "IntTrueDiv(k*n*s1, n*s1)"

        # Get the raw expression and simplify
        raw_expr = mm_f_m.node.expr
        simplified_expr_raw = sympy.simplify(raw_expr)
        assert str(simplified_expr_raw) == "IntTrueDiv(k*n*s1, n*s1)"

        simplified_expr_sympy = sympy.simplify(raw_expr)
        # The expression (k*n*s1) / (n*s1) should simplify to k
        assert str(simplified_expr_sympy) == "IntTrueDiv(k*n*s1, n*s1)"

        # Replace IntTrueDiv with standard division for sympy simplification
        simplified_expr_replace = raw_expr.replace(IntTrueDiv, lambda x, y: x / y)
        assert str(simplified_expr_replace) == "k"
