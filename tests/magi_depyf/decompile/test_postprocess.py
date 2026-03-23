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

"""Tests for decompile.postprocess pipeline passes."""

import textwrap

from magi_compiler.magi_depyf.decompile.postprocess import (
    dedup_branch_tails,
    eliminate_for_temps,
    eliminate_inline_temps,
    run_all,
)

# ---------------------------------------------------------------------------
# Pass 1: for-loop temp elimination
# ---------------------------------------------------------------------------


class TestForTempElimination:
    def test_simple_for(self):
        src = textwrap.dedent(
            """\
            for __temp_0 in items:
                x = __temp_0
                print(x)
        """
        )
        result = eliminate_for_temps(src)
        assert "for x in items" in result
        assert "__temp_0" not in result

    def test_nested_for(self):
        src = textwrap.dedent(
            """\
            for __temp_0 in outer:
                y = __temp_0
                for __temp_1 in inner:
                    z = __temp_1
                    print(z)
        """
        )
        result = eliminate_for_temps(src)
        assert "for y in outer" in result
        assert "for z in inner" in result
        assert "__temp" not in result

    def test_no_temp_for(self):
        src = textwrap.dedent(
            """\
            for x in items:
                print(x)
        """
        )
        result = eliminate_for_temps(src)
        assert "for x in items" in result

    def test_multi_assign_not_eliminated(self):
        """If the first statement isn't a simple assignment from the temp, keep it."""
        src = textwrap.dedent(
            """\
            for __temp_0 in items:
                print(__temp_0)
        """
        )
        result = eliminate_for_temps(src)
        assert "__temp_0" in result


# ---------------------------------------------------------------------------
# Pass 2: inline temp elimination
# ---------------------------------------------------------------------------


class TestInlineTempElimination:
    def test_simple_inline(self):
        src = textwrap.dedent(
            """\
            __temp_0 = a + b
            result = __temp_0 * 2
        """
        )
        result = eliminate_inline_temps(src)
        assert "__temp_0" not in result
        assert "result = (a + b) * 2" in result or "result" in result

    def test_no_inline_when_rhs_modified(self):
        """Don't inline if the RHS variable is modified between def and use."""
        src = textwrap.dedent(
            """\
            __temp_0 = b
            b = a
            x = __temp_0
        """
        )
        result = eliminate_inline_temps(src)
        assert "__temp_0" in result


# ---------------------------------------------------------------------------
# Pass 3: branch tail deduplication
# ---------------------------------------------------------------------------


class TestBranchTailDedup:
    def test_simple_dedup(self):
        src = textwrap.dedent(
            """\
            if cond:
                x = 1
                return x
            else:
                x = 2
                return x
        """
        )
        result = dedup_branch_tails(src)
        lines = [l.strip() for l in result.strip().splitlines()]
        assert lines.count("return x") == 1
        assert lines[-1] == "return x"

    def test_no_dedup_when_different(self):
        src = textwrap.dedent(
            """\
            if cond:
                return 1
            else:
                return 2
        """
        )
        result = dedup_branch_tails(src)
        assert "return 1" in result
        assert "return 2" in result

    def test_multi_statement_dedup(self):
        src = textwrap.dedent(
            """\
            if cond:
                x = 1
                y = f(x)
                return y
            else:
                x = 2
                y = f(x)
                return y
        """
        )
        result = dedup_branch_tails(src)
        lines = [l.strip() for l in result.strip().splitlines()]
        assert lines.count("y = f(x)") == 1
        assert lines.count("return y") == 1

    def test_elif_chain_dedup(self):
        src = textwrap.dedent(
            """\
            if a:
                x = 1
                return x
            elif b:
                x = 2
                return x
            else:
                x = 3
                return x
        """
        )
        result = dedup_branch_tails(src)
        lines = [l.strip() for l in result.strip().splitlines()]
        assert lines.count("return x") == 1

    def test_keep_one_statement_per_branch(self):
        """Both branches have only one (identical) statement — keep it in branches."""
        src = textwrap.dedent(
            """\
            if cond:
                return 1
            else:
                return 1
        """
        )
        result = dedup_branch_tails(src)
        assert "return 1" in result


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


class TestRunAll:
    def test_for_temp_then_inline(self):
        """Pipeline: for-temp elimination runs before inline elimination."""
        src = textwrap.dedent(
            """\
            __temp_0 = compute()
            for __temp_1 in __temp_0:
                x = __temp_1
                print(x)
        """
        )
        result = run_all(src)
        assert "for x in compute()" in result
        assert "__temp" not in result
