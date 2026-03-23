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

"""Roundtrip tests for control flow: if/elif/else, for loops."""

from magi_compiler.magi_depyf import decompile
from tests.magi_depyf.decompile.decompile_roundtrip.helpers import replaced_code

# ======================================================================
# if / elif / else
# ======================================================================


def test_IF():
    def f(a):
        if a == 0:
            return 0
        elif a == 1:
            return 1
        else:
            return 2

    ans = [f(i) for i in range(10)]
    with replaced_code(f, "COMPARE_OP"):
        assert [f(i) for i in range(10)] == ans


def test_compound_IF_and():
    def f(a, b):
        c = 1
        if a > 0 and b > 1:
            c += 1
        else:
            c += 2
        c += 3
        return c

    ans = [f(a, b) for a in range(-3, 3) for b in range(-3, 3)]
    with replaced_code(f, "POP_JUMP_IF_FALSE"):
        assert [f(a, b) for a in range(-3, 3) for b in range(-3, 3)] == ans


def test_compound_IF_or():
    def f(a, b):
        c = 1
        if a > 0 or b > 1:
            c += 1
        else:
            c += 2
        c += 3
        return c

    ans = [f(a, b) for a in range(-3, 3) for b in range(-3, 3)]
    with replaced_code(f, "POP_JUMP_IF_FALSE"):
        assert [f(a, b) for a in range(-3, 3) for b in range(-3, 3)] == ans


def test_IF_NONE():
    def f(a):
        if a is None:
            return 0
        elif a is not None:
            return 1

    ans = [f(i) for i in range(10)]
    with replaced_code(f, "POP_JUMP_IF_NONE"):
        assert [f(i) for i in range(10)] == ans


def test_ternary():
    def f(output_hidden_states):
        () if output_hidden_states else None

    ans = [f(i) for i in range(2)]
    with replaced_code(f, "POP_JUMP_IF_FALSE"):
        assert [f(i) for i in range(2)] == ans


def test_shortcircuit():
    def f(a, b):
        if a > 0 and b > 0:
            return a + b
        elif a > 1 or b > 2:
            return a - b
        else:
            return 2

    scope = {}
    exec(decompile(f), scope)
    for a in [-1, 0, 1, 2]:
        for b in [-1, 0, 1, 2]:
            assert f(a, b) == scope['f'](a, b)


def test_IF_return_in_both_branches():
    def f(x, y):
        if x:
            return 42
        else:
            return [x, y]

    ans = [f(x, y) for x in [True, False] for y in [1, 2]]
    with replaced_code(f, "POP_JUMP_IF_FALSE"):
        assert [f(x, y) for x in [True, False] for y in [1, 2]] == ans


def test_IF_nested():
    def f(a):
        if a > 10:
            return "big"
        elif a > 5:
            return "medium"
        elif a > 0:
            return "small"
        else:
            return "negative"

    ans = [f(i) for i in [-1, 0, 3, 7, 15]]
    with replaced_code(f, "COMPARE_OP"):
        assert [f(i) for i in [-1, 0, 3, 7, 15]] == ans


def test_IF_assign_then_use():
    def f(flag):
        if flag:
            x = 10
        else:
            x = 20
        return x + 1

    ans = [f(b) for b in [True, False]]
    with replaced_code(f, "POP_JUMP_IF_FALSE"):
        assert [f(b) for b in [True, False]] == ans


# ======================================================================
# For loop
# ======================================================================


def test_simple_for():
    def f(a):
        for i in range(5):
            a += i
        return a

    ans = [f(i) for i in range(10)]
    with replaced_code(f, "FOR_ITER"):
        assert [f(i) for i in range(10)] == ans


def test_for_with_break():
    def f(items):
        for x in items:
            if x < 0:
                break
        return x

    with replaced_code(f, "FOR_ITER"):
        assert f([1, 2, -1, 3]) == -1


def test_for_with_continue():
    def f(n):
        total = 0
        for i in range(n):
            if i % 2 == 0:
                continue
            total += i
        return total

    ans = f(10)
    with replaced_code(f, "FOR_ITER"):
        assert f(10) == ans


def test_for_nested():
    def f(n):
        total = 0
        for i in range(n):
            for j in range(i):
                total += j
        return total

    ans = f(5)
    with replaced_code(f, "FOR_ITER"):
        assert f(5) == ans


def test_for_with_enumerate():
    def f():
        total = 0
        for i, v in enumerate([10, 20, 30]):
            total += i * v
        return total

    ans = f()
    with replaced_code(f, "FOR_ITER"):
        assert f() == ans


def test_for_dict_items():
    def f():
        d = {"a": 1, "b": 2, "c": 3}
        total = 0
        for k, v in d.items():
            total += v
        return total

    ans = f()
    with replaced_code(f, "FOR_ITER"):
        assert f() == ans


def test_for_loop_accumulate():
    def f():
        total = 0
        for x in [10, 20, 30]:
            total += x
        return total

    ans = f()
    with replaced_code(f, "FOR_ITER"):
        assert f() == ans
