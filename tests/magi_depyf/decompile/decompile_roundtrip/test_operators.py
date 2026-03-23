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

"""Roundtrip tests for unary, binary, and in-place operators."""

from tests.magi_depyf.decompile.decompile_roundtrip.helpers import Point, point, replaced_code

# ======================================================================
# 1. Unary operators
# ======================================================================


def test_UNARY_POSITIVE():
    def f():
        x = 1
        return +x

    ans = f()
    with replaced_code(f, "CALL_INTRINSIC_1"):
        assert f() == ans


def test_UNARY_NEGATIVE():
    def f():
        x = 1
        return -x

    ans = f()
    with replaced_code(f, "UNARY_NEGATIVE"):
        assert f() == ans


def test_UNARY_NOT():
    def f():
        x = 1
        return not x

    ans = f()
    with replaced_code(f, "UNARY_NOT"):
        assert f() == ans


def test_UNARY_INVERT():
    def f():
        x = 1
        return ~x

    ans = f()
    with replaced_code(f, "UNARY_INVERT"):
        assert f() == ans


# ======================================================================
# 2. Binary operators
# ======================================================================


def test_BINARY_POWER():
    def f():
        a, b = 2, 3
        return (a**b) ** a

    ans = f()
    with replaced_code(f, ("BINARY_OP", "**")):
        assert f() == ans


def test_BINARY_MULTIPLY():
    def f():
        a, b = 2, 3
        return a * b * a

    ans = f()
    with replaced_code(f, ("BINARY_OP", "*")):
        assert f() == ans


def test_BINARY_MATRIX_MULTIPLY():
    def f():
        return point @ point

    ans = f()
    with replaced_code(f, ("BINARY_OP", "@")):
        assert f() == ans


def test_BINARY_FLOOR_DIVIDE():
    def f():
        a, b = 7, 3
        return (a // b) // a

    ans = f()
    with replaced_code(f, ("BINARY_OP", "//")):
        assert f() == ans


def test_BINARY_TRUE_DIVIDE():
    def f():
        a, b = 7, 3
        return (a / b) / a

    ans = f()
    with replaced_code(f, ("BINARY_OP", "/")):
        assert f() == ans


def test_BINARY_MODULO():
    def f():
        a, b = 10, 3
        return (a % b) % a

    ans = f()
    with replaced_code(f, ("BINARY_OP", "%")):
        assert f() == ans


def test_BINARY_ADD():
    def f():
        a, b = 2, 3
        return (a + b) + a

    ans = f()
    with replaced_code(f, ("BINARY_OP", "+")):
        assert f() == ans


def test_BINARY_SUBTRACT():
    def f():
        a, b = 5, 3
        return (a - b) - a

    ans = f()
    with replaced_code(f, ("BINARY_OP", "-")):
        assert f() == ans


def test_BINARY_SUBSCR():
    def f():
        a = (10, 20, 30)
        return a[1]

    ans = f()
    with replaced_code(f, "BINARY_SUBSCR"):
        assert f() == ans


def test_BINARY_LSHIFT():
    def f():
        a, b = 2, 3
        return (a << b) << 1

    ans = f()
    with replaced_code(f, ("BINARY_OP", "<<")):
        assert f() == ans


def test_BINARY_RSHIFT():
    def f():
        a, b = 16, 2
        return (a >> b) >> 1

    ans = f()
    with replaced_code(f, ("BINARY_OP", ">>")):
        assert f() == ans


def test_BINARY_AND():
    def f():
        a, b = 0b1100, 0b1010
        return (a & b) & 0b1111

    ans = f()
    with replaced_code(f, ("BINARY_OP", "&")):
        assert f() == ans


def test_BINARY_XOR():
    def f():
        a, b = 0b1100, 0b1010
        return (a ^ b) ^ 0b0001

    ans = f()
    with replaced_code(f, ("BINARY_OP", "^")):
        assert f() == ans


def test_BINARY_OR():
    def f():
        a, b = 0b1100, 0b1010
        return (a | b) | 0b0001

    ans = f()
    with replaced_code(f, ("BINARY_OP", "|")):
        assert f() == ans


# ======================================================================
# 3. In-place operators
# ======================================================================


def test_INPLACE_POWER():
    def f():
        a = 2
        a **= 3
        return a

    ans = f()
    with replaced_code(f, ("BINARY_OP", "**=")):
        assert f() == ans


def test_INPLACE_MULTIPLY():
    def f():
        a = 2
        a *= 3
        return a

    ans = f()
    with replaced_code(f, ("BINARY_OP", "*=")):
        assert f() == ans


def test_INPLACE_MATRIX_MULTIPLY():
    def f():
        p = Point(1, 2)
        p @= p
        return p

    ans = f()
    with replaced_code(f, ("BINARY_OP", "@=")):
        assert f() == ans


def test_INPLACE_FLOOR_DIVIDE():
    def f():
        a = 7
        a //= 3
        return a

    ans = f()
    with replaced_code(f, ("BINARY_OP", "//=")):
        assert f() == ans


def test_INPLACE_TRUE_DIVIDE():
    def f():
        a = 7
        a /= 2
        return a

    ans = f()
    with replaced_code(f, ("BINARY_OP", "/=")):
        assert f() == ans


def test_INPLACE_MODULO():
    def f():
        a = 10
        a %= 3
        return a

    ans = f()
    with replaced_code(f, ("BINARY_OP", "%=")):
        assert f() == ans


def test_INPLACE_ADD():
    def f():
        a = 2
        a += 3
        return a

    ans = f()
    with replaced_code(f, ("BINARY_OP", "+=")):
        assert f() == ans


def test_INPLACE_SUBTRACT():
    def f():
        a = 5
        a -= 3
        return a

    ans = f()
    with replaced_code(f, ("BINARY_OP", "-=")):
        assert f() == ans


def test_INPLACE_LSHIFT():
    def f():
        a = 1
        a <<= 3
        return a

    ans = f()
    with replaced_code(f, ("BINARY_OP", "<<=")):
        assert f() == ans


def test_INPLACE_RSHIFT():
    def f():
        a = 16
        a >>= 2
        return a

    ans = f()
    with replaced_code(f, ("BINARY_OP", ">>=")):
        assert f() == ans


def test_INPLACE_AND():
    def f():
        a = 0b1111
        a &= 0b1010
        return a

    ans = f()
    with replaced_code(f, ("BINARY_OP", "&=")):
        assert f() == ans


def test_INPLACE_XOR():
    def f():
        a = 0b1100
        a ^= 0b1010
        return a

    ans = f()
    with replaced_code(f, ("BINARY_OP", "^=")):
        assert f() == ans


def test_INPLACE_OR():
    def f():
        a = 0b1100
        a |= 0b0011
        return a

    ans = f()
    with replaced_code(f, ("BINARY_OP", "|=")):
        assert f() == ans
