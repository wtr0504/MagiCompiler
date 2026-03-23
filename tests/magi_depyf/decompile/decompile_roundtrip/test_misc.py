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

"""Roundtrip tests for miscellaneous opcodes:
LOAD_ATTR, IMPORT, MAKE_FUNCTION, SWAP, FORMAT, SLICE,
constants, STORE_GLOBAL, COPY, RETURN_CONST, etc.
"""

from magi_compiler.magi_depyf import decompile
from tests.magi_depyf.decompile.decompile_roundtrip.helpers import Point, replaced_code

# ======================================================================
# LOAD_ATTR / IMPORT
# ======================================================================


def test_LOAD_ATTR():
    def f():
        p = Point(1, 2)
        return p.x

    ans = f()
    with replaced_code(f, "LOAD_ATTR"):
        assert f() == ans


def test_IMPORT_NAME():
    def f():
        import functools
        from math import sqrt

        return functools.partial(sqrt, 0.3)()

    ans = f()
    with replaced_code(f, "IMPORT_NAME"):
        assert f() == ans


# ======================================================================
# MAKE_FUNCTION / nested functions
# ======================================================================


def test_MAKE_FUNCTION():
    def f(a):
        def g(b=3):
            return a + b

        return g(2)

    ans = [f(i) for i in range(10)]
    with replaced_code(f, "MAKE_FUNCTION"):
        assert [f(i) for i in range(10)] == ans


def test_nested_function():
    def f():
        def inner(x):
            return x * 2

        return inner(21)

    with replaced_code(f, "MAKE_FUNCTION"):
        assert f() == 42


# ======================================================================
# Swap / Rotate
# ======================================================================


def test_ROT_TWO():
    def f():
        a, b = 1, 2
        a, b = b, a
        return a, b

    ans = f()
    with replaced_code(f, "UNPACK_SEQUENCE"):
        assert f() == ans


def test_ROT_MULTI():
    def f():
        a, b, c, d = 1, 2, 3, 4
        a, b, c, d = d, c, b, a
        return a

    ans = f()
    with replaced_code(f, "UNPACK_SEQUENCE"):
        assert f() == ans


def test_SWAP_triple():
    def f():
        a, b, c = 1, 2, 3
        c, b, a = a, b, c
        return a, b, c

    ans = f()
    with replaced_code(f, "UNPACK_SEQUENCE"):
        assert f() == ans


# ======================================================================
# Format string / BUILD_STRING
# ======================================================================


def test_FORMAT_VALUE():
    def f():
        a, b, c = 1, 2, 3
        return f"{a} {b!r} {b!s} {b!a} {c:.2f}"

    ans = f()
    with replaced_code(f, "FORMAT_VALUE"):
        assert f() == ans


# ======================================================================
# BUILD_SLICE / BINARY_SLICE / STORE_SLICE
# ======================================================================


def test_BUILD_SLICE():
    def f():
        a = [1, 2, 3, 4, 5]
        return a[:] + a[1:] + a[:3] + a[1:3] + a[::-1]

    ans = f()
    with replaced_code(f, "BUILD_SLICE"):
        assert f() == ans


def test_BINARY_SLICE():
    def f():
        a = [10, 20, 30, 40, 50]
        return a[1:3], a[:2], a[3:]

    ans = f()
    with replaced_code(f, "BINARY_SLICE"):
        assert f() == ans


def test_STORE_SLICE():
    def f():
        a = [1, 2, 3, 4, 5]
        a[1:3] = [20, 30]
        return a

    ans = f()
    with replaced_code(f, "STORE_SLICE"):
        assert f() == ans


# ======================================================================
# Constants (various types)
# ======================================================================


def test_constants():
    def f():
        return (1, 2.5, "hello", True, None, b"bytes", (1, 2))

    with replaced_code(f, "RETURN_CONST"):
        assert f() == (1, 2.5, "hello", True, None, b"bytes", (1, 2))


# ======================================================================
# GET_LEN
# ======================================================================


def test_GET_LEN():
    def f():
        return len((1, 2, 3))

    ans = f()
    with replaced_code(f, ("LOAD_GLOBAL", "len")):
        assert f() == ans


# ======================================================================
# STORE_GLOBAL
# ======================================================================


def test_STORE_GLOBAL():
    def f():
        global len
        len = 1
        return len

    with replaced_code(f, "STORE_GLOBAL"):
        global len
        original_len = len
        f()
        assert len == 1
        len = original_len


# ======================================================================
# Class method with __class__
# ======================================================================


class A:
    def f(self):
        return __class__


def test_class_method():
    """__class__ is an implicit freevar; verify decompilation produces valid text."""
    src = decompile(A.f.__code__)
    assert "def f(self):" in src
    assert "__class__" in src


# ======================================================================
# COPY / SWAP patterns (Python 3.11+)
# ======================================================================


def test_COPY_simple():
    def f():
        a = [1, 2, 3]
        a[0] = a[1] = 99
        return a

    ans = f()
    with replaced_code(f, "COPY"):
        assert f() == ans


# ======================================================================
# LIST_APPEND (in comprehensions)
# ======================================================================


def test_LIST_APPEND_in_comp():
    def f(n):
        return [x * 2 for x in range(n) if x % 2 == 0]

    ans = f(10)
    with replaced_code(f, "LIST_APPEND"):
        assert f(10) == ans


# ======================================================================
# Mixed complex expressions
# ======================================================================


def test_complex_expression():
    def f(a, b):
        return (a + b) * (a - b) // (b + 1)

    ans = [f(a, b) for a in range(1, 5) for b in range(1, 5)]
    with replaced_code(f, "BINARY_OP"):
        assert [f(a, b) for a in range(1, 5) for b in range(1, 5)] == ans


def test_chained_method_calls():
    def f():
        return "  hello world  ".strip().upper().replace("O", "0")

    ans = f()
    with replaced_code(f, "LOAD_ATTR"):
        assert f() == ans


# ======================================================================
# RETURN_CONST (Python 3.12+)
# ======================================================================


def test_RETURN_CONST_none():
    def f():
        pass

    ans = f()
    with replaced_code(f, "RETURN_CONST"):
        assert f() == ans


def test_RETURN_CONST_string():
    def f(x):
        if x > 0:
            return "positive"
        else:
            return "negative"

    ans = [f(i) for i in [-1, 0, 1]]
    with replaced_code(f, "RETURN_CONST"):
        assert [f(i) for i in [-1, 0, 1]] == ans


def test_RETURN_CONST_number():
    def f(x):
        if x:
            return 42
        return 0

    ans = [f(b) for b in [True, False]]
    with replaced_code(f, "RETURN_CONST"):
        assert [f(b) for b in [True, False]] == ans


# ======================================================================
# LOAD_ASSERTION_ERROR
# ======================================================================


def test_LOAD_ASSERTION_ERROR():
    scope = {}
    exec(compile("def f(x):\n    assert x > 0, 'must be positive'\n    return x", "<test>", "exec"), scope)
    f = scope["f"]
    src = decompile(f)
    exec(src, scope)
    g = scope["f"]
    assert g(5) == 5


# ======================================================================
# Import patterns
# ======================================================================


def test_import_dotted():
    def f():
        import os.path

        return os.path.sep

    ans = f()
    with replaced_code(f, "IMPORT_NAME"):
        assert f() == ans


def test_import_from():
    def f():
        from math import sqrt

        return sqrt(4)

    ans = f()
    with replaced_code(f, "IMPORT_FROM"):
        assert f() == ans


# ======================================================================
# PUSH_NULL + LOAD_GLOBAL pattern (3.11+)
# ======================================================================


def test_global_function_call():
    def f():
        return len([1, 2, 3])

    ans = f()
    with replaced_code(f, ("LOAD_GLOBAL", "len")):
        assert f() == ans


# ======================================================================
# LOAD_ATTR normal
# ======================================================================


def test_LOAD_ATTR_normal():
    def f():
        import math

        return math.pi

    ans = f()
    with replaced_code(f, "LOAD_ATTR"):
        assert f() == ans
