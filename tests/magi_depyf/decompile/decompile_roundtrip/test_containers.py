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

"""Roundtrip tests for containers (tuple/list/set/dict), unpack, and comprehensions."""

from tests.magi_depyf.decompile.decompile_roundtrip.helpers import replaced_code

# ======================================================================
# Containers: tuple / list / set / dict
# ======================================================================


def test_BUILD_TUPLE():
    def f():
        a, b = 1, 2
        return (a, b), (a,)

    ans = f()
    with replaced_code(f, "BUILD_TUPLE"):
        assert f() == ans


def test_BUILD_LIST():
    def f():
        a, b = 1, 2
        return [a, b], [a]

    ans = f()
    with replaced_code(f, "BUILD_LIST"):
        assert f() == ans


def test_BUILD_SET():
    def f():
        a, b = 1, 2
        return {a, b}, {a}

    ans = f()
    with replaced_code(f, "BUILD_SET"):
        assert f() == ans


def test_BUILD_MAP():
    def f():
        a, b = 1, 2
        return {a: 1, 2: 3}, {b: a}

    ans = f()
    with replaced_code(f, "BUILD_MAP"):
        assert f() == ans


def test_BUILD_CONST_KEY_MAP():
    def f():
        return {5: 1, 2: 3}

    ans = f()
    with replaced_code(f, "BUILD_CONST_KEY_MAP"):
        assert f() == ans


def test_BUILD_CONST_KEY_MAP_string_keys():
    def f():
        return {"hello": 1, "world": 2}

    ans = f()
    with replaced_code(f, "BUILD_CONST_KEY_MAP"):
        assert f() == ans


def test_LIST_EXTEND():
    def f():
        return [1, 2, 3]

    ans = f()
    with replaced_code(f, "LIST_EXTEND"):
        assert f() == ans


def test_SET_UPDATE():
    def f():
        return {1, 2, 3}

    ans = f()
    with replaced_code(f, "SET_UPDATE"):
        assert f() == ans


def test_DICT_UPDATE():
    def f():
        a = {1: 2}
        b = {'a': 4}
        return {**a, **b}

    ans = f()
    with replaced_code(f, "DICT_UPDATE"):
        assert f() == ans


def test_DICT_MERGE():
    def f():
        a = {1: 2}
        b = {'a': 4}
        a.update(**b)
        return a

    ans = f()
    with replaced_code(f, "DICT_MERGE"):
        assert f() == ans


# ======================================================================
# Unpack
# ======================================================================


def test_UNPACK_SEQUENCE():
    def f():
        a, b = (1, 2)
        return a

    ans = f()
    with replaced_code(f, "UNPACK_SEQUENCE"):
        assert f() == ans


def test_UNPACK_SEQUENCE_one():
    def f():
        (a,) = (1,)
        return a

    ans = f()
    with replaced_code(f, "UNPACK_SEQUENCE"):
        assert f() == ans


def test_UNPACK_EX():
    def f():
        a, *b = (1, 2, 3)
        return b

    ans = f()
    with replaced_code(f, "UNPACK_EX"):
        assert f() == ans


# ======================================================================
# Comprehensions
# ======================================================================


def test_LIST_COMP():
    def f(a):
        return [i**2 for i in range(a)]

    ans = [f(i) for i in range(10)]
    with replaced_code(f, "LIST_APPEND"):
        assert [f(i) for i in range(10)] == ans


def test_SET_COMP():
    def f(a):
        return {i**2 for i in range(a)}

    ans = [f(i) for i in range(10)]
    with replaced_code(f, "SET_ADD"):
        assert [f(i) for i in range(10)] == ans


def test_MAP_COMP():
    def f(a):
        return {i: i**2 for i in range(a)}

    ans = [f(i) for i in range(10)]
    with replaced_code(f, "MAP_ADD"):
        assert [f(i) for i in range(10)] == ans


def test_NESTED_COMP():
    def f(a):
        return [{x: {_ for _ in range(x)} for x in range(i)} for i in range(a)]

    ans = [f(i) for i in range(5)]
    with replaced_code(f, "LIST_APPEND"):
        assert [f(i) for i in range(5)] == ans
