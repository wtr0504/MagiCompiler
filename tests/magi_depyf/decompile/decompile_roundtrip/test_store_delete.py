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

"""Roundtrip tests for STORE/DELETE operations (subscript, attr, name)."""

from copy import deepcopy

from tests.magi_depyf.decompile.decompile_roundtrip.helpers import Point, data_map, replaced_code


def test_STORE_SUBSCR():
    def f():
        p = Point(1, 2)
        p[0] = 99
        return p

    ans = f()
    with replaced_code(f, "STORE_SUBSCR"):
        assert f() == ans


def test_DELETE_SUBSCR():
    def f():
        a = deepcopy(data_map)
        del a[1]
        return a

    ans = f()
    with replaced_code(f, "DELETE_SUBSCR"):
        assert f() == ans


def test_STORE_ATTR():
    def f():
        p = Point(1, 2)
        p.x = 10
        return p

    ans = f()
    with replaced_code(f, "STORE_ATTR"):
        assert f() == ans


def test_DELETE_ATTR():
    def f():
        p = Point(1, 2)
        del p.x
        p.x = 99
        return p

    ans = f()
    with replaced_code(f, "DELETE_ATTR"):
        assert f() == ans


def test_DELETE_NAME():
    def f():
        a = 1
        del a
        a = 2
        return a

    ans = f()
    with replaced_code(f, "DELETE_FAST"):
        assert f() == ans
