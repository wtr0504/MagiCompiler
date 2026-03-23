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

"""Roundtrip tests for function calls (pure — no closures)."""

from tests.magi_depyf.decompile.decompile_roundtrip.helpers import replaced_code


def test_CALL_FUNCTION_NORMAL():
    def f(func):
        a = [1, 2, 3]
        b = {'a': 4}
        return func(1, a, b), func(a=a, b=b)

    def helper(a, b, c=1):
        return (a, b, c)

    ans = f(helper)
    with replaced_code(f, "CALL"):
        assert f(helper) == ans


def test_function_signature():
    def func(a, b, c=1, *, d=2):
        return (a, b, c, d)

    a, b = [1, 2, 3], {'a': 4}
    ans = func(1, a, b, d=5)
    with replaced_code(func, "BUILD_TUPLE"):
        assert func(1, a, b, d=5) == ans


def test_CALL_FUNCTION_EX():
    def f(func):
        a = [1, 2, 3]
        b = {'a': 4}
        return func(*a), func(**b), func(*a, **b)

    def helper(*args, **kwargs):
        return (args, kwargs)

    ans = f(helper)
    with replaced_code(f, "CALL_FUNCTION_EX"):
        assert f(helper) == ans


def test_var_args():
    def func(*args, **kwargs):
        return (args, kwargs)

    a, b = [1, 2, 3], {'a': 4}
    ans = func(1, a, b, d=5)
    with replaced_code(func, "BUILD_TUPLE"):
        assert func(1, a, b, d=5) == ans


def test_complex_signature():
    def func(a, b, *args, **kwargs):
        return (a, b, args, kwargs)

    a, b = [1, 2, 3], {'a': 4}
    ans = func(1, a, b, d=5)
    with replaced_code(func, "BUILD_TUPLE"):
        assert func(1, a, b, d=5) == ans


def test_call_with_kwargs():
    def f():
        return dict(a=1, b=2)

    with replaced_code(f, "KW_NAMES"):
        assert f() == {"a": 1, "b": 2}
