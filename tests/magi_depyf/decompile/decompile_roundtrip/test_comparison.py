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

"""Roundtrip tests for comparison / identity / membership operators."""

from tests.magi_depyf.decompile.decompile_roundtrip.helpers import replaced_code


def test_COMPARE_OP():
    def f():
        return (3 == 3) + (1 < 2) + (2 > 1) + (2 >= 2) + (1 <= 2) + (1 != 2)

    ans = f()
    with replaced_code(f, "COMPARE_OP"):
        assert f() == ans


def test_IS_OP():
    def f():
        return (int is int), (int is not float)

    ans = f()
    with replaced_code(f, "IS_OP"):
        assert f() == ans


def test_CONTAINS_OP():
    def f():
        return (1 in [1, 2, 3]), (5 not in (6, 7, 4))

    ans = f()
    with replaced_code(f, "CONTAINS_OP"):
        assert f() == ans
