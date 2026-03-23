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

"""Introspector API tests: build_function_info, guard tree, writer, debug_compiled."""

from __future__ import annotations

import os
import shutil
import tempfile

import pytest

torch = pytest.importorskip("torch")

from magi_compiler.magi_depyf.inspect import debug_compiled, write_function
from magi_compiler.magi_depyf.inspect.introspect import Introspector


def _reset():
    torch._dynamo.reset()


class TestBuildFunctionInfo:
    def test_basic(self):
        _reset()

        def fn(x):
            return x + 1

        compiled = torch.compile(fn, backend="eager")
        compiled(torch.randn(4))

        info = Introspector.build_function_info(fn)
        assert info.name == "fn"
        assert len(info.entries) >= 1
        assert info.entries[0].decompiled_source
        assert "def" in info.entries[0].decompiled_source

    def test_guard_tree(self):
        _reset()

        def fn(x):
            return x * 2

        compiled = torch.compile(fn, backend="eager")
        compiled(torch.randn(3))

        info = Introspector.build_function_info(fn)
        entry = info.entries[0]
        assert entry.guard is not None
        assert entry.guard.tree is not None
        assert entry.guard.tree.type_name == "RootGuardManager"

    def test_format_output(self):
        _reset()

        def fn(x):
            return x + 1

        compiled = torch.compile(fn, backend="eager")
        compiled(torch.randn(3))

        info = Introspector.build_function_info(fn)
        text = info.format()
        assert "fn" in text
        assert "entry" in text


class TestWriter:
    def test_write_files(self):
        _reset()
        tmpdir = tempfile.mkdtemp(prefix="magi_depyf_test_")
        try:

            def fn(x, y):
                return x + y

            compiled = torch.compile(fn, backend="eager")
            compiled(torch.randn(3), torch.randn(3))

            info = Introspector.build_function_info(fn)
            root = write_function(info, tmpdir)

            assert root.exists()
            assert (root / "overview.md").exists()
            assert (root / "entry_0").exists()
            assert (root / "entry_0" / "decompiled_code.py").exists()
            assert (root / "entry_0" / "guards.txt").exists()

            overview = (root / "overview.md").read_text()
            assert "fn" in overview
            assert "entry\\[0\\]" in overview
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_debug_compiled_convenience(self):
        _reset()
        tmpdir = tempfile.mkdtemp(prefix="magi_depyf_test_")
        try:

            def fn(x):
                return x.sum()

            compiled = torch.compile(fn, backend="eager")
            compiled(torch.randn(5))

            info = debug_compiled(fn, output_dir=tmpdir)
            assert info.name == "fn"
            assert os.path.exists(os.path.join(tmpdir, "fn"))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
