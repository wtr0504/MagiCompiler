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

"""End-to-end inspect pipeline test: simulated torch.compile example.

Verifies the full pipeline by compiling a function (with and without
graph breaks) and checking:
  1. Hook is called and CaptureResult is produced
  2. fn and __resume functions exist and are decompiled
  3. Guards are obtained from CacheEntry
  4. Backend compiled_fn info is extracted
"""

from __future__ import annotations

import shutil
import tempfile

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from magi_compiler.magi_depyf.inspect import CaptureSession, write_function
from magi_compiler.magi_depyf.inspect.introspect import Introspector


def _reset():
    torch._dynamo.reset()


class TestSimpleFunction:
    """No graph break — single CacheEntry, single compiled_fn."""

    def setup_method(self):
        _reset()

        def fn(x, y):
            return x + y

        self.fn = fn
        compiled = torch.compile(fn, backend="eager")
        with CaptureSession() as session:
            compiled(torch.randn(4), torch.randn(4))
        self.session = session
        self.info = Introspector.build_function_info(fn)

    def test_hook_called(self):
        assert len(self.session.results) >= 1
        r = self.session.results[0]
        assert r.function_name == "fn"
        assert r.fn_globals is not None

    def test_fn_decompiled(self):
        assert len(self.info.entries) >= 1
        entry = self.info.entries[0]
        assert entry.decompiled_source
        assert "def" in entry.decompiled_source
        assert "__compiled_fn" in entry.decompiled_source

    def test_no_resume(self):
        entry = self.info.entries[0]
        assert len(entry.resume_fns) == 0

    def test_guards_obtained(self):
        entry = self.info.entries[0]
        assert entry.guard is not None
        assert entry.guard.tree is not None
        assert entry.guard.tree.type_name == "RootGuardManager"
        assert len(entry.guard.tree.leaf_guards) > 0

    def test_compiled_fn_obtained(self):
        entry = self.info.entries[0]
        assert len(entry.compiled_fns) >= 1
        cf = entry.compiled_fns[0]
        assert cf.name.startswith("__compiled_fn")
        assert cf.backend in ("eager", "inductor")
        assert cf.readable_code is not None or cf.graph_module_code is not None


class TestGraphBreakFunction:
    """print() causes graph break — produces resume functions."""

    def setup_method(self):
        _reset()

        def fn(x, y):
            z = x + y
            print("[test] z =", z.shape)
            return z * 2

        self.fn = fn
        compiled = torch.compile(fn, backend="eager")
        with CaptureSession() as session:
            compiled(torch.randn(4), torch.randn(4))
        self.session = session
        self.info = Introspector.build_function_info(fn)

    def test_hook_called_multiple_times(self):
        assert len(self.session.results) >= 2, f"Graph break should produce >=2 hook events, got {len(self.session.results)}"

    def test_resume_functions_exist(self):
        entry = self.info.entries[0]
        assert len(entry.resume_fns) >= 1, "Graph break should produce resume functions"

    def test_resume_decompiled(self):
        entry = self.info.entries[0]
        for rf in entry.resume_fns:
            assert rf.name.startswith("__resume")
            assert len(rf.entries) >= 1
            re = rf.entries[0]
            assert re.decompiled_source
            assert "def" in re.decompiled_source

    def test_resume_has_compiled_fn(self):
        entry = self.info.entries[0]
        for rf in entry.resume_fns:
            for re in rf.entries:
                assert len(re.compiled_fns) >= 1, f"Resume entry for {rf.name} should have compiled_fn"

    def test_resume_has_guards(self):
        entry = self.info.entries[0]
        for rf in entry.resume_fns:
            for re in rf.entries:
                assert re.guard is not None, f"Resume entry for {rf.name} should have guard info"


class TestModuleWithNoGrad:
    """nn.Module wrapped with torch.no_grad() — common pattern."""

    def setup_method(self):
        _reset()

        layer = nn.Linear(16, 8)
        layer.eval()

        def fn(x):
            with torch.no_grad():
                return layer(x)

        fn.__name__ = "linear_forward"
        self.fn = fn
        compiled = torch.compile(fn, backend="eager")
        with CaptureSession() as session:
            compiled(torch.randn(2, 16))
        self.session = session
        self.info = Introspector.build_function_info(fn)

    def test_hook_called(self):
        assert len(self.session.results) >= 1

    def test_fn_decompiled(self):
        entry = self.info.entries[0]
        assert "def" in entry.decompiled_source

    def test_compiled_fn_has_graph_module(self):
        entry = self.info.entries[0]
        for cf in entry.compiled_fns:
            assert cf.readable_code or cf.graph_module_code, f"compiled_fn {cf.name} should expose GraphModule code"


class TestWriteOutput:
    """Verify the full write pipeline produces correct file structure."""

    def test_write_simple(self):
        _reset()
        tmpdir = tempfile.mkdtemp(prefix="magi_test_")
        try:

            def fn(x):
                return x.sum()

            compiled = torch.compile(fn, backend="eager")
            compiled(torch.randn(5))
            info = Introspector.build_function_info(fn)
            root = write_function(info, tmpdir)

            assert (root / "overview.md").exists()
            assert (root / "entry_0" / "decompiled_code.py").exists()
            assert (root / "entry_0" / "guards.txt").exists()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_write_graph_break_has_resume_dirs(self):
        _reset()
        tmpdir = tempfile.mkdtemp(prefix="magi_test_")
        try:

            def fn(x, y):
                z = x + y
                print("[test]", z.shape)
                return z * 2

            compiled = torch.compile(fn, backend="eager")
            compiled(torch.randn(3), torch.randn(3))
            info = Introspector.build_function_info(fn)
            root = write_function(info, tmpdir)

            rfns_dir = root / "entry_0" / "resume_fns"
            assert rfns_dir.exists(), "Expected resume_fns/ directory"
            resume_dirs = [d for d in rfns_dir.iterdir() if d.is_dir() and d.name.startswith("__resume")]
            assert len(resume_dirs) >= 1, "Expected resume function subdirectories"

            for rd in resume_dirs:
                assert (rd / "entry_0" / "decompiled_code.py").exists()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
