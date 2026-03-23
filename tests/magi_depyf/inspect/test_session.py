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

"""CaptureSession and CaptureResult tests: dataclass fields, lifecycle, basic capture."""

from __future__ import annotations

import time
import types

import pytest

from magi_compiler.magi_depyf.inspect.result import CaptureResult
from magi_compiler.magi_depyf.inspect.session import CaptureSession


def _make_dummy_code(name="test_fn"):
    src = f"def {name}(x): return x + 1"
    code = compile(src, "<test>", "exec")
    for c in code.co_consts:
        if isinstance(c, types.CodeType) and c.co_name == name:
            return c
    raise RuntimeError(f"No code object named {name}")


# ── CaptureResult ─────────────────────────────────────────────────────────


class TestCaptureResult:
    def test_fields_populated(self):
        code = _make_dummy_code()
        r = CaptureResult(
            function_name="my_func",
            original_code=code,
            dynamo_code=code,
            decompiled_source="def my_func(x):\n    return x + 1\n",
            guards=["guard1", "guard2"],
            graph_source="graph code here",
        )
        assert r.function_name == "my_func"
        assert r.original_code is code
        assert r.dynamo_code is code
        assert "my_func" in r.decompiled_source
        assert r.guards == ["guard1", "guard2"]
        assert r.graph_source == "graph code here"
        assert isinstance(r.timestamp, float)
        assert r.timestamp > 0

    def test_summary(self):
        code = _make_dummy_code("sample")
        r = CaptureResult(
            function_name="sample",
            original_code=code,
            dynamo_code=code,
            decompiled_source="...",
            guards=["g1", "g2", "g3"],
            graph_source="some graph",
        )
        s = r.summary()
        assert "sample" in s
        assert "guards=3" in s
        assert "graph=yes" in s

    def test_defaults(self):
        code = _make_dummy_code()
        r = CaptureResult(function_name="fn", original_code=code, dynamo_code=code, decompiled_source="...")
        assert "graph=no" in r.summary()
        assert r.guards == []
        assert r.fn_globals is None

    def test_timestamp_auto(self):
        before = time.time()
        code = _make_dummy_code()
        r = CaptureResult(function_name="fn", original_code=code, dynamo_code=code, decompiled_source="...")
        after = time.time()
        assert before <= r.timestamp <= after


# ── CaptureSession lifecycle (no torch) ───────────────────────────────────


class TestCaptureSessionLifecycle:
    def test_init_state(self):
        s = CaptureSession()
        assert s.results == []
        assert s._hook_handle is None

    def test_clear(self):
        s = CaptureSession()
        s._results.append("fake")
        assert len(s.results) == 1
        s.clear()
        assert s.results == []

    def test_results_returns_copy(self):
        s = CaptureSession()
        s._results.append("item")
        r = s.results
        r.append("should not affect internal")
        assert len(s._results) == 1


# ── CaptureSession with torch ────────────────────────────────────────────

torch_available = False
try:
    import torch

    torch_available = True
except ImportError:
    pass


@pytest.mark.skipif(not torch_available, reason="torch not installed")
class TestCaptureSessionWithTorch:
    def test_capture_simple_compile(self):
        """Compile a simple function and verify CaptureResult contents."""

        def fn(x):
            return x + 1

        torch._dynamo.reset()
        with CaptureSession() as session:
            compiled = torch.compile(fn, backend="eager")
            compiled(torch.tensor([1.0, 2.0, 3.0]))

        assert len(session.results) >= 1
        r = session.results[0]
        assert isinstance(r, CaptureResult)
        assert isinstance(r.function_name, str)
        assert isinstance(r.original_code, types.CodeType)
        assert isinstance(r.dynamo_code, types.CodeType)
        assert isinstance(r.decompiled_source, str)
        assert "def" in r.decompiled_source or "Failed" in r.decompiled_source

    def test_hook_removed_after_exit(self):
        session = CaptureSession()
        session.__enter__()
        assert session._hook_handle is not None
        session.__exit__(None, None, None)
        assert session._hook_handle is None

    def test_multiple_captures(self):
        def fn_a(x):
            return x * 2

        def fn_b(x):
            return x - 1

        torch._dynamo.reset()
        with CaptureSession() as session:
            ca = torch.compile(fn_a, backend="eager")
            cb = torch.compile(fn_b, backend="eager")
            ca(torch.tensor([1.0]))
            cb(torch.tensor([2.0]))

        assert len(session.results) >= 2
