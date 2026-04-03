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

"""Tests verifying that InductorStandaloneAdaptor.compile() correctly patches
autograd_cache_allow_custom_autograd_functions=True around standalone_compile().

Test strategy
-------------
This file doubles as its own subprocess helper (see ``__main__`` at the bottom).
Each subprocess invocation:

  1. Installs a *spy* on ``torch._inductor.standalone_compile`` that records the
     value of ``autograd_cache_allow_custom_autograd_functions`` at call time
     without replacing the real implementation.
  2. Runs one full training step (forward + backward + optimizer) on a model that
     contains a ``torch.autograd.Function`` subclass, causing Dynamo to emit an
     ``autograd_function_apply`` HigherOrderOperator node.
  3. Writes a JSON payload with flag observations and compilation counters.

The pytest test then spawns two subprocess runs sharing the same cache directory:

  run 1 (warm)  – compiles and saves the artifact; flag must be True.
  run 2 (cache) – loads from disk; num_inductor_compiles must be 0.

Verifications
-------------
- flag_during_compile is True on every standalone_compile() call (run 1).
- num_compiled_artifacts_saved > 0 on run 1.
- num_inductor_compiles == 0 on run 2 (artifact loaded, no recompilation).
- "Failed to save compiled artifact" absent from stderr of run 1.
- loss values are numerically close between runs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

# ══════════════════════════════════════════════════════════════════════════════
# Subprocess helper — model definition + training step + spy
# ══════════════════════════════════════════════════════════════════════════════


def _subprocess_main() -> None:
    """Entry point executed in each child process."""
    import argparse
    from unittest.mock import patch

    import torch._functorch.config as functorch_config
    import torch._inductor as _inductor_mod
    import torch.nn as nn

    from magi_compiler import magi_compile
    from magi_compiler.config import CompileMode, get_compile_config
    from magi_compiler.utils import compilation_counter

    DEVICE = "cuda"
    DTYPE = torch.bfloat16
    HIDDEN = 16

    # ── Model with torch.autograd.Function ──────────────────────────────────
    class _ScaledSigmoid(torch.autograd.Function):
        """Custom autograd function.

        When Dynamo traces ``_ScaledSigmoid.apply(x)`` it emits an
        ``autograd_function_apply`` HigherOrderOperator node — the node that
        previously caused AOTAutograd caching to be bypassed.
        """

        @staticmethod
        def forward(ctx, x: torch.Tensor) -> torch.Tensor:
            ctx.save_for_backward(x)
            return torch.sigmoid(x) * 2.0

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
            (x,) = ctx.saved_tensors
            sig = torch.sigmoid(x)
            return grad_output * sig * (1.0 - sig) * 2.0

    @magi_compile(dynamic_arg_dims={"x": 0})
    class TrainingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(HIDDEN, HIDDEN, dtype=DTYPE, device=DEVICE)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return _ScaledSigmoid.apply(self.linear(x))

    # ── Parse args ──────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # ── Configure magi ──────────────────────────────────────────────────────
    config = get_compile_config()
    config.compile_mode = CompileMode.MAGI_COMPILE
    config.aot = False
    config.cache_root_dir = args.cache_root

    torch._dynamo.reset()
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    # ── Install spy on standalone_compile ───────────────────────────────────
    # Capture the real implementation before the patch replaces it.
    _real_standalone_compile = _inductor_mod.standalone_compile
    flag_during_compile: list[bool] = []

    def _spy_standalone_compile(graph, example_inputs, **kwargs):
        # Record flag state *inside* the standalone_compile call.
        # If the fix is in place, functorch_config.patch(...=True) is already
        # active here, so the flag must be True.
        flag_during_compile.append(functorch_config.autograd_cache_allow_custom_autograd_functions)
        return _real_standalone_compile(graph, example_inputs, **kwargs)

    # ── Training step ────────────────────────────────────────────────────────
    with patch("torch._inductor.standalone_compile", side_effect=_spy_standalone_compile):
        model = TrainingModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        x = torch.randn(4, HIDDEN, device=DEVICE, dtype=DTYPE)

        optimizer.zero_grad()
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

    # ── Write JSON payload ───────────────────────────────────────────────────
    payload = {
        "flag_during_compile": flag_during_compile,
        "all_flags_true": all(flag_during_compile),
        "num_standalone_compile_calls": len(flag_during_compile),
        "num_compiled_artifacts_saved": compilation_counter.num_compiled_artifacts_saved,
        "num_inductor_compiles": compilation_counter.num_inductor_compiles,
        "loss": float(loss.float().item()),
    }
    with open(args.output, "w") as f:
        json.dump(payload, f)


# ══════════════════════════════════════════════════════════════════════════════
# pytest test
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_autograd_cache_flag_and_cache_reuse(tmp_path: Path):
    """Training model with autograd.Function: artifact saved on run 1, loaded
    on run 2, and autograd_cache_allow_custom_autograd_functions is True during
    every standalone_compile() call."""

    cache_root = tmp_path / "cache"
    out1 = tmp_path / "run1.json"
    out2 = tmp_path / "run2.json"

    env = os.environ.copy()
    env["MAGI_LOGGING_LEVEL"] = "info"

    def _run(output: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                __file__,  # this file is its own helper
                "--cache-root",
                str(cache_root),
                "--output",
                str(output),
            ],
            env=env,
            capture_output=True,
            text=True,
        )

    # ── Run 1: warm cache ────────────────────────────────────────────────────
    p1 = _run(out1)
    assert p1.returncode == 0, f"run 1 failed\nstdout:\n{p1.stdout}\nstderr:\n{p1.stderr}"

    # The fix must prevent "Failed to save compiled artifact" from appearing.
    assert "Failed to save compiled artifact" not in p1.stderr, (
        "CompiledArtifact.save() still failing — autograd_function_apply bypass not fixed.\n" f"stderr:\n{p1.stderr}"
    )

    payload1 = json.loads(out1.read_text())

    # Flag check: every standalone_compile() call must see the flag as True.
    assert payload1["num_standalone_compile_calls"] > 0, "Spy was never called — standalone_compile was not intercepted."
    assert payload1["all_flags_true"], (
        "autograd_cache_allow_custom_autograd_functions was NOT True during "
        f"standalone_compile(); observed per-call values: {payload1['flag_during_compile']}"
    )

    # Artifact must have been saved.
    assert payload1["num_compiled_artifacts_saved"] > 0, (
        "Expected at least one artifact to be saved on the warm run, "
        f"got num_compiled_artifacts_saved={payload1['num_compiled_artifacts_saved']}"
    )

    # ── Run 2: cache hit ─────────────────────────────────────────────────────
    p2 = _run(out2)
    assert p2.returncode == 0, f"run 2 failed\nstdout:\n{p2.stdout}\nstderr:\n{p2.stderr}"

    payload2 = json.loads(out2.read_text())

    # No recompilation: PiecewiseCompiler.load() returns early before compile().
    assert payload2["num_inductor_compiles"] == 0, (
        "Expected 0 inductor compiles on the cache-hit run — artifact was not loaded.\n"
        f"num_inductor_compiles={payload2['num_inductor_compiles']}\n"
        f"stderr:\n{p2.stderr}"
    )

    # Numerical consistency.
    assert (
        abs(payload1["loss"] - payload2["loss"]) < 1e-2
    ), f"Loss mismatch between runs: run1={payload1['loss']}, run2={payload2['loss']}"


# ══════════════════════════════════════════════════════════════════════════════
# Entry point (subprocess mode)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _subprocess_main()
