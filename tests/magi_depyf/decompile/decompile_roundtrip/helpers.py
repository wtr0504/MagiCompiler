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

"""Shared helpers for decompile-roundtrip tests."""

import dis
from contextlib import contextmanager
from dataclasses import dataclass

from magi_compiler.magi_depyf import decompile
from magi_compiler.magi_depyf.decompile.recompiler import CodeRecompiler


def _assert_uses_op(func, *specs):
    """Assert *func*'s bytecode contains at least one instruction matching any spec.

    Each *spec* is either:
      - ``str``              -- match ``opname`` exactly
      - ``(opname, substr)`` -- match ``opname`` AND ``argrepr`` contains *substr*
    """
    for inst in dis.get_instructions(func.__code__):
        for spec in specs:
            if isinstance(spec, str):
                if inst.opname == spec:
                    return
            else:
                opname, substr = spec
                if inst.opname == opname and substr in str(inst.argrepr):
                    return
    found = sorted({f"{i.opname}({i.argrepr})" if i.argrepr else i.opname for i in dis.get_instructions(func.__code__)})
    spec_str = ", ".join(repr(s) for s in specs)
    raise AssertionError(f"Bytecode of {func.__name__} does not contain any of [{spec_str}].\n" f"Found: {found}")


def roundtrip_code(func):
    """Decompile *func*, compile the result, return the new CodeType."""
    old = func.__code__
    src = decompile(old)
    compiled = compile(src, filename=old.co_filename, mode="exec")
    codes = CodeRecompiler.collect_code_objects(compiled)
    return [c for c in codes if c.co_name == old.co_name][0]


@contextmanager
def replaced_code(func, *expected_ops):
    """Context-manager: run *func* with decompiled+recompiled code.

    If *expected_ops* are given, assert the bytecode contains at least one
    matching instruction **before** the roundtrip (catches constant-folding).
    """
    if expected_ops:
        _assert_uses_op(func, *expected_ops)
    old = func.__code__
    func.__code__ = roundtrip_code(func)
    try:
        yield
    finally:
        func.__code__ = old


@dataclass
class Point:
    x: int
    y: int

    def __matmul__(self, other):
        return self.x * other.x + self.y * other.y

    def __imatmul__(self, other):
        self.x = self.x * other.x
        self.y = self.y * other.y
        return self

    def __setitem__(self, key, value):
        if key == 0:
            self.x = value
        elif key == 1:
            self.y = value
        else:
            raise IndexError("Point only has two dimensions")


point = Point(1, 2)
data_map = {1: 2}
