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

"""Instruction dataclass API tests: from_dis, category queries, nop_."""

import dis

from magi_compiler.magi_depyf.decompile.bytecode.instruction import Instruction


def _make_inst(opname, **kw):
    opcode = dis.opmap.get(opname, 0)
    return Instruction(opcode=opcode, opname=opname, arg=0, argval=0, argrepr="", **kw)


class TestInstructionCategory:
    def test_is_load(self):
        assert _make_inst("LOAD_FAST").is_load
        assert _make_inst("LOAD_GLOBAL").is_load
        assert not _make_inst("STORE_FAST").is_load

    def test_is_store(self):
        assert _make_inst("STORE_FAST").is_store
        assert not _make_inst("LOAD_FAST").is_store

    def test_is_return(self):
        assert _make_inst("RETURN_VALUE").is_return
        assert _make_inst("RETURN_CONST").is_return
        assert not _make_inst("LOAD_FAST").is_return

    def test_is_nop(self):
        assert _make_inst("NOP").is_nop
        assert not _make_inst("LOAD_FAST").is_nop


class TestInstructionNop:
    def test_nop_mutates_in_place(self):
        inst = _make_inst("LOAD_FAST", offset=10)
        inst.nop_()
        assert inst.opname == "NOP"
        assert inst.is_nop


class TestInstructionFromDis:
    def test_roundtrip(self):
        def f(x):
            return x + 1

        stdlib_insts = list(dis.get_instructions(f.__code__))
        mine = [Instruction.from_dis(i) for i in stdlib_insts]
        assert len(mine) == len(stdlib_insts)
        for orig, converted in zip(stdlib_insts, mine):
            assert converted.opname == orig.opname
            assert converted.offset == orig.offset
