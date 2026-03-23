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

"""Unit tests for SourceEmitter (the core improvement over depyf)."""

from magi_compiler.magi_depyf.decompile.bytecode.source_emitter import LoopContext, SourceEmitter


class TestStackOperations:
    def test_push_pop(self):
        em = SourceEmitter()
        em.push("a")
        em.push("b")
        assert em.pop() == "b"
        assert em.pop() == "a"

    def test_peek(self):
        em = SourceEmitter()
        em.push("x")
        em.push("y")
        assert em.peek(0) == "y"
        assert em.peek(1) == "x"
        assert em.stack_size == 2

    def test_set_at(self):
        em = SourceEmitter()
        em.push("a")
        em.push("b")
        em.set_at(0, "B")
        assert em.peek() == "B"
        em.set_at(1, "A")
        assert em.stack == ["A", "B"]


class TestTempCounter:
    def test_instance_scoped(self):
        """Verify temp counter is per-instance, not class-level."""
        em1 = SourceEmitter()
        em2 = SourceEmitter()
        t1 = em1.make_temp()
        t2 = em2.make_temp()
        assert t1 == "__temp_1"
        assert t2 == "__temp_1"

    def test_replace_tos_with_temp(self):
        em = SourceEmitter()
        em.push("[1, 2, 3]")
        name = em.replace_tos_with_temp()
        assert name.startswith("__temp_")
        assert em.peek() == name
        assert f"{name} = [1, 2, 3]\n" in em.get_source()


class TestEmission:
    def test_emit_appends_newline(self):
        em = SourceEmitter()
        em.emit("x = 1")
        assert em.get_source() == "x = 1\n"

    def test_emit_raw(self):
        em = SourceEmitter()
        em.emit_raw("def f():\n    pass\n")
        assert em.get_source() == "def f():\n    pass\n"

    def test_indent(self):
        em = SourceEmitter(indent_size=4)
        assert em.indent("a = 1\nb = 2\n") == "    a = 1\n    b = 2\n"


class TestFork:
    def test_fork_shares_counter(self):
        em = SourceEmitter()
        em.make_temp()  # __temp_1
        with em.fork() as child:
            t = child.make_temp()
        assert t == "__temp_2"

    def test_fork_independent_source(self):
        em = SourceEmitter()
        em.emit("parent line")
        with em.fork(stack=["a"]) as child:
            child.emit("child line")
        assert "child line" in child.get_source()
        assert "child line" not in em.get_source()

    def test_fork_inherits_loop(self):
        em = SourceEmitter()
        em.loop = LoopContext(start_index=0, end_index=10)
        with em.fork() as child:
            assert child.loop is not None
            assert child.loop.start_index == 0

    def test_fork_explicit_loop(self):
        em = SourceEmitter()
        loop = LoopContext(start_index=5, end_index=15)
        with em.fork(loop=loop) as child:
            assert child.loop.start_index == 5
