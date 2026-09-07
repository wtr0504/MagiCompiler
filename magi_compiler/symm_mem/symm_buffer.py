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

"""Symmetric-memory windows for copy-engine weight all-gather.

Shards are suballocated from pooled windows; runtime gathers look them up by
device pointer. Which weights get an allocation is ``bind``'s job.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from magi_compiler.utils import magi_logger


class SymmBuffer:
    """One symmetric-memory window, suballocated to many weight shards.

    The driver caps windows at 128 per process regardless of size, so pooling is
    required. Every rank walks the same plan, so offset ``k`` is the same shard
    on every peer.
    """

    # 256 bf16 elems = 512B, which the copy engine wants for peak throughput.
    ALIGN = 256

    def __init__(self, dtype: torch.dtype, device: torch.device, group_name: str) -> None:
        self.dtype = dtype
        self.device = device
        self.group_name = group_name
        self.buf: torch.Tensor | None = None
        self.handle = None
        self._reserved = 0
        self._cursor = 0

    def reserve(self, numel: int) -> None:
        """Book space for one shard.  Call for every member before ``commit``."""
        self._reserved += self._round(numel)

    def commit(self) -> None:
        """Open the window.  One ``rendezvous`` for every shard it will hold."""
        import torch.distributed._symmetric_memory as symm_mem

        symm_mem.enable_symm_mem_for_group(self.group_name)
        self.buf = symm_mem.empty(self._reserved, dtype=self.dtype, device=self.device)
        self.handle = symm_mem.rendezvous(self.buf, self.group_name)

    def take(self, shape: torch.Size | tuple[int, ...]) -> torch.Tensor:
        """Hand out the next slot as a tensor whose storage starts at the slot.

        Not a slice of ``buf``: Dynamo memoized ``storage_offset == 0`` for these
        parameters, and a mid-window slice would contradict the shape env.
        """
        numel = 1
        for s in shape:
            numel *= int(s)
        off = self._cursor
        self._cursor += self._round(numel)
        if self._cursor > self._reserved:
            raise RuntimeError(
                f"symmetric buffer overflow: wanted {self._cursor} elems, reserved {self._reserved}. "
                "The sizing walk and the dispensing walk must visit the same shards in the same order."
            )
        return self.handle.get_buffer(self.handle.rank, tuple(int(s) for s in shape), self.dtype, off)

    @property
    def nbytes(self) -> int:
        return self._reserved * self.dtype.itemsize

    def offset_of(self, t: torch.Tensor) -> int:
        return (t.data_ptr() - self.buf.data_ptr()) // self.buf.element_size()

    def peer_views(self, t: torch.Tensor) -> list[torch.Tensor]:
        """``world_size`` views of the same shard, one per rank.

        They borrow the window mapping; ``self.buf`` keeps it alive.
        """
        off, shape = self.offset_of(t), tuple(int(s) for s in t.shape)
        return [self.handle.get_buffer(r, shape, self.dtype, off) for r in range(self.handle.world_size)]

    def contains(self, t: torch.Tensor) -> bool:
        if self.buf is None:
            return False
        base = self.buf.data_ptr()
        return base <= t.data_ptr() < base + self.nbytes

    @classmethod
    def _round(cls, numel: int) -> int:
        return (numel + cls.ALIGN - 1) // cls.ALIGN * cls.ALIGN


@dataclass(frozen=True)
class ShardEntry:
    """What a run-time gather needs to know about one local shard."""

    buffer: SymmBuffer
    offset: int
    local: torch.Tensor
    peer_views: tuple[torch.Tensor, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.local.shape)

    @property
    def dtype(self) -> torch.dtype:
        return self.local.dtype


_SHARD_REGISTRY: dict[int, ShardEntry] = {}
_BUFFERS: list[SymmBuffer] = []
_UNPUBLISHED = False


def open_buffer(dtype: torch.dtype, device: torch.device, group_name: str, numels) -> SymmBuffer:
    """Open one window big enough for ``numels``, and track it for ``publish``."""
    global _UNPUBLISHED
    buffer = SymmBuffer(dtype, device, group_name)
    for numel in numels:
        buffer.reserve(int(numel))
    buffer.commit()
    _BUFFERS.append(buffer)
    _UNPUBLISHED = True
    return buffer


def register_shard(local: torch.Tensor, buffer: SymmBuffer) -> ShardEntry:
    """Record a slot so the run-time gather can find its peer views."""
    entry = ShardEntry(buffer=buffer, offset=buffer.offset_of(local), local=local, peer_views=tuple(buffer.peer_views(local)))
    _SHARD_REGISTRY[local.data_ptr()] = entry
    return entry


def alloc_shard(shape, dtype: torch.dtype, device: torch.device, group_name: str) -> torch.Tensor:
    """One shard in a window of its own. Tests and the cost model only -- binding a whole model must pool."""
    shape = tuple(int(s) for s in shape)
    numel = 1
    for s in shape:
        numel *= s

    buffer = open_buffer(dtype, device, group_name, (numel,))
    shard = buffer.take(shape)
    register_shard(shard, buffer)
    return shard


def group_name_of(t) -> str:
    """The process group a sharded parameter must rendezvous on.

    Never defaulted to WORLD: dense and expert weights may sit on different meshes.
    """
    mesh = getattr(t, "device_mesh", None)
    names = getattr(mesh, "_dim_group_names", None) if mesh is not None else None
    if not names:
        raise RuntimeError(
            f"cannot resolve the process group of {type(t).__name__} (device_mesh={mesh!r}); "
            "copy-engine binding needs the mesh dim the weight is sharded over"
        )
    return names[0]


def lookup_shard(data_ptr: int) -> ShardEntry | None:
    """The registered shard starting at ``data_ptr``, or None if it is not one."""
    return _SHARD_REGISTRY.get(data_ptr)


def registered_buffers() -> list[SymmBuffer]:
    return list(_BUFFERS)


def find_shard_by_layout(shape, dtype: torch.dtype) -> torch.Tensor | None:
    """Any registered shard with this exact layout, for the runtime estimator's stand-in."""
    want = tuple(int(s) for s in shape)
    for entry in _SHARD_REGISTRY.values():
        if entry.shape == want and entry.dtype == dtype:
            return entry.local
    return None


def publish() -> None:
    """Barrier so every rank has copied its shards in before any peer reads.

    Once per bind, not per step: the weights never change again.
    """
    global _UNPUBLISHED
    if not _UNPUBLISHED:
        return
    if dist.is_available() and dist.is_initialized():
        torch.cuda.synchronize()
        dist.barrier()
    _UNPUBLISHED = False
    magi_logger.info(
        "SymmBuffer: published %d shard(s), %.1f MiB total", len(_BUFFERS), sum(b.nbytes for b in _BUFFERS) / 2**20
    )


def reset_registry() -> None:
    """Drop every window and shard.  Tests only -- frees the symmetric allocations."""
    global _UNPUBLISHED
    _SHARD_REGISTRY.clear()
    _BUFFERS.clear()
    _UNPUBLISHED = False
