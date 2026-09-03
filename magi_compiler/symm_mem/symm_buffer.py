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

from __future__ import annotations

import inspect
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn

from magi_compiler.utils import magi_logger

# Only hijack ``to_empty``'s lambda.  ``.cuda()`` / ``.to()`` / ``.float()`` also
# go through ``_apply``; intercepting those would change builder semantics.
_TO_EMPTY_LAMBDA = "Module.to_empty.<locals>.<lambda>"


class SymmBuffer:
    """One symmetric-memory window, suballocated to many weight shards.

    One window per (decorated block, dtype, process group): a single rendezvous,
    and because every rank walks the module tree in the same order, offset ``k``
    is the same shard on every peer.  Two meshes sharing a dtype get two windows.
    """

    # 256 bf16 elems = 512B, which the copy engine wants for peak throughput.
    ALIGN = 256

    def __init__(self, dtype: torch.dtype, device: torch.device, group_name: str) -> None:
        self.dtype = dtype
        self.device = device
        self.group_name = group_name
        self.buf: torch.Tensor | None = None
        self.handle = None
        self.peers: list[torch.Tensor] = []
        self._reserved = 0
        self._cursor = 0

    # -- build phase ------------------------------------------------------
    def reserve(self, numel: int) -> None:
        self._reserved += self._round(numel)

    def commit(self) -> None:
        import torch.distributed._symmetric_memory as symm_mem

        symm_mem.enable_symm_mem_for_group(self.group_name)
        self.buf = symm_mem.empty(self._reserved, dtype=self.dtype, device=self.device)
        self.handle = symm_mem.rendezvous(self.buf, self.group_name)
        # Slice each peer's whole window once.  VMM maps every window into one
        # VA space so these are directly copyable; NCCL-backend peer pointers are not.
        self.peers = [self.handle.get_buffer(r, (self._reserved,), self.dtype) for r in range(self.handle.world_size)]

    def take(self, shape: torch.Size | tuple[int, ...]) -> torch.Tensor:
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
        return self.buf[off : off + numel].view(shape)

    # -- query ------------------------------------------------------------
    @property
    def nbytes(self) -> int:
        return self._reserved * self.dtype.itemsize

    def offset_of(self, t: torch.Tensor) -> int:
        return (t.data_ptr() - self.buf.data_ptr()) // self.buf.element_size()

    def peer_views(self, t: torch.Tensor) -> list[torch.Tensor]:
        """``world_size`` views of the same shard, one per rank, in rank order."""
        off, numel = self.offset_of(t), t.numel()
        return [p[off : off + numel].view(t.shape) for p in self.peers]

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
    """What the run-time gather needs to know about one local shard."""

    buffer: SymmBuffer
    offset: int
    local: torch.Tensor
    peer_views: tuple[torch.Tensor, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.local.shape)


# Keyed by ``data_ptr()``: the gather op only sees a plain tensor.
_SHARD_REGISTRY: dict[int, ShardEntry] = {}
_BUFFERS: list[SymmBuffer] = []
_BARRIER_DONE = False


def register_shard(local: torch.Tensor, buffer: SymmBuffer) -> ShardEntry:
    entry = ShardEntry(buffer=buffer, offset=buffer.offset_of(local), local=local, peer_views=tuple(buffer.peer_views(local)))
    _SHARD_REGISTRY[local.data_ptr()] = entry
    return entry


def lookup_shard(data_ptr: int) -> ShardEntry | None:
    return _SHARD_REGISTRY.get(data_ptr)


def registered_buffers() -> list[SymmBuffer]:
    return list(_BUFFERS)


def find_shard_by_layout(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor | None:
    """Any registered shard with this layout -- the cost model cannot replay a gather on a generic ``empty`` (no peers)."""
    want = tuple(int(s) for s in shape)
    for entry in _SHARD_REGISTRY.values():
        if entry.shape == want and entry.local.dtype == dtype:
            return entry.local
    return None


def reset_registry() -> None:
    """Test-only: drop every buffer so a new model can be built in-process."""
    global _BARRIER_DONE
    _SHARD_REGISTRY.clear()
    _BUFFERS.clear()
    _BARRIER_DONE = False


def barrier_after_load() -> None:
    """Publish every rank's freshly written shards, once per process.

    Must run after weights are loaded and before the first peer read.
    """
    global _BARRIER_DONE
    if _BARRIER_DONE or not _BUFFERS:
        return
    if dist.is_available() and dist.is_initialized():
        torch.cuda.synchronize()
        dist.barrier()
    _BARRIER_DONE = True
    magi_logger.info(
        "SymmBuffer: published %d buffer(s), %.1f MiB, %d shards; steady state is barrier-free",
        len(_BUFFERS),
        sum(b.nbytes for b in _BUFFERS) / 2**20,
        len(_SHARD_REGISTRY),
    )


def _is_gatherable_shard(t: object) -> bool:
    """An EVENLY split Shard(0) DTensor on a 1-D mesh -- the only placement the
    copy-engine gather handles.

    Uneven ``Shard(0)`` (``dim0 % world != 0``) is excluded on purpose, for three
    reasons that all trace back to the trailing ranks owning fewer rows: the window
    would be sized from a rank-dependent local numel, so ``rendezvous`` rejects it
    outright once the difference survives ``ALIGN``; below that threshold it is worse
    than an error, because every shard after the uneven one lands at a different
    offset per rank while ``peer_views`` slices each peer at the LOCAL offset; and the
    gather copies one fixed-size slab per peer, which would read past a shorter
    peer's rows.  ``dim0`` and the mesh size are identical on every rank, so all
    ranks drop the same parameters and the offset walk stays symmetric.  Dropped
    weights keep their ordinary allocation and gather over NCCL.
    """
    from torch.distributed.tensor import DTensor, Shard

    if not isinstance(t, DTensor):
        return False
    placements = t.placements
    if not (len(placements) == 1 and isinstance(placements[0], Shard) and placements[0].dim == 0):
        return False
    return int(t.shape[0]) % int(t.device_mesh.size(0)) == 0


def _group_name_of(t) -> str | None:
    try:
        return t.device_mesh._dim_group_names[0]
    except Exception:  # noqa: BLE001
        return None


def _buffer_key(t) -> tuple[torch.dtype, str]:
    """One window per (dtype, process group). Same dtype on two meshes (gaga4 FSDP + edp) must not share a window."""
    group_name = _group_name_of(t)
    if group_name is None:
        raise RuntimeError(f"cannot resolve the process group of a Shard(0) parameter on mesh {t.device_mesh}")
    return (t.dtype, group_name)


def _apply_order_entries(mod: nn.Module):
    """``(owner, name, param)`` in ``_apply`` order: post-order, every ``_parameters`` entry.

    Not ``named_parameters()`` (pre-order, dedups shared tensors).  A different
    walk would break cross-rank offset symmetry.  Walking ``_parameters`` also
    finds SimpleFSDP weights in ``parametrizations.weight.original``.
    """
    for child in mod.children():
        yield from _apply_order_entries(child)
    for name, p in mod._parameters.items():
        if p is not None:
            yield mod, name, p


def _plan_buffers(shards: list, device: torch.device) -> dict[tuple[torch.dtype, str], SymmBuffer]:
    """Size and commit one window per (dtype, group). Dedup by identity so a tied weight reserves a single slot."""
    buffers: dict[tuple[torch.dtype, str], SymmBuffer] = {}
    seen: set[int] = set()
    for p in shards:
        if id(p) in seen:
            continue
        seen.add(id(p))
        key = _buffer_key(p)
        buffer = buffers.get(key)
        if buffer is None:
            buffer = buffers[key] = SymmBuffer(p.dtype, device, key[1])
        buffer.reserve(p._local_tensor.numel())

    for buffer in buffers.values():
        buffer.commit()  # the only collective, once per window
    _BUFFERS.extend(buffers.values())
    return buffers


def materialize_into_buffers(mod: nn.Module, device: torch.device) -> dict[tuple[torch.dtype, str], SymmBuffer]:
    """Size windows for ``mod``'s Shard(0) shards while they are still on meta. Non-gatherable params are left to the caller."""
    shards = [p for _, _, p in _apply_order_entries(mod) if _is_gatherable_shard(p)]
    if not shards:
        return {}
    return _plan_buffers(shards, device)


def migrate_to_buffers(root: nn.Module) -> dict[tuple[torch.dtype, str], SymmBuffer]:
    """Copy already-allocated Shard(0) shards into symmetric memory.

    Used when ``magi_compile(model, ...)`` is given a live model rather than a
    meta + ``to_empty`` path.  ``load_state_dict(assign=True)`` after this would
    replace buffer views with ordinary tensors; the gather then rejects them.
    """
    entries = [(m, n, p) for m, n, p in _apply_order_entries(root) if _is_gatherable_shard(p)]
    if not entries:
        return {}

    device = entries[0][2]._local_tensor.device
    if device.type != "cuda":
        raise RuntimeError(f"symmetric memory needs the shards on cuda, found {device}")
    buffers = _plan_buffers([p for _, _, p in entries], device)

    from torch.distributed.tensor import DTensor

    views: dict[int, torch.Tensor] = {}
    for owner, name, p in entries:
        local = views.get(id(p))
        if local is None:
            buffer = buffers[_buffer_key(p)]
            local = views[id(p)] = buffer.take(p._local_tensor.shape)
            local.copy_(p._local_tensor)
            register_shard(local, buffer)
        moved = DTensor.from_local(local, p.device_mesh, p.placements, run_check=False)
        owner.register_parameter(name, nn.Parameter(moved, requires_grad=p.requires_grad))

    magi_logger.info(
        "SymmBuffer: migrated %d shard(s) into %.1f MiB across %d window(s)",
        len(views),
        sum(b.nbytes for b in buffers.values()) / 2**20,
        len(buffers),
    )
    return buffers


def patch_symm_buffer_apply(cls: type[nn.Module]) -> None:
    """Install the ``_apply`` interception on a decorated class.

    Mirrors ``_patch_cpu_offload_apply``: take over for ``to_empty``'s lambda, delegate everything else.
    """
    if getattr(cls, "_magi_symm_apply_patched", False):
        return
    orig_apply = cls._apply
    magi_logger.info("SymmBuffer: intercepting %s._apply for copy-engine FSDP", cls.__name__)

    def _symm_apply(self, fn, recurse: bool = True):
        if getattr(fn, "__qualname__", "") != _TO_EMPTY_LAMBDA:
            return orig_apply(self, fn, recurse)
        if getattr(self, "_magi_symm_buffers", None) is not None:
            return orig_apply(self, fn, recurse)

        device = torch.device(inspect.getclosurevars(fn).nonlocals["device"])
        from torch.distributed.tensor import DTensor

        buffers = materialize_into_buffers(self, device)
        if not buffers:
            return orig_apply(self, fn, recurse)

        views: dict[int, torch.Tensor] = {}

        def materialize(t: torch.Tensor) -> torch.Tensor:
            if not _is_gatherable_shard(t):
                return torch.empty_like(t, device=device)
            # Tied weight: same view so tying survives materialization.
            local = views.get(id(t))
            if local is None:
                buffer = buffers[_buffer_key(t)]
                local = views[id(t)] = buffer.take(t._local_tensor.shape)
                register_shard(local, buffer)
            return DTensor.from_local(local, t.device_mesh, t.placements, run_check=False)

        # Do not forge to_empty's qualname: a nested decorated block must fail
        # the check above and delegate, so its params land in *this* buffer.
        out = orig_apply(self, materialize, recurse)
        self._magi_symm_buffers = buffers
        magi_logger.info(
            "SymmBuffer: %s materialized %d shard(s) into %.1f MiB across %d window(s)",
            cls.__name__,
            len(views),
            sum(b.nbytes for b in buffers.values()) / 2**20,
            len(buffers),
        )
        return out

    cls._apply = _symm_apply
    cls._magi_symm_apply_patched = True
