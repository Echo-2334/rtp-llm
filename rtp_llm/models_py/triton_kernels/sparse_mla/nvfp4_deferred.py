"""Bounded BF16 tails, sealed into clustered NVFP4 at 64 tokens.

The packed page always contains a portable ordinary-NVFP4 mirror for unfinished
rows (center ID 255). Local attention substitutes the BF16 tail. A page copied
by PD/prefix reuse remains readable without the source process's tail pool.
It resumes from that quantized mirror; BF16 transport is not implied.

All decisions are GPU-side, with fixed-shape launches for CUDA Graph replay.
The bounded LRU pool retains sealed shadows to support speculative rewrites.
If oversubscribed, the portable mirror remains correct; overflow is counted.
"""

from __future__ import annotations

import logging
import secrets
import time
import weakref

import torch
import triton
import triton.language as tl
from rtp_llm.models_py.triton_kernels.sparse_mla.nvfp4_cache import (
    CENTER_ID_OFFSET,
    TAG_OFFSET,
    _load_latent,
    _store_latent,
    _validate_cache,
)
from rtp_llm.models_py.triton_kernels.sparse_mla.nvfp4_cluster import (
    _center_offsets,
    _fit_centers,
    initialize_block_centers,
)

_STATES = weakref.WeakValueDictionary()


def _key(cache):
    return (cache.device.index, cache.data_ptr(), tuple(cache.shape))


def find_state(cache):
    return _STATES.get(_key(cache))


# TOKENS is a runtime argument: eager prefill batches have arbitrary lengths and
# specializing on it would recompile every kernel below for each new length.
@triton.jit(do_not_specialize=["TOKENS"])
def _protect(Slots, Mapping, Ages, Epoch, TOKENS, BLOCKS: tl.constexpr):
    i = tl.program_id(0)
    slot = tl.load(Slots + i).to(tl.int64)
    if slot >= 0 and slot < BLOCKS * 64:
        tail = tl.load(Mapping + slot // 64)
        if tail >= 0:
            tl.store(Ages + tail, tl.load(Epoch) + 1)


@triton.jit(do_not_specialize=["TOKENS"])
def _prepare(
    Cache,
    Slots,
    Mapping,
    Owners,
    Ages,
    Tags,
    Counts,
    Epoch,
    NextTag,
    RestoreEpoch,
    Prefix,
    FullEpoch,
    SealEpoch,
    Overflow,
    TOKENS,
    BLOCKS: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    POOL: tl.constexpr,
    POOL_TILE: tl.constexpr,
):
    epoch = tl.load(Epoch) + 1
    tl.store(Epoch, epoch)
    pool = tl.arange(0, POOL_TILE)
    i = 0
    while i < TOKENS:
        slot = tl.load(Slots + i).to(tl.int64)
        if slot >= 0 and slot < BLOCKS * 64:
            block = slot // 64
            lanes = tl.arange(0, 64)
            positions = i + lanes
            group_slots = tl.load(Slots + positions, positions < TOKENS, -1).to(
                tl.int64
            )
            in_group = (
                (positions < TOKENS) & (group_slots >= 0) & (group_slots // 64 == block)
            )
            count = tl.sum(in_group.to(tl.int32), 0)
            contiguous = (
                tl.sum((in_group & (group_slots != slot + lanes)).to(tl.int32), 0) == 0
            )
            j = i + count
            end = tl.max(tl.where(in_group, group_slots % 64, -1), 0)
            full_input = (slot % 64 == 0) & (count == 64) & (end == 63) & contiguous
            if full_input:
                tl.store(FullEpoch + block, epoch)
            else:
                tl.store(SealEpoch + block, tl.where(end == 63, epoch, 0))
                tail = tl.load(Mapping + block)
                old_tag = tl.load(
                    (Cache + block * 64 * ROW_BYTES + TAG_OFFSET).to(tl.pointer_type(tl.int64))
                )
                valid_tail = False
                reusable = False
                if tail >= 0:
                    reusable = tl.load(Owners + tail) == block
                    valid_tail = reusable & (tl.load(Tags + tail) == old_tag)
                reset = (slot % 64 == 0) | ~valid_tail
                if valid_tail:
                    reset = reset | (slot % 64 > tl.load(Counts + tail))
                if not reusable:
                    ages = tl.load(Ages + pool, pool < POOL, 0x7FFFFFFFFFFFFFFF)
                    owners = tl.load(Owners + pool, pool < POOL, -2)
                    priority = tl.where(owners == -1, -1, ages)
                    best_age = tl.min(priority, 0)
                    if best_age < epoch:
                        tail = tl.min(
                            tl.where(
                                (pool < POOL) & (priority == best_age), pool, 0x7FFFFFFF
                            ),
                            0,
                        )
                        old_owner = tl.load(Owners + tail)
                        if old_owner >= 0:
                            if tl.load(Mapping + old_owner) == tail:
                                tl.store(Mapping + old_owner, -1)
                        tl.store(Mapping + block, tail)
                        tl.store(Owners + tail, block)
                    else:
                        tail = -1
                        tl.store(Mapping + block, -1)
                        tl.atomic_add(Overflow, 1)
                if tail >= 0:
                    tl.store(Ages + tail, epoch)
                    if reset:
                        # Every allocation/reset needs its own identity. A write
                        # epoch is shared by multiple blocks; using it as a tag
                        # lets a copied page alias a different block's raw tail.
                        tag = tl.atomic_add(NextTag, 1) + 1
                        tl.store(Tags + tail, tag)
                        tl.store(RestoreEpoch + block, epoch)
                        tl.store(Prefix + block, slot % 64)
                    tl.store(Counts + tail, end + 1)
            i = j
        else:
            i += 1


@triton.jit(do_not_specialize=["TOKENS"])
def _restore(
    Cache,
    Slots,
    Mapping,
    Tags,
    Data,
    RestoreEpoch,
    Prefix,
    Epoch,
    TOKENS,
    BLOCKS: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    CENTERS: tl.constexpr,
):
    i = tl.program_id(0)
    slot = tl.load(Slots + i).to(tl.int64)
    previous = tl.load(Slots + i - 1, i > 0, -1).to(tl.int64)
    if slot >= 0 and slot < BLOCKS * 64 and (i == 0 or previous // 64 != slot // 64):
        block = slot // 64
        if tl.load(RestoreEpoch + block) == tl.load(Epoch):
            tail = tl.load(Mapping + block)
            if tail >= 0:
                rows = tl.arange(0, 64)
                features = tl.arange(0, 512)
                valid = rows < tl.load(Prefix + block)
                positions = block * 64 + rows
                values = _load_latent(
                    Cache, positions, valid, features, ROW_BYTES, CENTERS
                )
                tl.store(
                    Data
                    + tail.to(tl.int64) * 64 * 512
                    + rows[:, None] * 512
                    + features[None, :],
                    values,
                    valid[:, None],
                )
                tag = tl.load(Tags + tail)
                tl.store(
                    (Cache + positions * ROW_BYTES + TAG_OFFSET).to(tl.pointer_type(tl.int64)),
                    tag,
                    valid,
                )


@triton.jit
def _write_deferred(
    Latent,
    Rope,
    Cache,
    Slots,
    Data,
    Mapping,
    Tags,
    FullEpoch,
    Epoch,
    LATENT_STRIDE: tl.constexpr,
    ROPE_STRIDE: tl.constexpr,
    BLOCKS: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    CENTERS: tl.constexpr,
):
    i = tl.program_id(0)
    slot = tl.load(Slots + i).to(tl.int64)
    if slot >= 0 and slot < BLOCKS * 64:
        features = tl.arange(0, 512)
        values = tl.load(Latent + i * LATENT_STRIDE + features)
        full_input = tl.load(FullEpoch + slot // 64) == tl.load(Epoch)
        tag = tl.full((), 0, tl.int64)
        if not full_input:
            tail = tl.load(Mapping + slot // 64)
            if tail >= 0:
                tag = tl.load(Tags + tail)
                tl.store(
                    Data + tail.to(tl.int64) * 64 * 512 + (slot % 64) * 512 + features,
                    values,
                )
        _store_latent(Cache, slot, values, ROW_BYTES, CENTERS, full_input, tag)
        rope = tl.load(Rope + i * ROPE_STRIDE + tl.arange(0, 64))
        tl.store(
            (Cache + slot * ROW_BYTES + 292).to(tl.pointer_type(tl.bfloat16))
            + tl.arange(0, 64),
            rope,
        )


@triton.jit(do_not_specialize=["TOKENS"])
def _plainify_fallback(
    Cache,
    Slots,
    Mapping,
    SealEpoch,
    Epoch,
    TOKENS,
    BLOCKS: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    CENTERS: tl.constexpr,
):
    i = tl.program_id(0)
    row = tl.program_id(1)
    slot = tl.load(Slots + i).to(tl.int64)
    previous = tl.load(Slots + i - 1, i > 0, -1).to(tl.int64)
    if slot >= 0 and slot < BLOCKS * 64 and (i == 0 or previous // 64 != slot // 64):
        block = slot // 64
        if (
            tl.load(SealEpoch + block) == tl.load(Epoch)
            and tl.load(Mapping + block) < 0
        ):
            positions = tl.full((1,), block * 64 + row, tl.int64)
            features = tl.arange(0, 512)
            # Plain mirrors keep 0xFF in every ID byte; clustered IDs are <= 4.
            marker = tl.load(Cache + (block * 64 + row) * ROW_BYTES + CENTER_ID_OFFSET)
            if marker != 255:
                values = tl.reshape(
                    _load_latent(
                        Cache, positions, positions >= 0, features, ROW_BYTES, CENTERS
                    ),
                    (512,),
                )
                _store_latent(
                    Cache, block * 64 + row, values, ROW_BYTES, CENTERS, False, 0
                )


@triton.jit(do_not_specialize=["TOKENS"])
def _seal_centers(
    Cache,
    Slots,
    Data,
    Mapping,
    SealEpoch,
    Epoch,
    TOKENS,
    BLOCKS: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    CENTERS: tl.constexpr,
):
    i = tl.program_id(0)
    slot = tl.load(Slots + i).to(tl.int64)
    previous = tl.load(Slots + i - 1, i > 0, -1).to(tl.int64)
    if slot >= 0 and slot < BLOCKS * 64 and (i == 0 or previous // 64 != slot // 64):
        block = slot // 64
        if tl.load(SealEpoch + block) == tl.load(Epoch):
            rows = tl.arange(0, 64)
            features = tl.arange(0, 512)
            tail = tl.load(Mapping + block)
            if tail >= 0:
                values = tl.load(
                    Data
                    + tail.to(tl.int64) * 64 * 512
                    + rows[:, None] * 512
                    + features[None, :]
                )
            else:
                values = _load_latent(
                    Cache, block * 64 + rows, rows < 64, features, ROW_BYTES, CENTERS
                )
            centers = _fit_centers(values.to(tl.float32), rows < 64, CENTERS)
            clusters = tl.arange(0, CENTERS)
            offsets = _center_offsets(
                block, clusters[:, None], features[None, :], ROW_BYTES, CENTERS
            )
            tl.store((Cache + offsets).to(tl.pointer_type(tl.bfloat16)), centers)


@triton.jit(do_not_specialize=["TOKENS"])
def _seal_rows(
    Cache,
    Slots,
    Data,
    Mapping,
    Tags,
    SealEpoch,
    Epoch,
    TOKENS,
    BLOCKS: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    CENTERS: tl.constexpr,
):
    i = tl.program_id(0)
    row = tl.program_id(1)
    slot = tl.load(Slots + i).to(tl.int64)
    previous = tl.load(Slots + i - 1, i > 0, -1).to(tl.int64)
    if slot >= 0 and slot < BLOCKS * 64 and (i == 0 or previous // 64 != slot // 64):
        block = slot // 64
        if tl.load(SealEpoch + block) == tl.load(Epoch):
            tail = tl.load(Mapping + block)
            features = tl.arange(0, 512)
            tag = tl.full((), 0, tl.int64)
            if tail >= 0:
                values = tl.load(
                    Data + tail.to(tl.int64) * 64 * 512 + row * 512 + features
                )
                tag = tl.load(Tags + tail)
            else:
                positions = tl.full((1,), block * 64 + row, tl.int64)
                values = tl.reshape(
                    _load_latent(
                        Cache, positions, positions >= 0, features, ROW_BYTES, CENTERS
                    ),
                    (512,),
                )
            _store_latent(
                Cache, block * 64 + row, values, ROW_BYTES, CENTERS, True, tag
            )


class DeferredNvfp4State:
    """One fixed-size tail pool per layer cache. Create during eager warmup."""

    def __init__(self, cache, capacity=256):
        centers = _validate_cache(cache)
        if not centers:
            raise ValueError("Deferred clustering requires 1/2/4 cluster centers")
        if capacity <= 0:
            raise ValueError("NVFP4 tail capacity must be positive")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Create the deferred NVFP4 pool during eager warmup before CUDA Graph capture"
            )
        self.cache = cache
        self.capacity = int(capacity)
        device = cache.device
        blocks = cache.shape[0]
        self.data = torch.empty(
            (capacity, 64, 512), dtype=torch.bfloat16, device=device
        )
        self.mapping = torch.full((blocks,), -1, dtype=torch.int32, device=device)
        self.owners = torch.full((capacity,), -1, dtype=torch.int32, device=device)
        self.ages = torch.zeros((capacity,), dtype=torch.int64, device=device)
        self.tags = torch.zeros_like(self.ages)
        self.counts = torch.zeros((capacity,), dtype=torch.int32, device=device)
        # A namespace makes copied PD/prefix pages unable to alias stale local tails.
        self.epoch = torch.full(
            (1,), (secrets.randbits(30) + 1) << 32, dtype=torch.int64, device=device
        )
        self.next_tag = self.epoch.clone()
        self.restore_epoch = torch.zeros((blocks,), dtype=torch.int64, device=device)
        self.prefix = torch.zeros((blocks,), dtype=torch.int32, device=device)
        self.full_epoch = torch.zeros_like(self.restore_epoch)
        self.seal_epoch = torch.zeros_like(self.restore_epoch)
        self.overflow_count = torch.zeros((1,), dtype=torch.int64, device=device)
        self.reported_overflow = 0
        previous = find_state(cache)
        if previous is not None:
            raise ValueError("A deferred state already owns this cache; reuse it")
        _STATES[_key(cache)] = self

    def write(self, latent, rope, cache, slots):
        if _key(cache) != _key(self.cache):
            raise ValueError("Deferred state belongs to a different cache")
        if not slots.numel():
            return
        centers = _validate_cache(cache)
        sorted_slots, _ = torch.sort(slots.reshape(-1))
        n = slots.numel()
        blocks = cache.shape[0]
        width = cache.shape[-1]
        common = dict(TOKENS=n, BLOCKS=blocks)
        _protect[(n,)](slots, self.mapping, self.ages, self.epoch, **common)
        _prepare[(1,)](
            cache,
            sorted_slots,
            self.mapping,
            self.owners,
            self.ages,
            self.tags,
            self.counts,
            self.epoch,
            self.next_tag,
            self.restore_epoch,
            self.prefix,
            self.full_epoch,
            self.seal_epoch,
            self.overflow_count,
            **common,
            ROW_BYTES=width,
            POOL=self.capacity,
            POOL_TILE=triton.next_power_of_2(self.capacity),
            num_warps=4,
        )
        _restore[(n,)](
            cache,
            sorted_slots,
            self.mapping,
            self.tags,
            self.data,
            self.restore_epoch,
            self.prefix,
            self.epoch,
            **common,
            ROW_BYTES=width,
            CENTERS=centers,
            num_warps=8,
        )
        initialize_block_centers(latent, cache, slots, centers, full_only=True)
        _write_deferred[(n,)](
            latent,
            rope,
            cache,
            slots,
            self.data,
            self.mapping,
            self.tags,
            self.full_epoch,
            self.epoch,
            LATENT_STRIDE=latent.stride(0),
            ROPE_STRIDE=rope.stride(0),
            BLOCKS=blocks,
            ROW_BYTES=width,
            CENTERS=centers,
            num_warps=4,
        )
        _plainify_fallback[(n, 64)](
            cache,
            sorted_slots,
            self.mapping,
            self.seal_epoch,
            self.epoch,
            **common,
            ROW_BYTES=width,
            CENTERS=centers,
            num_warps=4,
        )
        _seal_centers[(n,)](
            cache,
            sorted_slots,
            self.data,
            self.mapping,
            self.seal_epoch,
            self.epoch,
            **common,
            ROW_BYTES=width,
            CENTERS=centers,
            num_warps=8,
        )
        _seal_rows[(n, 64)](
            cache,
            sorted_slots,
            self.data,
            self.mapping,
            self.tags,
            self.seal_epoch,
            self.epoch,
            **common,
            ROW_BYTES=width,
            CENTERS=centers,
            num_warps=4,
        )


class _OverflowMonitor:
    """Surfaces GPU-side tail overflow without synchronizing the forward path.

    A poll enqueues an async copy of every pool's counter; a later poll reads it
    once its event completes. It is a no-op while a CUDA Graph is capturing.
    """

    def __init__(self, interval=10.0):
        self.interval = interval
        self.next_poll = 0.0
        self.pending = None

    def poll(self):
        if torch.cuda.is_current_stream_capturing():
            return
        if self.pending is not None:
            event, host, states = self.pending
            if not event.query():
                return
            self.pending = None
            self._report(host.tolist(), states)
        now = time.monotonic()
        if now < self.next_poll:
            return
        device = torch.cuda.current_device()
        states = [s for s in list(_STATES.values()) if s.cache.device.index == device]
        if not states:
            return
        self.next_poll = now + self.interval
        counts = torch.cat([state.overflow_count for state in states])
        host = torch.empty(counts.shape, dtype=counts.dtype, pin_memory=True)
        host.copy_(counts, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self.pending = (event, host, states)

    @staticmethod
    def _report(values, states):
        new = 0
        for state, value in zip(states, values):
            new += max(value - state.reported_overflow, 0)
            state.reported_overflow = value
        if new:
            logging.warning(
                "Deferred NVFP4 tail pool overflowed %d times since last report "
                "(%d total over %d layers, capacity %d tails/layer); those rows "
                "fell back to plain NVFP4. Set RTP_MLA_NVFP4_TAIL_BLOCKS to at "
                "least the per-rank max_generate_batch_size.",
                new,
                sum(values),
                len(states),
                states[0].capacity,
            )


_OVERFLOW_MONITOR = _OverflowMonitor()


def poll_overflow():
    """Cheap and sync-free; call from eager code that runs every step."""
    _OVERFLOW_MONITOR.poll()
