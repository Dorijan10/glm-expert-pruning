#!/usr/bin/env python3
"""In-place F32 router I/O for a pruned GLM MoE GGUF.

The router tensors of this architecture are stored UNQUANTIZED (F32):

    blk.N.ffn_gate_inp.weight   ne=[n_embd, n_expert]   numpy (n_expert, n_embd)
    blk.N.exp_probs_b.bias      ne=[n_expert]           numpy (n_expert,)

confirmed from the parent model load: `blk.78.ffn_gate_inp.weight (size =
6291456 bytes)` == 6144 * 256 * 4. Two consequences drive this whole module:

1. A router edit needs NO requantization pass and is bit-exact. The goal brief
   assumed routers would have to be re-quantized; they do not, because they were
   never quantized. `glm_prune_gguf.py` already copies them with
   `raw_dtype=t.tensor_type`, so what is on disk is plain little-endian F32.

2. A router tensor never changes size and lives at a fixed file offset, so a
   candidate router can be applied by seeking into the existing 100.32 GiB
   artefact and overwriting 2.65 MB per layer (~199 MB for all 75 MoE layers)
   instead of re-running the slicer, which writes 100.32 GiB per candidate.
   That is ~0.19% of the bytes, and it is what makes a search loop feasible.

INVARIANT enforced here: a write may only ever touch the exact byte range of a
router tensor. Every intended range is checked against the byte ranges of ALL
tensors in the file, and the write is refused unless the only tensor it
intersects is its own target, with byte-identical start and end. The frozen
expert tensors (ffn_gate_exps / ffn_up_exps / ffn_down_exps) are therefore
untouchable by construction rather than by convention.

The `data_offset` assumption is self-checking: before writing, the bytes already
at the computed offset are compared against the memmapped tensor contents, and a
mismatch aborts. After writing, a fresh reader re-reads the region and compares
against the intended array.
"""
import hashlib
import os
import re

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType

ROUTER_SUFFIX = ('ffn_gate_inp.weight', 'exp_probs_b.bias')
EXPERT_FAMILIES = ('ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps')

# bytes the legacy path writes for one candidate: the whole artefact
RESLICE_BYTES = 100.32 * 2**30


def _layer_of(name):
    m = re.match(r'blk\.(\d+)\.', name)
    return int(m.group(1)) if m else None


def tensor_ranges(reader):
    """[(start, end, name)] byte range of every tensor's data in the file."""
    return [(int(t.data_offset), int(t.data_offset) + int(t.n_bytes), t.name)
            for t in reader.tensors]


def router_tensor_names(reader):
    """Sorted router tensor names present, grouped by layer: {il: [names]}."""
    out = {}
    for t in reader.tensors:
        il = _layer_of(t.name)
        if il is None:
            continue
        if any(t.name.endswith(s) for s in ROUTER_SUFFIX):
            out.setdefault(il, []).append(t.name)
    return {il: sorted(v) for il, v in sorted(out.items())}


def _assert_f32(t):
    if t.tensor_type != GGMLQuantizationType.F32:
        raise AssertionError(
            f"{t.name}: expected F32 router tensor, found {t.tensor_type.name}. "
            "In-place patching is only sound for unquantized tensors.")


def _assert_exclusive(ranges, name, start, end):
    """The intended write range must coincide exactly with tensor `name` and
    must not intersect any other tensor in the file."""
    hits = [(s, e, n) for (s, e, n) in ranges if s < end and start < e]
    if len(hits) != 1 or hits[0][2] != name:
        others = [n for (_, _, n) in hits if n != name]
        raise AssertionError(
            f"refusing write [{start},{end}) for {name}: it intersects "
            f"{len(hits)} tensor(s); foreign={others[:4]}")
    s, e, _ = hits[0]
    if (s, e) != (start, end):
        raise AssertionError(
            f"refusing write for {name}: range [{start},{end}) != tensor "
            f"range [{s},{e}) -- partial-tensor writes are not allowed")
    for fam in EXPERT_FAMILIES:
        if fam in name:
            raise AssertionError(f"refusing to write frozen expert tensor {name}")


def sha256_region(path, start, nbytes, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        f.seek(start)
        left = nbytes
        while left:
            b = f.read(min(chunk, left))
            if not b:
                raise IOError(f"short read at {start} ({left} bytes left)")
            h.update(b)
            left -= len(b)
    return h.hexdigest()


def dump_routers(path):
    """Read every router tensor out of `path` into {name: float32 ndarray}."""
    r = GGUFReader(path)
    out = {}
    for il, names in router_tensor_names(r).items():
        for n in names:
            t = next(t for t in r.tensors if t.name == n)
            _assert_f32(t)
            out[n] = np.array(t.data, dtype=np.float32, copy=True)
    return out


def slice_parent_routers(parent_path, keep):
    """Baseline bundle: the verbatim row-subset of the parent's routers, i.e.
    exactly what `glm_prune_gguf.py` wrote. Used to (a) confirm the artefact
    starts in the known post-slice state and (b) revert bit-exactly.

    `keep` is {il: [expert ids]}; numpy axis 0 is the expert axis (ne order has
    the expert axis last, numpy order reverses it), matching the slicer.
    """
    r = GGUFReader(parent_path)
    out = {}
    for il, names in router_tensor_names(r).items():
        if il not in keep:
            continue
        for n in names:
            t = next(t for t in r.tensors if t.name == n)
            _assert_f32(t)
            out[n] = np.array(t.data[keep[il]], dtype=np.float32, copy=True)
    return out


def patch_routers(path, edits, dry_run=False, verify=True):
    """Overwrite router byte ranges of `path` in place.

    edits: {tensor_name: ndarray} -- shape and byte length must match exactly.
    Returns (n_tensors, bytes_written).
    """
    r = GGUFReader(path)
    tmap = {t.name: t for t in r.tensors}
    ranges = tensor_ranges(r)
    fsize = os.path.getsize(path)

    plan = []
    for name, arr in sorted(edits.items()):
        if name not in tmap:
            raise KeyError(f"{name} not present in {path}")
        t = tmap[name]
        _assert_f32(t)
        if not any(name.endswith(s) for s in ROUTER_SUFFIX):
            raise AssertionError(f"{name} is not a router tensor")
        a = np.ascontiguousarray(arr, dtype=np.float32)
        if tuple(a.shape) != tuple(t.data.shape):
            raise AssertionError(
                f"{name}: edit shape {a.shape} != on-disk shape {t.data.shape}")
        if a.nbytes != int(t.n_bytes):
            raise AssertionError(
                f"{name}: edit is {a.nbytes} B, tensor is {int(t.n_bytes)} B")
        if not np.isfinite(a).all():
            raise AssertionError(f"{name}: edit contains non-finite values")
        start = int(t.data_offset)
        end = start + a.nbytes
        if end > fsize:
            raise AssertionError(f"{name}: range beyond EOF ({end} > {fsize})")
        _assert_exclusive(ranges, name, start, end)
        plan.append((name, t, a, start, end))

    total = sum(a.nbytes for _, _, a, _, _ in plan)
    if dry_run:
        print(f"[dry-run] would patch {len(plan)} tensors, {total:,} bytes")
        return len(plan), 0

    with open(path, 'r+b') as f:
        for name, t, a, start, _ in plan:
            # self-check the offset assumption before trusting it for a write
            f.seek(start)
            cur = f.read(a.nbytes)
            if cur != t.data.tobytes():
                raise AssertionError(
                    f"{name}: bytes at data_offset {start} do not match the "
                    "memmapped tensor -- offset assumption is wrong, aborting")
            f.seek(start)
            f.write(a.tobytes())
        f.flush()
        os.fsync(f.fileno())

    if verify:
        r2 = GGUFReader(path)
        m2 = {t.name: t for t in r2.tensors}
        for name, _, a, _, _ in plan:
            got = np.array(m2[name].data, dtype=np.float32)
            if not np.array_equal(got, a):
                raise AssertionError(f"{name}: readback differs from intent")
        if os.path.getsize(path) != fsize:
            raise AssertionError("file size changed during patch")

    pct = 100.0 * total / RESLICE_BYTES
    print(f"patched tensors={len(plan)} bytes={total:,} "
          f"({pct:.3f}% of a full re-slice) verified={verify}")
    return len(plan), total
