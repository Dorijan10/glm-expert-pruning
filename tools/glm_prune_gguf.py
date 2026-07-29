#!/usr/bin/env python3
"""Whole-expert prune of a GLM MoE GGUF. Surviving experts are copied
byte-for-byte (no requantization). Router rows, expert-count metadata and
optionally whole blocks (MTP) are sliced/dropped to match."""
import argparse, json, re, sys
import numpy as np
from gguf import GGUFReader, GGUFWriter

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True)
ap.add_argument('--dst', required=True)
ap.add_argument('--keeplist', required=True)
ap.add_argument('--drop-blocks', default='', help='comma list, e.g. 78 (MTP)')
ap.add_argument('--temp-file', action='store_true', help='spool via TMPDIR instead of RAM')
ap.add_argument('--dry-run', action='store_true')
a = ap.parse_args()

kl = json.load(open(a.keeplist))
NE_SRC = int(kl['n_expert'])
KEEP = {int(k): sorted(int(x) for x in v) for k, v in kl['keep'].items()}
K = len(next(iter(KEEP.values())))
assert all(len(v) == K for v in KEEP.values()), "non-uniform keep counts unsupported"
assert K >= 8, "K must be >= expert_used_count"
DROP = set(int(x) for x in a.drop_blocks.split(',') if x.strip())

EXP  = ('ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps')
ROUT = ('ffn_gate_inp', 'exp_probs_b')

r = GGUFReader(a.src)
arch = r.fields['general.architecture'].contents()
print(f"arch={arch} src_experts={NE_SRC} keep={K} moe_layers={len(KEEP)} drop_blocks={sorted(DROP)}")

w = None if a.dry_run else GGUFWriter(a.dst, arch, use_temp_file=a.temp_file)

n_ec = 0
for key, f in r.fields.items():
    if key.startswith('split.') or key.startswith('GGUF.') or key == 'general.architecture':
        continue
    val = f.contents()
    if key.endswith('.expert_count'):
        print(f"  KV {key}: {val} -> {K}"); val = K; n_ec += 1
    elif key.endswith('.block_count') and DROP:
        print(f"  KV {key}: {val} -> {val - len(DROP)}"); val = val - len(DROP)
    elif key.endswith('.nextn_predict_layers') and DROP:
        print(f"  KV {key}: {val} -> 0"); val = 0
    if w:
        vt = f.types[0]
        st = f.types[1] if len(f.types) > 1 else None
        try:
            w.add_key_value(key, val, vt, sub_type=st)
        except Exception as e:
            print(f"  ! KV {key} SKIPPED ({e})", file=sys.stderr)
assert n_ec == 1, f"expected exactly one *.expert_count key, found {n_ec}"

st = dict(exp=0, rout=0, copy=0, drop=0, bin=0, bout=0)
for t in r.tensors:
    name = t.name
    m  = re.match(r'blk\.(\d+)\.', name)
    il = int(m.group(1)) if m else None
    st['bin'] += int(t.n_bytes)
    if il is not None and il in DROP:
        st['drop'] += 1
        continue
    ne   = [int(x) for x in t.shape]      # ne order (expert is LAST)
    data = t.data                          # numpy, REVERSED order (expert is axis 0)
    if il is not None and il in KEEP and (any(e in name for e in EXP) or any(q in name for q in ROUT)):
        assert ne[-1] == NE_SRC, f"{name}: expected ne[-1]={NE_SRC}, got {ne}"
        assert data.shape[0] == NE_SRC, f"{name}: numpy axis0={data.shape[0]}"
        data  = data[KEEP[il]]
        ne[-1] = K
        st['exp' if any(e in name for e in EXP) else 'rout'] += 1
    else:
        st['copy'] += 1
    st['bout'] += int(np.asarray(data).nbytes)
    if w:
        w.add_tensor(name, np.ascontiguousarray(data), raw_dtype=t.tensor_type)

print(f"experts_sliced={st['exp']} routers_sliced={st['rout']} copied={st['copy']} dropped={st['drop']}")
print(f"bytes in={st['bin']/2**30:.2f} GiB  out={st['bout']/2**30:.2f} GiB")
if w:
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {a.dst}")
