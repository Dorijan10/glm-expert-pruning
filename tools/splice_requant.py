#!/usr/bin/env python3
"""Take change-list tensors from a requantized file, byte-copy everything else."""
import argparse, re, sys
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType

def tname(tt):
    try: return GGMLQuantizationType(int(tt)).name
    except Exception: return str(tt)

ap = argparse.ArgumentParser()
ap.add_argument('--orig', required=True)
ap.add_argument('--quant', required=True)
ap.add_argument('--dst', required=True)
ap.add_argument('--recipe', required=True)
ap.add_argument('--dry-run', action='store_true')
a = ap.parse_args()

pats = []
for ln in open(a.recipe):
    ln = ln.strip()
    if not ln or '=' not in ln: continue
    p = ln.rsplit('=', 1)[0]
    if '_exps' in p: continue          # experts stay original
    pats.append(re.compile(p))
print(f'change-list patterns: {len(pats)}')

ro, rq = GGUFReader(a.orig), GGUFReader(a.quant)
Q = {t.name: t for t in rq.tensors}
arch = ro.fields['general.architecture'].contents()
w = None if a.dry_run else GGUFWriter(a.dst, arch, use_temp_file=True)

for key, f in ro.fields.items():
    if key.startswith('split.') or key.startswith('GGUF.') or key == 'general.architecture':
        continue
    if w:
        vt = f.types[0]; st = f.types[1] if len(f.types) > 1 else None
        try: w.add_key_value(key, f.contents(), vt, sub_type=st)
        except Exception as e: print(f'  ! KV {key} SKIPPED ({e})', file=sys.stderr)

n_sw = n_cp = 0; b_in = b_out = 0; kinds = {}
for t in ro.tensors:
    b_in += int(t.n_bytes)
    if any(p.search(t.name) for p in pats):
        s = Q[t.name]
        assert list(s.shape) == list(t.shape), f'{t.name}: {s.shape} vs {t.shape}'
        n_sw += 1
        k = (tname(t.tensor_type), tname(s.tensor_type))
        kinds[k] = kinds.get(k, 0) + 1
    else:
        s = t; n_cp += 1
    b_out += int(s.n_bytes)
    if w: w.add_tensor(t.name, np.ascontiguousarray(s.data), raw_dtype=s.tensor_type)

for k, v in sorted(kinds.items()): print(f'  {k[0]:9s} -> {k[1]:9s}  n={v}')
print(f'swapped={n_sw} copied={n_cp}')
print(f'in={b_in/2**30:.2f} GiB  out={b_out/2**30:.2f} GiB  saved={(b_in-b_out)/2**30:.2f} GiB')
if w:
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f'wrote {a.dst}')
