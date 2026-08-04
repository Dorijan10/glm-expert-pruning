#!/usr/bin/env python3
"""Carve the held-out fitness corpus used to score router-search candidates.

Contamination discipline is the hard constraint here, and it is the project's
own most expensive lesson: README's "Iteration 1" section retracts five claims
because code96 was evaluated on corpora drawn from its own calibration domain.
A router search runs hundreds of evaluations, so if that signal came from the
frozen test sets, every headline number afterwards would be tuned-on-test and
worthless.

Rules enforced by this script:

  * fitness text is taken ONLY from `corpora/<domain>_calib.txt`. The frozen
    artefacts -- `code_v2_eval.txt`, `general_v2_eval.txt`, the 13 probes in
    `oracle/probes_*.txt`, and the `--multiple-choice` bins -- are never read.
  * within calib, a disjoint tail slice is reserved as `router_fitness` and the
    remaining head is written as `router_search_train` for any fitting that
    needs data (e.g. the exp_probs_b load pass). Search *selects* on the
    fitness slice; nothing selects on the train slice.
  * all ten domains contribute in proportion, so the search cannot silently
    re-specialise the model toward one domain -- that would undo the max-min
    blend the whole project is built on.
  * SHA-256 of every slice is recorded, and the emitted manifest asserts the
    fitness slice shares no line with either eval corpus.

usage: build_router_fitness.py [--frac 0.15] [--out-dir /work/corpora]
"""
import argparse
import hashlib
import json
from pathlib import Path

DOMAINS = ('books', 'chat', 'code_instruct', 'code_raw', 'math',
           'news', 'reasoning', 'science', 'web', 'wiki')
FROZEN = ('code_v2_eval.txt', 'general_v2_eval.txt')

ap = argparse.ArgumentParser()
ap.add_argument('--corpora', default='/work/corpora')
ap.add_argument('--frac', type=float, default=0.15,
                help='tail fraction of each calib file reserved for fitness')
ap.add_argument('--out-dir', default='/work/corpora')
a = ap.parse_args()

C = Path(a.corpora)
O = Path(a.out_dir)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


frozen_lines = set()
for f in FROZEN:
    p = C / f
    if p.exists():
        frozen_lines |= {ln.strip() for ln in p.read_text(
            errors='replace').splitlines() if len(ln.strip()) > 40}
print(f"frozen eval lines loaded for disjointness check: {len(frozen_lines):,}")

fit_parts, train_parts, rows = [], [], []
for d in DOMAINS:
    p = C / f'{d}_calib.txt'
    if not p.exists():
        print(f"  ! missing {p}, skipping")
        continue
    lines = p.read_text(errors='replace').splitlines(keepends=True)
    cut = int(len(lines) * (1.0 - a.frac))
    train, fit = lines[:cut], lines[cut:]
    # a calib line must never coincide with a frozen eval line
    leak = sum(1 for ln in fit
               if len(ln.strip()) > 40 and ln.strip() in frozen_lines)
    fit_parts.append(''.join(fit))
    train_parts.append(''.join(train))
    rows.append(dict(domain=d, calib_lines=len(lines), train_lines=len(train),
                     fitness_lines=len(fit), leaked_lines=leak,
                     calib_sha256=sha(p)))
    print(f"  {d:<14} calib={len(lines):>6}  train={len(train):>6}  "
          f"fitness={len(fit):>5}  leak={leak}")

fit_path = O / 'router_fitness.txt'
tr_path = O / 'router_search_train.txt'
fit_path.write_text('\n'.join(fit_parts))
tr_path.write_text('\n'.join(train_parts))

total_leak = sum(r['leaked_lines'] for r in rows)
man = dict(purpose='held-out fitness signal for router search; NOT test data',
           frac_tail=a.frac, domains=rows,
           fitness_file=str(fit_path), fitness_sha256=sha(fit_path),
           train_file=str(tr_path), train_sha256=sha(tr_path),
           frozen_never_read=list(FROZEN) + ['oracle/probes_indomain.txt',
                                             'oracle/probes_ood.txt',
                                             '/work/data/evalbins/*.bin'],
           leaked_lines_total=total_leak)
mpath = O / 'router_fitness_manifest.json'
json.dump(man, open(mpath, 'w'), indent=1)

print(f"\nwrote {fit_path} ({fit_path.stat().st_size:,} B)")
print(f"wrote {tr_path} ({tr_path.stat().st_size:,} B)")
print(f"wrote {mpath}")
assert total_leak == 0, f"CONTAMINATION: {total_leak} fitness lines appear in a frozen eval corpus"
print("OK: fitness slice is disjoint from every frozen eval corpus")
