#!/usr/bin/env python3
"""Phase G corpora v2 — English only, 6 domains, document-level disjoint splits."""
import gzip, hashlib, json, os, random, sys
from pathlib import Path

SEED = 20260728
OUT  = Path("/work/oracle/v2")
OUT.mkdir(parents=True, exist_ok=True)

# byte budgets (chars/token: code 3.2, prose 4.2, chat 4.0, math 3.5)
CALIB = {"code_raw":640_000, "code_instruct":640_000, "web":840_000,
         "wiki":840_000, "chat":800_000, "math":700_000,
         "books":840_000, "science":840_000, "news":840_000, "reasoning":1_000_000}
EVAL  = {"code_raw":240_000, "code_instruct":240_000, "web":170_000,
         "wiki":170_000, "chat":160_000, "math":140_000,
         "books":170_000, "science":170_000, "news":170_000, "reasoning":200_000}

TMPL = None
tp = OUT/"chat_template.jinja"
if tp.exists():
    from jinja2 import Template
    TMPL = Template(tp.read_text())

def as_chat(msgs):
    if TMPL:
        try:
            return TMPL.render(messages=msgs, add_generation_prompt=False, enable_thinking=False)
        except Exception as e:
            print("TEMPLATE FAIL, falling back:", e, file=sys.stderr)
    return "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in msgs)

def docs_code_raw():
    root = Path("/work/dev/llama.cpp")
    fs = sorted([p for ext in ("*.cpp","*.c","*.h","*.hpp","*.cu","*.py")
                 for p in root.rglob(ext) if ".git" not in str(p)])
    random.Random(SEED).shuffle(fs)   # sample the repo, not its alphabetical prefix
    for p in fs:
        try:
            t = p.read_text(errors="ignore")
            if len(t) > 200: yield t
        except Exception: pass

def docs_code_instruct():
    for p in sorted(Path("/work/data/evol-codealpaca").rglob("*.json*")):
        try:
            import pandas as pd
            df = pd.read_parquet(p) if p.suffix==".parquet" else pd.read_json(p, lines=p.suffix==".jsonl")
        except Exception: continue
        ic = "instruction" if "instruction" in df.columns else df.columns[0]
        oc = "output" if "output" in df.columns else df.columns[-1]
        for _, r in df.iterrows():
            yield as_chat([{"role":"user","content":str(r[ic])},
                           {"role":"assistant","content":str(r[oc])}])

def docs_web():
    p = next(Path("/work/data/c4").rglob("c4-train*.json.gz"))
    with gzip.open(p, "rt") as f:
        for line in f:
            t = json.loads(line).get("text","")
            if len(t) > 500: yield t

def docs_wiki():
    import pandas as pd
    buf = []
    for p in sorted(Path("/work/data/wikitext").rglob("*.parquet")):
        for t in pd.read_parquet(p)["text"]:
            if t.startswith(" = ") and not t.startswith(" = = "):
                if buf: yield "".join(buf)
                buf = [t]
            else: buf.append(t)
    if buf: yield "".join(buf)

def docs_chat():
    import pandas as pd
    for p in sorted(Path("/work/data/ultrachat").rglob("*.parquet")):
        for msgs in pd.read_parquet(p)["messages"]:
            m = [{"role":x["role"],"content":x["content"]} for x in msgs]
            if m: yield as_chat(m)

def docs_math():
    import pandas as pd
    for p in sorted(Path("/work/data/gsm8k").rglob("*train*.parquet")):
        df = pd.read_parquet(p)
        for _, r in df.iterrows():
            yield as_chat([{"role":"user","content":str(r["question"])},
                           {"role":"assistant","content":str(r["answer"])}])


def _parquet_docs(root, cols, minlen=500, maxlen=None):
    import pandas as pd
    for p in sorted(Path(root).rglob("*.parquet")):
        try: df = pd.read_parquet(p)
        except Exception: continue
        c = next((c for c in cols if c in df.columns), None)
        if c is None:
            print(f"WARN {p.name}: none of {cols} in {list(df.columns)[:6]}", file=sys.stderr); continue
        for t in df[c]:
            t = str(t)
            if len(t) > minlen: yield (t[:maxlen] if maxlen else t)

def docs_books():
    import pandas as pd, random as _r
    rnd = _r.Random(SEED)
    for p in sorted(Path("/work/data/gutenberg").rglob("*.parquet")):
        try: df = pd.read_parquet(p)
        except Exception: continue
        c = next((c for c in ("TEXT","text") if c in df.columns), None)
        if c is None: continue
        for t in df[c]:
            t = str(t)
            if len(t) < 20_000: continue          # need a real body
            lo = min(5_000, len(t)//10)           # skip PG boilerplate/front matter
            if len(t) > lo + 20_000:
                st = rnd.randrange(lo, len(t) - 20_000)
                yield t[st:st+20_000]             # random window from the body
            else:
                yield t[lo:]
def docs_science():  yield from _parquet_docs("/work/data/arxiv2",   ["markdown","abstract"], maxlen=20_000)
def docs_news():     yield from _parquet_docs("/work/data/news",     ["article","text"])

def docs_reasoning():
    # raw role-tagged render, KEEPING <think> spans — deliberately not the enable_thinking=False template
    import pandas as pd
    for p in sorted(Path("/work/data/mot").rglob("*.parquet")):
        try: df = pd.read_parquet(p)
        except Exception: continue
        if "messages" not in df.columns:
            print(f"WARN {p.name}: no messages col", file=sys.stderr); continue
        for msgs in df["messages"]:
            parts = ["[gMASK]<sop>"]
            for x in msgs: parts.append(f"<|{x['role']}|>{x['content']}")
            d = "".join(parts)[:20_000]   # more distinct traces at the same byte budget
            if len(d) > 500: yield d

SRC = {"code_raw":docs_code_raw, "code_instruct":docs_code_instruct,
       "web":docs_web, "wiki":docs_wiki, "chat":docs_chat, "math":docs_math,
       "books":docs_books, "science":docs_science, "news":docs_news, "reasoning":docs_reasoning}

MAXDOC = 60_000          # truncate pathological single documents
SLACK  = 1.05            # allow 5% overshoot, never a whole giant file

def h(x): return hashlib.sha256(x.encode("utf-8", "ignore")).hexdigest()

manifest = {"seed":SEED, "chat_template": "gguf" if TMPL else "fallback",
            "maxdoc":MAXDOC, "domains":{}}
for name, fn in SRC.items():
    need = CALIB[name] + EVAL[name]
    pool, total = [], 0
    for d in fn():
        if len(d) > MAXDOC: d = d[:MAXDOC]
        pool.append(d); total += len(d)
        if total > need * 4: break
    random.Random(SEED).shuffle(pool)
    cal, ev, cb, eb = [], [], 0, 0
    for d in pool:
        if cb < CALIB[name]:
            if cb + len(d) <= CALIB[name] * SLACK: cal.append(d); cb += len(d)
        elif eb < EVAL[name]:
            if eb + len(d) <= EVAL[name] * SLACK: ev.append(d); eb += len(d)
        else:
            break
    (OUT/f"{name}_calib.txt").write_text("\n\n".join(cal))
    with open(OUT/f"{name}_calib.jsonl","w") as fh:
        for d in cal: fh.write(json.dumps({"text": d}) + "\n")
    (OUT/f"{name}_eval.txt").write_text("\n\n".join(ev))
    manifest["domains"][name] = {
        "pool_docs":len(pool), "calib_docs":len(cal), "calib_bytes":cb,
        "eval_docs":len(ev), "eval_bytes":eb,
        "calib_sha":[h(d) for d in cal], "eval_sha":[h(d) for d in ev]}
    print(f"{name:15s} calib {cb:>9,}B/{len(cal):>5} docs   eval {eb:>9,}B/{len(ev):>5} docs")

# combined eval corpora for PPL scoring
def cat(names, dst):
    dst.write_text("\n\n".join((OUT/f"{n}_eval.txt").read_text() for n in names))
    print(f"{dst.name}: {dst.stat().st_size:,} B")
cat(["code_raw","code_instruct"], OUT/"code_v2_eval.txt")
cat(["web","wiki","chat","math"], OUT/"general_v2_eval.txt")

json.dump(manifest, open(OUT/"manifest.json","w"), indent=2)
print("MANIFEST", OUT/"manifest.json")
