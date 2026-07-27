import re, glob, os
from collections import Counter

def clean(t):
    t = re.sub(r'^\[ Prompt:.*$', '', t, flags=re.M)
    t = re.sub(r'^Exiting\.\.\.$', '', t, flags=re.M)
    return t.strip()

def distinct_n(words, n=4):
    if len(words) < n: return 1.0
    g = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    return len(set(g))/len(g)

def max_line_rep(t):
    lines = [l.strip() for l in t.splitlines() if len(l.strip()) > 15]
    return max(Counter(lines).values()) if lines else 0

def max_word_run(words):
    best = run = 1
    for i in range(1, len(words)):
        run = run+1 if words[i] == words[i-1] else 1
        best = max(best, run)
    return best

def parse(path):
    out, cur, buf = [], None, []
    for line in open(path, errors='replace'):
        m = re.match(r'=====\[(IN|OOD)-(\d+)\] (.*)', line)
        if m:
            if cur: out.append((*cur, ''.join(buf)))
            cur, buf = (m.group(1), int(m.group(2)), m.group(3)), []
        elif cur is not None:
            buf.append(line)
    if cur: out.append((*cur, ''.join(buf)))
    return out

order = ['parent','code96','code96_guard','code80','code108','gen96','random96']
files = {os.path.basename(p)[7:-4]: p for p in glob.glob('/work/logs/probes_*.txt')}
print(f"{'model':<10} {'set':<4} {'n':>2} {'distinct4':>10} {'maxrep':>7} {'wordrun':>8} {'words':>6} {'code':>7}  verdict")
print("-"*84)
detail = {}
for tag in order:
    if tag not in files: continue
    for kind in ('IN','OOD'):
        rows = [b for b in parse(files[tag]) if b[0]==kind]
        if not rows: continue
        ds, rs, us, ws, code, bad, badlist = [], [], [], [], 0, 0, []
        for _, i, prompt, body in rows:
            body = clean(body); w = body.split()
            d = distinct_n(w); r = max_line_rep(body); u = max_word_run(w)
            hc = 1 if ('```' in body or re.search(r'\b(return|for |while |def |int |void )', body)) else 0
            ds.append(d); rs.append(r); us.append(u); ws.append(len(w)); code += hc
            if d < 0.55 or r >= 4 or u >= 5:
                bad += 1; badlist.append(i)
        n=len(rows)
        v = "DEGENERATE" if bad >= n/2 else ("some-degen" if bad else "clean")
        detail[(tag,kind)] = badlist
        print(f"{tag:<10} {kind:<4} {n:>2} {sum(ds)/n:>10.3f} {max(rs):>7} {max(us):>8} {sum(ws)//n:>6} {code:>4}/{n}  {v} ({bad}/{n} bad)")
print("\ndistinct4 = unique 4-gram ratio (1.0 = none repeated; <0.55 = looping)")
print("maxrep    = most-repeated line;  wordrun = longest run of one repeated word")
print("code      = responses containing code-like content")
print("\nbad prompt indices:", {k:v for k,v in detail.items() if v})
