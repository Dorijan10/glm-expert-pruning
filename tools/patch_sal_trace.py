import sys
p = sys.argv[1]
s = open(p).read()
assert 'SAL_TRACE' not in s, "already patched"

def sub(old, new):
    global s
    assert s.count(old) == 1, f"anchor matched {s.count(old)}x, expected 1:\n{old[:80]}"
    s = s.replace(old, new)

# 1. state field
sub("    std::set<std::string> seen;\n};",
    "    std::set<std::string> seen;\n    FILE* trace = nullptr;\n};")

# 2. per-token trace emission, before the flush bookkeeping
sub("    S->topk.erase(il); S->gate.erase(il); S->n_flush++;",
    """    if (S->trace) {
        if (nu > 16) { fprintf(stderr,"[sal] nu=%d >16 unsupported\\n", nu); exit(1); }
        for (size_t t=0;t<ntok;++t){
            int32_t hdr[2] = { (int32_t)il, (int32_t)(S->tokens + t) };
            int32_t e[16]; float w[16];
            for (int j=0;j<nu;++j){
                const size_t k=t*(size_t)nu+j;
                e[j] = it->second[k];
                w[j] = k<ig->second.size() ? ig->second[k] : 0.0f;
            }
            fwrite(hdr, sizeof(int32_t), 2,  S->trace);
            fwrite(e,   sizeof(int32_t), nu, S->trace);
            fwrite(w,   sizeof(float),   nu, S->trace);
        }
    }
    S->topk.erase(il); S->gate.erase(il); S->n_flush++;""")

# 3. open on startup
sub('''    const char* outp = getenv("SAL_OUT") ? getenv("SAL_OUT") : "/work/logs/saliency.json";''',
    '''    const char* outp = getenv("SAL_OUT") ? getenv("SAL_OUT") : "/work/logs/saliency.json";
    if (const char* tp = getenv("SAL_TRACE")) {
        S.trace = fopen(tp, "wb");
        if (!S.trace) { fprintf(stderr,"cannot open SAL_TRACE %s\\n", tp); return 1; }
    }''')

# 4. close on exit
sub("""flushes)\\n",cur_out.c_str(),S.acc.size(),S.n_flush);
    }
    return 0;
}""",
    """flushes)\\n",cur_out.c_str(),S.acc.size(),S.n_flush);
    }
    if (S.trace) fclose(S.trace);
    return 0;
}""")

open(p,'w').write(s)
print("patched 4/4 anchors")
