import sys
p = sys.argv[1]; s = open(p).read()
assert 'SAL_HIDDEN' not in s, "already patched"
def sub(old, new):
    global s
    assert s.count(old) == 1, f"anchor matched {s.count(old)}x:\n{old[:90]}"
    s = s.replace(old, new)

sub("""    std::set<std::string> seen;
    FILE* trace = nullptr;
};""",
"""    std::set<std::string> seen;
    FILE* trace = nullptr;
    const char* hid_dir = nullptr;
    int hid_l0 = 3, hid_l1 = 77;
    std::map<int,FILE*> hid_f;
    std::map<int,long long> hid_rows, hid_sat;
    std::map<int,float> hid_max;
};""")

sub("        return base==S->n_topk || base==S->n_gate || (S->with_norms && base==S->n_down);",
"""        if (S->hid_dir && base=="ffn_norm" && il>=S->hid_l0 && il<=S->hid_l1) return true;
        return base==S->n_topk || base==S->n_gate || (S->with_norms && base==S->n_down);""")

sub("    static thread_local std::vector<uint8_t> buf;   // reused: avoids ~68M page faults/shard",
"""    static thread_local std::vector<uint8_t> buf;   // reused: avoids ~68M page faults/shard
    if (S->hid_dir && base=="ffn_norm" && il>=S->hid_l0 && il<=S->hid_l1){
        if (t->type != GGML_TYPE_F32){ fprintf(stderr,"[sal] ffn_norm type=%s not f32\\n", ggml_type_name(t->type)); exit(1); }
        if (!ggml_is_contiguous(t)){ fprintf(stderr,"[sal] ffn_norm not contiguous\\n"); exit(1); }
        fetch(t,buf);
        const size_t ne0=(size_t)t->ne[0], ntok=(size_t)t->ne[1], n=ne0*ntok;
        const float* pf=(const float*)buf.data();
        static thread_local std::vector<ggml_fp16_t> h16;
        h16.resize(n);
        float mx=0.0f; long long sat=0;
        for (size_t i=0;i<n;++i){
            const float v=pf[i]; const float a=fabsf(v);
            if (a>mx) mx=a;
            if (!std::isfinite(v) || a>65504.0f) sat++;
            h16[i]=ggml_fp32_to_fp16(v);
        }
        FILE*& f = S->hid_f[il];
        if (!f){
            char path[512];
            snprintf(path,sizeof(path),"%s/L%02d.f16",S->hid_dir,il);
            f=fopen(path,"wb");
            if (!f){ fprintf(stderr,"[sal] cannot open %s\\n",path); exit(1); }
        }
        if (fwrite(h16.data(),sizeof(ggml_fp16_t),n,f)!=n){ fprintf(stderr,"[sal] short write L%d\\n",il); exit(1); }
        S->hid_rows[il]+=(long long)ntok;
        S->hid_sat[il]+=sat;
        if (mx>S->hid_max[il]) S->hid_max[il]=mx;
        return true;
    }""")

sub("""    if (const char* tp = getenv("SAL_TRACE")) {""",
"""    if (const char* hd = getenv("SAL_HIDDEN")) {
        S.hid_dir = hd;
        if (const char* a = getenv("SAL_HID_L0")) S.hid_l0 = atoi(a);
        if (const char* a = getenv("SAL_HID_L1")) S.hid_l1 = atoi(a);
        fprintf(stderr,"[sal] HIDDEN dir=%s layers=%d..%d\\n", hd, S.hid_l0, S.hid_l1);
    }
    if (const char* tp = getenv("SAL_TRACE")) {""")

sub("    if (S.trace) fclose(S.trace);",
"""    if (S.trace) fclose(S.trace);
    if (S.hid_dir){
        long long tot=0, sat=0; float mx=0.0f;
        for (auto& kv : S.hid_f) fclose(kv.second);
        for (auto& kv : S.hid_rows) tot+=kv.second;
        for (auto& kv : S.hid_sat)  sat+=kv.second;
        for (auto& kv : S.hid_max)  if (kv.second>mx) mx=kv.second;
        fprintf(stderr,"[sal] HIDDEN layers=%zu rows_total=%lld max_abs=%.4f saturated=%lld\\n",
                S.hid_f.size(), tot, mx, sat);
        for (auto& kv : S.hid_rows)
            fprintf(stderr,"[sal] HID L%02d rows=%lld max=%.4f sat=%lld\\n",
                    kv.first, kv.second, S.hid_max[kv.first], S.hid_sat[kv.first]);
    }""")

open(p,'w').write(s)
print("patched 5/5 anchors")
