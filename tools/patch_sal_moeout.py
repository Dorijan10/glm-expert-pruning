import sys
p = sys.argv[1]; s = open(p).read()
assert 'SAL_MOEOUT' not in s, "already patched"
def sub(old, new):
    global s
    assert s.count(old) == 1, f"anchor matched {s.count(old)}x:\n{old[:90]}"
    s = s.replace(old, new)

sub("""    std::map<int,long long> hid_rows, hid_sat;
    std::map<int,float> hid_max;
};""",
"""    std::map<int,long long> hid_rows, hid_sat;
    std::map<int,float> hid_max;
    const char* moe_dir = nullptr;
    std::map<int,FILE*> moe_f;
    std::map<int,long long> moe_rows;
};""")

sub('        if (S->hid_dir && base=="ffn_norm" && il>=S->hid_l0 && il<=S->hid_l1) return true;',
'''        if (S->hid_dir && base=="ffn_norm" && il>=S->hid_l0 && il<=S->hid_l1) return true;
        if (S->moe_dir && base=="ffn_moe_out" && il>=S->hid_l0 && il<=S->hid_l1) return true;''')

sub('    if (S->hid_dir && base=="ffn_norm" && il>=S->hid_l0 && il<=S->hid_l1){',
'''    if (S->moe_dir && base=="ffn_moe_out" && il>=S->hid_l0 && il<=S->hid_l1){
        if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)){ fprintf(stderr,"[sal] ffn_moe_out unexpected\\n"); exit(1); }
        fetch(t,buf);
        const size_t ne0=(size_t)t->ne[0], ntok=(size_t)t->ne[1], n=ne0*ntok;
        const float* pf=(const float*)buf.data();
        static thread_local std::vector<ggml_fp16_t> m16;
        m16.resize(n);
        for (size_t i=0;i<n;++i) m16[i]=ggml_fp32_to_fp16(pf[i]);
        FILE*& f = S->moe_f[il];
        if (!f){ char path[512]; snprintf(path,sizeof(path),"%s/M%02d.f16",S->moe_dir,il);
                 f=fopen(path,"wb"); if(!f){ fprintf(stderr,"[sal] cannot open %s\\n",path); exit(1);} }
        if (fwrite(m16.data(),sizeof(ggml_fp16_t),n,f)!=n){ fprintf(stderr,"[sal] short write M%d\\n",il); exit(1); }
        S->moe_rows[il]+=(long long)ntok;
        return true;
    }
    if (S->hid_dir && base=="ffn_norm" && il>=S->hid_l0 && il<=S->hid_l1){''')

sub('    if (const char* hd = getenv("SAL_HIDDEN")) {',
'''    if (const char* md = getenv("SAL_MOEOUT")) {
        S.moe_dir = md;
        fprintf(stderr,"[sal] MOEOUT dir=%s\\n", md);
    }
    if (const char* hd = getenv("SAL_HIDDEN")) {''')

sub("    if (S.hid_dir){\n        long long tot=0, sat=0; float mx=0.0f;",
"""    if (S.moe_dir){
        long long mt=0;
        for (auto& kv : S.moe_f) fclose(kv.second);
        for (auto& kv : S.moe_rows) mt+=kv.second;
        fprintf(stderr,"[sal] MOEOUT layers=%zu rows_total=%lld\\n", S.moe_f.size(), mt);
    }
    if (S.hid_dir){
        long long tot=0, sat=0; float mx=0.0f;""")

open(p,'w').write(s)
print("patched 5/5 anchors")
