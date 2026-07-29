// MoE expert saliency observer (REAP criterion: gate x ||expert_output||)
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

struct sal_acc {
    std::vector<double> cnt, gate, sal;
    void init(int n){ if((int)cnt.size()!=n){ cnt.assign(n,0.0); gate.assign(n,0.0); sal.assign(n,0.0);} }
};
struct sal_state {
    bool list_mode=false, with_norms=true;
    std::string n_topk="ffn_moe_topk", n_gate="ffn_moe_weights_norm", n_down="ffn_moe_down";
    int n_expert=0; long long tokens=0, n_flush=0;
    std::map<int,sal_acc> acc;
    std::map<int,std::vector<int32_t>> topk;
    std::map<int,std::vector<float>> gate;
    std::map<int,int> nused;
    std::set<std::string> seen;
};

static bool split_name(const std::string& n, std::string& base, int& il){
    size_t p = n.rfind('-');
    if (p == std::string::npos || p+1 >= n.size()) return false;
    for (size_t i=p+1;i<n.size();++i) if (!isdigit((unsigned char)n[i])) return false;
    base = n.substr(0,p); il = std::stoi(n.substr(p+1)); return true;
}
static void fetch(const ggml_tensor* t, std::vector<uint8_t>& b){
    const size_t ts = ggml_type_size(t->type);
    const size_t ne0=(size_t)t->ne[0], ne1=(size_t)t->ne[1], ne2=(size_t)t->ne[2], ne3=(size_t)t->ne[3];
    b.resize(ne0*ne1*ne2*ne3*ts);
    if (ggml_is_contiguous(t)) { ggml_backend_tensor_get(t, b.data(), 0, b.size()); return; }
    size_t off = 0;
    for (size_t i3=0;i3<ne3;++i3) for (size_t i2=0;i2<ne2;++i2) for (size_t i1=0;i1<ne1;++i1){
        const size_t src = i1*t->nb[1] + i2*t->nb[2] + i3*t->nb[3];
        ggml_backend_tensor_get(t, b.data()+off, src, ne0*ts);
        off += ne0*ts;
    }
}
static float fat(const std::vector<uint8_t>& b, ggml_type ty, size_t i){
    if (ty==GGML_TYPE_F32) return ((const float*)b.data())[i];
    if (ty==GGML_TYPE_F16) return ggml_fp16_to_fp32(((const ggml_fp16_t*)b.data())[i]);
    return 0.0f;
}
static void flush_layer(sal_state* S, int il, const std::vector<float>* norms){
    auto it=S->topk.find(il); auto ig=S->gate.find(il);
    if (it==S->topk.end() || ig==S->gate.end()) return;
    const int nu=S->nused[il]; if (nu<=0) return;
    const size_t ntok=it->second.size()/nu;
    auto& A=S->acc[il]; A.init(S->n_expert);
    for (size_t t=0;t<ntok;++t) for (int j=0;j<nu;++j){
        const size_t k=t*(size_t)nu+j;
        const int e=it->second[k]; if (e<0||e>=S->n_expert) continue;
        const float g  = k<ig->second.size() ? ig->second[k] : 0.0f;
        const float nr = (norms && k<norms->size()) ? (*norms)[k] : 1.0f;
        A.cnt[e]+=1.0; A.gate[e]+=g; A.sal[e]+=(double)g*nr;
    }
    S->topk.erase(il); S->gate.erase(il); S->n_flush++;
}

static bool sal_cb(struct ggml_tensor* t, bool ask, void* ud){
    sal_state* S=(sal_state*)ud;
    const std::string name = t->name;
    std::string base; int il;
    if (!split_name(name, base, il)) { return ask ? false : true; }
    if (ask){
        if (S->list_mode){
            if (name.find("ffn_moe")!=std::string::npos && S->seen.insert(base).second)
                printf("TENSOR %-28s type=%-5s ne=[%lld,%lld,%lld,%lld] nb1=%zu contig=%d\n", base.c_str(),
                    ggml_type_name(t->type),(long long)t->ne[0],(long long)t->ne[1],
                    (long long)t->ne[2],(long long)t->ne[3], t->nb[1], (int)ggml_is_contiguous(t));
            return false;
        }
        return base==S->n_topk || base==S->n_gate || (S->with_norms && base==S->n_down);
    }
    static thread_local std::vector<uint8_t> buf;   // reused: avoids ~68M page faults/shard
    if (base==S->n_topk){
        fetch(t,buf);
        S->nused[il]=(int)t->ne[0];
        const int32_t* p=(const int32_t*)buf.data();
        S->topk[il].assign(p,p+(size_t)ggml_nelements(t));
        if (!S->with_norms) flush_layer(S,il,nullptr);
    } else if (base==S->n_gate){
        fetch(t,buf);
        const size_t n=(size_t)ggml_nelements(t);
        std::vector<float> g(n);
        for (size_t i=0;i<n;++i) g[i]=fat(buf,t->type,i);
        S->gate[il]=std::move(g);
        if (!S->with_norms) flush_layer(S,il,nullptr);
    } else if (S->with_norms && base==S->n_down){
        fetch(t,buf);
        const size_t ne0=(size_t)t->ne[0], ne1=(size_t)t->ne[1], ne2=(size_t)t->ne[2];
        std::vector<float> norms(ne1*ne2,0.0f);
        const long long nrow = (long long)(ne1*ne2);
        if (t->type == GGML_TYPE_F32) {
            const float* dptr = (const float*)buf.data();
            #pragma omp parallel for schedule(static)
            for (long long k=0;k<nrow;++k){
                const float* v = dptr + (size_t)k*ne0;
                double s=0.0;
                for (size_t i=0;i<ne0;++i){ const double x=v[i]; s+=x*x; }
                norms[k]=(float)std::sqrt(s);
            }
        } else {
            #pragma omp parallel for schedule(static)
            for (long long k=0;k<nrow;++k){
                double s=0.0; const size_t off=(size_t)k*ne0;
                for (size_t i=0;i<ne0;++i){ const double v=fat(buf,t->type,off+i); s+=v*v; }
                norms[k]=(float)std::sqrt(s);
            }
        }
        flush_layer(S,il,&norms);
    }
    return true;
}

int main(int argc, char** argv){
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;
    sal_state S;
    S.list_mode  = getenv("SAL_LIST")!=nullptr;
    S.with_norms = getenv("SAL_NO_NORMS")==nullptr;
    if (getenv("SAL_TOPK")) S.n_topk=getenv("SAL_TOPK");
    if (getenv("SAL_GATE")) S.n_gate=getenv("SAL_GATE");
    if (getenv("SAL_DOWN")) S.n_down=getenv("SAL_DOWN");
    const char* outp = getenv("SAL_OUT") ? getenv("SAL_OUT") : "/work/logs/saliency.json";

    params.cb_eval = sal_cb; params.cb_eval_user_data = &S; params.warmup = false;
    common_init_result_ptr res = common_init_from_params(params);
    if (!res) { fprintf(stderr,"init failed\n"); return 1; }
    llama_model* model = res->model(); llama_context* ctx = res->context();
    if (!model||!ctx){ fprintf(stderr,"no model/ctx\n"); return 1; }

    char arch[128]={0}, val[128]={0};
    llama_model_meta_val_str(model,"general.architecture",arch,sizeof(arch));
    std::string key = std::string(arch)+".expert_count";
    if (llama_model_meta_val_str(model,key.c_str(),val,sizeof(val))>0) S.n_expert=atoi(val);
    if (S.n_expert<=0) S.n_expert=256;
    fprintf(stderr,"[sal] arch=%s n_expert=%d norms=%d gate=%s\n",arch,S.n_expert,(int)S.with_norms,S.n_gate.c_str());

    std::ifstream f(params.prompt_file);
    std::string text((std::istreambuf_iterator<char>(f)),std::istreambuf_iterator<char>());
    if (text.empty()){ fprintf(stderr,"empty corpus: pass -f <file>\n"); return 1; }
    std::vector<llama_token> toks = common_tokenize(ctx,text,true,true);
    fprintf(stderr,"[sal] %zu tokens\n",toks.size());

    const int n_ctx=llama_n_ctx(ctx);
    const int n_bat=std::min(n_ctx,(int)params.n_batch);
    llama_batch batch = llama_batch_init(n_bat, 0, 1);
    int in_ctx=0;
    for (size_t pos=0; pos<toks.size(); ){
        const int n=(int)std::min((size_t)n_bat, toks.size()-pos);
        common_batch_clear(batch);
        for (int i=0;i<n;++i) common_batch_add(batch, toks[pos+i], in_ctx+i, {0}, true);
        if (llama_decode(ctx,batch)) { fprintf(stderr,"\ndecode failed at %zu\n",pos); break; }
        pos+=n; S.tokens+=n; in_ctx+=n;
        if (in_ctx + n_bat > n_ctx) { llama_memory_clear(llama_get_memory(ctx), true); in_ctx=0; }
        fprintf(stderr,"\r[sal] %zu/%zu flushes=%lld", pos, toks.size(), S.n_flush);
        if (S.list_mode && pos>=(size_t)n_bat) break;
    }
    llama_batch_free(batch);
    fprintf(stderr,"\n");
    if (S.list_mode) return 0;
    if (S.acc.empty()){ fprintf(stderr,"[sal] NO DATA - wrong tensor names? run SAL_LIST=1\n"); return 1; }

    FILE* o=fopen(outp,"w");
    fprintf(o,"{\n \"arch\":\"%s\",\n \"n_expert\":%d,\n \"tokens\":%lld,\n \"layers\":{\n",
        arch,S.n_expert,S.tokens);
    bool first=true;
    for (auto& kv : S.acc){
        if(!first) fprintf(o,",\n");
        first=false;
        fprintf(o,"  \"%d\":{\"cnt\":[",kv.first);
        for(int i=0;i<S.n_expert;++i) fprintf(o,"%s%.0f",i?",":"",kv.second.cnt[i]);
        fprintf(o,"],\"gate\":[");
        for(int i=0;i<S.n_expert;++i) fprintf(o,"%s%.6g",i?",":"",kv.second.gate[i]);
        fprintf(o,"],\"sal\":[");
        for(int i=0;i<S.n_expert;++i) fprintf(o,"%s%.6g",i?",":"",kv.second.sal[i]);
        fprintf(o,"]}");
    }
    fprintf(o,"\n }\n}\n"); fclose(o);
    fprintf(stderr,"[sal] wrote %s (%zu layers, %lld flushes)\n",outp,S.acc.size(),S.n_flush);
    return 0;
}
