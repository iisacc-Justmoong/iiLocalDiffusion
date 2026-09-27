#include <core/compute_workspace.h>
#include <model/common/ggml_block.hpp>
#include <ggml-metal.h>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
struct ProbeLinear : Linear {
    ProbeLinear(ggml_tensor *w, ggml_tensor *s) : Linear(64, 64, false) {
        params["weight"] = w; params["weight_scale"] = s; has_weight_scale = true;
    }
};
int main() {
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> metal(ggml_backend_metal_init(), ggml_backend_free);
    if (!metal) return 1;
    for (auto type : {GGML_TYPE_F8_E4M3, GGML_TYPE_F8_E5M2}) {
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({32*ggml_tensor_overhead()+ggml_graph_overhead_custom(64,false),nullptr,true}),ggml_free);
        auto *w=ggml_new_tensor_2d(ctx.get(),type,64,64);
        auto *s=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,1);
        auto *x=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,64,16);
        std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),metal.get()),ggml_backend_buffer_free);
        ggml_backend_buffer_set_usage(buffer.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::vector<uint8_t> ones(4096,type==GGML_TYPE_F8_E4M3 ? 0x38 : 0x3c);
        std::vector<float> input(1024,1.f);float scale=.5f;
        ggml_backend_tensor_set(w,ones.data(),0,ones.size());ggml_backend_tensor_set(x,input.data(),0,input.size()*4);ggml_backend_tensor_set(s,&scale,0,4);
        if (ggml_backend_supports_op(metal.get(),ggml_mul_mat(ctx.get(),w,x))) {
            std::cerr << "Metal falsely advertises FP8 matrix kernels\n";return 2;
        }
        ProbeLinear linear(w,s);GGMLRunnerContext runner;runner.backend=metal.get();runner.ggml_ctx=ctx.get();
        auto *out=linear.forward(&runner,x);auto *graph=ggml_new_graph_custom(ctx.get(),64,false);ggml_build_forward_expand(graph,out);
        sd::ComputeWorkspace workspace(metal.get());auto assign=[](ggml_backend_sched_t,ggml_cgraph*){};
        auto measured=workspace.measure(graph,0,[&](const ggml_tensor*t){return t==w||t==x||t==s ? metal.get():nullptr;},assign);
        if(!workspace.prepare(measured)||!workspace.allocate(graph,assign))return 3;
        const auto status=workspace.scheduler() ? ggml_backend_sched_graph_compute(workspace.scheduler(),graph) : ggml_backend_graph_compute(metal.get(),graph);
        if(status!=GGML_STATUS_SUCCESS)return 4;
        std::vector<float> result(1024);ggml_backend_tensor_get(out,result.data(),0,result.size()*4);workspace.segment_end();
        for(float v:result)if(!std::isfinite(v)||std::abs(v-32.f)>.001f)return 5;
    }
    std::cout<<"FP8 Linear uses BF16 conversion and preserves weight scale on Metal\n";
}
