#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-alloc.h>
#include <ggml-webgpu.h>
#include <array>
#include <atomic>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

void check(bool ok,const char *message) { if(!ok) throw std::runtime_error(message); }
void execute(ggml_backend_t backend) {
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(
        ggml_init({32*ggml_tensor_overhead()+ggml_graph_overhead_custom(32,false),nullptr,true}),ggml_free);
    check(bool(ctx),"Cannot create graph");
    auto *a=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,2,2);
    auto *b=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,2,1);
    auto *result=ggml_mul_mat(ctx.get(),a,b);
    auto *graph=ggml_new_graph_custom(ctx.get(),32,false); ggml_build_forward_expand(graph,result);
    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> storage(
        ggml_backend_alloc_ctx_tensors(ctx.get(),backend),ggml_backend_buffer_free);
    check(bool(storage),"Cannot allocate graph");
    const std::array<float,4> weights{1,2,3,4}; const std::array<float,2> input{5,6};
    std::array<float,2> output{};
    ggml_backend_tensor_set(a,weights.data(),0,sizeof(weights)); ggml_backend_tensor_set(b,input.data(),0,sizeof(input));
    check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"Dawn graph failed");
    ggml_backend_tensor_get(result,output.data(),0,sizeof(output));
    check(std::abs(output[0]-17.f)<.01f && std::abs(output[1]-39.f)<.01f,"Dawn graph result differs");
}
void executeConvolution(ggml_backend_t backend) {
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(
        ggml_init({32*ggml_tensor_overhead()+ggml_graph_overhead_custom(32,false),nullptr,true}),ggml_free);
    check(bool(ctx),"Cannot create convolution graph");
    auto *weights=ggml_new_tensor_4d(ctx.get(),GGML_TYPE_F32,3,3,1,1);
    auto *input=ggml_new_tensor_4d(ctx.get(),GGML_TYPE_F32,3,3,1,1);
    auto *result=ggml_conv_2d_direct(ctx.get(),weights,input,1,1,1,1,1,1);
    check(ggml_backend_dev_supports_op(ggml_backend_get_device(backend),result),"Dawn GPU does not support convolution");
    auto *graph=ggml_new_graph_custom(ctx.get(),32,false); ggml_build_forward_expand(graph,result);
    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> storage(
        ggml_backend_alloc_ctx_tensors(ctx.get(),backend),ggml_backend_buffer_free);
    check(bool(storage),"Cannot allocate convolution graph");
    const std::array<float,9> kernel{1,1,1,1,1,1,1,1,1}, pixels{1,2,3,4,5,6,7,8,9};
    std::array<float,9> output{};
    ggml_backend_tensor_set(weights,kernel.data(),0,sizeof(kernel)); ggml_backend_tensor_set(input,pixels.data(),0,sizeof(pixels));
    check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"Dawn convolution failed");
    ggml_backend_tensor_get(result,output.data(),0,sizeof(output));
    check(std::abs(output[0]-12.f)<.01f && std::abs(output[4]-45.f)<.01f && std::abs(output[8]-28.f)<.01f,
          "Dawn convolution result differs");
}
int main() {
    try {
        auto reg1=ggml_backend_webgpu_reg();
        check(reg1 && ggml_backend_reg_dev_count(reg1)>0,"Dawn GPU is unavailable");
        auto dev1=ggml_backend_reg_dev_get(reg1,0);
        std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> first(ggml_backend_dev_init(dev1,nullptr),ggml_backend_free);
        check(bool(first),"First Dawn session failed");
        auto reg2=ggml_backend_webgpu_reg();
        check(reg1==reg2 && ggml_backend_reg_dev_count(reg2)>0,"Repeated Dawn registry access lost its device");
        auto dev2=ggml_backend_reg_dev_get(reg2,0);
        std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> second(ggml_backend_dev_init(dev2,nullptr),ggml_backend_free);
        check(bool(second),"Second Dawn session failed");
        std::atomic_bool consistent{true}; std::vector<std::thread> threads;
        for(int i=0;i<4;++i) threads.emplace_back([&] {
            const auto reg=ggml_backend_webgpu_reg();
            if(reg!=reg1 || !ggml_backend_reg_dev_count(reg) || ggml_backend_reg_dev_get(reg,0)!=dev1)
                consistent=false;
        });
        for(auto &thread:threads) thread.join();
        check(consistent,"Concurrent Dawn registry access lost its device");
        execute(first.get()); executeConvolution(first.get());
        execute(second.get()); executeConvolution(second.get());
        std::cout << "Repeated native Dawn sessions passed\n";
    } catch(const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
