#pragma once
#include <algorithm>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace model_fixtures {
inline void checkFixture(bool ok,const char *message) { if(!ok) throw std::runtime_error(message); }
struct Tensor { std::string name; std::vector<int> shape; std::vector<float> data; std::string dtype="F32"; };
inline void writeModel(const std::filesystem::path &path, const std::vector<Tensor> &tensors) {
    std::ostringstream h; h << '{'; std::size_t offset=0;
    for (std::size_t i=0; i<tensors.size(); ++i) {
        const auto &t=tensors[i]; if (i) h << ',';
        h << '"' << t.name << "\":{\"dtype\":\"" << t.dtype << "\",\"shape\":[";
        std::size_t elements=1;
        for (std::size_t j=0; j<t.shape.size(); ++j) { if (j) h << ','; h << t.shape[j]; elements*=std::size_t(t.shape[j]); }
        checkFixture(elements==t.data.size(),"Bad synthetic tensor fixture");
        const std::size_t bytes=t.dtype=="I64" ? 8 : 4;
        h << "],\"data_offsets\":[" << offset << ',' << offset+t.data.size()*bytes << "]}";
        offset+=t.data.size()*bytes;
    }
    h << '}'; auto json=h.str(); while(json.size()%8) json+=' ';
    const std::uint64_t n=json.size();
    std::ofstream stream(path,std::ios::binary); stream.write(reinterpret_cast<const char*>(&n),sizeof(n));
    stream.write(json.data(),std::streamsize(json.size()));
    for (const auto &t : tensors) {
        if(t.dtype=="I64") for(float value:t.data) {
            const auto encoded=std::uint64_t(std::int64_t(value));
            for(unsigned i=0;i<8;++i) stream.put(char((encoded>>(8*i))&255));
        }
        else stream.write(reinterpret_cast<const char*>(t.data.data()),std::streamsize(t.data.size()*4));
    }
    checkFixture(bool(stream),"Cannot create synthetic weights");
}
inline void writeHalfModel(const std::filesystem::path &path,const std::vector<Tensor> &tensors,bool bf16) {
    std::ostringstream h; h << '{'; std::size_t offset=0;
    for(std::size_t i=0;i<tensors.size();++i) {
        const auto &t=tensors[i]; if(i) h << ',';
        h << '"' << t.name << "\":{\"dtype\":\"" << (bf16 ? "BF16" : "F16") << "\",\"shape\":[";
        for(std::size_t j=0;j<t.shape.size();++j) { if(j) h << ','; h << t.shape[j]; }
        h << "],\"data_offsets\":[" << offset << ',' << offset+t.data.size()*2 << "]}"; offset+=t.data.size()*2;
    }
    h << '}'; auto json=h.str(); while(json.size()%8) json+=' ';
    const std::uint64_t n=json.size(); std::ofstream stream(path,std::ios::binary);
    stream.write(reinterpret_cast<const char*>(&n),sizeof(n)); stream.write(json.data(),std::streamsize(json.size()));
    for(const auto &t:tensors) for(float v:t.data) {
        std::uint16_t bits=0;
        if(bf16) bits=std::uint16_t(std::bit_cast<std::uint32_t>(v)>>16);
        else if(v==1) bits=0x3c00; else if(v==.5f) bits=0x3800; else if(v==.25f) bits=0x3400;
        else if(v==-.25f) bits=0xb400; else checkFixture(v==0,"Unexpected half fixture value");
        stream.put(char(bits&255)); stream.put(char(bits>>8));
    }
    checkFixture(bool(stream),"Cannot create half-precision fixture");
}
inline std::vector<Tensor> srvggFixture() {
    std::vector<float> last(12*2*3*3,0);
    for (int o=0;o<12;++o) last[std::size_t(o*2*9+(o%2)*9+4)]=1;
    std::vector<float> bias(12); for(int i=0;i<12;++i) bias[std::size_t(i)]=float(i)*.01f;
    return {{"body.0.weight",{2,3,3,3},std::vector<float>(54,0)},
            {"body.0.bias",{2},{-.2f,.2f}}, {"body.1.weight",{2},{.1f,.5f}},
            {"body.2.weight",{12,2,3,3},last}, {"body.2.bias",{12},bias}};
}
inline std::vector<Tensor> swinFixture() {
    const auto zeros=[](std::size_t n){return std::vector<float>(n,0);};
    const auto identityConv=[](std::size_t out,std::size_t in) {
        std::vector<float> v(out*in*9,0);
        for(std::size_t i=0;i<std::min(out,in);++i) v[(i*in+i)*9+4]=1;
        return v;
    };
    std::vector<Tensor> t;
    auto add=[&](std::string name,std::vector<int> shape,std::vector<float> data) { t.push_back({std::move(name),std::move(shape),std::move(data)}); };
    add("conv_first.weight",{2,3,3,3},identityConv(2,3)); add("conv_first.bias",{2},zeros(2));
    const std::string b="layers.0.residual_group.blocks.0.";
    for(const auto *n : {"norm1.","norm2."}) { add(b+n+"weight",{2},{1,1}); add(b+n+"bias",{2},zeros(2)); }
    add(b+"attn.relative_position_bias_table",{9,1},zeros(9));
    auto qkv=zeros(12); qkv[8]=-1; qkv[11]=-1;
    add(b+"attn.qkv.weight",{6,2},qkv); add(b+"attn.qkv.bias",{6},zeros(6));
    add(b+"attn.proj.weight",{2,2},{1,0,0,1}); add(b+"attn.proj.bias",{2},zeros(2));
    for(const auto *n : {"mlp.fc1.","mlp.fc2."}) { add(b+n+"weight",{2,2},zeros(4)); add(b+n+"bias",{2},zeros(2)); }
    add("layers.0.conv.weight",{2,2,3,3},identityConv(2,2)); add("layers.0.conv.bias",{2},zeros(2));
    add("norm.weight",{2},{1,1}); add("norm.bias",{2},zeros(2));
    add("conv_after_body.weight",{2,2,3,3},identityConv(2,2)); add("conv_after_body.bias",{2},zeros(2));
    auto up=zeros(12*2*9);
    for(int i=0;i<4;++i) {up[std::size_t((i*2)*9+4)]=.1f; up[std::size_t(((i+4)*2+1)*9+4)]=.1f;}
    add("upsample.0.weight",{12,2,3,3},up); add("upsample.0.bias",{12},zeros(12));
    return t;
}
inline std::vector<Tensor> swinHeadFixture(bool nearest,bool four) {
    auto t=swinFixture();
    std::erase_if(t,[](const Tensor &x){return x.name.starts_with("upsample.0.");});
    const auto zeros=[](std::size_t n){return std::vector<float>(n,0);};
    const auto identity=[](std::size_t out,std::size_t in) {
        std::vector<float> w(out*in*9,0);
        for(std::size_t i=0;i<std::min(out,in);++i) w[(i*in+i)*9+4]=1;
        return w;
    };
    const auto conv=[&](const std::string &name,std::size_t out,std::size_t in,std::vector<float> weights) {
        t.push_back({name+".weight",{int(out),int(in),3,3},std::move(weights)});
        t.push_back({name+".bias",{int(out)},zeros(out)});
    };
    conv("conv_before_upsample.0",2,2,identity(2,2));
    if(nearest) {
        conv("conv_up1",2,2,identity(2,2));
        if(four) conv("conv_up2",2,2,identity(2,2));
        conv("conv_hr",2,2,identity(2,2));
    } else {
        std::vector<float> up(8*2*9,0);
        for(std::size_t c=0;c<2;++c) for(std::size_t sub=0;sub<4;++sub)
            up[((c*4+sub)*2+c)*9+4]=1;
        conv("upsample.0",8,2,std::move(up));
    }
    auto last=identity(3,2); for(auto &v:last) v*=.1f;
    conv("conv_last",3,2,std::move(last));
    return t;
}
inline std::vector<Tensor> swinResidual3Fixture() {
    const auto z=[](std::size_t n){return std::vector<float>(n,0);};
    std::vector<Tensor> t;
    const auto add=[&](std::string name,std::vector<int> shape,std::vector<float> values) { t.push_back({std::move(name),std::move(shape),std::move(values)}); };
    auto first=z(4*3*9); for(int c=0;c<3;++c) first[std::size_t((c*3+c)*9+4)]=1;
    add("conv_first.weight",{4,3,3,3},first); add("conv_first.bias",{4},z(4));
    const std::string b="layers.0.residual_group.blocks.0.";
    for(const auto *n:{"norm1.","norm2."}) {add(b+n+"weight",{4},{1,1,1,1});add(b+n+"bias",{4},z(4));}
    add(b+"attn.relative_position_bias_table",{9,1},z(9));
    add(b+"attn.qkv.weight",{12,4},z(48)); add(b+"attn.qkv.bias",{12},z(12));
    add(b+"attn.proj.weight",{4,4},z(16)); add(b+"attn.proj.bias",{4},z(4));
    for(const auto *n:{"mlp.fc1.","mlp.fc2."}) {add(b+n+"weight",{4,4},z(16));add(b+n+"bias",{4},z(4));}
    const auto residual=[&](const std::string &prefix) {
        add(prefix+"0.weight",{1,4,3,3},z(36)); add(prefix+"0.bias",{1},z(1));
        add(prefix+"2.weight",{1,1,1,1},z(1)); add(prefix+"2.bias",{1},z(1));
        add(prefix+"4.weight",{4,1,3,3},z(36)); add(prefix+"4.bias",{4},z(4));
    };
    residual("layers.0.conv."); residual("conv_after_body.");
    add("norm.weight",{4},{1,1,1,1}); add("norm.bias",{4},z(4));
    auto up=z(12*4*9);
    for(int c=0;c<3;++c) for(int sub=0;sub<4;++sub) up[std::size_t(((c*4+sub)*4+c)*9+4)]=1;
    add("upsample.0.weight",{12,4,3,3},up); add("upsample.0.bias",{12},z(12));
    return t;
}

}
