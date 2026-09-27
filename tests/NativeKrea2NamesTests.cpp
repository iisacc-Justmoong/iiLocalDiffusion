#include <name_conversion.h>
#include <iostream>
int main() {
    for (const auto *suffix : {"model.embed_tokens.weight", "model.layers.0.mlp.down_proj.weight", "model.layers.0.mlp.down_proj.weight_scale", "visual.blocks.0.attn.qkv.weight"}) {
        const std::string canonical = std::string("text_encoders.llm.") + suffix;
        const std::string comfy = std::string("text_encoders.qwen3vl_4b.transformer.") + suffix;
        if (convert_tensor_name(comfy, VERSION_KREA2) != convert_tensor_name(canonical, VERSION_KREA2)) {
            std::cerr << comfy << " was not normalized\n"; return 1;
        }
    }
    std::cout << "Krea2 embedded encoder names and quant scales normalized\n";
}
