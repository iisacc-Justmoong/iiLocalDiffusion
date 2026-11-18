#pragma once

#include <algorithm>
#include <array>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace iild::detail {

// The complete signature avoids treating a lone, commonly named patch tensor
// as a Wan model. All markers must occur under the same source namespace.
struct WanDiffusersSignature {
    static constexpr std::array<std::string_view, 5> prefixes{
        "", "model.diffusion_model.", "diffusion_model.", "transformer.", "model.high_noise_diffusion_model."};
    static constexpr std::array<std::string_view, 4> markers{
        "patch_embedding.weight", "blocks.0.attn1.to_q.weight", "blocks.0.attn2.norm_k.weight",
        "condition_embedder.time_embedder.linear_1.weight"};
    std::array<std::array<bool, markers.size()>, prefixes.size()> present{};
    void add(std::string_view name) {
        for (std::size_t i = 0; i < prefixes.size(); ++i) {
            if (!name.starts_with(prefixes[i])) continue;
            for (std::size_t j = 0; j < markers.size(); ++j)
                if (name.substr(prefixes[i].size()) == markers[j]) present[i][j] = true;
        }
    }
    [[nodiscard]] std::optional<std::string_view> prefix() const {
        for (std::size_t i = 0; i < prefixes.size(); ++i)
            if (std::all_of(present[i].begin(), present[i].end(), [](bool found) { return found; })) return prefixes[i];
        return std::nullopt;
    }
};

// Reverse the base Wan Diffusers naming convention without transforming bytes.
// Unrelated auxiliary names are deliberately left outside the diffusion mount.
[[nodiscard]] inline std::optional<std::string> wanDiffusersTensorName(
    std::string_view name, std::string_view sourcePrefix) {
    if (!name.starts_with(sourcePrefix)) return std::nullopt;
    name.remove_prefix(sourcePrefix.size());
    if (name == "scale_shift_table") return std::string("head.modulation");
    if (name.starts_with("proj_out.")) return "head.head." + std::string(name.substr(9));
    if (name.starts_with("patch_embedding.")) return std::string(name);
    for (const auto &[source, target] : std::array<std::pair<std::string_view, std::string_view>, 9>{{
        {"condition_embedder.time_embedder.linear_1.", "time_embedding.0."},
        {"condition_embedder.time_embedder.linear_2.", "time_embedding.2."},
        {"condition_embedder.text_embedder.linear_1.", "text_embedding.0."},
        {"condition_embedder.text_embedder.linear_2.", "text_embedding.2."},
        {"condition_embedder.time_proj.", "time_projection.1."},
        {"condition_embedder.image_embedder.norm1.", "img_emb.proj.0."},
        {"condition_embedder.image_embedder.ff.net.0.proj.", "img_emb.proj.1."},
        {"condition_embedder.image_embedder.ff.net.2.", "img_emb.proj.3."},
        {"condition_embedder.image_embedder.norm2.", "img_emb.proj.4."}}}) {
        if (name.starts_with(source)) return std::string(target) + std::string(name.substr(source.size()));
    }
    if (!name.starts_with("blocks.")) return std::nullopt;
    std::string result(name);
    const auto replace = [&](std::string_view source, std::string_view target) {
        const auto position = result.find(source);
        if (position != std::string::npos) result.replace(position, source.size(), target);
    };
    replace(".attn1.", ".self_attn.");
    replace(".attn2.", ".cross_attn.");
    replace(".to_out.0.", ".o.");
    replace(".to_q.", ".q.");
    replace(".to_k.", ".k.");
    replace(".to_v.", ".v.");
    replace(".add_k_proj.", ".k_img.");
    replace(".add_v_proj.", ".v_img.");
    replace(".norm_added_k.", ".norm_k_img.");
    replace(".ffn.net.0.proj.", ".ffn.0.");
    replace(".ffn.net.2.", ".ffn.2.");
    replace(".norm2.", ".norm__wan_swap.");
    replace(".norm3.", ".norm2.");
    replace(".norm__wan_swap.", ".norm3.");
    replace(".scale_shift_table", ".modulation");
    return result;
}

} // namespace iild::detail
