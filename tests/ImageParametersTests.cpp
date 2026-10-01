#include "Generation/ImageParameters.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace iiLocalDiffusion;
void check(bool condition) { if (!condition) throw std::runtime_error("parameter contract failed"); }
int main()
{
    try {
        auto p = ImageParameters::defaults();
        check(validateImageParameters(p).empty()); // Empty prompt is valid in an editable draft.
        p.values["watermark"] = true;
        check(nativeParameterIssues(p, false).empty());
        check(nativeParameterIssues(p, true).empty()); // Publication is common to both routes.
        p.values["watermark"] = false;
        p.values["freeU"] = true;
        check(nativeParameterIssues(p, false).empty());
        check(!nativeParameterIssues(p, true).empty());
        p.values["freeU"] = false;
        p.values["sampler"] = std::string("dpmpp_2m");
        p.values["scheduler"] = std::string("karras");
        p.values["eta"] = 0.5;
        p.values["clipSkip"] = std::int64_t(2);
        p.values["hiresFix"] = true;
        check(nativeParameterIssues(p, false).empty());
        check(!nativeParameterIssues(p, true).empty()); // Legacy worker is unchanged.
        for (const auto *mode : {"nearest", "bilinear", "bicubic", "lanczos"}) {
            p.values["upscaler"] = std::string(mode);
            check(nativeParameterIssues(p, false).empty());
        }
        p.values["upscaler"] = std::string("4x-ultra");
        check(!nativeParameterIssues(p, false).empty()); // Requires an actual upscaler model.
        p.values["upscalerModel"] = std::string("upscale.safetensors");
        check(nativeParameterIssues(p, false).empty());
        check(!nativeParameterIssues(p, true).empty());
        p = ImageParameters::defaults();
        p.values["refiner"] = true;
        check(!nativeParameterIssues(p, false).empty());
        p.values["refinerModel"] = std::string("refiner.safetensors");
        check(nativeParameterIssues(p, false).empty());
        check(!nativeParameterIssues(p, true).empty());
        p.values["refinerSwitch"] = 1.;
        check(nativeParameterIssues(p, false).empty());
        p.values["refiner"] = false;
        p.values["refinerModel"] = std::string{};
        check(nativeParameterIssues(p, false).empty());
        p = ImageParameters::defaults();
        p.values["detailer"] = true;
        check(!nativeParameterIssues(p, false).empty());
        p.values["detailerModel"] = std::string("detector.safetensors");
        check(nativeParameterIssues(p, false).empty());
        check(!nativeParameterIssues(p, true).empty());
        p.values["denoiseStrength"] = 0.;
        check(!nativeParameterIssues(p, false).empty());
        p = ImageParameters::defaults();
        check(!validateImageParameters(p, true).empty());
        p.values["prompt"] = std::string("a coastal house");
        check(validateImageParameters(p, true).empty());
        check(std::get<std::int64_t>(p.values.at("seed")) == -1);
        check(std::get<std::int64_t>(p.values.at("width")) == 1024);
        p.values["width"] = std::int64_t(1025);
        check(!validateImageParameters(p).empty());
        p = ImageParameters::defaults(); p.values["cfgScale"] = std::numeric_limits<double>::quiet_NaN();
        check(!validateImageParameters(p).empty());
        p = ImageParameters::defaults(); p.values["steps"] = true;
        check(!validateImageParameters(p).empty());
        p = ImageParameters::defaults(); p.values["typo"] = true;
        check(!validateImageParameters(p).empty());
        p = ImageParameters::defaults(); p.values["seed"] = std::int64_t(4294967295LL);
        check(!validateImageParameters(p).empty()); // Four outputs would overflow.
        p.values["outputCount"] = std::int64_t(1);
        check(validateImageParameters(p).empty());
        p = ImageParameters::defaults();
        p.references.push_back("image.png");
        check(nativeParameterIssues(p, false).empty());
        check(!nativeParameterIssues(p, true).empty());
        p.references.clear();
        p.values["promptWeighting"] = false;
        p.values["textualEmbeddings"] = std::string("style.safetensors");
        check(nativeParameterIssues(p, false).empty());
        check(!nativeParameterIssues(p, true).empty());
        p.references.resize(21, "image.png");
        check(!validateImageParameters(p).empty());
        p = ImageParameters::defaults();
        p.controls.push_back({"control-1", "Pose", "pose.png", "", "", 1.0, false, false, false});
        check(validateImageParameters(p).empty());
        p.controls.push_back(p.controls.front());
        check(!validateImageParameters(p).empty());
        p = ImageParameters::defaults();
        p.controls.push_back({"canny", "Canny", "input.png", "control.safetensors", "", 0.7, false, false, true});
        check(nativeParameterIssues(p, false).empty());
        check(!nativeParameterIssues(p, true).empty());
        p.controls.front().process = "Tile";
        check(nativeParameterIssues(p, false).empty());
        p.controls.front().ipAdapter = true;
        check(!nativeParameterIssues(p, false).empty());
        p.controls.front().ipAdapterModel = "adapter.safetensors";
        p.controls.front().ipAdapterVision = "vision.safetensors";
        check(nativeParameterIssues(p, false).empty());
        check(!nativeParameterIssues(p, true).empty());
        p.controls.front().process = "IP-Adapter";
        p.controls.front().model.clear();
        check(nativeParameterIssues(p, false).empty());
        p.controls.front().process = "None";
        check(nativeParameterIssues(p, false).empty());
        p.controls.front().process = "Canny"; p.controls.front().model = "control.safetensors";
        p.controls.front().ipAdapter = false;
        p.controls.front().regionalMask = true; p.controls.front().maskSource = "mask.png";
        check(nativeParameterIssues(p, false).empty());
        check(!nativeParameterIssues(p, true).empty());
        p.controls.front().maskSource.clear();
        check(!nativeParameterIssues(p, false).empty());
        p.controls.front().maskSource = "mask.png";
        p.controls.front().regionalMask = false;
        p.controls.push_back({"other", "Canny", "other.png", "other.safetensors", "", 1, false, false, true});
        check(nativeParameterIssues(p, false).empty());
        check(!nativeParameterIssues(p, true).empty());
        p.controls.back().applied = false;
        check(nativeParameterIssues(p, false).empty());
        p.controls.front().process = "Depth";
        check(!nativeParameterIssues(p, false).empty());
        p.controls.front().process = "Pose";
        check(!nativeParameterIssues(p, false, true).empty());
        p.controls.front().poseDetector = "detector.onnx";
        p.controls.front().poseModel = "pose.onnx";
        check(nativeParameterIssues(p, false, true).empty());
        check(!nativeParameterIssues(p, false, false).empty());
        check(!nativeParameterIssues(p, true, true).empty());
        p = ImageParameters::defaults();
        p.loras.push_back({"lora-1", "style.safetensors", "Style", -0.5});
        check(validateImageParameters(p).empty());
        check(nativeParameterIssues(p, false).empty());
        p.values["sampler"] = std::string("dpmpp_2m");
        check(nativeParameterIssues(p, false).empty());
        p.values["sampler"] = std::string("euler");
        p.values["transparentBackground"] = true;
        check(!nativeParameterIssues(p, false).empty());
        std::cout << "Image parameter contracts passed\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
