#include "NativeImageEngineReport.hpp"
#include "NativeDiffusion.hpp"
#include <algorithm>
#include <type_traits>

static_assert(std::is_standard_layout_v<IildResolvedImageParameters>);
static_assert(std::is_trivially_copyable_v<IildResolvedImageParameters>);

namespace iiLocalDiffusion::native_detail {
void captureImageEngineReport(const IildResolvedImageParameters *v, void *data) {
    auto &s = *static_cast<NativeImageStageSettings *>(data);
    if (v->phase == 1) {
        s.hiresResolved = true;
        s.hiresSchedulerSteps = v->hiresSchedulerSteps;
        s.hiresSteps = std::max(0, v->sigmasCount - 1);
        if (v->sigmas && v->sigmasCount > 0) s.hiresSigmas.assign(v->sigmas, v->sigmas + v->sigmasCount);
        return;
    }
    s.resolved = true;
    s.baseWidth = v->width; s.baseHeight = v->height;
    s.canvasWidth = v->canvasWidth; s.canvasHeight = v->canvasHeight;
    s.steps = v->steps; s.hiresSteps = v->hiresSteps;
    s.sampler = v->sampler; s.scheduler = v->scheduler; s.hiresUpscaler = v->hiresUpscaler;
    s.hires = v->hires != 0; s.imageToImage = v->imageToImage != 0; s.seed = v->seed;
    s.cfgScale = v->cfg; s.imageCfgScale = v->imageCfg;
    s.distilledGuidance = v->distilledGuidance; s.hiresStrength = v->hiresStrength;
    s.imageStrength = v->imageStrength;
    if (v->sigmas && v->sigmasCount > 0) s.sigmas.assign(v->sigmas, v->sigmas + v->sigmasCount);
}
}
