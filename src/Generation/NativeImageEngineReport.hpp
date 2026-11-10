#pragma once
#include <stdint.h>

// Private bridge to the pinned native engine. Its callback runs on the calling
// thread after the engine resolves model alignment, sampler and random seed.
struct IildResolvedImageParameters {
    int phase, hiresSchedulerSteps;
    int width, height, canvasWidth, canvasHeight, steps, hiresSteps;
    int sampler, scheduler, hiresUpscaler, hires, imageToImage;
    int64_t seed;
    float cfg, imageCfg, distilledGuidance, hiresStrength, imageStrength;
    const float *sigmas;
    int sigmasCount;
};
using IildImageReportCallback = void (*)(const IildResolvedImageParameters *, void *);
extern "C" void iild_set_image_report_callback(IildImageReportCallback, void *);
namespace iiLocalDiffusion::native_detail {
void captureImageEngineReport(const IildResolvedImageParameters *, void *);
}
