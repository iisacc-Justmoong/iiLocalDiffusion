#pragma once

// Private, inference-runtime-independent DWPose tensor/image boundary.
// RGB belongs to the caller; all returned tensors and hint pixels are owned.
#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace iiLocalDiffusion::native_detail {
struct PosePixels { std::span<const std::uint8_t> rgb; int width = 0, height = 0; };
struct PoseBox { float left = 0, top = 0, right = 0, bottom = 0, score = 0; };
struct PoseCrop {
    int width = 0, height = 0;
    float centerX = 0, centerY = 0, scaleX = 0, scaleY = 0;
    std::vector<float> chw;
};
struct PoseJoint { float x = 0, y = 0, confidence = 0; };
using WholeBodyPose = std::array<PoseJoint, 133>; // COCO-WholeBody, before neck insertion.

// YOLOX-L: [1,3,640,640] BGR, raw 0..255, top-left letterbox (114).
std::vector<float> preparePoseDetector(PosePixels image);
// YOLOX-L raw grid output [1,8400,85], strides 8/16/32. Person class only.
// Score > .3; inclusive-coordinate IoU NMS .45, matching the reference.
std::vector<PoseBox> decodePoseDetector(std::span<const float> output, int width, int height);
// RTMPose: padded person crop, RGB normalization, zero image border, CHW.
PoseCrop preparePoseCrop(PosePixels image, PoseBox box, int width, int height);
// SimCC [1,133,2*width] and [1,133,2*height], scores=min(axis maxima).
WholeBodyPose decodePoseSimcc(std::span<const float> x, std::span<const float> y, const PoseCrop &crop);
// OpenPose-compatible body/hand colors and face points on a black RGB canvas.
std::vector<std::uint8_t> drawPoseHint(std::span<const WholeBodyPose> poses, int width, int height);
}
