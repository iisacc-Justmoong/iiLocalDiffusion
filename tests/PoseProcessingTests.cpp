#include "Generation/PoseProcessing.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace iiLocalDiffusion::native_detail;
namespace {
void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
void close(float actual, float expected) { require(std::abs(actual - expected) < .001f, "numeric mismatch"); }
template<class F> void rejects(F call) {
    bool rejected = false;
    try { call(); } catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "malformed tensor or image accepted");
}
}
int main()
{
    try {
        const std::vector<std::uint8_t> rgb(320 * 160 * 3, 0);
        auto colored = rgb;
        for (size_t i = 0; i < colored.size(); i += 3) { colored[i] = 20; colored[i+1] = 80; colored[i+2] = 140; }
        const PosePixels image{colored, 320, 160};
        const auto input = preparePoseDetector(image);
        require(input.size() == 3 * 640 * 640, "detector input shape");
        close(input[0], 140); close(input[640*640], 80); close(input[2*640*640], 20);
        close(input[319*640+639], 140); close(input[320*640], 114);
        rejects([&] { preparePoseDetector({std::span(colored).first(3), 320, 160}); });
        rejects([&] { preparePoseDetector({{}, 0, 160}); });

        std::vector<float> detection(8400 * 85);
        auto row = [&](int index, float x, float y, float w, float h, float score, int stride, int gx, int gy) {
            auto *p = detection.data() + index*85;
            p[0] = x / stride - gx; p[1] = y / stride - gy;
            p[2] = std::log(w / stride); p[3] = std::log(h / stride); p[4] = 1; p[5] = score;
        };
        row(0, 100, 100, 40, 80, .9f, 8, 0, 0);
        row(1, 101, 101, 40, 80, .8f, 8, 1, 0); // NMS removes duplicate.
        row(6400, 300, 100, 40, 80, .7f, 16, 0, 0);
        row(8000, 500, 100, 40, 80, .6f, 32, 0, 0);
        row(2, 200, 100, 40, 80, .2f, 8, 2, 0);
        detection[2*85+6] = 1; // A confident non-person must not become a person.
        const auto boxes = decodePoseDetector(detection, 320, 160);
        require(boxes.size() == 3, "multi-person decoding, class selection or NMS");
        close(boxes[0].left, 40); close(boxes[0].top, 30); close(boxes[0].right, 60); close(boxes[0].bottom, 70);
        close(boxes[1].left, 140); close(boxes[2].left, 240);
        rejects([&] { decodePoseDetector(std::span(detection).first(84), 320, 160); });
        detection[0] = std::numeric_limits<float>::quiet_NaN();
        rejects([&] { decodePoseDetector(detection, 320, 160); });
        detection.assign(detection.size(), 0);
        require(decodePoseDetector(detection, 320, 160).empty(), "empty detection is not empty");

        const auto crop = preparePoseCrop(image, {80, 0, 240, 160, 1}, 192, 256);
        close(crop.centerX, 160); close(crop.centerY, 80);
        close(crop.scaleX, 200); close(crop.scaleY, 200.f*256/192);
        const size_t center = 128*192+96;
        close(crop.chw[center], (20-123.675f)/58.395f);
        close(crop.chw[192*256+center], (80-116.28f)/57.12f);
        close(crop.chw[2*192*256+center], (140-103.53f)/57.375f);
        close(crop.chw[0], -123.675f/58.395f); // Padded area, not clamped edge pixels.
        rejects([&] { preparePoseCrop(image, {10, 0, 0, 10, 1}, 192, 256); });
        rejects([&] { preparePoseCrop(image, boxes[0], 0, 256); });
        std::vector<float> simX(133*384), simY(133*512);
        for (int i = 0; i < 133; ++i) { simX[i*384+192] = .8f; simY[i*512+256] = .7f; }
        const auto pose = decodePoseSimcc(simX, simY, crop);
        close(pose[0].x, 160); close(pose[0].y, 80); close(pose[0].confidence, .7f);
        close(pose[132].x, 160); close(pose[132].y, 80);
        simX[0] = std::numeric_limits<float>::infinity();
        rejects([&] { decodePoseSimcc(simX, simY, crop); });
        rejects([&] { decodePoseSimcc({}, simY, crop); });

        WholeBodyPose person{};
        person[5] = {40, 40, .9f}; person[6] = {80, 40, .9f}; // Neck at 60,40.
        person[8] = {90, 60, .9f}; person[10] = {100, 80, .9f};
        person[23] = {20, 20, .9f}; // First face landmark.
        person[91] = {20, 90, .9f}; person[92] = {30, 90, .9f}; // Left hand.
        auto hint = drawPoseHint(std::span(&person, 1), 128, 128);
        auto pixel = [&](int x, int y, int channel) { return hint[(y*128+x)*3+channel]; };
        require(pixel(0,0,0) == 0 && pixel(0,0,1) == 0, "background must stay black");
        require(pixel(60,40,0) == 255 && pixel(60,40,1) == 85, "neck synthesis/body remap");
        require(pixel(20,20,0) == 255 && pixel(20,20,2) == 255, "face not rendered");
        require(pixel(20,90,0) == 0 && pixel(20,90,2) == 255, "hand not rendered");
        person[5].confidence = .3f; person[6].confidence = .3f;
        person[8].confidence = 0; person[10].confidence = 0;
        hint = drawPoseHint(std::span(&person, 1), 128, 128);
        require(pixel(60,40,0) == 0, "low-confidence neck was rendered");
        require(drawPoseHint({}, 128,128) == std::vector<std::uint8_t>(128*128*3), "no people must yield black hint");
        person[0].x = std::numeric_limits<float>::quiet_NaN();
        rejects([&] { drawPoseHint(std::span(&person, 1), 128,128); });
        std::cout << "Pose processing contracts passed\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
