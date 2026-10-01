// DWPose/MMPose tensor geometry and ControlNet pose conventions adapted in
// native C++ by iiLocalDiffusion. Copyright 2023 IDEA; 2018-2020 Open-MMLab.
// Apache-2.0; see docs/licenses/DWPose.txt. This implementation is modified:
// Qt/OpenCV/Python-free RGB boundary, bounded validation and analytic rasterizer.
#include "PoseProcessing.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
namespace iiLocalDiffusion::native_detail {
namespace {
void check(bool valid, const char *message) { if (!valid) throw std::invalid_argument(message); }
void dimensions(int width, int height)
{
    check(width > 0 && height > 0 && width <= 2048 && height <= 2048, "Pose image dimensions must be in [1,2048].");
}
void imageValid(PosePixels image)
{
    dimensions(image.width, image.height);
    check(image.rgb.size() == size_t(image.width) * image.height * 3, "Pose RGB span does not match dimensions.");
}
void cropValid(const PoseCrop &crop)
{
    check((crop.width == 192 && crop.height == 256) || (crop.width == 288 && crop.height == 384),
        "DWPose supports 192x256 or 288x384 pose inputs.");
    for (float v : {crop.centerX, crop.centerY, crop.scaleX, crop.scaleY})
        check(std::isfinite(v) && std::abs(v) < 1e6f, "Invalid pose crop transform.");
    check(crop.scaleX > 0 && crop.scaleY > 0, "Pose crop scale must be positive.");
}
float sample(PosePixels image, float x, float y, int channel, bool clampBorder)
{
    if (clampBorder) {
        x = std::clamp(x, 0.f, float(image.width-1));
        y = std::clamp(y, 0.f, float(image.height-1));
    }
    // Fast rejection also bounds all floating-to-integer conversions.
    if (x <= -1 || y <= -1 || x >= image.width || y >= image.height) return 0;
    const int ix = int(std::floor(x)), iy = int(std::floor(y));
    const float fx = x-ix, fy = y-iy;
    auto at = [&](int px, int py) -> float {
        if (px < 0 || py < 0 || px >= image.width || py >= image.height) return 0;
        return image.rgb[(size_t(py)*image.width+px)*3+channel];
    };
    return std::round((1-fy)*((1-fx)*at(ix,iy)+fx*at(ix+1,iy))
        + fy*((1-fx)*at(ix,iy+1)+fx*at(ix+1,iy+1)));
}
float iou(const PoseBox &a, const PoseBox &b)
{
    const float intersection = std::max(0.f, std::min(a.right,b.right)-std::max(a.left,b.left)+1)
        * std::max(0.f, std::min(a.bottom,b.bottom)-std::max(a.top,b.top)+1);
    const float areaA = (a.right-a.left+1)*(a.bottom-a.top+1);
    const float areaB = (b.right-b.left+1)*(b.bottom-b.top+1);
    return intersection / (areaA+areaB-intersection);
}
using Color = std::array<std::uint8_t,3>;
constexpr std::array<Color,18> bodyColors{{{255,0,0},{255,85,0},{255,170,0},{255,255,0},{170,255,0},{85,255,0},
    {0,255,0},{0,255,85},{0,255,170},{0,255,255},{0,170,255},{0,85,255},{0,0,255},{85,0,255},
    {170,0,255},{255,0,255},{255,0,170},{255,0,85}}};
constexpr std::array<std::array<int,2>,17> bodyEdges{{{1,2},{1,5},{2,3},{3,4},{5,6},{6,7},
    {1,8},{8,9},{9,10},{1,11},{11,12},{12,13},{1,0},{0,14},{14,16},{0,15},{15,17}}};
std::array<PoseJoint,18> body(const WholeBodyPose &p)
{
    constexpr int map[18]{0,0,6,8,10,5,7,9,12,14,16,11,13,15,2,1,4,3};
    std::array<PoseJoint,18> result;
    for (int i = 0; i < 18; ++i) result[i] = p[map[i]];
    result[1] = {(p[5].x+p[6].x)/2, (p[5].y+p[6].y)/2,
        p[5].confidence > .3f && p[6].confidence > .3f ? 1.f : 0.f};
    return result;
}
Color handColor(int index)
{
    const float h = float(index)*6/20;
    const int sector = int(h);
    const auto rising = std::uint8_t(255*(h-sector));
    const auto falling = std::uint8_t(255*(1-h+sector));
    switch (sector) {
    case 0: return {255,rising,0}; case 1: return {falling,255,0};
    case 2: return {0,255,rising}; case 3: return {0,falling,255};
    case 4: return {rising,0,255}; default: return {255,0,falling};
    }
}
class Canvas {
public:
    Canvas(int w, int h) : width(w), height(h), rgb(size_t(w)*h*3) {}
    void ellipse(float cx, float cy, float rx, float ry, float cosine, float sine, Color color)
    {
        if (rx <= 0 || ry <= 0) return;
        const float ex = std::abs(rx*cosine)+std::abs(ry*sine);
        const float ey = std::abs(rx*sine)+std::abs(ry*cosine);
        const int left = int(std::clamp(std::floor(cx-ex), 0.f, float(width)));
        const int right = int(std::clamp(std::ceil(cx+ex), -1.f, float(width-1)));
        const int top = int(std::clamp(std::floor(cy-ey), 0.f, float(height)));
        const int bottom = int(std::clamp(std::ceil(cy+ey), -1.f, float(height-1)));
        for (int y = top; y <= bottom; ++y) for (int x = left; x <= right; ++x) {
            const float u = ((x-cx)*cosine+(y-cy)*sine)/rx;
            const float v = (-(x-cx)*sine+(y-cy)*cosine)/ry;
            if (u*u+v*v <= 1) put(x,y,color);
        }
    }
    void circle(PoseJoint p, float radius, Color color) { ellipse(std::trunc(p.x),std::trunc(p.y),radius,radius,1,0,color); }
    void limb(PoseJoint a, PoseJoint b, Color color)
    {
        const float length = std::hypot(b.x-a.x,b.y-a.y);
        if (length < .001f) return;
        ellipse(std::trunc((a.x+b.x)/2),std::trunc((a.y+b.y)/2),std::trunc(length/2),4,
            (b.x-a.x)/length,(b.y-a.y)/length,color);
    }
    void line(PoseJoint a, PoseJoint b, Color color)
    {
        // Distance-to-segment rasterization, clipped before iteration.
        const float dx = b.x-a.x, dy = b.y-a.y, length2 = dx*dx+dy*dy;
        const int left = int(std::clamp(std::floor(std::min(a.x,b.x)-1),0.f,float(width)));
        const int right = int(std::clamp(std::ceil(std::max(a.x,b.x)+1),-1.f,float(width-1)));
        const int top = int(std::clamp(std::floor(std::min(a.y,b.y)-1),0.f,float(height)));
        const int bottom = int(std::clamp(std::ceil(std::max(a.y,b.y)+1),-1.f,float(height-1)));
        for (int y = top; y <= bottom; ++y) for (int x = left; x <= right; ++x) {
            const float t = length2 > 0 ? std::clamp(((x-a.x)*dx+(y-a.y)*dy)/length2,0.f,1.f) : 0;
            const float ex = x-a.x-t*dx, ey = y-a.y-t*dy;
            if (ex*ex+ey*ey <= 1) put(x,y,color);
        }
    }
    int width, height;
    std::vector<std::uint8_t> rgb;
private:
    void put(int x, int y, Color color) { std::copy(color.begin(),color.end(),rgb.begin()+(size_t(y)*width+x)*3); }
};
}

std::vector<float> preparePoseDetector(PosePixels image)
{
    imageValid(image);
    constexpr int side = 640;
    const float ratio = std::min(float(side)/image.width,float(side)/image.height);
    const int width = std::max(1,int(image.width*ratio)), height = std::max(1,int(image.height*ratio));
    std::vector<float> chw(3*side*side,114);
    for (int c = 0; c < 3; ++c) for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x)
        chw[c*side*side+y*side+x] = sample(image,(x+.5f)*image.width/width-.5f,
            (y+.5f)*image.height/height-.5f,2-c,true);
    return chw;
}

std::vector<PoseBox> decodePoseDetector(std::span<const float> output, int width, int height)
{
    dimensions(width,height);
    check(output.size() == 8400*85, "Expected YOLOX output [1,8400,85].");
    const float ratio = std::min(640.f/width,640.f/height);
    std::vector<PoseBox> boxes;
    size_t index = 0;
    for (int stride : {8,16,32}) for (int y = 0; y < 640/stride; ++y) for (int x = 0; x < 640/stride; ++x,++index) {
        const auto row = output.subspan(index*85,85);
        for (int c = 0; c < 6; ++c) check(std::isfinite(row[c]), "Non-finite YOLOX person output.");
        const float score = row[4]*row[5];
        if (score <= .3f) continue;
        check(row[4] >= 0 && row[4] <= 1 && row[5] >= 0 && row[5] <= 1, "Invalid YOLOX probability.");
        const float w = std::exp(row[2])*stride/ratio, h = std::exp(row[3])*stride/ratio;
        const float cx = (row[0]+x)*stride/ratio, cy = (row[1]+y)*stride/ratio;
        check(std::isfinite(w) && std::isfinite(h) && w > 0 && h > 0 && w < 1e5f && h < 1e5f
            && std::abs(cx) < 1e5f && std::abs(cy) < 1e5f, "Invalid YOLOX box geometry.");
        boxes.push_back({cx-w/2,cy-h/2,cx+w/2,cy+h/2,score});
    }
    std::stable_sort(boxes.begin(),boxes.end(),[](const auto &a, const auto &b) { return a.score > b.score; });
    std::vector<PoseBox> kept;
    for (const auto &box : boxes) {
        if (std::any_of(kept.begin(),kept.end(),[&](const auto &other) { return iou(box,other) > .45f; })) continue;
        check(kept.size() < 64, "Pose detection exceeds 64 people; no partial hint was produced.");
        kept.push_back(box);
    }
    return kept;
}

PoseCrop preparePoseCrop(PosePixels image, PoseBox box, int width, int height)
{
    imageValid(image);
    for (float v : {box.left,box.top,box.right,box.bottom})
        check(std::isfinite(v) && std::abs(v) < 1e5f, "Invalid pose bounding box.");
    check(box.right > box.left && box.bottom > box.top, "Pose bounding box must have positive area.");
    PoseCrop crop{width,height,(box.left+box.right)/2,(box.top+box.bottom)/2,
        (box.right-box.left)*1.25f,(box.bottom-box.top)*1.25f,{}};
    cropValid(crop);
    const float aspect = float(width)/height;
    if (crop.scaleX > crop.scaleY*aspect) crop.scaleY = crop.scaleX/aspect;
    else crop.scaleX = crop.scaleY*aspect;
    cropValid(crop);
    const size_t plane = size_t(width)*height;
    crop.chw.resize(3*plane);
    constexpr float mean[3]{123.675f,116.28f,103.53f}, deviation[3]{58.395f,57.12f,57.375f};
    for (int c = 0; c < 3; ++c) for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        const float sx = crop.centerX+(float(x)/width-.5f)*crop.scaleX;
        const float sy = crop.centerY+(float(y)/height-.5f)*crop.scaleY;
        crop.chw[c*plane+y*width+x] = (sample(image,sx,sy,c,false)-mean[c])/deviation[c];
    }
    return crop;
}

WholeBodyPose decodePoseSimcc(std::span<const float> x, std::span<const float> y, const PoseCrop &crop)
{
    cropValid(crop);
    check(x.size() == size_t(133)*2*crop.width && y.size() == size_t(133)*2*crop.height, "Invalid DWPose SimCC output shape.");
    for (auto axis : {x,y}) for (float value : axis) check(std::isfinite(value), "Non-finite SimCC output.");
    WholeBodyPose result;
    for (size_t i = 0; i < result.size(); ++i) {
        const auto sx = x.subspan(i*2*crop.width,2*crop.width), sy = y.subspan(i*2*crop.height,2*crop.height);
        const auto mx = std::max_element(sx.begin(),sx.end()), my = std::max_element(sy.begin(),sy.end());
        const float confidence = std::min(*mx,*my);
        const float px = confidence > 0 ? float(mx-sx.begin())/2 : -.5f;
        const float py = confidence > 0 ? float(my-sy.begin())/2 : -.5f;
        result[i] = {px/crop.width*crop.scaleX+crop.centerX-crop.scaleX/2,
            py/crop.height*crop.scaleY+crop.centerY-crop.scaleY/2,confidence};
    }
    return result;
}

std::vector<std::uint8_t> drawPoseHint(std::span<const WholeBodyPose> poses, int width, int height)
{
    dimensions(width,height);
    check(poses.size() <= 64, "Pose hint exceeds 64 people.");
    for (const auto &pose : poses) for (const auto &joint : pose)
        check(std::isfinite(joint.x) && std::isfinite(joint.y) && std::isfinite(joint.confidence)
            && std::abs(joint.x) < 1e6f && std::abs(joint.y) < 1e6f, "Invalid pose joint.");
    Canvas canvas(width,height);
    // Draw all limbs before attenuation and joints, matching ControlNet color semantics.
    for (size_t edge = 0; edge < bodyEdges.size(); ++edge) for (const auto &pose : poses) {
        const auto joints = body(pose);
        const auto a = joints[bodyEdges[edge][0]], b = joints[bodyEdges[edge][1]];
        if (a.confidence > .3f && b.confidence > .3f) canvas.limb(a,b,bodyColors[edge]);
    }
    for (auto &channel : canvas.rgb) channel = std::uint8_t(channel*.6f);
    for (size_t i = 0; i < 18; ++i) for (const auto &pose : poses) {
        const auto joint = body(pose)[i];
        if (joint.confidence > .3f) canvas.circle(joint,4,bodyColors[i]);
    }
    auto visible = [](PoseJoint p) { return p.confidence >= .3f && p.x >= 1 && p.y >= 1; };
    for (const auto &pose : poses) for (int start : {91,112}) {
        for (int finger = 0; finger < 5; ++finger) for (int segment = 0; segment < 4; ++segment) {
            const auto a = pose[start+(segment ? finger*4+segment : 0)];
            const auto b = pose[start+finger*4+segment+1];
            if (visible(a) && visible(b)) canvas.line(a,b,handColor(finger*4+segment));
        }
        for (int i = 0; i < 21; ++i) if (visible(pose[start+i])) canvas.circle(pose[start+i],4,{0,0,255});
    }
    for (const auto &pose : poses) for (int i = 23; i < 91; ++i)
        if (visible(pose[i])) canvas.circle(pose[i],3,{255,255,255});
    return std::move(canvas.rgb);
}
}
