#pragma once
#include <mutex>
namespace iiLocalDiffusion::native_detail {
// Upstream callbacks are process-global. Image and video must share this lock.
std::timed_mutex &nativeEngineMutex();
void nativeEngineOnThread(bool owns);
void releaseImageCacheForVideoLocked();
}
