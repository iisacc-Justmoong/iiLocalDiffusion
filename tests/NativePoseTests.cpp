#include "Generation/NativePose.hpp"
#include <iostream>
#include <stdexcept>
using namespace iiLocalDiffusion;
template<class F> void retired(F action) {
    try { action(); } catch (const std::runtime_error &error) {
        if (std::string(error.what()).find("retired") != std::string::npos) return;
        throw;
    }
    throw std::runtime_error("Retired model inference was accepted");
}
int main() {
    try {
        if (nativePoseAvailable()) throw std::runtime_error("Retired model inference is advertised");
        const std::atomic_bool cancelled{false};
        retired([&] { NativePoseSession session("missing-detector", "missing-pose", cancelled); });
        retired([&] { (void)processNativePose("missing-detector", "missing-pose", {}, cancelled); });
        releaseNativePoseCache();
        std::cout << "Retired pose model requests fail explicitly without loading a runtime\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
