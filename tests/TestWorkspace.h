#pragma once
#include <cstdlib>
#include <filesystem>

namespace iild_test {
inline std::filesystem::path workspace(const char *fallback, const char *name)
{
    const char *root = std::getenv("IIFILEPROVIDER_TEST_DIRECTORY");
    return root && *root ? std::filesystem::path(root) / name : std::filesystem::path(fallback);
}
}
