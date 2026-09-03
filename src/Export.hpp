#pragma once

#if defined(_WIN32)
#    if defined(iiLocalDiffusion_EXPORTS)
#        define IILD_EXPORT __declspec(dllexport)
#    else
#        define IILD_EXPORT __declspec(dllimport)
#    endif
#else
#    define IILD_EXPORT __attribute__((visibility("default")))
#endif
