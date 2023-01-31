#pragma once

#include "noa/core/Definitions.hpp"

// CUDA device code supports the assert macro.
#if defined(NOA_ENABLE_ASSERTS) && (!defined(__CUDA_ARCH__) || defined(NOA_CUDA_ENABLE_ASSERT))
    #include <cassert>
    #define NOA_ASSERT(check) assert(check)
#else
    #define NOA_ASSERT(check)
#endif
