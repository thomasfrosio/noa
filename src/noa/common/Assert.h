/// \file noa/common/Assert.h
/// \brief Assertions.
/// \author Thomas - ffyr2w
/// \date 20 Jul 2020

#pragma once

// CUDA device code supports the assert macro.
#if defined(NOA_ENABLE_ASSERTS)
    #include <cassert>
    #define NOA_ASSERT(check) assert(check);
#else
    #define NOA_ASSERT(check)
#endif
