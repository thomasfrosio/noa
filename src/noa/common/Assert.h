/// \file noa/common/Assert.h
/// \brief Assertions.
/// \author Thomas - ffyr2w
/// \date 20 Jul 2020

#pragma once

#include "noa/common/Definitions.h"

// __CUDA_ARCH__ should not be defined for host code. See https://stackoverflow.com/a/16073481
#if defined(NOA_ENABLE_ASSERTS) && !defined(__CUDA_ARCH__)
    #include "noa/common/Exception.h"
    #define NOA_ASSERT(check) if(!(check)) NOA_THROW("Debug assertion failed")
#else
    #define NOA_ASSERT(check)
#endif
