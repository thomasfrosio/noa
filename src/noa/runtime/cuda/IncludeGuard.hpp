#pragma once

#if defined(NOA_ENABLE_CUDA)
#   ifndef __CUDACC__
#       error "CUDA headers should only be included in CUDA translation units, which is not the case here!"
#   endif
#endif
