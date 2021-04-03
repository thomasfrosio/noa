#include "noa/Definitions.h"
#include "noa/gpu/cuda/Types.h"

namespace Noa::CUDA::Memory {
    template<class T>
    struct Shared {

        // specialize for double to avoid unaligned memory
        // access compile errors
        NOA_ID operator T*() {
            if constexpr (alignof(T) == 4) {
                extern __shared__ int shared_buffer[];
                return reinterpret_cast<T*>(shared_buffer);
            } else if constexpr (alignof(T) == 8) {
                extern __shared__ double shared_buffer[];
                return reinterpret_cast<T*>(shared_buffer);
            } else if constexpr (alignof(T) == 16) {
                extern __shared__ double2 shared_buffer[];
                return reinterpret_cast<T*>(shared_buffer);
            }
        }

        NOA_ID operator const T*() const {
            if constexpr (alignof(T) == 4) {
                extern __shared__ int shared_buffer[];
                return reinterpret_cast<T*>(shared_buffer);
            } else if constexpr (alignof(T) == 8) {
                extern __shared__ double shared_buffer[];
                return reinterpret_cast<T*>(shared_buffer);
            } else if constexpr (alignof(T) == 16) {
                extern __shared__ double2 shared_buffer[];
                return reinterpret_cast<T*>(shared_buffer);
            }
        }
    };
}
