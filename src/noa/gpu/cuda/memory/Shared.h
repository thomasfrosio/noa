#include "noa/Definitions.h"
#include "noa/gpu/cuda/Types.h"

namespace Noa::CUDA::Memory {

    /**
     * For using dynamically-sized (i.e. "extern" with unspecified-size array) shared memory in templated kernels, this
     * kind of utility is necessary to avoid errors. Also, since the documentation is unclear about the alignment and
     * whether it comes with any alignment guarantees other than the alignment of the type used in the declaration
     * (thus whether or not the __align__ attribute has any effect on shared memory), use double to ensure 16-byte
     * alignment, then cast to the desired type.
     *
     * @see https://stackoverflow.com/questions/27570552
     */
    template<class T>
    struct Shared {
        static NOA_FD T* getBlockResource() {
            static_assert(alignof(T) <= alignof(double2));
            extern __shared__ double2 buffer_align16[];
            return reinterpret_cast<T*>(buffer_align16);
        }
    };
}
