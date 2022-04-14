#include "noa/unified/Array.h"
#include "noa/unified/memory/Copy.h"
#include "noa/unified/memory/Transpose.h"

namespace noa::details {
    template<typename T>
    void arrayCopy(const Array<T>& src, const Array<T>& dst) {
        return memory::copy(src, dst);
    }

    template<typename T>
    void arrayTranspose(const Array<T>& src, const Array<T>& dst, uint4_t permutation) {
        return memory::transpose(src, dst, permutation);
    }

    #define NOA_INSTANTIATE_COPY_TRANSPOSE(T)                       \
    template void arrayCopy<T>(const Array<T>&, const Array<T>&);   \
    template void arrayTranspose<T>(const Array<T>&, const Array<T>&, uint4_t)

    NOA_INSTANTIATE_COPY_TRANSPOSE(bool);
    NOA_INSTANTIATE_COPY_TRANSPOSE(int8_t);
    NOA_INSTANTIATE_COPY_TRANSPOSE(int16_t);
    NOA_INSTANTIATE_COPY_TRANSPOSE(int32_t);
    NOA_INSTANTIATE_COPY_TRANSPOSE(int64_t);
    NOA_INSTANTIATE_COPY_TRANSPOSE(uint8_t);
    NOA_INSTANTIATE_COPY_TRANSPOSE(uint16_t);
    NOA_INSTANTIATE_COPY_TRANSPOSE(uint32_t);
    NOA_INSTANTIATE_COPY_TRANSPOSE(uint64_t);
    NOA_INSTANTIATE_COPY_TRANSPOSE(half_t);
    NOA_INSTANTIATE_COPY_TRANSPOSE(float);
    NOA_INSTANTIATE_COPY_TRANSPOSE(double);
    NOA_INSTANTIATE_COPY_TRANSPOSE(chalf_t);
    NOA_INSTANTIATE_COPY_TRANSPOSE(cfloat_t);
    NOA_INSTANTIATE_COPY_TRANSPOSE(cdouble_t);
}
