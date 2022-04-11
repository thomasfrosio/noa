#include "noa/unified/Array.h"
#include "noa/unified/memory/Copy.h"
#include "noa/unified/memory/Transpose.h"

namespace noa {
    std::ostream& operator<<(std::ostream& os, Allocator resource) {
        switch (resource) {
            case Allocator::NONE:
                return os << "NONE";
            case Allocator::DEFAULT:
                return os << "DEFAULT";
            case Allocator::DEFAULT_ASYNC:
                return os << "DEFAULT_ASYNC";
            case Allocator::PITCHED:
                return os << "PITCHED";
            case Allocator::PINNED:
                return os << "PINNED";
            case Allocator::MANAGED:
                return os << "MANAGED";
            case Allocator::MANAGED_GLOBAL:
                return os << "MANAGED_GLOBAL";
        }
        return os;
    }
}

namespace noa::details {
    template<typename T>
    void arrayCopy(const Array<T>& src, Array<T>& dst) {
        return memory::copy(src, dst);
    }

    template<typename T>
    void arrayTranspose(const Array<T>& src, Array<T>& dst, uint4_t permutation) {
        return memory::transpose(src, dst, permutation);
    }

    #define NOA_INSTANTIATE_COPY_TRANSPOSE(T)               \
    template void arrayCopy<T>(const Array<T>&, Array<T>&); \
    template void arrayTranspose<T>(const Array<T>&, Array<T>&, uint4_t)

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
