#include <noa/unified/Array.hpp>
#include <noa/unified/Reduce.hpp>

namespace test1 {
    auto get_min0(float* ptr, size_t size) -> float {
        float min = std::numeric_limits<float>::max();
        for (size_t i{}; i < size; ++i) {
            min = std::min(ptr[i], min);
        }
        return min;
    }

    auto get_min1(float* ptr, noa::Shape<noa::i64, 4> shape, int n_threads) -> float {
        float o{};
        auto output_accessor = noa::make_tuple(noa::AccessorRestrictContiguous<float, 1, noa::i64>(&o));
        noa::cpu::reduce_ewise<noa::cpu::ReduceEwiseConfig<>>(
            shape, noa::ReduceMin{},
            noa::make_tuple(noa::AccessorRestrict<float, 4, noa::i64>(ptr, shape.strides())),
            noa::make_tuple(noa::AccessorValue(std::numeric_limits<float>::max())),
            output_accessor, n_threads);
        return o;
    }

    auto get_min2(noa::View<float> ptr) -> float {
        return noa::min_max(ptr).first;
    }
}
