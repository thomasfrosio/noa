#include <algorithm>

#include "noa/common/Profiler.h"
#include "noa/cpu/math/ArithmeticsComposite.h"

namespace noa::cpu::math {
    template<typename T>
    void multiplyAddArray(const T* inputs, const T* multipliers, const T* addends, T* outputs,
                          size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        for (uint batch{0}; batch < batches; ++batch)
            for (size_t idx{0}; idx < elements; ++idx)
                outputs[batch * elements + idx] = inputs[batch * elements + idx] * multipliers[idx] + addends[idx];
    }

    template<typename T>
    void squaredDistanceFromValue(const T* inputs, const T* values, T* outputs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        for (uint batch = 0; batch < batches; ++batch) {
            const T& value = values[batch];
            size_t batch_offset = elements * static_cast<size_t>(batch);
            std::transform(inputs + batch_offset, inputs + batch_offset + elements, outputs + batch_offset,
                           [value](const T& a) -> T {
                               T distance = a - value;
                               return distance * distance;
                           });
        }
    }

    template<typename T>
    void squaredDistanceFromArray(const T* inputs, const T* array, T* outputs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        for (uint batch = 0; batch < batches; ++batch) {
            size_t batch_offset = elements * static_cast<size_t>(batch);
            std::transform(inputs + batch_offset, inputs + batch_offset + elements, array, outputs + batch_offset,
                           [](const T& a, const T& b) -> T {
                               T distance = a - b;
                               return distance * distance;
                           });
        }
    }

    #define NOA_INSTANTIATE_ARITHMETICS_COMPOSITE_(T)                                   \
    template void multiplyAddArray<T>(const T*, const T*, const T*, T*, size_t, uint);  \
    template void squaredDistanceFromValue<T>(const T*, const T*, T*, size_t, uint);    \
    template void squaredDistanceFromArray<T>(const T*, const T*, T*, size_t, uint)

    NOA_INSTANTIATE_ARITHMETICS_COMPOSITE_(int);
    NOA_INSTANTIATE_ARITHMETICS_COMPOSITE_(long);
    NOA_INSTANTIATE_ARITHMETICS_COMPOSITE_(long long);
    NOA_INSTANTIATE_ARITHMETICS_COMPOSITE_(unsigned int);
    NOA_INSTANTIATE_ARITHMETICS_COMPOSITE_(unsigned long);
    NOA_INSTANTIATE_ARITHMETICS_COMPOSITE_(unsigned long long);
    NOA_INSTANTIATE_ARITHMETICS_COMPOSITE_(float);
    NOA_INSTANTIATE_ARITHMETICS_COMPOSITE_(double);
}
