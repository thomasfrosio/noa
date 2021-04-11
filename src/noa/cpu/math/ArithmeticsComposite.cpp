#include <algorithm>
#include <execution>

#include "noa/Profiler.h"
#include "noa/cpu/math/ArithmeticsComposite.h"

namespace Noa::Math {
    template<typename T>
    void multiplyAddArray(T* inputs, T* multipliers, T* addends, T* outputs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        for (uint batch{0}; batch < batches; ++batch)
            for (size_t idx{0}; idx < elements; ++idx)
                outputs[batch * elements + idx] = inputs[batch * elements + idx] * multipliers[idx] + addends[idx];
    }
    template void multiplyAddArray<int>(int*, int*, int*, int*, size_t, uint);
    template void multiplyAddArray<uint>(uint*, uint*, uint*, uint*, size_t, uint);
    template void multiplyAddArray<float>(float*, float*, float*, float*, size_t, uint);
    template void multiplyAddArray<double>(double*, double*, double*, double*, size_t, uint);

    template<typename T>
    void squaredDistanceFromValue(T* inputs, T* values, T* outputs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        for (uint batch = 0; batch < batches; ++batch) {
            T& value = values[batch];
            size_t batch_offset = elements * static_cast<size_t>(batch);
            std::transform(std::execution::par_unseq,
                           inputs + batch_offset, inputs + batch_offset + elements, outputs + batch_offset,
                           [value](T a) -> T {
                               T distance = a - value;
                               return distance * distance;
                           });
        }
    }
    template void squaredDistanceFromValue<int>(int*, int*, int*, size_t, uint);
    template void squaredDistanceFromValue<uint>(uint*, uint*, uint*, size_t, uint);
    template void squaredDistanceFromValue<float>(float*, float*, float*, size_t, uint);
    template void squaredDistanceFromValue<double>(double*, double*, double*, size_t, uint);

    template<typename T>
    void squaredDistanceFromArray(T* inputs, T* array, T* outputs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        for (uint batch = 0; batch < batches; ++batch) {
            size_t batch_offset = elements * static_cast<size_t>(batch);
            std::transform(std::execution::par_unseq,
                           inputs + batch_offset, inputs + batch_offset + elements, array, outputs + batch_offset,
                           [](T a, T b) {
                               T distance = a - b;
                               return distance * distance;
                           });
        }
    }
    template void squaredDistanceFromArray<int>(int*, int*, int*, size_t, uint);
    template void squaredDistanceFromArray<uint>(uint*, uint*, uint*, size_t, uint);
    template void squaredDistanceFromArray<float>(float*, float*, float*, size_t, uint);
    template void squaredDistanceFromArray<double>(double*, double*, double*, size_t, uint);
}
