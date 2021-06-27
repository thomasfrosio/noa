#include <algorithm>
#include "noa/cpu/math/Arithmetics.h"

namespace noa::math::details {
    template<int OPERATION, typename T, typename U>
    void applyValue(const T* arrays, const U* values, T* outputs, size_t elements, uint batches) {
        for (uint batch = 0; batch < batches; ++batch) {
            const U& value = values[batch];
            size_t batch_offset = elements * static_cast<size_t>(batch);

            auto operation = [&value](const T& element) {
                if constexpr (OPERATION == ADD)
                    return element + value;
                else if constexpr (OPERATION == SUBTRACT)
                    return element - value;
                else if constexpr (OPERATION == MULTIPLY)
                    return element * value;
                else if constexpr (OPERATION == DIVIDE)
                    return element / value;
                else
                    noa::traits::always_false_v<T>;
            };

            std::transform(arrays + batch_offset, arrays + batch_offset + elements,
                           outputs + batch_offset, operation);
        }
    }

    template<int OPERATION, typename T, typename U>
    void applyArray(const T* arrays, const U* weights, T* outputs, size_t elements, uint batches) {
        auto operation = [](const T& value, const U& weight) -> T {
            if constexpr (OPERATION == ADD)
                return value + weight;
            else if constexpr (OPERATION == SUBTRACT)
                return value - weight;
            else if constexpr (OPERATION == MULTIPLY)
                return value * weight;
            else if constexpr (OPERATION == DIVIDE)
                return value / weight;
            else if constexpr (OPERATION == DIVIDE_SAFE) {
                if constexpr (std::is_floating_point_v<U>)
                    return math::abs(weight) < static_cast<U>(1e-15) ? T(0) : value / weight;
                else if constexpr (std::is_integral_v<U>)
                    return weight == 0 ? 0 : value / weight;
                else
                    noa::traits::always_false_v<T>;
            }
            else
                noa::traits::always_false_v<T>;
        };

        for (uint batch = 0; batch < batches; ++batch) {
            size_t batch_offset = elements * static_cast<size_t>(batch);
            std::transform(arrays + batch_offset, arrays + batch_offset + elements,
                           weights, outputs + batch_offset, operation);
        }
    }

    #define INSTANTIATE_APPLY(T, U)                                                             \
    template void applyValue<details::ADD, T, U>(const T*, const U*, T*, size_t, uint);         \
    template void applyValue<details::SUBTRACT, T, U>(const T*, const U*, T*, size_t, uint);    \
    template void applyValue<details::MULTIPLY, T, U>(const T*, const U*, T*, size_t, uint);    \
    template void applyValue<details::DIVIDE, T, U>(const T*, const U*, T*, size_t, uint);      \
    template void applyArray<details::ADD, T, U>(const T*, const U*, T*, size_t, uint);         \
    template void applyArray<details::SUBTRACT, T, U>(const T*, const U*, T*, size_t, uint);    \
    template void applyArray<details::MULTIPLY, T, U>(const T*, const U*, T*, size_t, uint);    \
    template void applyArray<details::DIVIDE, T, U>(const T*, const U*, T*, size_t, uint)

    INSTANTIATE_APPLY(int, int);
    INSTANTIATE_APPLY(uint, uint);
    INSTANTIATE_APPLY(float, float);
    INSTANTIATE_APPLY(double, double);
    INSTANTIATE_APPLY(cfloat_t, cfloat_t);
    INSTANTIATE_APPLY(cdouble_t, cdouble_t);
    INSTANTIATE_APPLY(cfloat_t, float);
    INSTANTIATE_APPLY(cdouble_t, double);

    #define INSTANTIATE_DIVIDE_SAFE(T, U) \
    template void applyArray<details::DIVIDE_SAFE, T, U>(const T*, const U*, T*, size_t, uint)

    INSTANTIATE_DIVIDE_SAFE(int, int);
    INSTANTIATE_DIVIDE_SAFE(uint, uint);
    INSTANTIATE_DIVIDE_SAFE(float, float);
    INSTANTIATE_DIVIDE_SAFE(double, double);
    INSTANTIATE_DIVIDE_SAFE(cfloat_t, float);
    INSTANTIATE_DIVIDE_SAFE(cdouble_t, double);
}
