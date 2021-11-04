#include <algorithm>

#include "noa/common/Math.h"
#include "noa/cpu/math/Arithmetics.h"

namespace noa::cpu::math::details {
    template<int OPERATION, typename T, typename U>
    void applyValue(const T* arrays, const U* values, T* outputs, size_t elements, size_t batches) {
        for (size_t batch = 0; batch < batches; ++batch) {
            const U& value = values[batch];
            size_t batch_offset = elements * batch;

            auto operation = [value](const T& element) {
                if constexpr (OPERATION == ADD)
                    return element + value;
                else if constexpr (OPERATION == SUBTRACT)
                    return element - value;
                else if constexpr (OPERATION == MULTIPLY)
                    return element * value;
                else if constexpr (OPERATION == DIVIDE)
                    return element / value;
                else
                    static_assert(noa::traits::always_false_v<T>);
            };

            std::transform(arrays + batch_offset, arrays + batch_offset + elements,
                           outputs + batch_offset, operation);
        }
    }

    template<int OPERATION, typename T, typename U>
    void applyArray(const T* arrays, const U* weights, T* outputs, size_t elements, size_t batches) {
        auto operation = [](const T& value, const U& weight) -> T {
            if constexpr (OPERATION == ADD) {
                return value + weight;
            } else if constexpr (OPERATION == SUBTRACT) {
                return value - weight;
            } else if constexpr (OPERATION == MULTIPLY) {
                return value * weight;
            } else if constexpr (OPERATION == DIVIDE) {
                return value / weight;
            } else if constexpr (OPERATION == DIVIDE_SAFE) {
                if constexpr (std::is_floating_point_v<U>)
                    return noa::math::abs(weight) < noa::math::Limits<U>::epsilon() ?
                           static_cast<T>(0) : value / weight;
                else if constexpr (std::is_integral_v<U>)
                    return weight == 0 ? 0 : value / weight;
                else
                    static_assert(noa::traits::always_false_v<T>);
            } else {
                static_assert(noa::traits::always_false_v<T>);
            }
        };

        for (size_t batch = 0; batch < batches; ++batch) {
            size_t batch_offset = elements * batch;
            std::transform(arrays + batch_offset, arrays + batch_offset + elements,
                           weights, outputs + batch_offset, operation);
        }
    }

    #define NOA_INSTANTIATE_APPLY_(T, U)                                                        \
    template void applyValue<details::ADD, T, U>(const T*, const U*, T*, size_t, size_t);       \
    template void applyValue<details::SUBTRACT, T, U>(const T*, const U*, T*, size_t, size_t);  \
    template void applyValue<details::MULTIPLY, T, U>(const T*, const U*, T*, size_t, size_t);  \
    template void applyValue<details::DIVIDE, T, U>(const T*, const U*, T*, size_t, size_t);    \
    template void applyArray<details::ADD, T, U>(const T*, const U*, T*, size_t, size_t);       \
    template void applyArray<details::SUBTRACT, T, U>(const T*, const U*, T*, size_t, size_t);  \
    template void applyArray<details::MULTIPLY, T, U>(const T*, const U*, T*, size_t, size_t);  \
    template void applyArray<details::DIVIDE, T, U>(const T*, const U*, T*, size_t, size_t)

    NOA_INSTANTIATE_APPLY_(int, int);
    NOA_INSTANTIATE_APPLY_(long, long);
    NOA_INSTANTIATE_APPLY_(long long, long long);
    NOA_INSTANTIATE_APPLY_(unsigned int, unsigned int);
    NOA_INSTANTIATE_APPLY_(unsigned long, unsigned long);
    NOA_INSTANTIATE_APPLY_(unsigned long long, unsigned long long);
    NOA_INSTANTIATE_APPLY_(float, float);
    NOA_INSTANTIATE_APPLY_(double, double);
    NOA_INSTANTIATE_APPLY_(cfloat_t, cfloat_t);
    NOA_INSTANTIATE_APPLY_(cdouble_t, cdouble_t);
    NOA_INSTANTIATE_APPLY_(cfloat_t, float);
    NOA_INSTANTIATE_APPLY_(cdouble_t, double);

    #define NOA_INSTANTIATE_DIVIDE_SAFE_(T, U) \
    template void applyArray<details::DIVIDE_SAFE, T, U>(const T*, const U*, T*, size_t, size_t)

    NOA_INSTANTIATE_DIVIDE_SAFE_(int, int);
    NOA_INSTANTIATE_DIVIDE_SAFE_(long, long);
    NOA_INSTANTIATE_DIVIDE_SAFE_(long long, long long);
    NOA_INSTANTIATE_DIVIDE_SAFE_(unsigned int, unsigned int);
    NOA_INSTANTIATE_DIVIDE_SAFE_(unsigned long, unsigned long);
    NOA_INSTANTIATE_DIVIDE_SAFE_(unsigned long long, unsigned long long);
    NOA_INSTANTIATE_DIVIDE_SAFE_(float, float);
    NOA_INSTANTIATE_DIVIDE_SAFE_(double, double);
    NOA_INSTANTIATE_DIVIDE_SAFE_(cfloat_t, float);
    NOA_INSTANTIATE_DIVIDE_SAFE_(cdouble_t, double);
}
