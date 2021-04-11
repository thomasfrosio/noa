#include <algorithm>
#include <execution>

#include "noa/cpu/math/Arithmetics.h"

namespace Noa::Math::Details {
    template<int OPERATION, typename T, typename U>
    void applyValue(T* arrays, U* values, T* output, size_t elements, uint batches) {
        for (uint batch = 0; batch < batches; ++batch) {
            U& value = values[batch];
            size_t batch_offset = elements * static_cast<size_t>(batch);

            auto operation = [&value](const T& element) -> T {
                if constexpr (OPERATION == ADD)
                    return element + value;
                else if constexpr (OPERATION == SUBTRACT)
                    return element - value;
                else if constexpr (OPERATION == MULTIPLY)
                    return element * value;
                else if constexpr (OPERATION == DIVIDE)
                    return element / value;
                else
                    Noa::Traits::always_false_v<T>;
            };

            std::transform(std::execution::par_unseq, arrays + batch_offset, arrays + batch_offset + elements,
                           output + batch_offset, operation);
        }
    }

    template<int OPERATION, typename T, typename U>
    void applyArray(T* arrays, U* weights, T* output, size_t elements, uint batches) {
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
                    return Math::abs(weight) < static_cast<U>(1e-15) ? T(0) : value / weight;
                else if constexpr (std::is_integral_v<U>)
                    return weight == 0 ? 0 : value / weight;
                else
                    Noa::Traits::always_false_v<T>;
            }
            else
                Noa::Traits::always_false_v<T>;
        };

        for (uint batch = 0; batch < batches; ++batch) {
            size_t batch_offset = elements * static_cast<size_t>(batch);
            std::transform(std::execution::par_unseq, arrays + batch_offset, arrays + batch_offset + elements,
                           weights, output + batch_offset, operation);
        }
    }
}

// INSTANTIATIONS:
namespace Noa::Math::Details {
    #define INSTANTIATE_APPLY(T, U)                                                 \
    template void applyValue<Details::ADD, T, U>(T*, U*, T*, size_t, uint);         \
    template void applyValue<Details::SUBTRACT, T, U>(T*, U*, T*, size_t, uint);    \
    template void applyValue<Details::MULTIPLY, T, U>(T*, U*, T*, size_t, uint);    \
    template void applyValue<Details::DIVIDE, T, U>(T*, U*, T*, size_t, uint);      \
    template void applyArray<Details::ADD, T, U>(T*, U*, T*, size_t, uint);         \
    template void applyArray<Details::SUBTRACT, T, U>(T*, U*, T*, size_t, uint);    \
    template void applyArray<Details::MULTIPLY, T, U>(T*, U*, T*, size_t, uint);    \
    template void applyArray<Details::DIVIDE, T, U>(T*, U*, T*, size_t, uint)

    INSTANTIATE_APPLY(char, char);
    INSTANTIATE_APPLY(unsigned char, unsigned char);
    INSTANTIATE_APPLY(int, int);
    INSTANTIATE_APPLY(uint, uint);
    INSTANTIATE_APPLY(float, float);
    INSTANTIATE_APPLY(double, double);
    INSTANTIATE_APPLY(cfloat_t, cfloat_t);
    INSTANTIATE_APPLY(cdouble_t, cdouble_t);
    INSTANTIATE_APPLY(cfloat_t, float);
    INSTANTIATE_APPLY(cdouble_t, double);

    #define INSTANTIATE_DIVIDE_SAFE(T, U)                                           \
    template void applyArray<Details::DIVIDE_SAFE, T, U>(T*, U*, T*, size_t, uint)

    INSTANTIATE_DIVIDE_SAFE(char, char);
    INSTANTIATE_DIVIDE_SAFE(unsigned char, unsigned char);
    INSTANTIATE_DIVIDE_SAFE(int, int);
    INSTANTIATE_DIVIDE_SAFE(uint, uint);
    INSTANTIATE_DIVIDE_SAFE(float, float);
    INSTANTIATE_DIVIDE_SAFE(double, double);
    INSTANTIATE_DIVIDE_SAFE(cfloat_t, float);
    INSTANTIATE_DIVIDE_SAFE(cdouble_t, double);
}

