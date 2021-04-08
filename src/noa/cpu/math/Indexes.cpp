#include "noa/cpu/math/Indexes.h"

namespace Noa::Math {
    template<typename T>
    std::pair<size_t, T> firstMin(T* input, size_t elements) {
        NOA_PROFILE_FUNCTION();
        size_t min_index = 0;
        T min_value = input[0];
        for (size_t idx = 1; idx < elements; ++idx) {
            if (input[idx] < min_value) {
                min_value = input[idx];
                min_index = idx;
            }
        }
        return {min_index, min_value};
    }
    template std::pair<size_t, char> firstMin<char>(char*, size_t);
    template std::pair<size_t, unsigned char> firstMin<unsigned char>(unsigned char*, size_t);
    template std::pair<size_t, int> firstMin<int>(int*, size_t);
    template std::pair<size_t, uint> firstMin<uint>(uint*, size_t);
    template std::pair<size_t, float> firstMin<float>(float*, size_t);
    template std::pair<size_t, double> firstMin<double>(double*, size_t);

    template<typename T>
    std::pair<size_t, T> firstMax(T* input, size_t elements) {
        NOA_PROFILE_FUNCTION();
        size_t max_index = 0;
        T max_value = input[0];
        for (size_t idx = 1; idx < elements; ++idx) {
            if (max_value < input[idx]) {
                max_value = input[idx];
                max_index = idx;
            }
        }
        return {max_index, max_value};
    }
    template std::pair<size_t, char> firstMax<char>(char*, size_t);
    template std::pair<size_t, unsigned char> firstMax<unsigned char>(unsigned char*, size_t);
    template std::pair<size_t, int> firstMax<int>(int*, size_t);
    template std::pair<size_t, uint> firstMax<uint>(uint*, size_t);
    template std::pair<size_t, float> firstMax<float>(float*, size_t);
    template std::pair<size_t, double> firstMax<double>(double*, size_t);
}

