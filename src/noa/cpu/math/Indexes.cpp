#include "noa/cpu/math/Indexes.h"

namespace Noa::Math {
    template<typename T>
    void firstMin(T* inputs, size_t* output_indexes, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        for (uint batch = 0; batch < batches; ++batch) {
            T* tmp = inputs + elements * batch;
            size_t min_index = 0;
            T min_value = *tmp;
            for (size_t idx = 1; idx < elements; ++idx) {
                if (tmp[idx] < min_value) {
                    min_value = tmp[idx];
                    min_index = idx;
                }
            }
            output_indexes[batch] = min_index;
        }
    }
    template void firstMin<char>(char*, size_t*, size_t, uint);
    template void firstMin<unsigned char>(unsigned char*, size_t*, size_t, uint);
    template void firstMin<int>(int*, size_t*, size_t, uint);
    template void firstMin<uint>(uint*, size_t*, size_t, uint);

    template<typename T>
    void firstMax(T* inputs, size_t* output_indexes, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        for (uint batch = 0; batch < batches; ++batch) {
            T* tmp = inputs + elements * batch;
            size_t max_index = 0;
            T max_value = *tmp;
            for (size_t idx = 1; idx < elements; ++idx) {
                if (max_value < tmp[idx]) {
                    max_value = tmp[idx];
                    max_index = idx;
                }
            }
            output_indexes[batch] = max_index;
        }
    }
    template void firstMax<char>(char*, size_t*, size_t, uint);
    template void firstMax<unsigned char>(unsigned char*, size_t*, size_t, uint);
    template void firstMax<int>(int*, size_t*, size_t, uint);
    template void firstMax<uint>(uint*, size_t*, size_t, uint);

    template<typename T>
    void lastMin(T* inputs, size_t* output_indexes, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        for (uint batch = 0; batch < batches; ++batch) {
            T* tmp = inputs + elements * batch;
            size_t min_index = 0;
            T min_value = *tmp;
            for (size_t idx = 1; idx < elements; ++idx) {
                if (tmp[idx] <= min_value) {
                    min_value = tmp[idx];
                    min_index = idx;
                }
            }
            output_indexes[batch] = min_index;
        }
    }
    template void lastMin<char>(char*, size_t*, size_t, uint);
    template void lastMin<unsigned char>(unsigned char*, size_t*, size_t, uint);
    template void lastMin<int>(int*, size_t*, size_t, uint);
    template void lastMin<uint>(uint*, size_t*, size_t, uint);

    template<typename T>
    void lastMax(T* inputs, size_t* output_indexes, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        for (uint batch = 0; batch < batches; ++batch) {
            T* tmp = inputs + elements * batch;
            size_t max_index = 0;
            T max_value = *tmp;
            for (size_t idx = 1; idx < elements; ++idx) {
                if (max_value <= tmp[idx]) {
                    max_value = tmp[idx];
                    max_index = idx;
                }
            }
            output_indexes[batch] = max_index;
        }
    }
    template void lastMax<char>(char*, size_t*, size_t, uint);
    template void lastMax<unsigned char>(unsigned char*, size_t*, size_t, uint);
    template void lastMax<int>(int*, size_t*, size_t, uint);
    template void lastMax<uint>(uint*, size_t*, size_t, uint);
}
