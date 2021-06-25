#include "noa/cpu/math/Indexes.h"

namespace noa::math {
    template<typename T>
    void firstMin(const T* inputs, size_t* output_indexes, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        for (uint batch = 0; batch < batches; ++batch) {
            const T* tmp = inputs + elements * batch;
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

    template<typename T>
    void firstMax(const T* inputs, size_t* output_indexes, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        for (uint batch = 0; batch < batches; ++batch) {
            const T* tmp = inputs + elements * batch;
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

    template<typename T>
    void lastMin(const T* inputs, size_t* output_indexes, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        for (uint batch = 0; batch < batches; ++batch) {
            const T* tmp = inputs + elements * batch;
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

    template<typename T>
    void lastMax(const T* inputs, size_t* output_indexes, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        for (uint batch = 0; batch < batches; ++batch) {
            const T* tmp = inputs + elements * batch;
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

    #define INSTANTIATE_INDEXES(T)                                  \
    template void firstMin<T>(const T*, size_t*, size_t, uint);     \
    template void firstMax<T>(const T*, size_t*, size_t, uint);     \
    template void lastMin<T>(const T*, size_t*, size_t, uint);      \
    template void lastMax<T>(const T*, size_t*, size_t, uint)

    INSTANTIATE_INDEXES(char);
    INSTANTIATE_INDEXES(short);
    INSTANTIATE_INDEXES(int);
    INSTANTIATE_INDEXES(long);
    INSTANTIATE_INDEXES(long long);
    INSTANTIATE_INDEXES(unsigned char);
    INSTANTIATE_INDEXES(unsigned short);
    INSTANTIATE_INDEXES(unsigned int);
    INSTANTIATE_INDEXES(unsigned long);
    INSTANTIATE_INDEXES(unsigned long long);
}
