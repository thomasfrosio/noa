#include "noa/common/Assert.h"
#include "noa/common/Profiler.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/filter/Convolve.h"

namespace {
    using namespace ::noa;

    template<typename T, int DIM>
    void convolve_(const T* __restrict input, T* __restrict output, int3_t shape,
                   const T* __restrict filter, int3_t filter_size) {
        const int3_t HALO = filter_size / 2;

        int2_t offset;
        for (int z = 0; z < shape.z; ++z) {
            offset[1] = z * shape.y * shape.x; // offset to current page
            for (int y = 0; y < shape.y; ++y) {
                offset[0] = offset[1] + y * shape.x; // offset to current row
                for (int x = 0; x < shape.x; ++x, ++output) {

                    T conv = 0;
                    if constexpr (DIM == 0) {
                        for (int w_x = 0; w_x < filter_size.x; ++w_x) {
                            int idx_x = x - HALO.x + w_x;
                            if (idx_x >= 0 && idx_x < shape.x)
                                conv += input[offset[0] + idx_x] * filter[w_x];
                        }
                    } else if constexpr (DIM == 1) {
                        for (int w_y = 0; w_y < filter_size.y; ++w_y) {
                            int idx_y = y - HALO.y + w_y;
                            if (idx_y < 0 || idx_y >= shape.y)
                                continue;
                            int tmp = w_y * filter_size.x;
                            for (int w_x = 0; w_x < filter_size.x; ++w_x) {
                                int idx_x = x - HALO.x + w_x;
                                if (idx_x >= 0 && idx_x < shape.x)
                                    conv += input[offset[1] + idx_y * shape.x + idx_x] * filter[tmp + w_x];
                            }
                        }
                    } else if constexpr (DIM == 2) {
                        for (int w_z = 0; w_z < filter_size.z; ++w_z) {
                            int idx_z = z - HALO.z + w_z;
                            if (idx_z < 0 || idx_z >= shape.z)
                                continue;
                            int tmp_z = w_z * filter_size.y * filter_size.x;
                            for (int w_y = 0; w_y < filter_size.y; ++w_y) {
                                int idx_y = y - HALO.y + w_y;
                                if (idx_y < 0 || idx_y >= shape.y)
                                    continue;
                                int tmp = tmp_z + w_y * filter_size.x;
                                for (int w_x = 0; w_x < filter_size.x; ++w_x) {
                                    int idx_x = x - HALO.x + w_x;
                                    if (idx_x >= 0 && idx_x < shape.x)
                                        conv += input[(idx_z * shape.y + idx_y) * shape.x + idx_x] * filter[tmp + w_x];
                                }
                            }
                        }
                    }
                    *output = conv;
                }
            }
        }
    }

    template<typename T, int DIM>
    void convolve_(const T* __restrict input, T* __restrict output, int3_t shape,
                   const T* __restrict filter, int filter_size) {
        const int HALO = filter_size / 2;

        int2_t offset;
        for (int z = 0; z < shape.z; ++z) {
            offset[1] = z * shape.y * shape.x; // offset to current page
            for (int y = 0; y < shape.y; ++y) {
                offset[0] = offset[1] + y * shape.x; // offset to current row
                for (int x = 0; x < shape.x; ++x, ++output) {

                    T conv = 0;
                    if constexpr (DIM == 0) {
                        for (int w_x = 0; w_x < filter_size; ++w_x) {
                            int idx_x = x - HALO + w_x;
                            if (idx_x >= 0 && idx_x < shape.x)
                                conv += input[offset[0] + idx_x] * filter[w_x];
                        }
                    } else if constexpr (DIM == 1) {
                        for (int w_y = 0; w_y < filter_size; ++w_y) {
                            int idx_y = y - HALO + w_y;
                            if (idx_y >= 0 && idx_y < shape.y)
                                conv += input[offset[1] + idx_y * shape.x + x] * filter[w_y];
                        }
                    } else if constexpr (DIM == 2) {
                        for (int w_z = 0; w_z < filter_size; ++w_z) {
                            int idx_z = z - HALO + w_z;
                            if (idx_z >= 0 && idx_z < shape.z)
                                conv += input[(idx_z * shape.y + y) * shape.x + x] * filter[w_z];
                        }
                    }
                    *output = conv;
                }
            }
        }
    }
}

namespace noa::cpu::filter {
    template<typename T>
    void convolve1(const T* inputs, T* outputs, size3_t shape, size_t batches,
                   const T* filter, size_t filter_size) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);
        size_t elements = noa::elements(shape);
        if (filter_size == 1)
            return memory::copy(inputs, outputs, elements * batches);

        const int3_t int_shape(shape);
        const int3_t int_filter_size(filter_size, 0, 0);
        for (size_t batch = 0; batch < batches; ++batch) {
            size_t offset = batch * elements;
            convolve_<T, 0>(inputs + offset, outputs + offset, int_shape, filter, int_filter_size);
        }
    }

    template<typename T>
    void convolve2(const T* inputs, T* outputs, size3_t shape, size_t batches,
                   const T* filter, size2_t filter_shape) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);
        size_t elements = noa::elements(shape);
        if (all(filter_shape == size_t{1}))
            return memory::copy(inputs, outputs, elements * batches);

        const int3_t int_shape(shape);
        const int3_t int_filter_size(filter_shape.x, filter_shape.y, 0);
        for (size_t batch = 0; batch < batches; ++batch) {
            size_t offset = batch * elements;
            convolve_<T, 1>(inputs + offset, outputs + offset, int_shape, filter, int_filter_size);
        }
    }

    template<typename T>
    void convolve3(const T* inputs, T* outputs, size3_t shape, size_t batches,
                   const T* filter, size3_t filter_shape) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);
        size_t elements = noa::elements(shape);
        if (all(filter_shape == size_t{1}))
            return memory::copy(inputs, outputs, elements * batches);

        const int3_t int_shape(shape);
        const int3_t int_filter_size(filter_shape);
        for (size_t batch = 0; batch < batches; ++batch) {
            size_t offset = batch * elements;
            convolve_<T, 2>(inputs + offset, outputs + offset, int_shape, filter, int_filter_size);
        }
    }

    template<typename T>
    void convolve(const T* inputs, T* outputs, size3_t shape, size_t batches,
                  const T* filter0, size_t filter0_size,
                  const T* filter1, size_t filter1_size,
                  const T* filter2, size_t filter2_size,
                  T* tmp) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);
        const int3_t int_shape(shape);
        const int3_t filter_size(filter0_size, filter1_size, filter2_size);

        for (size_t batch = 0; batch < batches; ++batch) {
            size_t offset = batch * elements(shape);
            const T* input = inputs + offset;
            T* output = outputs + offset;

            if (filter0 && filter1 && filter2) {
                convolve_<T, 0>(input, output, int_shape, filter0, filter_size[0]);
                convolve_<T, 1>(output, tmp, int_shape, filter1, filter_size[1]);
                convolve_<T, 2>(tmp, output, int_shape, filter2, filter_size[2]);
            } else if (filter0 && filter1) {
                convolve_<T, 0>(input, tmp, int_shape, filter0, filter_size[0]);
                convolve_<T, 1>(tmp, output, int_shape, filter1, filter_size[1]);
            } else if (filter1 && filter2) {
                convolve_<T, 1>(input, tmp, int_shape, filter1, filter_size[1]);
                convolve_<T, 2>(tmp, output, int_shape, filter2, filter_size[2]);
            } else if (filter0 && filter2) {
                convolve_<T, 0>(input, tmp, int_shape, filter0, filter_size[0]);
                convolve_<T, 2>(tmp, output, int_shape, filter2, filter_size[2]);
            } else if (filter0) {
                convolve_<T, 0>(input, output, int_shape, filter0, filter_size[0]);
            } else if (filter1) {
                convolve_<T, 1>(input, output, int_shape, filter1, filter_size[1]);
            } else if (filter2) {
                convolve_<T, 2>(input, output, int_shape, filter2, filter_size[2]);
            }
        }
    }

    #define NOA_INSTANTIATE_CONV_(T)                                                \
    template void convolve1<T>(const T*, T*, size3_t, size_t, const T*, size_t);    \
    template void convolve2<T>(const T*, T*, size3_t, size_t, const T*, size2_t);   \
    template void convolve3<T>(const T*, T*, size3_t, size_t, const T*, size3_t);   \
    template void convolve<T>(const T*, T*, size3_t, size_t, const T*, size_t, const T*, size_t, const T*, size_t, T*)

    NOA_INSTANTIATE_CONV_(float);
    NOA_INSTANTIATE_CONV_(double);
}
