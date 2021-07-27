#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/filter/Convolve.h"

namespace {
    using namespace ::noa;

    template<typename T, int DIM>
    void convolve_(const T* input, T* output, int3_t shape, const T* filter, int3_t filter_size) {
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
    void convolve_(const T* input, T* output, int3_t shape, const T* filter, int filter_size) {
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

namespace noa::filter {
    template<typename T>
    void convolve1(const T* inputs, T* outputs, size3_t shape, uint batches,
                   const T* filter, uint filter_size) {
        size_t elements = getElements(shape);
        if (filter_size == 1) {
            memory::copy(inputs, outputs, elements * batches);
            return;
        }

        int3_t tmp_shape(shape);
        int3_t tmp_filter_size(filter_size, 0, 0);
        for (uint batch = 0; batch < batches; ++batch) {
            size_t offset = batch * elements;
            convolve_<T, 0>(inputs + offset, outputs + offset, tmp_shape, filter, tmp_filter_size);
        }
    }

    template<typename T>
    void convolve2(const T* inputs, T* outputs, size3_t shape, uint batches,
                   const T* filter, uint2_t filter_shape) {
        size_t elements = getElements(shape);
        if (all(filter_shape == 1U)) {
            memory::copy(inputs, outputs, elements * batches);
            return;
        }

        int3_t tmp_shape(shape);
        int3_t tmp_filter_size(filter_shape.x, filter_shape.y, 0);
        for (uint batch = 0; batch < batches; ++batch) {
            size_t offset = batch * elements;
            convolve_<T, 1>(inputs + offset, outputs + offset, tmp_shape, filter, tmp_filter_size);
        }
    }

    template<typename T>
    void convolve3(const T* inputs, T* outputs, size3_t shape, uint batches,
                   const T* filter, uint3_t filter_shape) {
        size_t elements = getElements(shape);
        if (all(filter_shape == 1U)) {
            memory::copy(inputs, outputs, elements * batches);
            return;
        }

        int3_t tmp_shape(shape);
        int3_t tmp_filter_size(filter_shape);
        for (uint batch = 0; batch < batches; ++batch) {
            size_t offset = batch * elements;
            convolve_<T, 2>(inputs + offset, outputs + offset, tmp_shape, filter, tmp_filter_size);
        }
    }

    template<typename T>
    void convolve(const T* inputs, T* outputs, size3_t shape, uint batches,
                  const T* filter0, uint filter0_size,
                  const T* filter1, uint filter1_size,
                  const T* filter2, uint filter2_size,
                  T* tmp) {
        int3_t tmp_shape(shape);
        size_t elements = getElements(shape);
        int3_t filter_size(filter0_size, filter1_size, filter2_size);

        for (uint batch = 0; batch < batches; ++batch) {
            size_t offset = batch * elements;
            const T* input = inputs + offset;
            T* output = outputs + offset;

            if (filter0 && filter1 && filter2) {
                convolve_<T, 0>(input, output, tmp_shape, filter0, filter_size[0]);
                convolve_<T, 1>(output, tmp, tmp_shape, filter1, filter_size[1]);
                convolve_<T, 2>(tmp, output, tmp_shape, filter2, filter_size[2]);
            } else if (filter0 && filter1) {
                convolve_<T, 0>(input, tmp, tmp_shape, filter0, filter_size[0]);
                convolve_<T, 1>(tmp, output, tmp_shape, filter1, filter_size[1]);
            } else if (filter1 && filter2) {
                convolve_<T, 1>(input, tmp, tmp_shape, filter1, filter_size[1]);
                convolve_<T, 2>(tmp, output, tmp_shape, filter2, filter_size[2]);
            } else if (filter0 && filter2) {
                convolve_<T, 0>(input, tmp, tmp_shape, filter0, filter_size[0]);
                convolve_<T, 2>(tmp, output, tmp_shape, filter2, filter_size[2]);
            } else if (filter0) {
                convolve_<T, 0>(input, output, tmp_shape, filter0, filter_size[0]);
            } else if (filter1) {
                convolve_<T, 1>(input, output, tmp_shape, filter1, filter_size[1]);
            } else if (filter2) {
                convolve_<T, 2>(input, output, tmp_shape, filter2, filter_size[2]);
            }
        }
    }

    #define NOA_INSTANTIATE_CONV_(T)                                            \
    template void convolve1<T>(const T*, T*, size3_t, uint, const T*, uint);    \
    template void convolve2<T>(const T*, T*, size3_t, uint, const T*, uint2_t); \
    template void convolve3<T>(const T*, T*, size3_t, uint, const T*, uint3_t); \
    template void convolve<T>(const T*, T*, size3_t, uint, const T*, uint, const T*, uint, const T*, uint, T*)

    NOA_INSTANTIATE_CONV_(float);
    NOA_INSTANTIATE_CONV_(double);
}
