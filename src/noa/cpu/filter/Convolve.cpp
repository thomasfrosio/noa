#include "noa/common/Assert.h"
#include "noa/common/Profiler.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/filter/Convolve.h"

namespace {
    using namespace ::noa;

    template<typename T,typename U, int DIM>
    void convolve_(const T* __restrict inputs, size3_t input_pitch,
                   T* __restrict outputs, size3_t output_pitch, size3_t shape, size_t batches,
                   const U* __restrict filter, int3_t filter_size, size_t threads) {
        const int3_t int_shape(shape);
        const int3_t HALO = filter_size / 2;

        // If half precision, convert filter and do the accumulation is single-precision.
        // Without this, half precision would be ~10times slower. With this preprocessing, it is only 2times slower.
        using Comp = std::conditional_t<std::is_same_v<half_t, T>, float, T>;
        cpu::memory::PtrHost<Comp> buffer;
        const Comp* kernel;
        if constexpr (std::is_same_v<half_t, T> && std::is_same_v<half_t, U>) {
            buffer.reset(static_cast<size_t>(elements(filter_size)));
            for (size_t i = 0; i < buffer.size(); ++i)
                buffer[i] = static_cast<Comp>(filter[i]);
            kernel = buffer.get();
        } else {
            kernel = filter;
        }

        #pragma omp parallel for num_threads(threads) default(none) collapse(4) \
        shared(inputs, input_pitch, outputs, output_pitch, batches, filter_size, int_shape, HALO, kernel)

        for (size_t batch = 0; batch < batches; ++batch) {
            for (int z = 0; z < int_shape.z; ++z) {
                for (int y = 0; y < int_shape.y; ++y) {
                    for (int x = 0; x < int_shape.x; ++x) {

                        const T* input = inputs + batch * elements(input_pitch);
                        T* output = outputs + batch * elements(output_pitch);
                        output += index(x, y, z, output_pitch);

                        Comp conv = 0;
                        if constexpr (DIM == 0) {
                            for (int w_x = 0; w_x < filter_size.x; ++w_x) {
                                int idx_x = x - HALO.x + w_x;
                                if (idx_x >= 0 && idx_x < int_shape.x)
                                    conv += static_cast<Comp>(input[index(idx_x, y, z, input_pitch)]) *
                                            kernel[w_x];
                            }
                        } else if constexpr (DIM == 1) {
                            for (int w_y = 0; w_y < filter_size.y; ++w_y) {
                                int idx_y = y - HALO.y + w_y;
                                if (idx_y < 0 || idx_y >= int_shape.y)
                                    continue;
                                int tmp = w_y * filter_size.x;
                                for (int w_x = 0; w_x < filter_size.x; ++w_x) {
                                    int idx_x = x - HALO.x + w_x;
                                    if (idx_x >= 0 && idx_x < int_shape.x)
                                        conv += static_cast<Comp>(input[index(idx_x, idx_y, z, input_pitch)]) *
                                                kernel[tmp + w_x];
                                }
                            }
                        } else if constexpr (DIM == 2) {
                            for (int w_z = 0; w_z < filter_size.z; ++w_z) {
                                int idx_z = z - HALO.z + w_z;
                                if (idx_z < 0 || idx_z >= int_shape.z)
                                    continue;
                                int tmp_z = w_z * filter_size.y * filter_size.x;
                                for (int w_y = 0; w_y < filter_size.y; ++w_y) {
                                    int idx_y = y - HALO.y + w_y;
                                    if (idx_y < 0 || idx_y >= int_shape.y)
                                        continue;
                                    int tmp = tmp_z + w_y * filter_size.x;
                                    for (int w_x = 0; w_x < filter_size.x; ++w_x) {
                                        int idx_x = x - HALO.x + w_x;
                                        if (idx_x >= 0 && idx_x < int_shape.x)
                                            conv += static_cast<Comp>(input[index(idx_x, idx_y, idx_z, input_pitch)]) *
                                                    kernel[tmp + w_x];
                                    }
                                }
                            }
                        }
                        *output = static_cast<T>(conv);
                    }
                }
            }
        }
    }

    template<typename T, typename U, int DIM>
    void convolveSep_(const T* __restrict inputs, size3_t input_pitch, T* __restrict outputs, size3_t output_pitch,
                      size3_t shape, size_t batches, const U* __restrict filter, int filter_size, size_t threads) {
        const int3_t int_shape(shape);
        const int HALO = filter_size / 2;

        // If half precision, convert filter and do the accumulation is single-precision.
        // Without this, half precision would be ~10times slower. With this preprocessing, it is only 2times slower.
        using Comp = std::conditional_t<std::is_same_v<half_t, T>, float, T>;
        cpu::memory::PtrHost<Comp> buffer;
        const Comp* kernel;
        if constexpr (std::is_same_v<half_t, T> && std::is_same_v<half_t, U>) {
            buffer.reset(static_cast<size_t>(filter_size));
            for (size_t i = 0; i < buffer.size(); ++i)
                buffer[i] = static_cast<Comp>(filter[i]);
            kernel = buffer.get();
        } else {
            kernel = filter;
        }

        #pragma omp parallel for num_threads(threads) default(none) collapse(4) \
        shared(inputs, input_pitch, outputs, output_pitch, batches, filter_size, int_shape, HALO, kernel)

        for (size_t batch = 0; batch < batches; ++batch) {
            for (int z = 0; z < int_shape.z; ++z) {
                for (int y = 0; y < int_shape.y; ++y) {
                    for (int x = 0; x < int_shape.x; ++x) {

                        const T* input = inputs + batch * elements(input_pitch);
                        T* output = outputs + batch * elements(output_pitch);
                        output += index(x, y, z, output_pitch);

                        Comp conv = 0;
                        if constexpr (DIM == 0) {
                            for (int w_x = 0; w_x < filter_size; ++w_x) {
                                int idx_x = x - HALO + w_x;
                                if (idx_x >= 0 && idx_x < int_shape.x)
                                    conv += static_cast<Comp>(input[index(idx_x, y, z, input_pitch)]) * kernel[w_x];
                            }
                        } else if constexpr (DIM == 1) {
                            for (int w_y = 0; w_y < filter_size; ++w_y) {
                                int idx_y = y - HALO + w_y;
                                if (idx_y >= 0 && idx_y < int_shape.y)
                                    conv += static_cast<Comp>(input[index(x, idx_y, z, input_pitch)]) * kernel[w_y];
                            }
                        } else if constexpr (DIM == 2) {
                            for (int w_z = 0; w_z < filter_size; ++w_z) {
                                int idx_z = z - HALO + w_z;
                                if (idx_z >= 0 && idx_z < int_shape.z)
                                    conv += static_cast<Comp>(input[index(x, y, idx_z, input_pitch)]) * kernel[w_z];
                            }
                        }
                        *output = static_cast<T>(conv);
                    }
                }
            }
        }
    }
}

namespace noa::cpu::filter {
    template<typename T, typename U>
    void convolve1(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch,
                   size3_t shape, size_t batches, const U* filter, size_t filter_size, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);
        NOA_ASSERT(filter_size % 2); // only odd sizes
        if (filter_size == 1)
            return memory::copy(inputs, input_pitch, outputs, output_pitch, shape, batches, stream);

        stream.enqueue(convolve_<T, U, 0>,
                       inputs, input_pitch, outputs, output_pitch, shape, batches,
                       filter, int3_t(filter_size, 1, 1), stream.threads());
    }

    template<typename T, typename U>
    void convolve2(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch,
                   size3_t shape, size_t batches, const U* filter, size2_t filter_shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);
        NOA_ASSERT(all((filter_shape % 2) == 1)); // only odd sizes
        if (all(filter_shape == 1))
            return memory::copy(inputs, input_pitch, outputs, output_pitch, shape, batches, stream);

        stream.enqueue(convolve_<T, U, 1>,
                       inputs, input_pitch, outputs, output_pitch, shape, batches,
                       filter, int3_t(filter_shape.x, filter_shape.y, 1), stream.threads());
    }

    template<typename T, typename U>
    void convolve3(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch,
                   size3_t shape, size_t batches, const U* filter, size3_t filter_shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);
        NOA_ASSERT(all((filter_shape % 2) == 1)); // only odd sizes
        if (all(filter_shape == 1))
            return memory::copy(inputs, input_pitch, outputs, output_pitch, shape, batches, stream);

        stream.enqueue(convolve_<T, U, 2>,
                       inputs, input_pitch, outputs, output_pitch, shape, batches,
                       filter, int3_t(filter_shape), stream.threads());
    }

    template<typename T, typename U>
    void convolve(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch,
                  size3_t shape, size_t batches,
                  const U* filter0, size_t filter0_size,
                  const U* filter1, size_t filter1_size,
                  const U* filter2, size_t filter2_size,
                  T* tmp, size3_t tmp_pitch, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);
        const int3_t fs(filter0_size, filter1_size, filter2_size);
        size_t threads = stream.threads();

        if (filter0 && filter1 && filter2) {
            NOA_ASSERT(filter0_size % 2);
            NOA_ASSERT(filter1_size % 2);
            NOA_ASSERT(filter2_size % 2);
            stream.enqueue([=]() {
                convolveSep_<T, U, 0>(inputs, input_pitch, outputs, output_pitch, shape, batches, filter0, fs[0], threads);
                convolveSep_<T, U, 1>(outputs, output_pitch, tmp, tmp_pitch, shape, batches, filter1, fs[1], threads);
                convolveSep_<T, U, 2>(tmp, tmp_pitch, outputs, output_pitch, shape, batches, filter2, fs[2], threads);
            });
        } else if (filter0 && filter1) {
            NOA_ASSERT(filter0_size % 2);
            NOA_ASSERT(filter1_size % 2);
            stream.enqueue([=]() {
                convolveSep_<T, U, 0>(inputs, input_pitch, tmp, tmp_pitch, shape, batches, filter0, fs[0], threads);
                convolveSep_<T, U, 1>(tmp, tmp_pitch, outputs, output_pitch, shape, batches, filter1, fs[1], threads);
            });
        } else if (filter1 && filter2) {
            NOA_ASSERT(filter1_size % 2);
            NOA_ASSERT(filter2_size % 2);
            stream.enqueue([=]() {
                convolveSep_<T, U, 1>(inputs, input_pitch, tmp, tmp_pitch, shape, batches, filter1, fs[1], threads);
                convolveSep_<T, U, 2>(tmp, tmp_pitch, outputs, output_pitch, shape, batches, filter2, fs[2], threads);
            });
        } else if (filter0 && filter2) {
            NOA_ASSERT(filter0_size % 2);
            NOA_ASSERT(filter2_size % 2);
            stream.enqueue([=]() {
                convolveSep_<T, U, 0>(inputs, input_pitch, tmp, tmp_pitch, shape, batches, filter0, fs[0], threads);
                convolveSep_<T, U, 2>(tmp, tmp_pitch, outputs, output_pitch, shape, batches, filter2, fs[2], threads);
            });
        } else if (filter0) {
            NOA_ASSERT(filter0_size % 2);
            stream.enqueue(convolveSep_<T, U, 0>,
                           inputs, input_pitch, outputs, output_pitch, shape, batches, filter0, fs[0], threads);
        } else if (filter1) {
            NOA_ASSERT(filter1_size % 2);
            stream.enqueue(convolveSep_<T, U, 1>,
                           inputs, input_pitch, outputs, output_pitch, shape, batches, filter1, fs[1], threads);
        } else if (filter2) {
            NOA_ASSERT(filter2_size % 2);
            stream.enqueue(convolveSep_<T, U, 2>,
                           inputs, input_pitch, outputs, output_pitch, shape, batches, filter2, fs[2], threads);
        }
    }

    #define NOA_INSTANTIATE_CONV_(T)                                                                            \
    template void convolve1<T>(const T*, size3_t, T*, size3_t, size3_t, size_t, const T*, size_t, Stream&);     \
    template void convolve2<T>(const T*, size3_t, T*, size3_t, size3_t, size_t, const T*, size2_t, Stream&);    \
    template void convolve3<T>(const T*, size3_t, T*, size3_t, size3_t, size_t, const T*, size3_t, Stream&);    \
    template void convolve<T>(const T*, size3_t, T*, size3_t, size3_t, size_t, const T*, size_t, const T*, size_t, const T*, size_t, T*, size3_t, Stream&)

    NOA_INSTANTIATE_CONV_(half_t);
    NOA_INSTANTIATE_CONV_(float);
    NOA_INSTANTIATE_CONV_(double);

    template void convolve1<half_t>(const half_t*, size3_t, half_t*, size3_t, size3_t, size_t, const float*, size_t, Stream&);
    template void convolve2<half_t>(const half_t*, size3_t, half_t*, size3_t, size3_t, size_t, const float*, size2_t, Stream&);
    template void convolve3<half_t>(const half_t*, size3_t, half_t*, size3_t, size3_t, size_t, const float*, size3_t, Stream&);
    template void convolve<half_t>(const half_t*, size3_t, half_t*, size3_t, size3_t, size_t, const float*, size_t,  const float*, size_t,  const float*, size_t, half_t*, size3_t, Stream&);
}
