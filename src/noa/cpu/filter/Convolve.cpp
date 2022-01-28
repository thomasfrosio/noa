#include "noa/common/Assert.h"
#include "noa/common/Profiler.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/math/Ewise.h"
#include "noa/cpu/filter/Convolve.h"

namespace {
    using namespace ::noa;

    template<typename T, typename U, int DIM>
    void convolve_(const T* __restrict input, size4_t input_stride,
                   T* __restrict output, size4_t output_stride, size4_t shape,
                   const U* __restrict filter, int3_t filter_size, size_t threads) {
        const int4_t int_shape(shape);
        const int3_t HALO = filter_size / 2;

        // If half precision, convert filter and do the accumulation is single-precision.
        // Without this, half precision would be ~10times slower. With this preprocessing, it is only 2times slower.
        using Comp = std::conditional_t<std::is_same_v<half_t, T>, float, T>;
        cpu::memory::PtrHost<Comp> buffer;
        const Comp* kernel;
        if constexpr (std::is_same_v<half_t, T> && std::is_same_v<half_t, U>) {
            buffer.reset(static_cast<size_t>(filter_size.elements()));
            for (size_t i = 0; i < buffer.size(); ++i)
                buffer[i] = static_cast<Comp>(filter[i]);
            kernel = buffer.get();
        } else {
            kernel = filter;
        }

        #pragma omp parallel for num_threads(threads) default(none) collapse(3) \
        shared(input, input_stride, output, output_stride, filter_size, int_shape, HALO, kernel)

        for (int i = 0; i < int_shape[0]; ++i) {
            for (int j = 0; j < int_shape[1]; ++j) {
                for (int k = 0; k < int_shape[2]; ++k) {
                    for (int l = 0; l < int_shape[3]; ++l) {

                        Comp conv = 0;
                        if constexpr (DIM == 0) {
                            for (int wl = 0; wl < filter_size[2]; ++wl) {
                                int il = l - HALO[2] + wl;
                                if (il >= 0 && il < int_shape[3])
                                    conv += static_cast<Comp>(input[at(i, j, k, il, input_stride)]) *
                                            kernel[wl];
                            }
                        } else if constexpr (DIM == 1) {
                            for (int wk = 0; wk < filter_size[1]; ++wk) {
                                int ik = k - HALO[1] + wk;
                                if (ik < 0 || ik >= int_shape[2])
                                    continue;
                                int tmp = wk * filter_size[2];
                                for (int wl = 0; wl < filter_size[2]; ++wl) {
                                    int il = l - HALO[2] + wl;
                                    if (il >= 0 && il < int_shape[3])
                                        conv += static_cast<Comp>(input[at(i, j, ik, il, input_stride)]) *
                                                kernel[tmp + wl];
                                }
                            }
                        } else if constexpr (DIM == 2) {
                            for (int wj = 0; wj < filter_size[0]; ++wj) {
                                int ij = j - HALO[0] + wj;
                                if (ij < 0 || ij >= int_shape[1])
                                    continue;
                                int tmp_z = wj * filter_size[1] * filter_size[2];
                                for (int wk = 0; wk < filter_size[1]; ++wk) {
                                    int ik = k - HALO[1] + wk;
                                    if (ik < 0 || ik >= int_shape[2])
                                        continue;
                                    int tmp = tmp_z + wk * filter_size[2];
                                    for (int wl = 0; wl < filter_size[2]; ++wl) {
                                        int il = l - HALO[2] + wl;
                                        if (il >= 0 && il < int_shape[3])
                                            conv += static_cast<Comp>(input[at(i, ij, ik, il, input_stride)]) *
                                                    kernel[tmp + wl];
                                    }
                                }
                            }
                        }
                        output[at(i, j, k, l, output_stride)] = static_cast<T>(conv);
                    }
                }
            }
        }
    }

    template<typename T, typename U, int DIM>
    void convolveSep_(const T* __restrict input, size4_t input_stride, T* __restrict output, size4_t output_stride,
                      size4_t shape, const U* __restrict filter, int filter_size, size_t threads) {
        const int4_t int_shape(shape);
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

        #pragma omp parallel for num_threads(threads) default(none) collapse(3) \
        shared(input, input_stride, output, output_stride, filter_size, int_shape, HALO, kernel)

        for (int i = 0; i < int_shape[0]; ++i) {
            for (int j = 0; j < int_shape[1]; ++j) {
                for (int k = 0; k < int_shape[2]; ++k) {
                    for (int l = 0; l < int_shape[3]; ++l) {

                        Comp conv = 0;
                        if constexpr (DIM == 2) {
                            for (int wl = 0; wl < filter_size; ++wl) {
                                int il = l - HALO + wl;
                                if (il >= 0 && il < int_shape[3])
                                    conv += static_cast<Comp>(input[at(i, j, k, il, input_stride)]) * kernel[wl];
                            }
                        } else if constexpr (DIM == 1) {
                            for (int wk = 0; wk < filter_size; ++wk) {
                                int ik = k - HALO + wk;
                                if (ik >= 0 && ik < int_shape[2])
                                    conv += static_cast<Comp>(input[at(i, j, ik, l, input_stride)]) * kernel[wk];
                            }
                        } else if constexpr (DIM == 0) {
                            for (int wj = 0; wj < filter_size; ++wj) {
                                int ij = j - HALO + wj;
                                if (ij >= 0 && ij < int_shape[1])
                                    conv += static_cast<Comp>(input[at(i, ij, k, l, input_stride)]) * kernel[wj];
                            }
                        }
                        output[at(i, j, k, l, output_stride)] = static_cast<T>(conv);
                    }
                }
            }
        }
    }
}

namespace noa::cpu::filter {
    template<typename T, typename U>
    void convolve1(const T* input, size4_t input_stride, T* output, size4_t output_stride,
                   size4_t shape, const U* filter, size_t filter_size, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output);
        NOA_ASSERT(filter_size % 2);
        if (filter_size == 1)
            return math::ewise(input, input_stride, static_cast<T>(filter[0]), output, output_stride, shape,
                               noa::math::multiply_t{}, stream);

        stream.enqueue(convolve_<T, U, 0>,
                       input, input_stride, output, output_stride, shape,
                       filter, int3_t(1, 1, filter_size), stream.threads());
    }

    template<typename T, typename U>
    void convolve2(const T* input, size4_t input_stride, T* output, size4_t output_stride,
                   size4_t shape, const U* filter, size2_t filter_shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output);
        NOA_ASSERT(all((filter_shape % 2) == 1));
        if (all(filter_shape == 1))
            return math::ewise(input, input_stride, static_cast<T>(filter[0]), output, output_stride, shape,
                               noa::math::multiply_t{}, stream);

        stream.enqueue(convolve_<T, U, 1>,
                       input, input_stride, output, output_stride, shape,
                       filter, int3_t(1, filter_shape[0], filter_shape[1]), stream.threads());
    }

    template<typename T, typename U>
    void convolve3(const T* input, size4_t input_stride, T* output, size4_t output_stride,
                   size4_t shape, const U* filter, size3_t filter_shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output);
        NOA_ASSERT(all((filter_shape % 2) == 1));
        if (all(filter_shape == 1))
            return math::ewise(input, input_stride, static_cast<T>(filter[0]), output, output_stride, shape,
                               noa::math::multiply_t{}, stream);

        stream.enqueue(convolve_<T, U, 2>,
                       input, input_stride, output, output_stride, shape,
                       filter, int3_t(filter_shape), stream.threads());
    }

    template<typename T, typename U>
    void convolve(const T* input, size4_t input_stride, T* output, size4_t output_stride, size4_t shape,
                  const U* filter0, size_t filter0_size,
                  const U* filter1, size_t filter1_size,
                  const U* filter2, size_t filter2_size,
                  T* tmp, size4_t tmp_stride, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output);
        const int3_t fs(filter0_size, filter1_size, filter2_size);
        size_t threads = stream.threads();

        if (filter0 && filter1 && filter2) {
            NOA_ASSERT(filter0_size % 2);
            NOA_ASSERT(filter1_size % 2);
            NOA_ASSERT(filter2_size % 2);
            stream.enqueue([=]() {
                convolveSep_<T, U, 0>(input, input_stride, output, output_stride, shape, filter0, fs[0], threads);
                convolveSep_<T, U, 1>(output, output_stride, tmp, tmp_stride, shape, filter1, fs[1], threads);
                convolveSep_<T, U, 2>(tmp, tmp_stride, output, output_stride, shape, filter2, fs[2], threads);
            });
        } else if (filter0 && filter1) {
            NOA_ASSERT(filter0_size % 2);
            NOA_ASSERT(filter1_size % 2);
            stream.enqueue([=]() {
                convolveSep_<T, U, 0>(input, input_stride, tmp, tmp_stride, shape, filter0, fs[0], threads);
                convolveSep_<T, U, 1>(tmp, tmp_stride, output, output_stride, shape, filter1, fs[1], threads);
            });
        } else if (filter1 && filter2) {
            NOA_ASSERT(filter1_size % 2);
            NOA_ASSERT(filter2_size % 2);
            stream.enqueue([=]() {
                convolveSep_<T, U, 1>(input, input_stride, tmp, tmp_stride, shape, filter1, fs[1], threads);
                convolveSep_<T, U, 2>(tmp, tmp_stride, output, output_stride, shape, filter2, fs[2], threads);
            });
        } else if (filter0 && filter2) {
            NOA_ASSERT(filter0_size % 2);
            NOA_ASSERT(filter2_size % 2);
            stream.enqueue([=]() {
                convolveSep_<T, U, 0>(input, input_stride, tmp, tmp_stride, shape, filter0, fs[0], threads);
                convolveSep_<T, U, 2>(tmp, tmp_stride, output, output_stride, shape, filter2, fs[2], threads);
            });
        } else if (filter0) {
            NOA_ASSERT(filter0_size % 2);
            stream.enqueue(convolveSep_<T, U, 0>,
                           input, input_stride, output, output_stride, shape, filter0, fs[0], threads);
        } else if (filter1) {
            NOA_ASSERT(filter1_size % 2);
            stream.enqueue(convolveSep_<T, U, 1>,
                           input, input_stride, output, output_stride, shape, filter1, fs[1], threads);
        } else if (filter2) {
            NOA_ASSERT(filter2_size % 2);
            stream.enqueue(convolveSep_<T, U, 2>,
                           input, input_stride, output, output_stride, shape, filter2, fs[2], threads);
        }
    }

    #define NOA_INSTANTIATE_CONV_(T, U)                                                                 \
    template void convolve1<T>(const T*, size4_t, T*, size4_t, size4_t, const U*, size_t, Stream&);     \
    template void convolve2<T>(const T*, size4_t, T*, size4_t, size4_t, const U*, size2_t, Stream&);    \
    template void convolve3<T>(const T*, size4_t, T*, size4_t, size4_t, const U*, size3_t, Stream&);    \
    template void convolve<T>(const T*, size4_t, T*, size4_t, size4_t, const U*, size_t, const U*, size_t, const U*, size_t, T*, size4_t, Stream&)

    NOA_INSTANTIATE_CONV_(half_t, half_t);
    NOA_INSTANTIATE_CONV_(half_t, float);
    NOA_INSTANTIATE_CONV_(float, float);
    NOA_INSTANTIATE_CONV_(double, double);
}
