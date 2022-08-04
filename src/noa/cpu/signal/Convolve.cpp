#include "noa/common/Assert.h"
#include "noa/cpu/math/Ewise.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/signal/Convolve.h"

namespace {
    using namespace ::noa;

    template<typename T, typename U, int DIM>
    void convolve_(const T* __restrict input, size4_t input_strides,
                   T* __restrict output, size4_t output_strides, size4_t shape,
                   const U* __restrict filter, int3_t filter_size, size_t threads) {
        const int4_t int_shape(shape);
        const int3_t HALO = filter_size / 2;

        // If half precision, convert filter and do the accumulation is single-precision.
        // Without this, half precision would be ~10times slower. With this preprocessing, it is only 2times slower.
        using Comp = std::conditional_t<std::is_same_v<half_t, T>, float, T>;
        cpu::memory::PtrHost<Comp> buffer;
        const Comp* kernel;
        if constexpr (std::is_same_v<half_t, T> && std::is_same_v<half_t, U>) {
            buffer = cpu::memory::PtrHost<Comp>{static_cast<size_t>(filter_size.elements())};
            for (size_t i = 0; i < buffer.size(); ++i)
                buffer[i] = static_cast<Comp>(filter[i]);
            kernel = buffer.get();
        } else {
            kernel = filter;
        }

        #pragma omp parallel for num_threads(threads) default(none) collapse(4) \
        shared(input, input_strides, output, output_strides, filter_size, int_shape, HALO, kernel)

        for (int i = 0; i < int_shape[0]; ++i) {
            for (int j = 0; j < int_shape[1]; ++j) {
                for (int k = 0; k < int_shape[2]; ++k) {
                    for (int l = 0; l < int_shape[3]; ++l) {
                        using namespace ::noa::indexing;

                        Comp conv = 0;
                        if constexpr (DIM == 1) {
                            for (int wl = 0; wl < filter_size[2]; ++wl) {
                                int il = l - HALO[2] + wl;
                                if (il >= 0 && il < int_shape[3])
                                    conv += static_cast<Comp>(input[at(i, j, k, il, input_strides)]) *
                                            kernel[wl];
                            }
                        } else if constexpr (DIM == 2) {
                            for (int wk = 0; wk < filter_size[1]; ++wk) {
                                int ik = k - HALO[1] + wk;
                                if (ik < 0 || ik >= int_shape[2])
                                    continue;
                                int tmp = wk * filter_size[2];
                                for (int wl = 0; wl < filter_size[2]; ++wl) {
                                    int il = l - HALO[2] + wl;
                                    if (il >= 0 && il < int_shape[3])
                                        conv += static_cast<Comp>(input[at(i, j, ik, il, input_strides)]) *
                                                kernel[tmp + wl];
                                }
                            }
                        } else if constexpr (DIM == 3) {
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
                                            conv += static_cast<Comp>(input[at(i, ij, ik, il, input_strides)]) *
                                                    kernel[tmp + wl];
                                    }
                                }
                            }
                        }
                        output[at(i, j, k, l, output_strides)] = static_cast<T>(conv);
                    }
                }
            }
        }
    }

    template<typename T, typename U, int DIM>
    void convolveSep_(const T* __restrict input, size4_t input_strides,
                      T* __restrict output, size4_t output_strides, size4_t shape,
                      const U* __restrict filter, int filter_size, size_t threads) {
        const int4_t int_shape(shape);
        const int HALO = filter_size / 2;

        // If half precision, convert filter and do the accumulation is single-precision.
        // Without this, half precision would be ~10times slower. With this preprocessing, it is only 2times slower.
        using Comp = std::conditional_t<std::is_same_v<half_t, T>, float, T>;
        cpu::memory::PtrHost<Comp> buffer;
        const Comp* kernel;
        if constexpr (std::is_same_v<half_t, T> && std::is_same_v<half_t, U>) {
            buffer = cpu::memory::PtrHost<Comp>{static_cast<size_t>(filter_size)};
            for (size_t i = 0; i < buffer.size(); ++i)
                buffer[i] = static_cast<Comp>(filter[i]);
            kernel = buffer.get();
        } else {
            kernel = filter;
        }

        #pragma omp parallel for num_threads(threads) default(none) collapse(4) \
        shared(input, input_strides, output, output_strides, filter_size, int_shape, HALO, kernel)

        for (int i = 0; i < int_shape[0]; ++i) {
            for (int j = 0; j < int_shape[1]; ++j) {
                for (int k = 0; k < int_shape[2]; ++k) {
                    for (int l = 0; l < int_shape[3]; ++l) {
                        using namespace ::noa::indexing;

                        Comp conv = 0;
                        if constexpr (DIM == 2) {
                            for (int wl = 0; wl < filter_size; ++wl) {
                                int il = l - HALO + wl;
                                if (il >= 0 && il < int_shape[3])
                                    conv += static_cast<Comp>(input[at(i, j, k, il, input_strides)]) * kernel[wl];
                            }
                        } else if constexpr (DIM == 1) {
                            for (int wk = 0; wk < filter_size; ++wk) {
                                int ik = k - HALO + wk;
                                if (ik >= 0 && ik < int_shape[2])
                                    conv += static_cast<Comp>(input[at(i, j, ik, l, input_strides)]) * kernel[wk];
                            }
                        } else if constexpr (DIM == 0) {
                            for (int wj = 0; wj < filter_size; ++wj) {
                                int ij = j - HALO + wj;
                                if (ij >= 0 && ij < int_shape[1])
                                    conv += static_cast<Comp>(input[at(i, ij, k, l, input_strides)]) * kernel[wj];
                            }
                        }
                        output[at(i, j, k, l, output_strides)] = static_cast<T>(conv);
                    }
                }
            }
        }
    }
}

namespace noa::cpu::signal {
    template<typename T, typename U, typename>
    void convolve1(const shared_t<T[]>& input, size4_t input_strides,
                   const shared_t<T[]>& output, size4_t output_strides, size4_t shape,
                   const shared_t<U[]>& filter, size_t filter_size, Stream& stream) {
        NOA_ASSERT(input != output);
        NOA_ASSERT(filter_size % 2);
        if (filter_size == 1)
            return math::ewise(input, input_strides, static_cast<T>(filter[0]),
                               output, output_strides, shape,
                               noa::math::multiply_t{}, stream);

        stream.enqueue([=](){
            convolve_<T, U, 1>(input.get(), input_strides,
                               output.get(), output_strides, shape,
                               filter.get(), int3_t{1, 1, filter_size}, stream.threads());
        });
    }

    template<typename T, typename U, typename>
    void convolve2(const shared_t<T[]>& input, size4_t input_strides,
                   const shared_t<T[]>& output, size4_t output_strides, size4_t shape,
                   const shared_t<U[]>& filter, size2_t filter_shape, Stream& stream) {
        NOA_ASSERT(input != output);
        NOA_ASSERT(all((filter_shape % 2) == 1));
        if (all(filter_shape == 1))
            return math::ewise(input, input_strides, static_cast<T>(filter[0]), output, output_strides, shape,
                               noa::math::multiply_t{}, stream);

        stream.enqueue([=](){
            convolve_<T, U, 2>(input.get(), input_strides,
                               output.get(), output_strides, shape,
                               filter.get(), int3_t{1, filter_shape[0], filter_shape[1]}, stream.threads());
        });
    }

    template<typename T, typename U, typename>
    void convolve3(const shared_t<T[]>& input, size4_t input_strides,
                   const shared_t<T[]>& output, size4_t output_strides, size4_t shape,
                   const shared_t<U[]>& filter, size3_t filter_shape, Stream& stream) {
        NOA_ASSERT(input != output);
        NOA_ASSERT(all((filter_shape % 2) == 1));
        if (all(filter_shape == 1))
            return math::ewise(input, input_strides, static_cast<T>(filter[0]), output, output_strides, shape,
                               noa::math::multiply_t{}, stream);

        stream.enqueue([=](){
            convolve_<T, U, 3>(input.get(), input_strides,
                               output.get(), output_strides, shape,
                               filter.get(), int3_t{filter_shape}, stream.threads());
        });
    }

    template<typename T, typename U, typename>
    void convolve(const shared_t<T[]>& input, size4_t input_strides,
                  const shared_t<T[]>& output, size4_t output_strides, size4_t shape,
                  const shared_t<U[]>& filter0, size_t filter0_size,
                  const shared_t<U[]>& filter1, size_t filter1_size,
                  const shared_t<U[]>& filter2, size_t filter2_size, Stream& stream,
                  const shared_t<T[]>& tmp, size4_t tmp_strides) {
        NOA_ASSERT(input != output);
        const int3_t fs{filter0_size, filter1_size, filter2_size};
        const size_t threads = stream.threads();

        int count = 0;
        if (filter0)
            count += 1;
        if (filter1)
            count += 1;
        if (filter2)
            count += 1;
        const bool allocate = !tmp && count > 1;
        const shared_t<T[]> buffer = allocate ? memory::PtrHost<T>::alloc(shape.elements()) : tmp;
        const size4_t buffer_strides = allocate ? shape.strides() : tmp_strides;

        if (filter0 && filter1 && filter2) {
            NOA_ASSERT(filter0_size % 2);
            NOA_ASSERT(filter1_size % 2);
            NOA_ASSERT(filter2_size % 2);
            stream.enqueue([=]() {
                convolveSep_<T, U, 0>(input.get(), input_strides,
                                      output.get(), output_strides, shape,
                                      filter0.get(), fs[0], threads);
                convolveSep_<T, U, 1>(output.get(), output_strides,
                                      buffer.get(), buffer_strides, shape,
                                      filter1.get(), fs[1], threads);
                convolveSep_<T, U, 2>(buffer.get(), buffer_strides,
                                      output.get(), output_strides, shape,
                                      filter2.get(), fs[2], threads);
            });
        } else if (filter0 && filter1) {
            NOA_ASSERT(filter0_size % 2);
            NOA_ASSERT(filter1_size % 2);
            stream.enqueue([=]() {
                convolveSep_<T, U, 0>(input.get(), input_strides,
                                      buffer.get(), buffer_strides, shape,
                                      filter0.get(), fs[0], threads);
                convolveSep_<T, U, 1>(buffer.get(), buffer_strides,
                                      output.get(), output_strides, shape,
                                      filter1.get(), fs[1], threads);
            });
        } else if (filter1 && filter2) {
            NOA_ASSERT(filter1_size % 2);
            NOA_ASSERT(filter2_size % 2);
            stream.enqueue([=]() {
                convolveSep_<T, U, 1>(input.get(), input_strides,
                                      buffer.get(), buffer_strides, shape,
                                      filter1.get(), fs[1], threads);
                convolveSep_<T, U, 2>(buffer.get(), buffer_strides,
                                      output.get(), output_strides, shape,
                                      filter2.get(), fs[2], threads);
            });
        } else if (filter0 && filter2) {
            NOA_ASSERT(filter0_size % 2);
            NOA_ASSERT(filter2_size % 2);
            stream.enqueue([=]() {
                convolveSep_<T, U, 0>(input.get(), input_strides,
                                      buffer.get(), buffer_strides, shape,
                                      filter0.get(), fs[0], threads);
                convolveSep_<T, U, 2>(buffer.get(), buffer_strides,
                                      output.get(), output_strides, shape,
                                      filter2.get(), fs[2], threads);
            });
        } else if (filter0) {
            NOA_ASSERT(filter0_size % 2);
            stream.enqueue(convolveSep_<T, U, 0>,
                           input.get(), input_strides,
                           output.get(), output_strides, shape,
                           filter0.get(), fs[0], threads);
        } else if (filter1) {
            NOA_ASSERT(filter1_size % 2);
            stream.enqueue(convolveSep_<T, U, 1>,
                           input.get(), input_strides,
                           output.get(), output_strides, shape,
                           filter1.get(), fs[1], threads);
        } else if (filter2) {
            NOA_ASSERT(filter2_size % 2);
            stream.enqueue(convolveSep_<T, U, 2>,
                           input.get(), input_strides,
                           output.get(), output_strides, shape,
                           filter2.get(), fs[2], threads);
        }
    }

    template<typename T, typename U, typename>
    void convolve(const shared_t<T[]>& input, size4_t input_strides,
                  const shared_t<T[]>& output, size4_t output_strides, size4_t shape,
                  const shared_t<U[]>& filter, size3_t filter_shape, Stream& stream)  {
        NOA_ASSERT(all(filter_shape >= 1));
        const size_t ndim = filter_shape.ndim();

        // If there's a single dimension, use separable convolution kernels:
        if (ndim == 1 || (ndim == 3 && filter_shape[1] == 1 && filter_shape[2])) {
            if (all(filter_shape == 1)) {
                math::ewise(input, input_strides, static_cast<T>(filter[0]),
                            output, output_strides, shape,
                            noa::math::multiply_t{}, stream);
            } else if (filter_shape[2] > 1) {
                convolve<T, U>(input, input_strides, output, output_strides, shape,
                               nullptr, 0, nullptr, 0, filter, filter_shape[2], stream);
            } else if (filter_shape[1] > 1) {
                convolve<T, U>(input, input_strides, output, output_strides, shape,
                               nullptr, 0, filter, filter_shape[1], nullptr, 0, stream);
            } else {
                convolve<T, U>(input, input_strides, output, output_strides, shape,
                               filter, filter_shape[0], nullptr, 0, nullptr, 0, stream);
            }
            return;
        } else if (ndim == 2) {
            return convolve2(input, input_strides, output, output_strides,
                             shape, filter, {filter_shape[1], filter_shape[2]}, stream);
        } else {
            return convolve3(input, input_strides, output, output_strides,
                             shape, filter, filter_shape, stream);
        }
    }

    #define NOA_INSTANTIATE_CONV_(T, U)                                                                                                                 \
    template void convolve1<T, U, void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, const shared_t<U[]>&, size_t, Stream&);  \
    template void convolve2<T, U, void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, const shared_t<U[]>&, size2_t, Stream&); \
    template void convolve3<T, U, void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, const shared_t<U[]>&, size3_t, Stream&); \
    template void convolve<T, U, void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t,                                           \
                                       const shared_t<U[]>&, size_t,                                                                                    \
                                       const shared_t<U[]>&, size_t,                                                                                    \
                                       const shared_t<U[]>&, size_t, Stream&,                                                                           \
                                       const shared_t<T[]>&, size4_t);                                                                                  \
    template void convolve<T, U, void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t,                                           \
                                       const shared_t<U[]>&, size3_t, Stream&)

    NOA_INSTANTIATE_CONV_(half_t, half_t);
    NOA_INSTANTIATE_CONV_(half_t, float);
    NOA_INSTANTIATE_CONV_(float, float);
    NOA_INSTANTIATE_CONV_(double, double);
}
