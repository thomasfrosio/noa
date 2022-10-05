#include "noa/common/Assert.h"
#include "noa/cpu/math/Ewise.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/signal/Convolve.h"

namespace {
    using namespace ::noa;

    template<typename T, typename U, int DIM>
    void convolve_(AccessorRestrict<const T, 4, dim_t> input,
                   AccessorRestrict<T, 4, dim_t> output, dim4_t shape,
                   const U* __restrict filter, long3_t filter_size, dim_t threads) {
        const long4_t l_shape(shape);
        const long3_t HALO = filter_size / 2;

        // If half precision, convert filter and do the accumulation is single-precision.
        // Without this, half precision would be ~10times slower. With this preprocessing, it is only 2times slower.
        using Comp = std::conditional_t<std::is_same_v<half_t, T>, float, T>;
        cpu::memory::PtrHost<Comp> buffer;
        const Comp* __restrict kernel;
        if constexpr (std::is_same_v<half_t, T> && std::is_same_v<half_t, U>) {
            buffer = cpu::memory::PtrHost<Comp>(filter_size.elements());
            for (dim_t i = 0; i < buffer.size(); ++i)
                buffer[i] = static_cast<Comp>(filter[i]);
            kernel = buffer.get();
        } else {
            kernel = filter;
        }

        #pragma omp parallel for num_threads(threads) default(none) collapse(4) \
        shared(input, output, filter_size, l_shape, HALO, kernel)

        for (int64_t i = 0; i < l_shape[0]; ++i) {
            for (int64_t j = 0; j < l_shape[1]; ++j) {
                for (int64_t k = 0; k < l_shape[2]; ++k) {
                    for (int64_t l = 0; l < l_shape[3]; ++l) {

                        Comp conv = 0;
                        if constexpr (DIM == 1) {
                            for (int64_t wl = 0; wl < filter_size[2]; ++wl) {
                                const int64_t il = l - HALO[2] + wl;
                                if (il >= 0 && il < l_shape[3])
                                    conv += static_cast<Comp>(input(i, j, k, il)) * kernel[wl];
                            }
                        } else if constexpr (DIM == 2) {
                            for (int64_t wk = 0; wk < filter_size[1]; ++wk) {
                                const int64_t ik = k - HALO[1] + wk;
                                if (ik < 0 || ik >= l_shape[2])
                                    continue;
                                const int64_t tmp = wk * filter_size[2];
                                for (int64_t wl = 0; wl < filter_size[2]; ++wl) {
                                    const int64_t il = l - HALO[2] + wl;
                                    if (il >= 0 && il < l_shape[3])
                                        conv += static_cast<Comp>(input(i, j, ik, il)) * kernel[tmp + wl];
                                }
                            }
                        } else if constexpr (DIM == 3) {
                            for (int64_t wj = 0; wj < filter_size[0]; ++wj) {
                                const int64_t ij = j - HALO[0] + wj;
                                if (ij < 0 || ij >= l_shape[1])
                                    continue;
                                const int64_t tmp_z = wj * filter_size[1] * filter_size[2];
                                for (int64_t wk = 0; wk < filter_size[1]; ++wk) {
                                    const int64_t ik = k - HALO[1] + wk;
                                    if (ik < 0 || ik >= l_shape[2])
                                        continue;
                                    const int64_t tmp = tmp_z + wk * filter_size[2];
                                    for (int64_t wl = 0; wl < filter_size[2]; ++wl) {
                                        const int64_t il = l - HALO[2] + wl;
                                        if (il >= 0 && il < l_shape[3])
                                            conv += static_cast<Comp>(input(i, ij, ik, il)) * kernel[tmp + wl];
                                    }
                                }
                            }
                        }
                        output(i, j, k, l) = static_cast<T>(conv);
                    }
                }
            }
        }
    }

    template<typename T, typename U, int DIM>
    void convolveSep_(AccessorRestrict<const T, 4, dim_t> input,
                      AccessorRestrict<T, 4, dim_t> output, dim4_t shape,
                      const U* __restrict filter, int64_t filter_size, dim_t threads) {
        const long4_t l_shape(shape);
        const int64_t HALO = filter_size / 2;

        // If half precision, convert filter and do the accumulation is single-precision.
        // Without this, half precision would be ~10times slower. With this preprocessing, it is only 2times slower.
        using Comp = std::conditional_t<std::is_same_v<half_t, T>, float, T>;
        cpu::memory::PtrHost<Comp> buffer;
        const Comp* __restrict kernel;
        if constexpr (std::is_same_v<half_t, T> && std::is_same_v<half_t, U>) {
            buffer = cpu::memory::PtrHost<Comp>(filter_size);
            for (dim_t i = 0; i < buffer.size(); ++i)
                buffer[i] = static_cast<Comp>(filter[i]);
            kernel = buffer.get();
        } else {
            kernel = filter;
        }

        #pragma omp parallel for num_threads(threads) default(none) collapse(4) \
        shared(input, output, filter_size, l_shape, HALO, kernel)

        for (int64_t i = 0; i < l_shape[0]; ++i) {
            for (int64_t j = 0; j < l_shape[1]; ++j) {
                for (int64_t k = 0; k < l_shape[2]; ++k) {
                    for (int64_t l = 0; l < l_shape[3]; ++l) {

                        Comp conv = 0;
                        if constexpr (DIM == 2) {
                            for (int64_t wl = 0; wl < filter_size; ++wl) {
                                const int64_t il = l - HALO + wl;
                                if (il >= 0 && il < l_shape[3])
                                    conv += static_cast<Comp>(input(i, j, k, il)) * kernel[wl];
                            }
                        } else if constexpr (DIM == 1) {
                            for (int64_t wk = 0; wk < filter_size; ++wk) {
                                const int64_t ik = k - HALO + wk;
                                if (ik >= 0 && ik < l_shape[2])
                                    conv += static_cast<Comp>(input(i, j, ik, l)) * kernel[wk];
                            }
                        } else if constexpr (DIM == 0) {
                            for (int64_t wj = 0; wj < filter_size; ++wj) {
                                const int64_t ij = j - HALO + wj;
                                if (ij >= 0 && ij < l_shape[1])
                                    conv += static_cast<Comp>(input(i, ij, k, l)) * kernel[wj];
                            }
                        }
                        output(i, j, k, l) = static_cast<T>(conv);
                    }
                }
            }
        }
    }
}

namespace noa::cpu::signal {
    template<typename T, typename U, typename>
    void convolve1(const shared_t<T[]>& input, dim4_t input_strides,
                   const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                   const shared_t<U[]>& filter, dim_t filter_size, Stream& stream) {
        NOA_ASSERT(input != output && all(shape > 0));
        NOA_ASSERT(filter_size % 2);
        if (filter_size == 1)
            return math::ewise(input, input_strides, static_cast<T>(filter[0]),
                               output, output_strides, shape,
                               noa::math::multiply_t{}, stream);

        stream.enqueue([=](){
            convolve_<T, U, 1>({input.get(), input_strides},
                               {output.get(), output_strides}, shape,
                               filter.get(), long3_t{1, 1, filter_size}, stream.threads());
        });
    }

    template<typename T, typename U, typename>
    void convolve2(const shared_t<T[]>& input, dim4_t input_strides,
                   const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                   const shared_t<U[]>& filter, dim2_t filter_shape, Stream& stream) {
        NOA_ASSERT(input != output && all(shape > 0));
        NOA_ASSERT(all((filter_shape % 2) == 1));
        if (all(filter_shape == 1))
            return math::ewise(input, input_strides, static_cast<T>(filter[0]), output, output_strides, shape,
                               noa::math::multiply_t{}, stream);

        stream.enqueue([=](){
            convolve_<T, U, 2>({input.get(), input_strides},
                               {output.get(), output_strides}, shape,
                               filter.get(), long3_t{1, filter_shape[0], filter_shape[1]}, stream.threads());
        });
    }

    template<typename T, typename U, typename>
    void convolve3(const shared_t<T[]>& input, dim4_t input_strides,
                   const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                   const shared_t<U[]>& filter, dim3_t filter_shape, Stream& stream) {
        NOA_ASSERT(input != output && all(shape > 0));
        NOA_ASSERT(all((filter_shape % 2) == 1));
        if (all(filter_shape == 1))
            return math::ewise(input, input_strides, static_cast<T>(filter[0]), output, output_strides, shape,
                               noa::math::multiply_t{}, stream);

        stream.enqueue([=](){
            convolve_<T, U, 3>({input.get(), input_strides},
                               {output.get(), output_strides}, shape,
                               filter.get(), long3_t{filter_shape}, stream.threads());
        });
    }

    template<typename T, typename U, typename>
    void convolve(const shared_t<T[]>& input, dim4_t input_strides,
                  const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                  const shared_t<U[]>& filter0, dim_t filter0_size,
                  const shared_t<U[]>& filter1, dim_t filter1_size,
                  const shared_t<U[]>& filter2, dim_t filter2_size, Stream& stream,
                  const shared_t<T[]>& tmp, dim4_t tmp_strides) {
        NOA_ASSERT(input != output && all(shape > 0));
        const long3_t fs{filter0_size, filter1_size, filter2_size};
        const dim_t threads = stream.threads();

        int count = 0;
        if (filter0)
            count += 1;
        if (filter1)
            count += 1;
        if (filter2)
            count += 1;
        const bool allocate = !tmp && count > 1;
        const shared_t<T[]> buffer = allocate ? memory::PtrHost<T>::alloc(shape.elements()) : tmp;
        const dim4_t buffer_strides = allocate ? shape.strides() : tmp_strides;

        if (filter0 && filter1 && filter2) {
            NOA_ASSERT(filter0_size % 2);
            NOA_ASSERT(filter1_size % 2);
            NOA_ASSERT(filter2_size % 2);
            stream.enqueue([=]() {
                convolveSep_<T, U, 0>({input.get(), input_strides},
                                      {output.get(), output_strides}, shape,
                                      filter0.get(), fs[0], threads);
                convolveSep_<T, U, 1>({output.get(), output_strides},
                                      {buffer.get(), buffer_strides}, shape,
                                      filter1.get(), fs[1], threads);
                convolveSep_<T, U, 2>({buffer.get(), buffer_strides},
                                      {output.get(), output_strides}, shape,
                                      filter2.get(), fs[2], threads);
            });
        } else if (filter0 && filter1) {
            NOA_ASSERT(filter0_size % 2);
            NOA_ASSERT(filter1_size % 2);
            stream.enqueue([=]() {
                convolveSep_<T, U, 0>({input.get(), input_strides},
                                      {buffer.get(), buffer_strides}, shape,
                                      filter0.get(), fs[0], threads);
                convolveSep_<T, U, 1>({buffer.get(), buffer_strides},
                                      {output.get(), output_strides}, shape,
                                      filter1.get(), fs[1], threads);
            });
        } else if (filter1 && filter2) {
            NOA_ASSERT(filter1_size % 2);
            NOA_ASSERT(filter2_size % 2);
            stream.enqueue([=]() {
                convolveSep_<T, U, 1>({input.get(), input_strides},
                                      {buffer.get(), buffer_strides}, shape,
                                      filter1.get(), fs[1], threads);
                convolveSep_<T, U, 2>({buffer.get(), buffer_strides},
                                      {output.get(), output_strides}, shape,
                                      filter2.get(), fs[2], threads);
            });
        } else if (filter0 && filter2) {
            NOA_ASSERT(filter0_size % 2);
            NOA_ASSERT(filter2_size % 2);
            stream.enqueue([=]() {
                convolveSep_<T, U, 0>({input.get(), input_strides},
                                      {buffer.get(), buffer_strides}, shape,
                                      filter0.get(), fs[0], threads);
                convolveSep_<T, U, 2>({buffer.get(), buffer_strides},
                                      {output.get(), output_strides}, shape,
                                      filter2.get(), fs[2], threads);
            });
        } else if (filter0) {
            NOA_ASSERT(filter0_size % 2);
            stream.enqueue(convolveSep_<T, U, 0>,
                           AccessorRestrict<const T, 4, dim_t>(input.get(), input_strides),
                           AccessorRestrict<T, 4, dim_t>(output.get(), output_strides), shape,
                           filter0.get(), fs[0], threads);
        } else if (filter1) {
            NOA_ASSERT(filter1_size % 2);
            stream.enqueue(convolveSep_<T, U, 1>,
                           AccessorRestrict<const T, 4, dim_t>(input.get(), input_strides),
                           AccessorRestrict<T, 4, dim_t>(output.get(), output_strides), shape,
                           filter1.get(), fs[1], threads);
        } else if (filter2) {
            NOA_ASSERT(filter2_size % 2);
            stream.enqueue(convolveSep_<T, U, 2>,
                           AccessorRestrict<const T, 4, dim_t>(input.get(), input_strides),
                           AccessorRestrict<T, 4, dim_t>(output.get(), output_strides), shape,
                           filter2.get(), fs[2], threads);
        }
    }

    template<typename T, typename U, typename>
    void convolve(const shared_t<T[]>& input, dim4_t input_strides,
                  const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                  const shared_t<U[]>& filter, dim3_t filter_shape, Stream& stream)  {
        NOA_ASSERT(all(filter_shape >= 1) && all(shape > 0));
        const dim_t ndim = filter_shape.ndim();

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

    #define NOA_INSTANTIATE_CONV_(T, U)                                                                                                             \
    template void convolve1<T, U, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<U[]>&, dim_t, Stream&);  \
    template void convolve2<T, U, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<U[]>&, dim2_t, Stream&); \
    template void convolve3<T, U, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<U[]>&, dim3_t, Stream&); \
    template void convolve<T, U, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t,                                          \
                                       const shared_t<U[]>&, dim_t,                                                                                 \
                                       const shared_t<U[]>&, dim_t,                                                                                 \
                                       const shared_t<U[]>&, dim_t, Stream&,                                                                        \
                                       const shared_t<T[]>&, dim4_t);                                                                               \
    template void convolve<T, U, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t,                                          \
                                       const shared_t<U[]>&, dim3_t, Stream&)

    NOA_INSTANTIATE_CONV_(half_t, half_t);
    NOA_INSTANTIATE_CONV_(half_t, float);
    NOA_INSTANTIATE_CONV_(float, float);
    NOA_INSTANTIATE_CONV_(double, double);
}
