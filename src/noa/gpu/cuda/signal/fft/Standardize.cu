#include "noa/gpu/cuda/math/Ewise.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/Set.h"
#include "noa/gpu/cuda/signal/fft/Standardize.h"
#include "noa/gpu/cuda/util/ReduceUnary.cuh"

namespace {
    using namespace ::noa;
    using Norm = noa::fft::Norm;

    template<fft::Remap REMAP, typename T>
    void standardizeFull_(const shared_t<Complex<T>[]>& input, size4_t input_strides,
                          const shared_t<Complex<T>[]>& output, size4_t output_strides,
                          size4_t shape, Norm norm, cuda::Stream& stream) {
        NOA_ASSERT(shape.ndim() <= 3);
        const auto count = static_cast<T>(math::prod(shape));
        const auto scale = norm == fft::NORM_FORWARD ? 1 : norm == fft::NORM_ORTHO ? math::sqrt(count) : count;

        size4_t index_dc{};
        if constexpr (REMAP == fft::Remap::FC2FC) {
            index_dc = {0,
                        math::FFTShift(size_t{0}, shape[1]),
                        math::FFTShift(size_t{0}, shape[2]),
                        math::FFTShift(size_t{0}, shape[3])};
        }

        T factor;
        T* null{};
        cuda::util::reduce(
                "signal::fft::standardize",
                input.get(), uint4_t(input_strides), uint4_t(shape),
                math::abs_squared_t{}, math::plus_t{}, T{0},
                &factor, 1, math::copy_t{}, null, 1, math::copy_t{}, true, true, stream);
        Complex<T> dc;
        cuda::memory::copy(input.get() + indexing::at(index_dc, input_strides), &dc, 1, stream);

        stream.synchronize();
        factor = math::sqrt(factor - math::abs(dc) * math::abs(dc)) / scale;

        cuda::math::ewise(input, input_strides, 1 / factor, output, output_strides, shape, math::multiply_t{}, stream);
        cuda::memory::set(output.get() + indexing::at(index_dc, output_strides), 1, Complex<T>{0}, stream);
        stream.attach(output);
    }

    template<fft::Remap REMAP, typename T>
    void standardizeHalf_(const shared_t<Complex<T>[]>& input, size4_t input_strides,
                          const shared_t<Complex<T>[]>& output, size4_t output_strides,
                          size4_t shape, size4_t shape_fft, Norm norm, cuda::Stream& stream) {
        NOA_ASSERT(shape.ndim() <= 3);
        using namespace noa::indexing;
        const auto count = static_cast<T>(math::prod(shape));
        const auto scale = norm == fft::NORM_FORWARD ? 1 : norm == fft::NORM_ORTHO ? math::sqrt(count) : count;

        const Subregion original(shape_fft, input_strides);
        const bool even = !(shape[3] % 2);
        size4_t index_dc{};
        if constexpr (REMAP == fft::Remap::HC2HC)
            index_dc = {0, math::FFTShift(size_t{0}, shape[1]), math::FFTShift(size_t{0}, shape[2]), 0};

        // Reduce unique chunk:
        auto subregion = original(ellipsis_t{}, slice_t{1, original.shape()[3] - even});
        T factor0;
        T* null{};
        cuda::util::reduce(
                "signal::fft::standardize",
                input.get() + subregion.offset(), uint4_t(subregion.strides()), uint4_t(subregion.shape()),
                math::abs_squared_t{}, math::plus_t{}, T{0},
                &factor0, 1, math::copy_t{}, null, 1, math::copy_t{}, true, true, stream);

        // Reduce common column/plane containing the DC:
        subregion = original(ellipsis_t{}, 0);
        T factor1;
        cuda::util::reduce(
                "signal::fft::standardize",
                input.get() + subregion.offset(), uint4_t(subregion.strides()), uint4_t(subregion.shape()),
                math::abs_squared_t{}, math::plus_t{}, T{0},
                &factor1, 1, math::copy_t{}, null, 1, math::copy_t{}, true, true, stream);
        Complex<T> dc;
        cuda::memory::copy(input.get() + indexing::at(index_dc, input_strides), &dc, 1, stream);

        T factor2{0};
        if (even) {
            // Reduce common column/plane containing the real Nyquist:
            subregion = original(ellipsis_t{}, -1);
            cuda::util::reduce(
                    "signal::fft::standardize",
                    input.get() + subregion.offset(), uint4_t(subregion.strides()), uint4_t(subregion.shape()),
                    math::abs_squared_t{}, math::plus_t{}, T{0},
                    &factor2, 1, math::copy_t{}, null, 1, math::copy_t{}, true, true, stream);
        }

        stream.synchronize();
        factor1 -= math::abs(dc) * math::abs(dc);
        const T factor = math::sqrt(2 * factor0 + factor1 + factor2) / scale;

        cuda::math::ewise(input, input_strides, 1 / factor,
                          output, output_strides, shape_fft, math::multiply_t{}, stream);
        cuda::memory::set(output.get() + indexing::at(index_dc, output_strides), 1, Complex<T>{0}, stream);
        stream.attach(output);
    }
}

namespace noa::cuda::signal::fft {
    template<Remap REMAP, typename T, typename>
    void standardize(const shared_t<T[]>& input, size4_t input_strides,
                     const shared_t<T[]>& output, size4_t output_strides,
                     size4_t shape, Norm norm, Stream& stream) {
        using real_t = traits::value_type_t<T>;
        const size4_t shape_{1, shape[1], shape[2], shape[3]};
        const size4_t shape_fft = REMAP == Remap::F2F || REMAP == Remap::FC2FC ? shape_ : shape_.fft();

        // TODO The reduction can be batched, so maybe loop through batched only for the final ewise?
        for (size_t batch = 0; batch < shape[0]; ++batch) {
            if constexpr (REMAP == Remap::F2F || REMAP == Remap::FC2FC) {
                standardizeFull_<REMAP>(input, input_strides, output, output_strides,
                                        shape_, norm, stream);
            } else if constexpr (REMAP == Remap::H2H || REMAP == Remap::HC2HC) {
                standardizeHalf_<REMAP>(input, input_strides, output, output_strides,
                                        shape_, shape_fft, norm, stream);
            } else {
                static_assert(traits::always_false_v<T>);
            }
        }
    }

    #define INSTANTIATE_STANDARDIZE_(T)                                                                                                     \
    template void standardize<Remap::F2F, T, void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, Norm, Stream&);   \
    template void standardize<Remap::FC2FC, T, void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, Norm, Stream&); \
    template void standardize<Remap::H2H, T, void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, Norm, Stream&);   \
    template void standardize<Remap::HC2HC, T, void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, Norm, Stream&)

    INSTANTIATE_STANDARDIZE_(cfloat_t);
    INSTANTIATE_STANDARDIZE_(cdouble_t);
}
