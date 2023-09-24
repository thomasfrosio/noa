#include "noa/core/fft/Frequency.hpp"
#include "noa/core/utils/Indexing.hpp"
#include "noa/gpu/cuda/memory/Copy.hpp"
#include "noa/gpu/cuda/memory/Set.hpp"
#include "noa/gpu/cuda/signal/fft/Standardize.hpp"
#include "noa/gpu/cuda/utils/EwiseBinary.cuh"
#include "noa/gpu/cuda/utils/ReduceUnary.cuh"

namespace {
    using namespace ::noa;
    using Remap = noa::fft::Remap;
    using Norm = noa::fft::Norm;

    template<Remap REMAP, typename T>
    void standardize_full_(
            const Complex<T>* input, const Strides4<i64>& input_strides,
            Complex<T>* output, const Strides4<i64>& output_strides,
            const Shape4<i64>& shape, Norm norm, noa::cuda::Stream& stream
    ) {
        const auto count = static_cast<T>(shape.elements());
        const auto scale = norm == Norm::FORWARD ? 1 : norm == Norm::ORTHO ? noa::math::sqrt(count) : count;

        Vec4<i64> index_dc{0};
        if constexpr (REMAP == Remap::FC2FC) {
            index_dc = {0,
                        noa::fft::fftshift(i64{0}, shape[1]),
                        noa::fft::fftshift(i64{0}, shape[2]),
                        noa::fft::fftshift(i64{0}, shape[3])};
        }

        // Compute the energy of the input.
        T factor;
        noa::cuda::utils::reduce_unary(
                input, input_strides, shape, &factor, Strides1<i64>{1},
                T{0}, noa::abs_squared_t{}, noa::plus_t{}, {},
                true, true, stream);
        Complex<T> dc;
        noa::cuda::memory::copy(input + noa::indexing::at(index_dc, input_strides), &dc, 1, stream);
        stream.synchronize();
        factor = 1 / (noa::math::sqrt(factor - noa::abs_squared_t{}(dc)) / scale); // anticipate dc=0

        noa::cuda::utils::ewise_binary(
                input, input_strides, factor, output, output_strides,
                shape, stream, noa::multiply_t{});
        noa::cuda::memory::set(output + noa::indexing::at(index_dc, output_strides), 1, Complex<T>{0}, stream);
    }

    template<Remap REMAP, typename T>
    void standardize_half_(
            const Complex<T>* input, const Strides4<i64>& input_strides,
            Complex<T>* output, const Strides4<i64>& output_strides,
            const Shape4<i64>& shape, const Shape4<i64>& shape_fft,
            Norm norm, noa::cuda::Stream& stream
    ) {
        const auto count = static_cast<T>(shape.elements());
        const auto scale = norm == Norm::FORWARD ? 1 : norm == Norm::ORTHO ? math::sqrt(count) : count;

        const bool even = !(shape[3] % 2);
        Vec4<i64> index_dc{};
        if constexpr (REMAP == fft::Remap::HC2HC) {
            index_dc = {0,
                        noa::fft::fftshift(i64{0}, shape[1]),
                        noa::fft::fftshift(i64{0}, shape[2]),
                        0};
        }

        // Encode the original input layout:
        using namespace noa::indexing;
        const auto original = SubregionIndexer(shape_fft, input_strides);

        // Reduce unique chunk:
        T factor0;
        auto subregion = original.extract_subregion(Ellipsis{}, Slice{1, original.shape[3] - even});
        stream.synchronize();
        noa::cuda::utils::reduce_unary(
                input + subregion.offset, subregion.strides, subregion.shape,
                &factor0, Strides1<i64>{1}, T{0},
                noa::abs_squared_t{}, noa::plus_t{}, {},
                true, true, stream);

        // Reduce common column/plane containing the DC:
        subregion = original.extract_subregion(Ellipsis{}, 0);
        T factor1;
        noa::cuda::utils::reduce_unary(
                input + subregion.offset, subregion.strides, subregion.shape,
                &factor1, Strides1<i64>{1}, T{0},
                noa::abs_squared_t{}, noa::plus_t{}, {},
                true, true, stream);
        Complex<T> dc;
        noa::cuda::memory::copy(input + at(index_dc, input_strides), &dc, 1, stream);

        T factor2{0};
        if (even) {
            // Reduce common column/plane containing the real Nyquist:
            subregion = original.extract_subregion(Ellipsis{}, -1);
            noa::cuda::utils::reduce_unary(
                    input + subregion.offset, subregion.strides, subregion.shape,
                    &factor2, Strides1<i64>{1}, T{0},
                    noa::abs_squared_t{}, noa::plus_t{}, {},
                    true, true, stream);
        }

        // Standardize.
        stream.synchronize();
        factor1 -= noa::abs_squared_t{}(dc); // anticipate dc=0
        const T factor = scale / math::sqrt(2 * factor0 + factor1 + factor2);

        noa::cuda::utils::ewise_binary(
                input, input_strides, factor,
                output, output_strides, shape_fft,
                stream, noa::multiply_t{});
        noa::cuda::memory::set(output + at(index_dc, output_strides), 1, Complex<T>{0}, stream);
    }
}

namespace noa::cuda::signal::fft {
    template<noa::fft::Remap REMAP, typename T, typename>
    void standardize_ifft(
            const T* input, const Strides4<i64>& input_strides,
            T* output, const Strides4<i64>& output_strides,
            const Shape4<i64>& shape, noa::fft::Norm norm, Stream& stream
    ) {
        const auto shape_unbatched = shape.pop_front().push_front(1);
        const auto shape_unbatched_fft =
                REMAP == noa::fft::F2F || REMAP == noa::fft::FC2FC ?
                shape_unbatched : shape_unbatched.rfft();

        for (i64 batch = 0; batch < shape[0]; ++batch) {
            if constexpr (REMAP == noa::fft::F2F || REMAP == noa::fft::FC2FC) {
                standardize_full_<REMAP>(
                        input, input_strides, output, output_strides,
                        shape_unbatched, norm, stream);
            } else if constexpr (REMAP == noa::fft::H2H || REMAP == noa::fft::HC2HC) {
                standardize_half_<REMAP>(
                        input, input_strides, output, output_strides,
                        shape_unbatched, shape_unbatched_fft, norm, stream);
            } else {
                static_assert(nt::always_false_v<T>);
            }
        }
    }

    #define INSTANTIATE_STANDARDIZE_(T)                                                                                                                     \
    template void standardize_ifft<Remap::F2F, T>(const T*, const Strides4<i64>&, T*, const Strides4<i64>&, const Shape4<i64>&, noa::fft::Norm, Stream&);   \
    template void standardize_ifft<Remap::FC2FC, T>(const T*, const Strides4<i64>&, T*, const Strides4<i64>&, const Shape4<i64>&, noa::fft::Norm, Stream&); \
    template void standardize_ifft<Remap::H2H, T>(const T*, const Strides4<i64>&, T*, const Strides4<i64>&, const Shape4<i64>&, noa::fft::Norm, Stream&);   \
    template void standardize_ifft<Remap::HC2HC, T>(const T*, const Strides4<i64>&, T*, const Strides4<i64>&, const Shape4<i64>&, noa::fft::Norm, Stream&)

//    INSTANTIATE_STANDARDIZE_(c32);
//    INSTANTIATE_STANDARDIZE_(c64);
}
