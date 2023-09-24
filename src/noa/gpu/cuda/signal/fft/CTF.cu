#include "noa/algorithms/signal/CTF.hpp"
#include "noa/gpu/cuda/signal/fft/CTF.hpp"
#include "noa/gpu/cuda/utils/Iwise.cuh"

namespace noa::cuda::signal::fft {
    template<noa::fft::Remap REMAP, typename Input, typename Output, typename CTFIsotropic>
    void ctf_isotropic(
            const Input* input, Strides4<i64> input_strides,
            Output* output, Strides4<i64> output_strides, Shape4<i64> shape,
            const CTFIsotropic& ctf, bool ctf_abs, bool ctf_square, Stream& stream
    ) {
        constexpr bool IS_HALF = static_cast<u8>(REMAP) & noa::fft::Layout::SRC_HALF;
        switch (shape.ndim()) {
            case 1: {
                const i64 index = noa::indexing::non_empty_dhw_dimension(shape);
                const auto kernel = noa::algorithm::signal::fft::ctf<REMAP, 1, f32>(
                        input, input_strides.filter(0, index),
                        output, output_strides.filter(0, index),
                        shape.filter(index), ctf, ctf_abs, ctf_square);

                auto iwise_shape = shape.filter(0, index);
                if constexpr (IS_HALF)
                    iwise_shape = iwise_shape.rfft();
                noa::cuda::utils::iwise_2d(iwise_shape, kernel, stream);
                break;
            }
            case 2: {
                // Reorder HW dimensions to rightmost.
                const auto order = noa::indexing::order(output_strides.filter(2, 3), shape.filter(2, 3));
                if (noa::any(order != Vec2<i64>{0, 1})) {
                    std::swap(input_strides[2], input_strides[3]);
                    std::swap(output_strides[2], output_strides[3]);
                    std::swap(shape[2], shape[3]);
                }
                const auto kernel = noa::algorithm::signal::fft::ctf<REMAP, 2, f32>(
                        input, input_strides.filter(0, 2, 3),
                        output, output_strides.filter(0, 2, 3),
                        shape.filter(2, 3), ctf, ctf_abs, ctf_square);

                auto iwise_shape = shape.filter(0, 2, 3);
                if constexpr (IS_HALF)
                    iwise_shape = iwise_shape.rfft();
                noa::cuda::utils::iwise_3d(iwise_shape, kernel, stream);
                break;
            }
            case 3: {
                // Reorder BHW dimensions to rightmost.
                const auto order = noa::indexing::order(output_strides.pop_front(), shape.pop_front());
                if (noa::any(order != Vec3<i64>{0, 1, 2})) {
                    const auto order_3d = (order + 1).push_front(0);
                    input_strides = noa::indexing::reorder(input_strides, order_3d);
                    output_strides = noa::indexing::reorder(output_strides, order_3d);
                    shape = noa::indexing::reorder(shape, order_3d);
                }
                const auto kernel = noa::algorithm::signal::fft::ctf<REMAP, 3, f32>(
                        input, input_strides,
                        output, output_strides,
                        shape.pop_front(), ctf, ctf_abs, ctf_square);

                auto iwise_shape = shape;
                if constexpr (IS_HALF)
                    iwise_shape = iwise_shape.rfft();
                noa::cuda::utils::iwise_4d(iwise_shape, kernel, stream);
                break;
            }
        }
    }

    template<noa::fft::Remap REMAP, typename Output, typename CTFIsotropic>
    void ctf_isotropic(
            Output* output, Strides4<i64> output_strides, Shape4<i64> shape,
            const CTFIsotropic& ctf, bool ctf_abs, bool ctf_square,
            const Vec2<f32>& fftfreq_range, bool fftfreq_range_endpoint, Stream& stream
    ) {
        constexpr bool IS_HALF = static_cast<u8>(REMAP) & noa::fft::Layout::SRC_HALF;
        switch (shape.ndim()) {
            case 1: {
                const i64 index = noa::indexing::non_empty_dhw_dimension(shape);
                const auto kernel = noa::algorithm::signal::fft::ctf<REMAP, 1>(
                        output, output_strides.filter(0, index),
                        shape.filter(index), ctf, ctf_abs, ctf_square,
                        fftfreq_range, fftfreq_range_endpoint);

                auto iwise_shape = shape.filter(0, index);
                if constexpr (IS_HALF)
                    iwise_shape = iwise_shape.rfft();
                noa::cuda::utils::iwise_2d(iwise_shape, kernel, stream);
                break;
            }
            case 2: {
                // Reorder HW dimensions to rightmost.
                const auto order = noa::indexing::order(output_strides.filter(2, 3), shape.filter(2, 3));
                if (noa::any(order != Vec2<i64>{0, 1})) {
                    std::swap(output_strides[2], output_strides[3]);
                    std::swap(shape[2], shape[3]);
                }
                const auto kernel = noa::algorithm::signal::fft::ctf<REMAP, 2>(
                        output, output_strides.filter(0, 2, 3),
                        shape.filter(2, 3), ctf, ctf_abs, ctf_square,
                        fftfreq_range, fftfreq_range_endpoint);

                auto iwise_shape = shape.filter(0, 2, 3);
                if constexpr (IS_HALF)
                    iwise_shape = iwise_shape.rfft();
                noa::cuda::utils::iwise_3d(iwise_shape, kernel, stream);
                break;
            }
            case 3: {
                // Reorder BHW dimensions to rightmost.
                const auto order = noa::indexing::order(output_strides.pop_front(), shape.pop_front());
                if (noa::any(order != Vec3<i64>{0, 1, 2})) {
                    const auto order_3d = (order + 1).push_front(0);
                    output_strides = noa::indexing::reorder(output_strides, order_3d);
                    shape = noa::indexing::reorder(shape, order_3d);
                }
                const auto kernel = noa::algorithm::signal::fft::ctf<REMAP, 3>(
                        output, output_strides,
                        shape.pop_front(), ctf, ctf_abs, ctf_square,
                        fftfreq_range, fftfreq_range_endpoint);

                auto iwise_shape = shape;
                if constexpr (IS_HALF)
                    iwise_shape = iwise_shape.rfft();
                noa::cuda::utils::iwise_4d(iwise_shape, kernel, stream);
                break;
            }
        }
    }

    template<noa::fft::Remap REMAP, typename Input, typename Output, typename CTFAnisotropic>
    void ctf_anisotropic(
            const Input* input, const Strides4<i64>& input_strides,
            Output* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const CTFAnisotropic& ctf, bool ctf_abs, bool ctf_square, Stream& stream
    ) {
        NOA_ASSERT(shape.ndim() == 2);

        constexpr bool IS_HALF = static_cast<u8>(REMAP) & noa::fft::Layout::SRC_HALF;
        const auto kernel = noa::algorithm::signal::fft::ctf<REMAP, 2, f32>(
                input, input_strides.filter(0, 2, 3),
                output, output_strides.filter(0, 2, 3),
                shape.filter(2, 3), ctf, ctf_abs, ctf_square);

        auto iwise_shape = shape.filter(0, 2, 3);
        if constexpr (IS_HALF)
            iwise_shape = iwise_shape.rfft();
        noa::cuda::utils::iwise_3d(iwise_shape, kernel, stream);
    }

    template<noa::fft::Remap REMAP, typename Output, typename CTFAnisotropic>
    void ctf_anisotropic(
            Output* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const CTFAnisotropic& ctf, bool ctf_abs, bool ctf_square,
            const Vec2<f32>& fftfreq_range, bool fftfreq_range_endpoint, Stream& stream
    ) {
        NOA_ASSERT(shape.ndim() == 2);

        constexpr bool IS_HALF = static_cast<u8>(REMAP) & noa::fft::Layout::SRC_HALF;
        const auto kernel = noa::algorithm::signal::fft::ctf<REMAP, 2>(
                output, output_strides.filter(0, 2, 3),
                shape.filter(2, 3), ctf, ctf_abs, ctf_square,
                fftfreq_range, fftfreq_range_endpoint);

        auto iwise_shape = shape.filter(0, 2, 3);
        if constexpr (IS_HALF)
            iwise_shape = iwise_shape.rfft();
        noa::cuda::utils::iwise_3d(iwise_shape, kernel, stream);
    }

    #define NOA_INSTANTIATE_CTF_ISOTROPIC(Remap, Input, Output, CTF)    \
    template void ctf_isotropic<Remap, Input, Output, CTF>(             \
            const Input*, Strides4<i64>,                                \
            Output*, Strides4<i64>,                                     \
            Shape4<i64>, CTF const&, bool, bool, Stream&)

    #define NOA_INSTANTIATE_CTF_ANISOTROPIC(Remap, Input, Output, CTF)  \
    template void ctf_anisotropic<Remap, Input, Output, CTF>(           \
            const Input*, const Strides4<i64>&,                         \
            Output*, const Strides4<i64>&,                              \
            const Shape4<i64>&, CTF const&, bool, bool, Stream&)

    #define NOA_INSTANTIATE_CTF_ALL(Remap, Input, Output)                                               \
    NOA_INSTANTIATE_CTF_ISOTROPIC(Remap, Input, Output, noa::signal::fft::CTFIsotropic<f64>);           \
    NOA_INSTANTIATE_CTF_ISOTROPIC(Remap, Input, Output, const noa::signal::fft::CTFIsotropic<f64>*);    \
    NOA_INSTANTIATE_CTF_ANISOTROPIC(Remap, Input, Output, noa::signal::fft::CTFAnisotropic<f64>);       \
    NOA_INSTANTIATE_CTF_ANISOTROPIC(Remap, Input, Output, const noa::signal::fft::CTFAnisotropic<f64>*)

//    NOA_INSTANTIATE_CTF_ISOTROPIC(Remap, Input, Output, noa::signal::fft::CTFIsotropic<f32>);           \
//    NOA_INSTANTIATE_CTF_ISOTROPIC(Remap, Input, Output, const noa::signal::fft::CTFIsotropic<f32>*);    \
//    NOA_INSTANTIATE_CTF_ANISOTROPIC(Remap, Input, Output, noa::signal::fft::CTFAnisotropic<f32>);       \
//    NOA_INSTANTIATE_CTF_ANISOTROPIC(Remap, Input, Output, const noa::signal::fft::CTFAnisotropic<f32>*);\

    #define NOA_INSTANTIATE_CTF_ALL_REMAP(Input, Output)                \
    NOA_INSTANTIATE_CTF_ALL(noa::fft::Remap::H2H, Input, Output);
//    NOA_INSTANTIATE_CTF_ALL(noa::fft::Remap::HC2HC, Input, Output);     \
//    NOA_INSTANTIATE_CTF_ALL(noa::fft::Remap::HC2H, Input, Output);      \
//    NOA_INSTANTIATE_CTF_ALL(noa::fft::Remap::H2HC, Input, Output);      \
//    NOA_INSTANTIATE_CTF_ALL(noa::fft::Remap::F2F, Input, Output);       \
//    NOA_INSTANTIATE_CTF_ALL(noa::fft::Remap::FC2FC, Input, Output);     \
//    NOA_INSTANTIATE_CTF_ALL(noa::fft::Remap::FC2F, Input, Output);      \
//    NOA_INSTANTIATE_CTF_ALL(noa::fft::Remap::F2FC, Input, Output)

    NOA_INSTANTIATE_CTF_ALL_REMAP(f32, f32);
//    NOA_INSTANTIATE_CTF_ALL_REMAP(f64, f64);
    NOA_INSTANTIATE_CTF_ALL_REMAP(c32, c32);
//    NOA_INSTANTIATE_CTF_ALL_REMAP(c64, c64);
//    NOA_INSTANTIATE_CTF_ALL_REMAP(c32, f32);
//    NOA_INSTANTIATE_CTF_ALL_REMAP(c64, f64);

    #define NOA_INSTANTIATE_CTF_RANGE_ISOTROPIC(Remap, Output, CTF) \
    template void ctf_isotropic<Remap, Output, CTF>(                \
            Output*, Strides4<i64>,                                 \
            Shape4<i64>, CTF const&, bool, bool,                    \
            const Vec2<f32>&, bool, Stream&)

    #define NOA_INSTANTIATE_CTF_RANGE_ANISOTROPIC(Remap, Output, CTF)   \
    template void ctf_anisotropic<Remap, Output, CTF>(                  \
            Output*, const Strides4<i64>&,                              \
            const Shape4<i64>&, CTF const&, bool, bool,                 \
            const Vec2<f32>&, bool, Stream&)

    #define NOA_INSTANTIATE_CTF_REMAP_ALL(Remap, Output)                                               \
    NOA_INSTANTIATE_CTF_RANGE_ISOTROPIC(Remap, Output, noa::signal::fft::CTFIsotropic<f64>);           \
    NOA_INSTANTIATE_CTF_RANGE_ISOTROPIC(Remap, Output, const noa::signal::fft::CTFIsotropic<f64>*);    \
    NOA_INSTANTIATE_CTF_RANGE_ANISOTROPIC(Remap, Output, noa::signal::fft::CTFAnisotropic<f64>);       \
    NOA_INSTANTIATE_CTF_RANGE_ANISOTROPIC(Remap, Output, const noa::signal::fft::CTFAnisotropic<f64>*)

//    NOA_INSTANTIATE_CTF_RANGE_ISOTROPIC(Remap, Output, noa::signal::fft::CTFIsotropic<f32>);           \
//    NOA_INSTANTIATE_CTF_RANGE_ISOTROPIC(Remap, Output, const noa::signal::fft::CTFIsotropic<f32>*);    \
//    NOA_INSTANTIATE_CTF_RANGE_ANISOTROPIC(Remap, Output, noa::signal::fft::CTFAnisotropic<f32>);       \
//    NOA_INSTANTIATE_CTF_RANGE_ANISOTROPIC(Remap, Output, const noa::signal::fft::CTFAnisotropic<f32>*);\

    #define NOA_INSTANTIATE_CTF_RANGE_ALL_REMAP(Output)             \
    NOA_INSTANTIATE_CTF_REMAP_ALL(noa::fft::Remap::H2H, Output);
//    NOA_INSTANTIATE_CTF_REMAP_ALL(noa::fft::Remap::HC2HC, Output);  \
//    NOA_INSTANTIATE_CTF_REMAP_ALL(noa::fft::Remap::F2F, Output);    \
//    NOA_INSTANTIATE_CTF_REMAP_ALL(noa::fft::Remap::FC2FC, Output)

    NOA_INSTANTIATE_CTF_RANGE_ALL_REMAP(f32);
//    NOA_INSTANTIATE_CTF_RANGE_ALL_REMAP(f64);
    NOA_INSTANTIATE_CTF_RANGE_ALL_REMAP(c32);
//    NOA_INSTANTIATE_CTF_RANGE_ALL_REMAP(c64);
}
