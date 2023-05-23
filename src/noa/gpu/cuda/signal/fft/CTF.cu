#include "noa/algorithms/signal/CTF.hpp"
#include "noa/gpu/cuda/signal/fft/CTF.hpp"
#include "noa/gpu/cuda/utils/Iwise.cuh"

namespace noa::cuda::signal::fft {
    template<noa::fft::Remap REMAP, typename Input, typename Output, typename CTFIsotropic, typename>
    void ctf_isotropic(
            const Input* input, const Strides4<i64>& input_strides,
            Output* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const CTFIsotropic& ctf, bool ctf_square, bool ctf_abs, Stream& stream) {
        constexpr bool IS_HALF = static_cast<u8>(REMAP) & noa::fft::Layout::SRC_HALF;
        switch (shape.ndim()) {
            case 1: {
                // The unified API should have reshaped to row vectors.
                const auto kernel = noa::algorithm::signal::fft::ctf<REMAP, 1, f32>(
                        input, input_strides.filter(0, 3),
                        output, output_strides.filter(0, 3),
                        shape.filter(3), ctf, ctf_square, ctf_abs);

                auto iwise_shape = shape.filter(0, 3);
                if constexpr (IS_HALF)
                    iwise_shape = iwise_shape.rfft();
                noa::cuda::utils::iwise_2d("ctf_isotropic", iwise_shape, kernel, stream);
                break;
            }
            case 2: {
                const auto kernel = noa::algorithm::signal::fft::ctf<REMAP, 2, f32>(
                        input, input_strides.filter(0, 2, 3),
                        output, output_strides.filter(0, 2, 3),
                        shape.filter(2, 3), ctf, ctf_square, ctf_abs);

                auto iwise_shape = shape.filter(0, 2, 3);
                if constexpr (IS_HALF)
                    iwise_shape = iwise_shape.rfft();
                noa::cuda::utils::iwise_3d("ctf_isotropic", iwise_shape, kernel, stream);
                break;
            }
            case 3: {
                const auto kernel = noa::algorithm::signal::fft::ctf<REMAP, 3, f32>(
                        input, input_strides,
                        output, output_strides,
                        shape.pop_front(), ctf, ctf_square, ctf_abs);

                auto iwise_shape = shape;
                if constexpr (IS_HALF)
                    iwise_shape = iwise_shape.rfft();
                noa::cuda::utils::iwise_4d("ctf_isotropic", iwise_shape, kernel, stream);
                break;
            }
        }
    }

    template<noa::fft::Remap REMAP, typename Input, typename Output, typename CTFAnisotropic, typename>
    void ctf_anisotropic(
            const Input* input, const Strides4<i64>& input_strides,
            Output* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const CTFAnisotropic& ctf, bool ctf_square, bool ctf_abs, Stream& stream) {

        constexpr bool IS_HALF = static_cast<u8>(REMAP) & noa::fft::Layout::SRC_HALF;
        const auto kernel = noa::algorithm::signal::fft::ctf<REMAP, 2, f32>(
                input, input_strides.filter(0, 2, 3),
                output, output_strides.filter(0, 2, 3),
                shape.filter(2, 3), ctf, ctf_square, ctf_abs);

        auto iwise_shape = shape.filter(0, 2, 3);
        if constexpr (IS_HALF)
            iwise_shape = iwise_shape.rfft();
        noa::cuda::utils::iwise_3d("ctf_anisotropic", iwise_shape, kernel, stream);
    }

    #define NOA_INSTANTIATE_CTF_ISOTROPIC(Remap, Input, Output, CTF)    \
    template void ctf_isotropic<Remap, Input, Output, CTF, void>(       \
            const Input*, const Strides4<i64>&,                         \
            Output*, const Strides4<i64>&,                              \
            const Shape4<i64>&, CTF const&, bool, bool, Stream&)

    #define NOA_INSTANTIATE_CTF_ANISOTROPIC(Remap, Input, Output, CTF)  \
    template void ctf_anisotropic<Remap, Input, Output, CTF, void>(     \
            const Input*, const Strides4<i64>&,                         \
            Output*, const Strides4<i64>&,                              \
            const Shape4<i64>&, CTF const&, bool, bool, Stream&)

    #define NOA_INSTANTIATE_CTF_ALL(Remap, Input, Output)                                               \
    NOA_INSTANTIATE_CTF_ISOTROPIC(Remap, Input, Output, noa::signal::fft::CTFIsotropic<f32>);           \
    NOA_INSTANTIATE_CTF_ISOTROPIC(Remap, Input, Output, noa::signal::fft::CTFIsotropic<f64>);           \
    NOA_INSTANTIATE_CTF_ISOTROPIC(Remap, Input, Output, const noa::signal::fft::CTFIsotropic<f32>*);    \
    NOA_INSTANTIATE_CTF_ISOTROPIC(Remap, Input, Output, const noa::signal::fft::CTFIsotropic<f64>*);    \
    NOA_INSTANTIATE_CTF_ANISOTROPIC(Remap, Input, Output, noa::signal::fft::CTFAnisotropic<f32>);       \
    NOA_INSTANTIATE_CTF_ANISOTROPIC(Remap, Input, Output, noa::signal::fft::CTFAnisotropic<f64>);       \
    NOA_INSTANTIATE_CTF_ANISOTROPIC(Remap, Input, Output, const noa::signal::fft::CTFAnisotropic<f32>*);\
    NOA_INSTANTIATE_CTF_ANISOTROPIC(Remap, Input, Output, const noa::signal::fft::CTFAnisotropic<f64>*)

    #define NOA_INSTANTIATE_CTF_ALL_REMAP(Input, Output)                \
    NOA_INSTANTIATE_CTF_ALL(noa::fft::Remap::H2H, Input, Output);       \
    NOA_INSTANTIATE_CTF_ALL(noa::fft::Remap::HC2HC, Input, Output);     \
    NOA_INSTANTIATE_CTF_ALL(noa::fft::Remap::HC2H, Input, Output);      \
    NOA_INSTANTIATE_CTF_ALL(noa::fft::Remap::H2HC, Input, Output);      \
    NOA_INSTANTIATE_CTF_ALL(noa::fft::Remap::F2F, Input, Output);       \
    NOA_INSTANTIATE_CTF_ALL(noa::fft::Remap::FC2FC, Input, Output);     \
    NOA_INSTANTIATE_CTF_ALL(noa::fft::Remap::FC2F, Input, Output);      \
    NOA_INSTANTIATE_CTF_ALL(noa::fft::Remap::F2FC, Input, Output)

    NOA_INSTANTIATE_CTF_ALL_REMAP(f32, f32);
    NOA_INSTANTIATE_CTF_ALL_REMAP(f64, f64);
    NOA_INSTANTIATE_CTF_ALL_REMAP(c32, c32);
    NOA_INSTANTIATE_CTF_ALL_REMAP(c64, c64);
    NOA_INSTANTIATE_CTF_ALL_REMAP(c32, f32);
    NOA_INSTANTIATE_CTF_ALL_REMAP(c64, f64);
}
