#pragma once

#include "noa/cpu/signal/fft/FSC.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/signal/fft/FSC.hpp"
#endif

#include "noa/unified/Array.hpp"
#include "noa/unified/signal/fft/FSC.hpp"

namespace noa::signal::fft::details {
    template<typename Lhs, typename Rhs, typename Output, typename Cones = Empty>
    void check_fsc_parameters(const Lhs& lhs, const Rhs& rhs, const Output& fsc, const Shape4<i64>& shape,
                              const Cones& cone_directions = {}) {
        NOA_CHECK(!lhs.is_empty() && !rhs.is_empty() && !fsc.is_empty(), "Empty array detected");
        NOA_CHECK(lhs.get() != rhs.get(), "Computing the FSC on the same array is not allowed");

        NOA_CHECK(noa::all(rhs.shape() == shape.rfft()),
                  "Given the logical shape {}, the expected non-redundant shape should be {}, but got {}",
                  shape, shape.rfft(), rhs.shape());
        NOA_CHECK(noa::all(lhs.shape() == rhs.shape()),
                  "The two input arrays should have the same shape. Got lhs:{} and rhs:{}",
                  lhs.shape(), rhs.shape());

        const Device device = fsc.device();
        NOA_CHECK(device == lhs.device() && device == rhs.device(),
                  "The input and output arrays must be on the same device, but got lhs:{}, rhs:{}, fsc:{}",
                  lhs.device(), rhs.device(), device);

        i64 cones{1};
        if constexpr (!std::is_empty_v<Cones>) {
            NOA_CHECK(!cone_directions.is_empty(), "Empty array detected");
            NOA_CHECK(device == cone_directions.device(),
                      "The input and output arrays must be on the same device, "
                      "but got fsc:{}, cones:{}",
                      device, cone_directions.device());
            NOA_CHECK(noa::indexing::is_contiguous_vector(cone_directions.shape()),
                      "The cone directions should be specified as a contiguous vector, but got shape:{}, strides:{}",
                      cone_directions.shape(), cone_directions.strides());
            cones = cone_directions.elements();
        }

        const auto shell_count = noa::math::min(lhs.shape()) / 2 + 1;
        const auto expected_shape = Shape4<i64>{shape[0], 1, cones, shell_count};
        NOA_CHECK(noa::all(fsc.shape() == expected_shape) && fsc.is_contiguous(),
                  "The FSC does not have the correct shape. Given the input shape {}, and the number of cones ({}),"
                  "the expected shape is {}, but got {}",
                  shape, cones, expected_shape, fsc.shape());
    }
}

namespace noa::signal::fft {
    /// Computes the isotropic Fourier Shell Correlation between \p lhs and \p rhs.
    /// \tparam REMAP   Whether the input non-redundant FFTs are centered. Should be H2H or HC2HC.
    /// \param[in] lhs  Left-hand side.
    /// \param[in] rhs  Right-hand side. Should have the same shape as \p lhs.
    /// \param[out] fsc The output FSC. Should be a (batched) vector of size min(shape) // 2 + 1.
    /// \param shape    Logical shape of \p lhs and \p rhs.
    ///                 It should be a cubic or rectangular (batched) volume.
    template<noa::fft::Remap REMAP, typename Lhs, typename Rhs, typename Output, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_almost_any_v<Lhs, c32, c64> &&
             noa::traits::is_array_or_view_of_almost_any_v<Rhs, c32, c64> &&
             noa::traits::is_array_or_view_of_any_v<Output, f32, f64> &&
             noa::traits::are_almost_same_value_type_v<Lhs, Rhs> &&
             noa::traits::are_almost_same_value_type_v<noa::traits::value_type_t<Lhs>, Output> &&
             (REMAP == noa::fft::Remap::H2H || REMAP == noa::fft::Remap::HC2HC)>>
    void isotropic_fsc(const Lhs& lhs, const Rhs& rhs, const Output& fsc, const Shape4<i64>& shape) {
        details::check_fsc_parameters(lhs, rhs, fsc, shape);

        const Device device = fsc.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::signal::fft::isotropic_fsc<REMAP>(
                        lhs.get(), lhs.strides(),
                        rhs.get(), rhs.strides(),
                        fsc.get(), shape,
                        threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::signal::fft::isotropic_fsc<REMAP>(
                    lhs.get(), lhs.strides(),
                    rhs.get(), rhs.strides(),
                    fsc.get(), shape,
                    cuda_stream);
            cuda_stream.enqueue_attach(lhs.share(), rhs.share(), fsc.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Computes the isotropic Fourier Shell Correlation between \p lhs and \p rhs.
    /// \tparam REMAP   Whether the input non-redundant FFTs are centered. Should be H2H or HC2HC.
    /// \param[in] lhs  Left-hand side.
    /// \param[in] rhs  Right-hand side. Should have the same shape as \p lhs.
    /// \param shape    Logical shape of \p lhs and \p rhs.
    ///                 It should be a cubic or rectangular (batched) volume.
    /// \return A (batched) row vector with the FSC. The number of shells is min(shape) // 2 + 1.
    template<noa::fft::Remap REMAP, typename Lhs, typename Rhs, typename = std::enable_if_t<
            noa::traits::is_array_or_view_of_almost_any_v<Lhs, c32, c64> &&
            noa::traits::is_array_or_view_of_almost_any_v<Rhs, c32, c64> &&
            noa::traits::are_almost_same_value_type_v<Lhs, Rhs> &&
            (REMAP == noa::fft::Remap::H2H || REMAP == noa::fft::Remap::HC2HC)>>
    auto isotropic_fsc(const Lhs& lhs, const Rhs& rhs, const Shape4<i64>& shape) {
        const auto shell_count = noa::math::min(lhs.shape()) / 2 + 1;
        const auto expected_shape = Shape4<i64>{shape[0], 1, 1, shell_count};

        using value_t = typename Lhs::value_type;
        Array<value_t> fsc(expected_shape, rhs.options());
        isotropic_fsc<REMAP>(lhs, rhs, fsc, shape);
        return fsc;
    }

    /// Computes the anisotropic Fourier Shell Correlation between \p lhs and \p rhs.
    /// \tparam REMAP               Whether the input non-redundant FFTs are centered. Should be H2H or HC2HC.
    /// \param[in] lhs              Left-hand side.
    /// \param[in] rhs              Right-hand side. Should have the same shape as \p lhs.
    /// \param[out] fsc             The output FSC. A row-major table with the FSC of shape (batch, 1, cones, shells).
    ///                             Each row contains the shell values. There's one row per cone.
    ///                             Each column is a shell, with the number of shells set to min(shape) // 2 + 1.
    ///                             There's one table per input batch.
    /// \param shape                Logical shape of \p lhs and \p rhs.
    ///                             It should be a cubic or rectangular (batched) volume.
    /// \param[in] cone_directions  DHW normalized direction(s) of the cone(s).
    /// \param cone_aperture        Cone aperture, in radians.
    template<noa::fft::Remap REMAP, typename Lhs, typename Rhs,
             typename Output, typename Cones, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_almost_any_v<Lhs, c32, c64> &&
             noa::traits::is_array_or_view_of_almost_any_v<Rhs, c32, c64> &&
             noa::traits::is_array_or_view_of_any_v<Output, f32, f64> &&
             noa::traits::are_almost_same_value_type_v<Lhs, Rhs> &&
             noa::traits::are_almost_same_value_type_v<noa::traits::value_type_t<Lhs>, Output> &&
             noa::traits::is_array_or_view_of_almost_any_v<Cones, Vec3<f32>> &&
             (REMAP == noa::fft::Remap::H2H || REMAP == noa::fft::Remap::HC2HC)>>
    void anisotropic_fsc(const Lhs& lhs, const Rhs& rhs, const Output& fsc, const Shape4<i64>& shape,
                         const Cones& cone_directions, f32 cone_aperture) {
        details::check_fsc_parameters(lhs, rhs, fsc, shape, cone_directions);

        const Device device = fsc.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::signal::fft::anisotropic_fsc<REMAP>(
                        lhs.get(), lhs.strides(),
                        rhs.get(), rhs.strides(),
                        fsc.get(), shape,
                        cone_directions.get(), cone_directions.elements(), cone_aperture,
                        threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::signal::fft::anisotropic_fsc<REMAP>(
                    lhs.get(), lhs.strides(),
                    rhs.get(), rhs.strides(),
                    fsc.get(), shape,
                    cone_directions.get(), cone_directions.elements(), cone_aperture,
                    cuda_stream);
            cuda_stream.enqueue_attach(lhs.share(), rhs.share(), fsc.share(), cone_directions.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Computes the anisotropic/conical Fourier Shell Correlation between \p lhs and \p rhs.
    /// \tparam REMAP               Whether the input non-redundant FFTs are centered. Should be H2H or HC2HC.
    /// \param[in] lhs              Left-hand side.
    /// \param[in] rhs              Right-hand side. Should have the same shape as \p lhs.
    /// \param shape                Logical shape of \p lhs and \p rhs.
    ///                             It should be a cubic or rectangular (batched) volume.
    /// \param[in] cone_directions  DHW normalized direction(s) of the cone(s).
    /// \param cone_aperture        Cone aperture, in radians.
    /// \return A row-major (batched) table with the FSC of shape (batch, 1, cones, shells).
    ///         Each row contains the shell values. There's one row per cone.
    ///         Each column is a shell, with the number of shells set to min(shape) // 2 + 1.
    ///         There's one table per input batch.
    template<noa::fft::Remap REMAP, typename Lhs, typename Rhs, typename Cones, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_almost_any_v<Lhs, c32, c64> &&
             noa::traits::is_array_or_view_of_almost_any_v<Rhs, c32, c64> &&
             noa::traits::are_almost_same_value_type_v<Lhs, Rhs> &&
             noa::traits::is_array_or_view_of_almost_any_v<Cones, Vec3<f32>> &&
             (REMAP == noa::fft::Remap::H2H || REMAP == noa::fft::Remap::HC2HC)>>
    auto anisotropic_fsc(const Lhs& lhs, const Rhs& rhs, const Shape4<i64>& shape,
                         const Cones& cone_directions, f32 cone_aperture) {
        const auto shell_count = noa::math::min(lhs.shape()) / 2 + 1;
        const auto expected_shape = Shape4<i64>{shape[0], 1, cone_directions.elements(), shell_count};

        using value_t = typename Lhs::value_type;
        Array<value_t> fsc(expected_shape, rhs.options());
        anisotropic_fsc<REMAP>(lhs, rhs, fsc, shape, cone_directions, cone_aperture);
        return fsc;
    }
}
