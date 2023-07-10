#pragma once

#include "noa/cpu/signal/fft/Correlate.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/signal/fft/Correlate.hpp"
#endif

#include "noa/unified/Array.hpp"
#include "noa/unified/fft/Transform.hpp"

namespace noa::signal::fft::details {
    using Remap = ::noa::fft::Remap;

    template<Remap REMAP, size_t N>
    constexpr bool is_valid_xpeak_v =
            (REMAP == Remap::F2F || REMAP == Remap::FC2FC) &&
            (N == 1 || N == 2 || N == 3);

    template<Remap REMAP>
    constexpr bool is_valid_xcorr_remap_v =
            (REMAP == Remap::H2H || REMAP == Remap::HC2HC ||
             REMAP == Remap::F2F || REMAP == Remap::FC2FC);

    template<size_t NDIM, typename Input, typename PeakCoord = Empty, typename PeakValue = Empty>
    void check_xpeak_parameters(const Input& xmap, const Vec<i64, NDIM>& peak_radius, PeakMode peak_mode,
                                const PeakCoord& peak_coordinates = {},
                                const PeakValue& peak_values = {}) {
        NOA_CHECK(!xmap.is_empty(), "Empty array detected");
        NOA_CHECK(noa::all(xmap.strides() > 0), "The cross-correlation map should not be broadcast");
        NOA_CHECK(xmap.shape().ndim() == NDIM,
                  "The cross-correlation map(s) shape doesn't match the ndim. Got shape {} and expected ndim is {}",
                  xmap.shape(), NDIM);

        if constexpr (noa::traits::is_array_or_view_v<PeakCoord>) {
            if (!peak_coordinates.is_empty()) {
                NOA_CHECK(noa::indexing::is_contiguous_vector(peak_coordinates) &&
                          peak_coordinates.elements() == xmap.shape()[0],
                          "The number of peak coordinates, specified as a contiguous vector, should be equal to "
                          "the number of batches in the cross-correlation map. Got {} peak coordinates and {} output "
                          "batches", peak_coordinates.elements(), xmap.shape()[0]);
                NOA_CHECK(xmap.device() == peak_coordinates.device(),
                          "The cross-correlation map and output peak coordinates must be on the same device, "
                          "but got xmap:{} and peak_coordinates:{}", xmap.device(), peak_coordinates.device());
            }
        }

        if constexpr (noa::traits::is_array_or_view_v<PeakValue>) {
            if (!peak_values.is_empty()) {
                NOA_CHECK(noa::indexing::is_contiguous_vector(peak_values) &&
                          peak_values.elements() == xmap.shape()[0],
                          "The number of peak values, specified as a contiguous vector, should be equal to "
                          "the number of batches in the cross-correlation map. Got {} peak values and {} output "
                          "batches", peak_values.elements(), xmap.shape()[0]);
                NOA_CHECK(xmap.device() == peak_values.device(),
                          "The cross-correlation map and output peak values must be on the same device, "
                          "but got xmap:{} and peak_values:{}", xmap.device(), peak_values.device());
            }
        }

        constexpr i64 CUDA_COM_LIMIT = NDIM == 1 ? 64 : NDIM == 2 ? 8 : 2;
        const i64 peak_radius_limit = xmap.device().is_gpu() && peak_mode == PeakMode::COM ? CUDA_COM_LIMIT : 64;
        NOA_CHECK(noa::all(peak_radius > 0 && peak_radius <= peak_radius_limit),
                  "The registration radius should be a small positive value (less than {}), but got {}",
                  peak_radius_limit, peak_radius);
    }
}

// TODO Add xmap_autocorrelate() to compute the xcorr and the auto-correlation of the lhs and rhs?

namespace noa::signal::fft {
    using Remap = ::noa::fft::Remap;
    using Norm = ::noa::fft::Norm;

    /// Computes the cross-correlation map.
    /// \tparam REMAP           Whether the output map should be centered. Should be H2F or H2FC.
    /// \tparam Real            float or double.
    /// \param[in] lhs          Left-hand side non-redundant and non-centered FFT argument.
    /// \param[in,out] rhs      Right-hand side non-redundant and non-centered FFT argument.
    ///                         Overwritten by default (see \p buffer).
    /// \param[out] output      Cross-correlation map.
    ///                         If REMAP is H2F, the central peak is at indexes {n, 0, 0, 0}.
    ///                         If REMAP is H2FC, the central peal is at indexes {n, shape[1]/2, shape[2]/2, shape[3]/2}.
    /// \param correlation_mode Correlation mode to use. Remember that DOUBLE_PHASE_CORRELATION doubles the shifts.
    /// \param fft_norm         Normalization mode to use for the C2R transform producing the final output.
    ///                         This should match the mode that was used to compute the input transforms.
    /// \param[out] buffer      Buffer that can fit \p shape.rfft() complex elements. It is overwritten.
    ///                         Can be \p lhs or \p rhs. If empty, use \p rhs instead.
    template<Remap REMAP, typename Lhs, typename Rhs, typename Output,
             typename Buffer = View<noa::traits::mutable_value_type_t<Rhs>>, typename = std::enable_if_t<
                    noa::traits::is_array_or_view_of_almost_any_v<Lhs, c32, c64> &&
                    noa::traits::is_array_or_view_of_any_v<Rhs, c32, c64> &&
                    noa::traits::is_array_or_view_of_any_v<Output, f32, f64> &&
                    noa::traits::are_almost_same_value_type_v<Lhs, Rhs, Buffer> &&
                    noa::traits::are_same_value_type_v<noa::traits::value_type_t<Rhs>, Output> &&
                    (REMAP == Remap::H2F || REMAP == Remap::H2FC)>>
    void xmap(const Lhs& lhs, const Rhs& rhs, const Output& output,
              CorrelationMode correlation_mode = CorrelationMode::CONVENTIONAL,
              Norm fft_norm = noa::fft::NORM_DEFAULT,
              const Buffer& buffer = {}) {
        NOA_CHECK(!lhs.is_empty() && !rhs.is_empty() && !output.is_empty(), "Empty array detected");

        const auto expected_shape = output.shape().rfft();
        auto lhs_strides = lhs.strides();
        if (!noa::indexing::broadcast(lhs.shape(), lhs_strides, expected_shape)) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      lhs.shape(), expected_shape);
        }
        auto rhs_strides = rhs.strides();
        if (!noa::indexing::broadcast(rhs.shape(), rhs_strides, expected_shape)) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      rhs.shape(), expected_shape);
        }

        const Device device = output.device();
        NOA_CHECK(device == lhs.device() && device == rhs.device(),
                  "The lhs, rhs and output arrays must be on the same device, but got lhs:{}, rhs:{} and output:{}",
                  lhs.device(), rhs.device(), device);

        if (buffer.is_empty()) {
            NOA_CHECK(noa::all(rhs_strides >= 0),
                      "Since no temporary buffer is passed, the rhs input will be overwritten and "
                      "should not have any strides equal to 0, but got {}", rhs_strides);
        } else {
            NOA_CHECK(device == buffer.device(),
                      "The temporary and output arrays must be on the same device, buffer:{} and output:{}",
                      buffer.device(), device);
            NOA_CHECK(noa::all(buffer.shape() >= expected_shape) && noa::all(buffer.strides() >= 0),
                      "The temporary buffer should be able to fit an array of shape {}, but got effective shape of {}",
                      expected_shape, noa::indexing::effective_shape(buffer.shape(), buffer.strides()));
        }

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::signal::fft::xmap<REMAP>(
                        lhs.get(), lhs_strides, rhs.get(), rhs_strides,
                        output.get(), output.strides(), output.shape(),
                        correlation_mode, fft_norm,
                        buffer.get(), buffer.strides(), threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::signal::fft::xmap<REMAP>(
                    lhs.get(), lhs_strides, rhs.get(), rhs_strides,
                    output.get(), output.strides(), output.shape(),
                    correlation_mode, fft_norm,
                    buffer.get(), buffer.strides(), cuda_stream);
            cuda_stream.enqueue_attach(lhs.share(), rhs.share(), output.share(), buffer.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Find the highest peak in a cross-correlation line.
    /// \tparam REMAP                   Whether \p xmap is centered. Should be F2F or FC2FC.
    /// \param[in,out] xmap             1d, 2d or 3d cross-correlation map.
    ///                                 It can be overwritten depending on \p xmap_radius.
    /// \param[out] peak_coordinates    Output ((D)H)W coordinate of the highest peak. One per batch or empty.
    /// \param[out] peak_values         Output value of the highest peak. One per batch or empty.
    /// \param xmap_radius              ((D)H)W radius of the smooth elliptic mask to apply (in-place) to \p xmap.
    ///                                 This is used to restrict the peak position relative to the center of \p xmap.
    ///                                 If negative or 0, it is ignored.
    /// \param peak_mode                Registration mode to use for subpixel accuracy.
    /// \param peak_radius              ((D)H)W radius of the registration window, centered on the peak.
    /// \note On the GPU, \p peak_radius is limited to 64 (1d), 8 (2d), or 2 (3d) with \p peak_mode PeakMode::COM.
    template<Remap REMAP, size_t N, typename Input,
             typename PeakCoord = View<Vec<f32, N>>,
             typename PeakValue = View<noa::traits::mutable_value_type_t<Input>>,
             typename = std::enable_if_t<
                     noa::traits::is_array_or_view_of_almost_any_v<Input, f32, f64> &&
                     noa::traits::is_array_or_view_of_any_v<PeakCoord, Vec<f32, N>> &&
                     noa::traits::is_array_or_view_of_any_v<PeakValue, f32, f64> &&
                     noa::traits::are_almost_same_value_type_v<Input, PeakValue> &&
                     details::is_valid_xpeak_v<REMAP, N>>>
    void xpeak(const Input& xmap,
               const PeakCoord& peak_coordinates,
               const PeakValue& peak_values = {},
               const Vec<f32, N>& xmap_radius = Vec<f32, N>{-1},
               PeakMode peak_mode = PeakMode::PARABOLA_1D,
               const Vec<i64, N>& peak_radius = Vec<i64, N>{1}) {
        details::check_xpeak_parameters(xmap, peak_radius, peak_mode, peak_coordinates, peak_values);

        const Device& device = xmap.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                if constexpr (N == 1) {
                    cpu::signal::fft::xpeak_1d<REMAP>(
                            xmap.get(), xmap.strides(), xmap.shape(), xmap_radius,
                            peak_coordinates.get(), peak_values.get(),
                            peak_mode, peak_radius, threads);
                } else if constexpr (N == 2) {
                    cpu::signal::fft::xpeak_2d<REMAP>(
                            xmap.get(), xmap.strides(), xmap.shape(), xmap_radius,
                            peak_coordinates.get(), peak_values.get(),
                            peak_mode, peak_radius, threads);
                } else {
                    cpu::signal::fft::xpeak_3d<REMAP>(
                            xmap.get(), xmap.strides(), xmap.shape(), xmap_radius,
                            peak_coordinates.get(), peak_values.get(),
                            peak_mode, peak_radius, threads);
                }
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            if constexpr (N == 1) {
                cuda::signal::fft::xpeak_1d<REMAP>(
                        xmap.get(), xmap.strides(), xmap.shape(), xmap_radius,
                        peak_coordinates.get(), peak_values.get(),
                        peak_mode, peak_radius, cuda_stream);
            } else if constexpr (N == 2) {
                cuda::signal::fft::xpeak_2d<REMAP>(
                        xmap.get(), xmap.strides(), xmap.shape(), xmap_radius,
                        peak_coordinates.get(), peak_values.get(),
                        peak_mode, peak_radius, cuda_stream);
            } else {
                cuda::signal::fft::xpeak_3d<REMAP>(
                        xmap.get(), xmap.strides(), xmap.shape(), xmap_radius,
                        peak_coordinates.get(), peak_values.get(),
                        peak_mode, peak_radius, cuda_stream);
            }
            cuda_stream.enqueue_attach(xmap.share(), peak_coordinates.share(), peak_values.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename Input,
             typename PeakCoord = View<Vec1<f32>>,
             typename PeakValue = View<noa::traits::mutable_value_type_t<Input>>,
             typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, 1>>>
    void xpeak_1d(const Input& xmap,
                  const PeakCoord& peak_coordinates,
                  const PeakValue& peak_values = {},
                  const Vec1<f32>& xmap_radius = Vec1<f32>{-1},
                  PeakMode peak_mode = PeakMode::PARABOLA_1D,
                  const Vec1<i64>& peak_radius = Vec1<i64>{1}) {
        xpeak<REMAP, 1>(xmap, peak_coordinates, peak_values, xmap_radius, peak_mode, peak_radius);
    }

    template<Remap REMAP, typename Input,
             typename PeakCoord = View<Vec2<f32>>,
             typename PeakValue = View<noa::traits::mutable_value_type_t<Input>>,
             typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, 2>>>
    void xpeak_2d(const Input& xmap,
                  const PeakCoord& peak_coordinates,
                  const PeakValue& peak_values = {},
                  const Vec2<f32>& xmap_radius = Vec2<f32>{-1},
                  PeakMode peak_mode = PeakMode::PARABOLA_1D,
                  const Vec2<i64>& peak_radius = Vec2<i64>{1}) {
        xpeak<REMAP, 2>(xmap, peak_coordinates, peak_values, xmap_radius, peak_mode, peak_radius);
    }

    template<Remap REMAP, typename Input,
             typename PeakCoord = View<Vec3<f32>>,
             typename PeakValue = View<noa::traits::mutable_value_type_t<Input>>,
             typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, 2>>>
    void xpeak_3d(const Input& xmap,
                  const PeakCoord& peak_coordinates,
                  const PeakValue& peak_values = {},
                  const Vec3<f32>& xmap_radius = Vec3<f32>{-1},
                  PeakMode peak_mode = PeakMode::PARABOLA_1D,
                  const Vec3<i64>& peak_radius = Vec3<i64>{1}) {
        xpeak<REMAP, 3>(xmap, peak_coordinates, peak_values, xmap_radius, peak_mode, peak_radius);
    }

    /// Returns a pair of the ((D)H)W coordinate and the value of the highest peak in a cross-correlation map.
    /// \tparam REMAP       Whether \p xmap is centered. Should be F2F or FC2FC.
    /// \param[in,out] xmap 1d, 2d, or 3d cross-correlation map. It can be overwritten depending on \p xmap_radius.
    /// \param xmap_radius  ((D)H)W radius of the smooth elliptic mask to apply (in-place) to \p xmap.
    ///                     This is used to restrict the peak position relative to the center of \p xmap.
    ///                     If negative or 0, it is ignored.
    /// \param peak_mode    Registration mode to use for subpixel accuracy.
    /// \param peak_radius  ((D)H)W radius of the registration window, centered on the peak.
    /// \note On the GPU, \p peak_radius is limited to 64 (1d), 8 (2d), or 2 (3d) with \p peak_mode PeakMode::COM.
    template<Remap REMAP, size_t N, typename Input, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_any_v<Input, f32, f64> &&
             details::is_valid_xpeak_v<REMAP, N>>>
    [[nodiscard]] auto xpeak(const Input& xmap,
                             const Vec<f32, N>& xmap_radius = Vec<f32, N>{0},
                             PeakMode peak_mode = PeakMode::PARABOLA_1D,
                             const Vec<i64, N>& peak_radius = Vec<i64, N>{1}) {
        details::check_xpeak_parameters(xmap, peak_radius, peak_mode);
        NOA_CHECK(!xmap.shape().is_batched(),
                  "The input cross-correlation cannot be batched, but got shape {}",
                  xmap.shape());

        const Device& device = xmap.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            cpu_stream.synchronize();
            const auto threads = cpu_stream.thread_limit();
            if constexpr (N == 1) {
                return cpu::signal::fft::xpeak_1d<REMAP>(
                        xmap.get(), xmap.strides(), xmap.shape(), xmap_radius,
                        peak_mode, peak_radius, threads);
            } else if constexpr (N == 2) {
                return cpu::signal::fft::xpeak_2d<REMAP>(
                        xmap.get(), xmap.strides(), xmap.shape(), xmap_radius,
                        peak_mode, peak_radius, threads);
            } else {
                return cpu::signal::fft::xpeak_3d<REMAP>(
                        xmap.get(), xmap.strides(), xmap.shape(), xmap_radius,
                        peak_mode, peak_radius, threads);
            }
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            if constexpr (N == 1) {
                return cuda::signal::fft::xpeak_1d<REMAP>(
                        xmap.get(), xmap.strides(), xmap.shape(), xmap_radius,
                        peak_mode, peak_radius, cuda_stream);
            } else if constexpr (N == 2) {
                return cuda::signal::fft::xpeak_2d<REMAP>(
                        xmap.get(), xmap.strides(), xmap.shape(), xmap_radius,
                        peak_mode, peak_radius, cuda_stream);
            } else {
                return cuda::signal::fft::xpeak_3d<REMAP>(
                        xmap.get(), xmap.strides(), xmap.shape(), xmap_radius,
                        peak_mode, peak_radius, cuda_stream);
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename Input, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_any_v<Input, f32, f64> &&
             details::is_valid_xpeak_v<REMAP, 1>>>
    [[nodiscard]] auto xpeak_1d(const Input& xmap,
                                const Vec1<f32>& xmap_radius = Vec1<f32>{0},
                                PeakMode peak_mode = PeakMode::PARABOLA_1D,
                                const Vec1<i64>& peak_radius = Vec1<i64>{1}) {
        return xpeak<REMAP, 1>(xmap, xmap_radius, peak_mode, peak_radius);
    }

    template<Remap REMAP, typename Input, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_any_v<Input, f32, f64> &&
             details::is_valid_xpeak_v<REMAP, 2>>>
    [[nodiscard]] auto xpeak_2d(const Input& xmap,
                                const Vec2<f32>& xmap_radius = Vec2<f32>{0},
                                PeakMode peak_mode = PeakMode::PARABOLA_1D,
                                const Vec2<i64>& peak_radius = Vec2<i64>{1}) {
        return xpeak<REMAP, 2>(xmap, xmap_radius, peak_mode, peak_radius);
    }

    template<Remap REMAP, typename Input, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_any_v<Input, f32, f64> &&
             details::is_valid_xpeak_v<REMAP, 3>>>
    [[nodiscard]] auto xpeak_3d(const Input& xmap,
                                const Vec3<f32>& xmap_radius = Vec3<f32>{0},
                                PeakMode peak_mode = PeakMode::PARABOLA_1D,
                                const Vec3<i64>& peak_radius = Vec3<i64>{1}) {
        return xpeak<REMAP, 3>(xmap, xmap_radius, peak_mode, peak_radius);
    }

    /// Computes the cross-correlation coefficient(s).
    /// \tparam REMAP               Layout of \p lhs and \p rhs. Should be H2H, HC2HC, F2F or FC2FC.
    /// \param[in] lhs              Left-hand side FFT.
    /// \param[in] rhs              Right-hand side FFT.
    /// \param shape                BDHW logical shape.
    /// \param[out] coefficients    Cross-correlation coefficient(s). One per batch.
    ///                             It should be dereferenceable by the CPU.
    template<Remap REMAP, typename Lhs, typename Rhs, typename Output, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_almost_any_v<Lhs, c32, c64> &&
             noa::traits::is_array_or_view_of_almost_any_v<Rhs, c32, c64> &&
             noa::traits::is_array_or_view_of_any_v<Output, f32, f64> &&
             noa::traits::are_almost_same_value_type_v<Lhs, Rhs> &&
             noa::traits::are_almost_same_value_type_v<noa::traits::value_type_t<Lhs>, Output> &&
             details::is_valid_xcorr_remap_v<REMAP>>>
    void xcorr(const Lhs& lhs, const Rhs& rhs, const Shape4<i64>& shape, const Output& coefficients) {
        NOA_CHECK(!lhs.is_empty() && !rhs.is_empty() && !coefficients.is_empty(), "Empty array detected");
        NOA_CHECK(noa::indexing::is_contiguous_vector(coefficients) && coefficients.elements() == shape[0],
                  "The number of coefficients, specified as a contiguous vector, should be equal to the number "
                  "of batches. Got {} coefficients and {} output batches", coefficients.elements(), shape[0]);

        constexpr bool SRC_IS_HALF = noa::traits::to_underlying(REMAP) & noa::fft::Layout::SRC_HALF;
        const auto expected_shape = SRC_IS_HALF ? shape.rfft() : shape;
        auto lhs_strides = lhs.strides();
        if (!noa::indexing::broadcast(lhs.shape(), lhs_strides, expected_shape)) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      lhs.shape(), expected_shape);
        }
        auto rhs_strides = rhs.strides();
        if (!noa::indexing::broadcast(rhs.shape(), rhs_strides, expected_shape)) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      rhs.shape(), expected_shape);
        }

        NOA_CHECK(lhs.device() == rhs.device(),
                  "The lhs and rhs input arrays should be on the same device, but got lhs:{} and rhs:{}",
                  lhs.device(), rhs.device());
        NOA_CHECK(coefficients.is_dereferenceable(), "The coefficients should be accessible to the CPU");
        if (coefficients.device() != lhs.device())
            Stream::current(coefficients.device()).synchronize();

        Stream& stream = Stream::current(lhs.device());
        if (stream.device().is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::signal::fft::xcorr<REMAP>(
                        lhs.get(), lhs_strides, rhs.get(), rhs_strides,
                        shape, coefficients.get(), threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::signal::fft::xcorr<REMAP>(
                    lhs.get(), lhs_strides, rhs.get(), rhs_strides,
                    shape, coefficients.get(), cuda_stream);
            cuda_stream.enqueue_attach(lhs.share(), rhs.share(), coefficients.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Computes the cross-correlation coefficient.
    /// \tparam REMAP   Layout of \p lhs and \p rhs. Should be H2H, HC2HC, F2F or FC2FC.
    /// \param[in] lhs  Left-hand side FFT.
    /// \param[in] rhs  Right-hand side FFT.
    /// \param shape    BDHW logical shape. Should not be batched.
    template<Remap REMAP, typename Lhs, typename Rhs, typename = std::enable_if_t<
            noa::traits::is_array_or_view_of_almost_any_v<Lhs, c32, c64> &&
            noa::traits::is_array_or_view_of_almost_any_v<Rhs, c32, c64> &&
            noa::traits::are_almost_same_value_type_v<Lhs, Rhs> &&
            details::is_valid_xcorr_remap_v<REMAP>>>
    [[nodiscard]] auto xcorr(const Lhs& lhs, const Rhs& rhs, const Shape4<i64>& shape) {
        NOA_CHECK(!lhs.is_empty() && !rhs.is_empty(), "Empty array detected");
        NOA_CHECK(!shape.is_batched(), "The input shape should not be batched");

        constexpr bool SRC_IS_HALF = noa::traits::to_underlying(REMAP) & noa::fft::Layout::SRC_HALF;
        const auto expected_shape = SRC_IS_HALF ? shape.rfft() : shape;
        auto lhs_strides = lhs.strides();
        if (!noa::indexing::broadcast(lhs.shape(), lhs_strides, expected_shape)) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      lhs.shape(), expected_shape);
        }
        auto rhs_strides = rhs.strides();
        if (!noa::indexing::broadcast(rhs.shape(), rhs_strides, expected_shape)) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      rhs.shape(), expected_shape);
        }

        NOA_CHECK(lhs.device() == rhs.device(),
                  "The lhs and rhs input arrays should be on the same device, but got lhs:{} and rhs:{}",
                  lhs.device(), rhs.device());

        const Device device = lhs.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            cpu_stream.synchronize();
            const auto threads = cpu_stream.thread_limit();
            return cpu::signal::fft::xcorr<REMAP>(
                    lhs.share(), lhs_strides, rhs.share(), rhs_strides, shape, threads);
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::signal::fft::xcorr<REMAP>(
                    lhs.share(), lhs_strides, rhs.share(), rhs_strides, shape, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
