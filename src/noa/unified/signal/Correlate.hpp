#pragma once

#include "noa/core/signal/Correlation.hpp"
#include "noa/core/geometry/Shape.hpp"
#include "noa/unified/Array.hpp"
#include "noa/unified/fft/Transform.hpp"
#include "noa/unified/signal/PhaseShift.hpp"
#include "noa/unified/ReduceEwise.hpp"
#include "noa/unified/ReduceAxesEwise.hpp"

#include "noa/cpu/signal/Correlate.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/signal/Correlate.hpp"
#endif

namespace noa::signal::guts {
    template<size_t NDIM, typename Input, typename PeakCoord = Empty, typename PeakValue = Empty>
    void check_xpeak_parameters(
            const Input& xmap,
            const Vec<i64, NDIM>& peak_radius,
            CorrelationRegistration peak_mode,
            const PeakCoord& peak_coordinates = {},
            const PeakValue& peak_values = {}
    ) {
        check(not xmap.is_empty(), "Empty array detected");
        check(all(xmap.strides() > 0), "The cross-correlation map should not be broadcasted");
        check(xmap.shape().ndim() == NDIM,
              "The cross-correlation map(s) shape doesn't match the ndim. Got shape={} and expected ndim={}",
              xmap.shape(), NDIM);

        if constexpr (nt::is_varray_v<PeakCoord>) {
            if (not peak_coordinates.is_empty()) {
                check(ni::is_contiguous_vector(peak_coordinates) and
                      peak_coordinates.elements() == xmap.shape()[0],
                      "The number of peak coordinates, specified as a contiguous vector, should be equal to "
                      "the batch size of the cross-correlation map. Got n_peaks={} and batch={}",
                      peak_coordinates.elements(), xmap.shape()[0]);
                check(xmap.device() == peak_coordinates.device(),
                      "The cross-correlation map and output peak coordinates must be on the same device, "
                      "but got xmap:device={} and peak_coordinates:device={}",
                      xmap.device(), peak_coordinates.device());
            }
        }

        if constexpr (nt::is_varray_v<PeakValue>) {
            if (not peak_values.is_empty()) {
                check(ni::is_contiguous_vector(peak_values) and
                      peak_values.elements() == xmap.shape()[0],
                      "The number of peak values, specified as a contiguous vector, should be equal to "
                      "the batch size of the cross-correlation map. Got n_peaks={} and batch={}",
                      peak_values.elements(), xmap.shape()[0]);
                check(xmap.device() == peak_values.device(),
                      "The cross-correlation map and output peak values must be on the same device, "
                      "but got xmap:device={} and peak_values:device={}",
                      xmap.device(), peak_values.device());
            }
        }

        constexpr i64 GPU_LIMIT = NDIM == 1 ? 64 : NDIM == 2 ? 8 : 2;
        const i64 peak_radius_limit = xmap.device().is_gpu() and peak_mode == CorrelationRegistration::COM ? GPU_LIMIT : 64;
        check(all(peak_radius > 0 and peak_radius <= peak_radius_limit),
              "The registration radius should be a small positive value (less than {}), but got {}",
              peak_radius_limit, peak_radius);
    }
}

namespace noa::signal {
    /// Computes the cross-correlation score(s).
    /// \param[in] lhs      Left-hand side.
    /// \param[in] rhs      Right-hand side.
    /// \param[out] scores  Cross-correlation scores(s). One per batch.
    /// \param normalize    Whether the inputs should be L2-norm normalized before computing the scores.
    template<typename Lhs, typename Rhs, typename Output>
    requires ((nt::are_varray_of_real_v<Lhs, Rhs, Output> or
               nt::are_varray_of_complex_v<Lhs, Rhs, Output>) and
              nt::is_varray_of_mutable_v<Output>)
    void cross_correlation_score(const Lhs& lhs, const Rhs& rhs, const Output& scores, bool normalize = false) {
        check(not lhs.is_empty() and not rhs.is_empty() and not scores.is_empty(), "Empty array detected");
        check(all(lhs.shape() == rhs.shape()),
              "Inputs should have the same shape, but got lhs:shape={}, rhs:shape={}",
              lhs.shape(), rhs.shape());
        check(lhs.device() == rhs.device() and rhs.device() == scores.device(),
              "The input arrays should be on the same device, but got lhs:device={}, rhs:device={} and  scores:device={}",
              lhs.device(), rhs.device(), scores.device());

        const auto batch = lhs.shape()[0];
        check(ni::is_contiguous_vector(scores) and scores.elements() == batch,
              "The number of scores, specified as a contiguous vector, should be equal to the batch size. "
              "Got scores:shape={}, scores:strides={}, and batch={}",
              scores.shape(), scores.strides(), batch);

        using output_t = nt::value_type_t<Output>;
        using lhs_real_t = nt::mutable_value_type_twice_t<Lhs>;
        using rhs_real_t = nt::mutable_value_type_twice_t<Rhs>;
        if (normalize) {
            const auto options = ArrayOption{lhs.device(), Allocator(MemoryResource::DEFAULT_ASYNC)};
            Array<lhs_real_t> lhs_norms({batch, 1, 1, 1}, options);
            Array<rhs_real_t> rhs_norms({batch, 1, 1, 1}, options);
            reduce_axes_ewise(
                    wrap(lhs, rhs),
                    wrap(f64{}, f64{}),
                    wrap(lhs_norms.view(), rhs_norms.view()),
                    guts::CrossCorrelationL2Norm{});
            reduce_axes_ewise(
                    wrap(lhs, rhs, std::move(lhs_norms), std::move(rhs_norms)),
                    wrap(output_t{}),
                    wrap(scores.flat(0)),
                    guts::CrossCorrelationScore{});
        } else {
            reduce_axes_ewise(
                    wrap(lhs, rhs),
                    wrap(output_t{}),
                    wrap(scores.flat(0)),
                    guts::CrossCorrelationScore{});
        }
    }

    /// Computes the cross-correlation coefficient.
    /// \param[in] lhs      Left-hand side.
    /// \param[in] rhs      Right-hand side.
    /// \param normalize    Whether the inputs should be L2-norm normalized before computing the score.
    template<typename Lhs, typename Rhs>
    requires (nt::are_varray_of_real_v<Lhs, Rhs> or nt::are_varray_of_complex_v<Lhs, Rhs>)
    [[nodiscard]] auto cross_correlation_score(const Lhs& lhs, const Rhs& rhs, bool normalize = false) {
        check(not lhs.is_empty() and not rhs.is_empty(), "Empty array detected");
        check(all(lhs.shape() == rhs.shape()) and not rhs.shape().is_batched(),
              "Arrays should have the same shape and should not be batched, but got lhs:shape={}, rhs:shape={}",
              lhs.shape(), rhs.shape());
        check(lhs.device() == rhs.device(),
              "The lhs and rhs input arrays should be on the same device, but got lhs:{} and rhs:{}",
              lhs.device(), rhs.device());

        using output_t = std::conditional_t<nt::is_complex_v<Lhs>, c64, f64>;
        using lhs_real_t = nt::mutable_value_type_twice_t<Lhs>;
        using rhs_real_t = nt::mutable_value_type_twice_t<Rhs>;

        output_t score;
        if (normalize) {
            lhs_real_t lhs_norm;
            rhs_real_t rhs_norm;
            reduce_ewise(
                    wrap(lhs, rhs),
                    wrap(f64{}, f64{}),
                    wrap(lhs_norm, rhs_norm),
                    guts::CrossCorrelationL2Norm{});
            reduce_ewise(
                    wrap(lhs, rhs, lhs_norm, rhs_norm),
                    wrap(output_t{}),
                    wrap(score),
                    guts::CrossCorrelationScore{});
        } else {
            reduce_ewise(
                    wrap(lhs, rhs),
                    wrap(output_t{}),
                    wrap(score),
                    guts::CrossCorrelationScore{});
        }
    }

    struct CrossCorrelationMapOptions {
        /// Correlation mode to use. Remember that DOUBLE_PHASE_CORRELATION doubles the shifts.
        Correlation mode = Correlation::CONVENTIONAL;

        /// Normalization mode to use for the C2R transform producing the final output.
        /// This should match the mode that was used to compute the input transforms.
        noa::fft::Norm ifft_norm = noa::fft::NORM_DEFAULT;

        /// Whether the C2T transform should be cached.
        bool ifft_cache_plan{true};
    };

    /// Computes the cross-correlation map.
    /// \tparam REMAP       Whether the output map should be centered. Should be H2F or H2FC.
    /// \tparam Real        float or double.
    /// \param[in] lhs      Non-centered rFFT of the signal to cross-correlate.
    /// \param[in,out] rhs  Non-centered rFFT of the signal to cross-correlate.
    ///                     Overwritten by default (see \p buffer).
    /// \param[out] output  Cross-correlation map.
    ///                     If REMAP is H2F, the zero lag is at {n, 0, 0, 0}.
    ///                     If REMAP is H2FC, the zero lag is at {n, shape[1]/2, shape[2]/2, shape[3]/2}.
    /// \param options      Correlation mode and ifft options.
    /// \param[out] buffer  Buffer that can fit \p shape.rfft() complex elements. It is overwritten.
    ///                     Can be \p lhs or \p rhs. If empty, use \p rhs instead.
    ///
    /// \note As mentioned above, this function takes the rFFT of the real inputs to correlate.
    ///       The score with zero lag can be computed more efficiently wth cross_correlation_score.
    ///       If other lags are to be selected (which is the entire point of this function), the inputs
    ///       should be zero-padded before taking the irfft to cancel the circular convolution effect of
    ///       the DFT.
    template<noa::fft::RemapInterface REMAP, typename Lhs, typename Rhs, typename Output,
             typename Buffer = View<nt::mutable_value_type_t<Rhs>>>
    requires(nt::are_varray_of_complex_v<Lhs, Rhs, Buffer> and
             nt::are_almost_same_value_type_v<Lhs, Rhs, Buffer> and
             nt::are_varray_of_mutable_v<Rhs, Buffer> and
             nt::is_varray_of_any_v<Output, nt::value_type_t<Rhs>> and
             (REMAP.remap == noa::fft::Remap::H2F or REMAP.remap == noa::fft::Remap::H2FC))
    void cross_correlation_map(
            const Lhs& lhs, const Rhs& rhs, const Output& output,
            const CrossCorrelationMapOptions& options = {},
            const Buffer& buffer = {}
    ) {
        check(not lhs.is_empty() and not rhs.is_empty() and not output.is_empty(), "Empty array detected");

        const auto expected_shape = output.shape().rfft();
        auto lhs_strides = lhs.strides();
        check(ni::broadcast(lhs.shape(), lhs_strides, expected_shape),
              "Cannot broadcast an array of shape {} into an array of shape {}",
              lhs.shape(), expected_shape
        );
        auto rhs_strides = rhs.strides();
        check(ni::broadcast(rhs.shape(), rhs_strides, expected_shape),
              "Cannot broadcast an array of shape {} into an array of shape {}",
              rhs.shape(), expected_shape
        );

        const Device device = output.device();
        check(device == lhs.device() and device == rhs.device(),
              "The lhs, rhs and output arrays must be on the same device, but got lhs:{}, rhs:{} and output:{}",
              lhs.device(), rhs.device(), device);

        using complex_t = nt::value_type_t<Buffer>;
        using real_t = nt::value_type_t<complex_t>;
        View<complex_t> tmp;
        if (buffer.is_empty()) {
            check(all(rhs_strides >= 0),
                  "Since no temporary buffer is passed, the rhs input will be overwritten and "
                  "should not have any strides equal to 0, but got {}", rhs_strides);
            tmp = View<complex_t>(rhs.get(), rhs.shape(), rhs.strides(), rhs.options());
        } else {
            check(device == buffer.device(),
                  "The temporary and output arrays must be on the same device, buffer:{} and output:{}",
                  buffer.device(), device);
            check(all(buffer.shape() >= expected_shape) and all(buffer.strides() >= 0),
                  "The temporary buffer should be able to fit an array of shape {}, but got effective shape of {}",
                  expected_shape, ni::effective_shape(buffer.shape(), buffer.strides()));
            tmp = View<complex_t>(buffer.get(), buffer.shape(), buffer.strides(), buffer.options());
        }

        constexpr auto EPSILON = static_cast<real_t>(1e-13);

        // TODO Add normalization with auto-correlation?
        //      IMO it's always simpler to normalize the real inputs,
        //      so not sure how useful this would be.
        switch (options.mode) {
            case Correlation::CONVENTIONAL:
                ewise(wrap(lhs, rhs), tmp, MultiplyConjugate{});
                break;
            case Correlation::PHASE:
                ewise(wrap(lhs, rhs), tmp,
                      []NOA_HD<typename R>(const Complex<R>& l, const Complex<R>& r, Complex<R>& o) {
                          const Complex<R> product = l * conj(r);
                          const R magnitude = abs(product);
                          o = product / (magnitude + EPSILON);
                          // The epsilon could be scaled by the max(abs(rhs)), but this seems to be useful only
                          // for input values close to zero (less than 1e-10). In most cases, this is fine.
                      });
                break;
            case Correlation::DOUBLE_PHASE:
                ewise(wrap(lhs, rhs), tmp,
                      []NOA_HD<typename R>(const Complex<R>& l, const Complex<R>& r, Complex<R>& o) {
                          const Complex<R> product = l * conj(r);
                          const Complex<R> product_sqd = {product.real * product.real, product.imag * product.imag};
                          const R magnitude = sqrt(product_sqd.real + product_sqd.imag) + EPSILON;
                          o = {(product_sqd.real - product_sqd.imag) / magnitude,
                               (2 * product.real * product.imag) / magnitude};
                      });
                break;
            case Correlation::MUTUAL:
                ewise(wrap(lhs, rhs), tmp,
                      []NOA_HD<typename R>(const Complex<R>& l, const Complex<R>& r, Complex<R>& o) {
                          const Complex<R> product = l * conj(r);
                          const R magnitude_sqrt = sqrt(abs(product));
                          o = product / (magnitude_sqrt + EPSILON);
                      });
                break;
        }

        using namespace noa::fft;
        if constexpr (REMAP.remap == Remap::H2FC) {
            const auto shape = output.shape();
            const auto shape_3d = shape.pop_front();
            if (shape_3d.ndim() == 3) {
                phase_shift_3d<Remap::H2H>(tmp, tmp, shape, (shape_3d / 2).vec.template as<f32>());
            } else {
                phase_shift_2d<Remap::H2H>(tmp, tmp, shape, (shape_3d.pop_front() / 2).vec.template as<f32>());
            }
        }

        // In case this is an Array, pass the original object.
        if (buffer.is_empty())
            c2r(rhs, output, {.norm=options.ifft_norm, .cache_plan=options.ifft_cache_plan});
        else {
            c2r(buffer, output, {.norm=options.ifft_norm, .cache_plan=options.ifft_cache_plan});
        }
    }

    template<size_t N>
    struct CrossCorrelationPeakOptions {
        CorrelationRegistration mode = CorrelationRegistration::PARABOLA_1D;
        Vec<i64, N> registration_radius{1};
        Vec<f64, N> maximum_lag{-1};
    };

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
    template<noa::fft::RemapInterface REMAP, size_t N, typename Input,
             typename PeakCoord = View<Vec<f32, N>>,
             typename PeakValue = View<nt::mutable_value_type_t<Input>>>
    requires (nt::is_varray_of_almost_any_v<Input, f32, f64> &&
              nt::is_varray_of_any_v<PeakCoord, Vec<f32, N>> &&
              nt::is_varray_of_any_v<PeakValue, f32, f64> &&
              nt::are_almost_same_value_type_v<Input, PeakValue>)
    void cross_correlation_peak(
            const Input& xmap,
            const PeakCoord& peak_coordinates,
            const PeakValue& peak_values = {},
            const CrossCorrelationPeakOptions<N>& options = {}
    ) {
        guts::check_xpeak_parameters(xmap, options.registration_radius, options.mode, peak_coordinates, peak_values);

        const Device& device = xmap.device();
        const auto shape = xmap.shape();

        using value_t = nt::mutable_value_type_t<Input>;
        if (options.maximum_lag >= 0) {
            using ellipse_t = noa::geometry::Ellipse<N, value_t, f32, true>;
            using accessor_t = AccessorI64<value_t, 4>;
            using op_t = noa::geometry::DrawGeometricShape<
                    N, REMAP.remap, i64, f32, ellipse_t, Empty, Multiply, accessor_t, accessor_t>;

            ellipse_t ellipse();
            iwise(xmap.shape(), device, op_t{xmap.accessor(), xmap.accessor(), shape.pop_front(), ellipse, {}, {}});
        }
        // TODO If centered, select xmap subregion within options.maximum_lag.

        // FIXME Mask xmap with maximum_lag
        Array<i64> peak_offsets;
        reduce_axes_iwise(xmap, i64{}, peak_offsets, ReduceFirstMax{});


        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                using namespace noa::cpu::signal;
                if constexpr (N == 1) {
                    subpixel_registration_1d<REMAP.remap>(
                            xmap.get(), xmap.strides(), xmap.shape(),
                            peak_offsets.get(), peak_coordinates.get(), peak_values.get(),
                            options.mode, options.registration_radius);
                } else if constexpr (N == 2) {
                    subpixel_registration_2d<REMAP.remap>(
                            xmap.get(), xmap.strides(), xmap.shape(),
                            peak_offsets.get(), peak_coordinates.get(), peak_values.get(),
                            options.mode, options.registration_radius);
                } else {
                    subpixel_registration_3d<REMAP.remap>(
                            xmap.get(), xmap.strides(), xmap.shape(),
                            peak_offsets.get(), peak_coordinates.get(), peak_values.get(),
                            options.mode, options.registration_radius);
                }
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            using namespace noa::cuda::signal;
            if constexpr (N == 1) {
                subpixel_registration_1d<REMAP.remap>(
                        xmap.get(), xmap.strides(), xmap.shape(),
                        peak_offsets.get(), peak_values.get(),
                        options.mode, options.registration_radius, cuda_stream);
            } else if constexpr (N == 2) {
                subpixel_registration_2d<REMAP.remap>(
                        xmap.get(), xmap.strides(), xmap.shape(),
                        peak_offsets.get(), peak_values.get(),
                        options.mode, options.registration_radius, cuda_stream);
            } else {
                subpixel_registration_3d<REMAP.remap>(
                        xmap.get(), xmap.strides(), xmap.shape(),
                        peak_offsets.get(), peak_values.get(),
                        options.mode, options.registration_radius, cuda_stream);
            }
            cuda_stream.enqueue_attach(xmap, peak_offsets, peak_coordinates, peak_values);
            #else
            panic("No GPU backend detected");
            #endif
        }
    }

    template<noa::fft::RemapInterface REMAP, typename Input,
             typename PeakCoord = View<Vec1<f32>>,
             typename PeakValue = View<nt::mutable_value_type_t<Input>>>
    void cross_correlation_peak_1d(
            const Input& xmap,
            const PeakCoord& peak_coordinates,
            const PeakValue& peak_values = {},
            const CrossCorrelationPeakOptions<1>& options = {}
    ) {
        cross_correlation_peak<REMAP, 1>(xmap, peak_coordinates, peak_values, options);
    }

    template<noa::fft::RemapInterface REMAP, typename Input,
             typename PeakCoord = View<Vec2<f32>>,
             typename PeakValue = View<nt::mutable_value_type_t<Input>>>
    void cross_correlation_peak_2d(
            const Input& xmap,
            const PeakCoord& peak_coordinates,
            const PeakValue& peak_values = {},
            const CrossCorrelationPeakOptions<3>& options = {}
    ) {
        cross_correlation_peak<REMAP, 3>(xmap, peak_coordinates, peak_values, options);
    }

    template<noa::fft::RemapInterface REMAP, typename Input,
             typename PeakCoord = View<Vec3<f32>>,
             typename PeakValue = View<nt::mutable_value_type_t<Input>>>
    void cross_correlation_peak_3d(
            const Input& xmap,
            const PeakCoord& peak_coordinates,
            const PeakValue& peak_values = {},
            const CrossCorrelationPeakOptions<3>& options = {}
    ) {
        cross_correlation_peak<REMAP, 3>(xmap, peak_coordinates, peak_values, options);
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
             nt::is_varray_of_any_v<Input, f32, f64> &&
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
             nt::is_varray_of_any_v<Input, f32, f64> &&
             details::is_valid_xpeak_v<REMAP, 1>>>
    [[nodiscard]] auto xpeak_1d(const Input& xmap,
                                const Vec1<f32>& xmap_radius = Vec1<f32>{0},
                                PeakMode peak_mode = PeakMode::PARABOLA_1D,
                                const Vec1<i64>& peak_radius = Vec1<i64>{1}) {
        return xpeak<REMAP, 1>(xmap, xmap_radius, peak_mode, peak_radius);
    }

    template<Remap REMAP, typename Input, typename = std::enable_if_t<
             nt::is_varray_of_any_v<Input, f32, f64> &&
             details::is_valid_xpeak_v<REMAP, 2>>>
    [[nodiscard]] auto xpeak_2d(const Input& xmap,
                                const Vec2<f32>& xmap_radius = Vec2<f32>{0},
                                PeakMode peak_mode = PeakMode::PARABOLA_1D,
                                const Vec2<i64>& peak_radius = Vec2<i64>{1}) {
        return xpeak<REMAP, 2>(xmap, xmap_radius, peak_mode, peak_radius);
    }

    template<Remap REMAP, typename Input, typename = std::enable_if_t<
             nt::is_varray_of_any_v<Input, f32, f64> &&
             details::is_valid_xpeak_v<REMAP, 3>>>
    [[nodiscard]] auto xpeak_3d(const Input& xmap,
                                const Vec3<f32>& xmap_radius = Vec3<f32>{0},
                                PeakMode peak_mode = PeakMode::PARABOLA_1D,
                                const Vec3<i64>& peak_radius = Vec3<i64>{1}) {
        return xpeak<REMAP, 3>(xmap, xmap_radius, peak_mode, peak_radius);
    }
}
