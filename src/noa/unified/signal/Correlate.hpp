#pragma once

#include "noa/core/types/Complex.hpp"
#include "noa/core/types/Vec.hpp"
#include "noa/core/types/Pair.hpp"
#include "noa/core/math/LeastSquare.hpp"
#include "noa/core/geometry/DrawShape.hpp"

#include "noa/unified/Array.hpp"
#include "noa/unified/fft/Transform.hpp"
#include "noa/unified/signal/PhaseShift.hpp"
#include "noa/unified/ReduceEwise.hpp"
#include "noa/unified/ReduceAxesEwise.hpp"
#include "noa/unified/ReduceAxesIwise.hpp"

namespace noa::signal::guts {
    struct CrossCorrelationL2Norm {
        using enable_vectorization = bool;
        using remove_default_final = bool;

        static constexpr void init(const auto& lhs, const auto& rhs, f64& lhs_sum, f64& rhs_sum) {
            lhs_sum += static_cast<f64>(abs_squared(lhs));
            rhs_sum += static_cast<f64>(abs_squared(rhs));
        }

        static constexpr void join(f64 lhs_isum, f64 rhs_isum, f64& lhs_sum, f64& rhs_sum) {
            lhs_sum += lhs_isum;
            rhs_sum += rhs_isum;
        }

        template<typename T>
        static constexpr void final(f64 lhs_sum, f64 rhs_sum, T& lhs_norm, T& rhs_norm) {
            lhs_norm = static_cast<T>(sqrt(lhs_sum));
            rhs_norm = static_cast<T>(sqrt(rhs_sum));
        }
    };

    struct CrossCorrelationScore {
        using enable_vectorization = bool;

        template<typename T>
        static constexpr void init(auto lhs, auto rhs, T& sum) {
            if constexpr (nt::complex<decltype(rhs)>) {
                sum += static_cast<T>(lhs * conj(rhs));
            } else {
                sum += static_cast<T>(lhs * rhs);
            }
        }

        static constexpr void init(const auto& lhs, const auto& rhs, auto lhs_norm, auto rhs_norm, auto& sum) {
            init(lhs / lhs_norm, rhs / rhs_norm, sum);
        }

        static constexpr void join(auto isum, auto& sum) {
            sum += isum;
        }
    };

    template<Correlation MODE>
    struct CrossCorrelationMap {
        template<typename R>
        constexpr void operator()(const Complex<R>& l, const Complex<R>& r, Complex<R>& o) {
            constexpr auto EPSILON = static_cast<R>(1e-13);

            if constexpr (MODE == Correlation::CONVENTIONAL) {
                o = l * conj(r);
            } else if constexpr (MODE == Correlation::PHASE) {
                const Complex<R> product = l * conj(r);
                const R magnitude = abs(product);
                o = product / (magnitude + EPSILON);
                // The epsilon could be scaled by the max(abs(rhs)), but this seems to be useful only
                // for input values close to zero (less than 1e-10). In most cases, this is fine.
            } else if constexpr (MODE == Correlation::DOUBLE_PHASE) {
                const Complex<R> product = l * conj(r);
                const Complex<R> product_sqd = {product.real * product.real, product.imag * product.imag};
                const R magnitude = sqrt(product_sqd.real + product_sqd.imag) + EPSILON;
                o = {(product_sqd.real - product_sqd.imag) / magnitude,
                     (2 * product.real * product.imag) / magnitude};
            } else if constexpr (MODE == Correlation::MUTUAL) {
                const Complex<R> product = l * conj(r);
                const R magnitude_sqrt = sqrt(abs(product));
                o = product / (magnitude_sqrt + EPSILON);
            } else {
                static_assert(nt::always_false<>);
            }
        }
    };

    template<size_t N, bool IS_CENTERED, size_t REGISTRATION_RADIUS_LIMIT,
             nt::readable_nd<N + 1> Input,
             nt::writable_nd_optional<1> PeakCoordinates,
             nt::writable_nd_optional<1> PeakValues>
    struct ReducePeak {
        using input_type = Input;
        using peak_coordinates_type = PeakCoordinates;
        using peak_values_type = PeakValues;
        static_assert(nt::same_mutable_value_type<peak_values_type, input_type>);

        using index_type = nt::index_type_t<input_type>;
        using coord_type = nt::value_type_t<peak_coordinates_type>;
        using value_type = nt::mutable_value_type_t<input_type>;
        static_assert(nt::vec_real_size<coord_type, N>);

        using reduced_type = Pair<value_type, index_type>;
        using index_n_type = Vec<index_type, N>;
        using shape_type = Shape<index_type, N>;

        using subregion_offset_type = Vec<index_type, N>;
        using ellipse_type = noa::geometry::guts::DrawEllipse<N, f32, false>;

    public:
        constexpr ReducePeak(
            const input_type& input,
            const peak_coordinates_type& peak_coordinates,
            const peak_values_type& peak_values,
            const Shape<index_type, N + 1>& shape,
            const index_n_type& registration_radius,
            const index_n_type& subregion_offset,
            const index_n_type& maximum_lag,
            bool apply_ellipse
        ) : m_input(input),
            m_peak_coordinates(peak_coordinates),
            m_peak_values(peak_values),
            m_shape(shape.pop_front()),
            m_batch(shape[0]),
            m_registration_radius(registration_radius),
            m_apply_ellipse(apply_ellipse)
        {
            if (m_apply_ellipse) {
                m_subregion_offset = subregion_offset;
                const auto center = (m_shape.vec / 2).template as<f32>(); // DC position
                constexpr auto cvalue = 1.f;
                constexpr auto is_inverted = false;
                m_ellipse = ellipse_type(center, maximum_lag.template as<f32>(), cvalue, is_inverted);
            }
            NOA_ASSERT(all(registration_radius <= static_cast<index_type>(REGISTRATION_RADIUS_LIMIT)));
        }

    public:
        template<nt::vec_of_size<N + 1> S> // nvcc segfaults if the concept is used with the auto syntax
        constexpr void init(const S& subregion_indices, reduced_type& reduced) const {
            auto batch = subregion_indices[0];
            auto indices = subregion_indices.pop_front();

            f32 mask{1};
            if (m_apply_ellipse) {
                indices += m_subregion_offset;
                mask = m_ellipse(Vec<f32, N>::from_vec(indices));
            }

            if constexpr (not IS_CENTERED)
                indices = noa::fft::ifftshift(indices, m_shape);
            const auto batched_indices = indices.push_front(batch);

            const auto value = m_input(batched_indices) * static_cast<value_type>(mask);
            const auto offset = m_input.offset_at(batched_indices);
            join(reduced_type{value, offset}, reduced);
        }

        static constexpr void join(const reduced_type& current, reduced_type& reduced) {
            // If multiple peaks, the selected peak is one of those, but with no guarantee about which one.
            if (current.first > reduced.first)
                reduced = current;
        }

        constexpr void final(const reduced_type& reduced) { // single-threaded, one thread per batch
            const auto peak_indices = ni::offset2index(reduced.second, m_shape.push_front(m_batch));
            const auto batch = peak_indices[0];

            auto [peak_value, peak_coordinate] = subpixel_registration_using_1d_parabola_(
                m_input[batch], peak_indices.pop_front(), reduced.first);
            if (m_peak_coordinates)
                m_peak_coordinates(batch) = static_cast<coord_type>(peak_coordinate);
            if (m_peak_values)
                m_peak_values(batch) = static_cast<value_type>(peak_value);
        }

    private:
        constexpr auto subpixel_registration_using_1d_parabola_(
            auto input, const index_n_type& peak_indices, const value_type& original_value
        ) {
            Vec<value_type, REGISTRATION_RADIUS_LIMIT * 2 + 1> buffer;

            f64 peak_value{};
            Vec<f64, N> peak_coordinate;
            for (size_t dim{}; dim < N; ++dim) {
                // Reduce the problem to 1d by offsetting to the peak location, except for the current dimension.
                const auto* input_line = input.get();
                for (size_t i{}; i < N; ++i)
                    input_line += (peak_indices[i] * input.strides()[i]) * (dim != i);

                auto peak_radius = m_registration_radius[dim];
                auto peak_window = Span(buffer.data(), peak_radius * 2 + 1);
                auto peak_index = peak_indices[dim];
                const i64 input_size = m_shape[dim];
                const i64 input_stride = static_cast<index_type>(input.strides()[dim]);

                // If non-centered, the peak window can be split across two separate quadrants.
                // As such, retrieve the frequency and if it is a valid frequency, convert back
                // to an index and compute the memory offset.
                const i64 peak_frequency = noa::fft::index2frequency<IS_CENTERED>(peak_index, input_size);
                for (i64 i = -peak_radius, c{}; i <= peak_radius; ++i, ++c) {
                    if (i == 0) {
                        peak_window[c] = original_value;
                        continue;
                    }
                    const i64 frequency = peak_frequency + i;
                    if (-input_size / 2 <= frequency and frequency <= (input_size - 1) / 2) {
                        const i64 index = noa::fft::frequency2index<IS_CENTERED>(frequency, input_size);
                        peak_window[c] = input_line[index * input_stride];
                    }
                }

                if constexpr (not IS_CENTERED)
                    peak_index = noa::fft::fftshift(peak_index, input_size);

                // Subpixel registration.
                if (peak_radius == 1) {
                    auto [x, y] = lstsq_fit_quadratic_vertex_3points(peak_window[0], peak_window[1], peak_window[2]);
                    // Add x directly, since it's relative to peak_index.
                    peak_coordinate[dim] = static_cast<f64>(peak_index) + static_cast<f64>(x);
                    peak_value += static_cast<f64>(y);
                } else {
                    QuadraticCurve<f64> curve = lstsq_fit_quadratic(peak_window.as_const());
                    if (abs(curve.a) < 1e-6) {
                        const f64 x = -curve.b / (2 * curve.a);
                        const f64 y = curve.a * x * x + curve.b * x + curve.c;
                        // x is within [0, size-1], so we need to subtract by peak_radius.
                        peak_coordinate[dim] = static_cast<f64>(peak_index - peak_radius) + x;
                        peak_value += static_cast<f64>(y);
                    } else {
                        peak_coordinate[dim] = static_cast<f64>(peak_index);
                        peak_value += static_cast<f64>(peak_window[peak_radius]);
                    }
                }
            }
            peak_value /= static_cast<f64>(N); // take the average
            return Pair{peak_value, peak_coordinate};
        }

    private:
        input_type m_input;
        peak_coordinates_type m_peak_coordinates;
        peak_values_type m_peak_values;
        shape_type m_shape;
        index_type m_batch;
        index_n_type m_registration_radius;
        ellipse_type m_ellipse;
        subregion_offset_type m_subregion_offset;
        bool m_apply_ellipse;
    };

    template<size_t NDIM, typename Input, typename PeakCoord, typename PeakValue>
    void check_cross_correlation_peak_parameters(
        const Input& xmap,
        const PeakCoord& peak_coordinates = {},
        const PeakValue& peak_values = {}
    ) {
        check(not xmap.is_empty(), "Empty array detected");
        check(not xmap.strides().is_broadcast(), "The cross-correlation map should not be broadcasted");
        check(xmap.shape().ndim() == NDIM,
              "The cross-correlation map(s) shape doesn't match the ndim. Got shape={} and expected ndim={}",
              xmap.shape(), NDIM);

        if constexpr (nt::is_varray_v<PeakCoord>) {
            if (not peak_coordinates.is_empty()) {
                check(ni::is_contiguous_vector(peak_coordinates) and
                      peak_coordinates.n_elements() == xmap.shape()[0],
                      "The number of peak coordinates, specified as a contiguous vector, should be equal to "
                      "the batch size of the cross-correlation map. Got n_peaks={} and batch={}",
                      peak_coordinates.n_elements(), xmap.shape()[0]);
                check(xmap.device() == peak_coordinates.device(),
                      "The cross-correlation map and output peak coordinates must be on the same device, "
                      "but got cross_correlation_map:device={} and peak_coordinates:device={}",
                      xmap.device(), peak_coordinates.device());
            }
        }

        if constexpr (nt::is_varray_v<PeakValue>) {
            if (not peak_values.is_empty()) {
                check(ni::is_contiguous_vector(peak_values) and
                      peak_values.n_elements() == xmap.shape()[0],
                      "The number of peak values, specified as a contiguous vector, should be equal to "
                      "the batch size of the cross-correlation map. Got n_peaks={} and batch={}",
                      peak_values.n_elements(), xmap.shape()[0]);
                check(xmap.device() == peak_values.device(),
                      "The cross-correlation map and output peak values must be on the same device, "
                      "but got cross_correlation_map:device={} and peak_values:device={}",
                      xmap.device(), peak_values.device());
            }
        }
    }
}

namespace noa::signal {
    /// Computes the cross-correlation score(s).
    /// \param[in] lhs      Left-hand side.
    /// \param[in] rhs      Right-hand side.
    /// \param[out] scores  Cross-correlation scores(s). One per batch.
    /// \param normalize    Whether the inputs should be L2-norm normalized before computing the scores.
    template<typename Lhs, typename Rhs, nt::writable_varray_decay Output>
    requires (nt::varray_decay_of_real<Lhs, Rhs, Output> or
              nt::varray_decay_of_complex<Lhs, Rhs, Output>)
    void cross_correlation_score(
        Lhs&& lhs,
        Rhs&& rhs,
        Output&& scores,
        bool normalize = false
    ) {
        check(not lhs.is_empty() and not rhs.is_empty() and not scores.is_empty(), "Empty array detected");
        check(vall(Equal{}, lhs.shape(), rhs.shape()),
              "Inputs should have the same shape, but got lhs:shape={}, rhs:shape={}",
              lhs.shape(), rhs.shape());
        check(lhs.device() == rhs.device() and rhs.device() == scores.device(),
              "The input arrays should be on the same device, but got lhs:device={}, rhs:device={} and  scores:device={}",
              lhs.device(), rhs.device(), scores.device());

        const auto batch = lhs.shape()[0];
        check(ni::is_contiguous_vector(scores) and scores.n_elements() == batch,
              "The number of scores, specified as a contiguous vector, should be equal to the batch size. "
              "Got scores:shape={}, scores:strides={}, and batch={}",
              scores.shape(), scores.strides(), batch);

        using output_t = nt::value_type_t<Output>;
        using lhs_real_t = nt::mutable_value_type_twice_t<Lhs>;
        using rhs_real_t = nt::mutable_value_type_twice_t<Rhs>;
        if (normalize) {
            const auto options = ArrayOption{.device = lhs.device(), .allocator = Allocator::DEFAULT_ASYNC};
            auto lhs_norms = Array<lhs_real_t>({batch, 1, 1, 1}, options);
            auto rhs_norms = Array<rhs_real_t>({batch, 1, 1, 1}, options);
            reduce_axes_ewise(
                wrap(std::forward<Lhs>(lhs), std::forward<Rhs>(rhs)),
                wrap(f64{}, f64{}),
                wrap(lhs_norms.view(), rhs_norms.view()),
                guts::CrossCorrelationL2Norm{});
            reduce_axes_ewise(
                wrap(std::forward<Lhs>(lhs), std::forward<Rhs>(rhs), std::move(lhs_norms), std::move(rhs_norms)),
                output_t{},
                std::forward<Output>(scores).flat(0),
                guts::CrossCorrelationScore{});
        } else {
            reduce_axes_ewise(
                wrap(std::forward<Lhs>(lhs), std::forward<Rhs>(rhs)),
                output_t{},
                std::forward<Output>(scores).flat(0),
                guts::CrossCorrelationScore{});
        }
    }

    /// Computes the cross-correlation score.
    /// \param[in] lhs      Left-hand side.
    /// \param[in] rhs      Right-hand side.
    /// \param normalize    Whether the inputs should be L2-norm normalized before computing the score.
    template<typename Lhs, typename Rhs>
    requires (nt::varray_decay_of_real<Lhs, Rhs> or nt::varray_decay_of_complex<Lhs, Rhs>)
    [[nodiscard]] auto cross_correlation_score(Lhs&& lhs, Rhs&& rhs, bool normalize = false) {
        check(not lhs.is_empty() and not rhs.is_empty(), "Empty array detected");
        check(vall(Equal{}, lhs.shape(), rhs.shape()) and not rhs.shape().is_batched(),
              "Arrays should have the same shape and should not be batched, but got lhs:shape={}, rhs:shape={}",
              lhs.shape(), rhs.shape());
        check(lhs.device() == rhs.device(),
              "The lhs and rhs input arrays should be on the same device, but got lhs:device={} and rhs:device={}",
              lhs.device(), rhs.device());

        using output_t = std::conditional_t<nt::complex<Lhs>, c64, f64>;
        using lhs_real_t = nt::mutable_value_type_twice_t<Lhs>;
        using rhs_real_t = nt::mutable_value_type_twice_t<Rhs>;

        output_t score;
        if (normalize) {
            lhs_real_t lhs_norm;
            rhs_real_t rhs_norm;
            reduce_ewise(
                wrap(std::forward<Lhs>(lhs), std::forward<Rhs>(rhs)),
                wrap(f64{}, f64{}),
                wrap(lhs_norm, rhs_norm),
                guts::CrossCorrelationL2Norm{});
            reduce_ewise(
                wrap(std::forward<Lhs>(lhs), std::forward<Rhs>(rhs), lhs_norm, rhs_norm),
                output_t{},
                score,
                guts::CrossCorrelationScore{});
        } else {
            reduce_ewise(
                wrap(std::forward<Lhs>(lhs), std::forward<Rhs>(rhs)),
                output_t{},
                score,
                guts::CrossCorrelationScore{});
        }
        return score;
    }

    struct CrossCorrelationMapOptions {
        /// Correlation mode to use. Remember that DOUBLE_PHASE_CORRELATION doubles the lags/shifts.
        Correlation mode = Correlation::CONVENTIONAL;

        /// Normalization mode to use for the C2R transform producing the final output.
        /// This should match the mode that was used to compute the input transforms.
        noa::fft::Norm ifft_norm = noa::fft::NORM_DEFAULT;

        /// Whether the C2R transform should be cached.
        bool ifft_cache_plan{true};
    };

    /// Computes the cross-correlation map.
    /// \tparam REMAP       Whether the output map should be centered. Should be H2F or H2FC.
    /// \param[in] lhs      Non-centered rFFT of the signal to cross-correlate.
    /// \param[in,out] rhs  Non-centered rFFT of the signal to cross-correlate.
    ///                     Overwritten by default (see \p buffer).
    /// \param[out] output  Cross-correlation map.
    ///                     If \p REMAP is H2F, the zero lag is at {n, 0, 0, 0}.
    ///                     If \p REMAP is H2FC, the zero lag is at {n, shape[1]/2, shape[2]/2, shape[3]/2}.
    /// \param options      Correlation mode and ifft options.
    /// \param[out] buffer  Buffer of the same shape as the inputs (no broadcasting allowed). It is overwritten.
    ///                     Can be \p lhs or \p rhs. If empty, use \p rhs instead (the default).
    ///
    /// \note As mentioned above, this function takes the rFFT of the real inputs to correlate.
    ///       The score with zero lag can be computed more efficiently with the cross_correlation_score function.
    ///       If other lags are to be selected (which is the entire point of this function), the inputs
    ///       should be zero-padded before taking the rFFT to cancel the circular convolution effect of
    ///       the DFT. The amount of padding along a dimension is equal to the maximum lag allowed.
    ///
    /// \note As opposed to the cross_correlation_score function, this function does not take a normalization flag.
    ///       To get the normalized cross-correlation scores, simply L2 normalize the inputs before computing their
    ///       rFFT. To output scores with the correct scaling, use .ifft_norm=ORTHO|BACKWARD (using noa::fft::FORWARD
    ///       outputs scores divided by a scaling factor of 1/n_elements).
    template<Remap REMAP, typename Lhs, typename Rhs, typename Output,
             typename Buffer = View<nt::mutable_value_type_t<Rhs>>>
    requires(nt::varray_decay_of_complex<Lhs, Rhs, Buffer> and
             nt::varray_decay_of_almost_same_type<Lhs, Rhs, Buffer> and
             nt::writable_varray_decay<Rhs, Buffer> and
             nt::writable_varray_decay_of_any<Output, nt::value_type_twice_t<Rhs>> and
             (REMAP == Remap::H2F or REMAP == Remap::H2FC))
    void cross_correlation_map(
        Lhs&& lhs, Rhs&& rhs, Output&& output,
        const CrossCorrelationMapOptions& options = {},
        Buffer&& buffer = {}
    ) {
        const Device device = output.device();
        const auto expected_shape = output.shape().rfft();
        check(not lhs.is_empty() and not rhs.is_empty() and not output.is_empty(), "Empty array detected");
        check(device == lhs.device() and device == rhs.device(),
              "The lhs, rhs and output arrays must be on the same device, "
              "but got lhs:device={}, rhs:device={} and output:device={}",
              lhs.device(), rhs.device(), device);

        using complex_t = nt::value_type_t<Buffer>;
        View<complex_t> tmp;
        if (buffer.is_empty()) {
            check(ni::are_elements_unique(rhs.strides(), expected_shape),
                  "Since no temporary buffer is passed, the rhs input is used as buffer, "
                  "thus should have unique elements (e.g. no broadcasting) with a shape of {}, "
                  "but got rhs:shape={}, rhs:strides={}", expected_shape, rhs.shape(), rhs.strides());
            tmp = rhs.view();
        } else {
            check(device == buffer.device(),
                  "The temporary and output arrays must be on the same device, buffer:device={} and output:device={}",
                  buffer.device(), device);
            check(vall(Equal{}, buffer.shape(), expected_shape) and
                  ni::are_elements_unique(buffer.strides(), buffer.shape()),
                  "Given an output map of shape {}, the buffer should be of shape {} and have unique elements, "
                  "but got buffer:shape={}, buffer:strides={}",
                  output.shape(), expected_shape, buffer.shape(), buffer.strides());
            tmp = buffer.view();
        }

        // TODO Add normalization with auto-correlation?
        //      IMO it's always simpler to normalize the real inputs,
        //      so not sure how useful this would be.
        switch (options.mode) {
            case Correlation::CONVENTIONAL:
                ewise(wrap(std::forward<Lhs>(lhs), rhs), tmp,
                      guts::CrossCorrelationMap<Correlation::CONVENTIONAL>{});
                break;
            case Correlation::PHASE:
                ewise(wrap(std::forward<Lhs>(lhs), rhs), tmp,
                      guts::CrossCorrelationMap<Correlation::PHASE>{});
                break;
            case Correlation::DOUBLE_PHASE:
                ewise(wrap(std::forward<Lhs>(lhs), rhs), tmp,
                      guts::CrossCorrelationMap<Correlation::DOUBLE_PHASE>{});
                break;
            case Correlation::MUTUAL:
                ewise(wrap(std::forward<Lhs>(lhs), rhs), tmp,
                      guts::CrossCorrelationMap<Correlation::MUTUAL>{});
                break;
        }

        if constexpr (REMAP == Remap::H2FC) {
            using real_t = nt::value_type_t<complex_t>;
            const Shape4<i64> shape = output.shape();
            if (shape.ndim() == 3) {
                phase_shift_3d<Remap::H2H>(tmp, tmp, shape, (shape.pop_front<1>() / 2).vec.as<real_t>());
            } else {
                phase_shift_2d<Remap::H2H>(tmp, tmp, shape, (shape.pop_front<2>() / 2).vec.as<real_t>());
            }
        }

        if (buffer.is_empty())
            noa::fft::c2r(std::forward<Rhs>(rhs), std::forward<Output>(output),
                          {.norm = options.ifft_norm, .cache_plan = options.ifft_cache_plan});
        else {
            noa::fft::c2r(std::forward<Buffer>(buffer), std::forward<Output>(output),
                          {.norm = options.ifft_norm, .cache_plan = options.ifft_cache_plan});
        }
    }

    template<size_t N>
    struct CrossCorrelationPeakOptions {
        /// ((D)H)W radius of the registration window, centered on the peak.
        Vec<i64, N> registration_radius{Vec<i64, N>::from_value(1)};

        /// ((D)H)W maximum lag allowed, i.e. the peak is selected within this elliptical radius.
        /// If negative or 0, it is ignored. Otherwise, an elliptical mask is applied on the
        /// centered cross-correlation map before selecting the peak.
        Vec<f64, N> maximum_lag{Vec<f64, N>::from_value(-1)};
    };

    /// Find the cross-correlation peak(s) of the cross-correlation map(s).
    /// \tparam REMAP                       Whether \p xmap is centered. Should be F2F or FC2FC.
    /// \param[in] cross_correlation_map    1d, 2d or 3d cross-correlation map.
    /// \param[out] peak_coordinates        Output ((D)H)W coordinate of the highest peak. One per batch or empty.
    /// \param[out] peak_values             Output value of the highest peak. One per batch or empty.
    /// \param options                      Picking and registration options.
    template<Remap REMAP, size_t N,
             nt::readable_varray_decay_of_almost_any<f32, f64> Input,
             nt::writable_varray_decay_of_any<Vec<f32, N>, Vec<f64, N>> PeakCoord = View<Vec<f64, N>>,
             nt::writable_varray_decay_of_almost_same_type<Input> PeakValue = View<nt::mutable_value_type_t<Input>>>
    requires (1 <= N and N <= 3)
    void cross_correlation_peak(
        Input&& cross_correlation_map,
        PeakCoord&& peak_coordinates,
        PeakValue&& peak_values = {},
        const CrossCorrelationPeakOptions<N>& options = {}
    ) {
        constexpr size_t REGISTRATION_RADIUS_LIMIT = 4;
        guts::check_cross_correlation_peak_parameters<N>(cross_correlation_map, peak_coordinates, peak_values);
        check(all(options.registration_radius > 0 and options.registration_radius <= REGISTRATION_RADIUS_LIMIT),
              "The registration radius should be a small positive value (less than {}), but got {}",
              REGISTRATION_RADIUS_LIMIT, options.registration_radius);

        const Device& device = cross_correlation_map.device();
        const auto shape = cross_correlation_map.shape();
        auto filter_nd = [&shape](auto v) {
            if constexpr (N == 1) return v.filter(0, shape[2] > 1 ? 2 : 3);
            if constexpr (N == 2) return v.filter(0, 2, 3);
            if constexpr (N == 3) return v;
        };

        using value_t = nt::mutable_value_type_t<Input>;
        using index_t = nt::index_type_t<Input>;
        using input_accessor_t = AccessorRestrictI64<const value_t, N + 1>;
        using peak_values_accessor_t = AccessorRestrictContiguousI64<value_t, 1>;
        using peak_coordinates_accessor_t = AccessorRestrictContiguousI64<nt::value_type_t<PeakCoord>, 1>;

        auto input_accessor = input_accessor_t(cross_correlation_map.get(), filter_nd(cross_correlation_map.strides()));
        auto peak_values_accessor = peak_values_accessor_t(peak_values.get());
        auto peak_coordinates_accessor = peak_coordinates_accessor_t(peak_coordinates.get());
        auto shape_nd = filter_nd(cross_correlation_map.shape());
        auto initial_reduction_value = Pair{std::numeric_limits<value_t>::min(), index_t{}};

        auto reduce_axes = ReduceAxes::all();
        reduce_axes[3 - N] = false; // batch equivalent

        bool apply_ellipse{};
        auto maximum_allowed_lag = shape_nd.vec.pop_front() / 2;
        Vec<index_t, N> maximum_lag;
        Vec<index_t, N> subregion_offset;
        Shape<index_t, N> subregion_shape;
        for (size_t i{}; i < N; ++i) {
            if (options.maximum_lag[i] <= 0) {
                maximum_lag[i] = maximum_allowed_lag[i];
            } else {
                auto lag_index = static_cast<index_t>(ceil(options.maximum_lag[i]));
                maximum_lag[i] = min(maximum_allowed_lag[i], lag_index);
                apply_ellipse = true;
            }

            // The reduction is done within the subregion that goes up to the maximum lag
            // (and centered on the original map center, of course).
            subregion_offset[i] = maximum_allowed_lag[i] - maximum_lag[i];
            subregion_shape[i] = min(maximum_lag[i] * 2 + 1, shape_nd[i + 1]);
        }

        using reducer_t = guts::ReducePeak<
            N, REMAP.is_xc2xx(), REGISTRATION_RADIUS_LIMIT,
            input_accessor_t, peak_coordinates_accessor_t, peak_values_accessor_t>;

        reduce_axes_iwise(
            subregion_shape.push_front(shape[0]), device, initial_reduction_value, reduce_axes,
            reducer_t(input_accessor, peak_coordinates_accessor, peak_values_accessor,
                      shape_nd, options.registration_radius,
                      subregion_offset, maximum_lag, apply_ellipse)
        );
    }

    template<Remap REMAP, size_t N, nt::readable_varray_decay_of_almost_any<f32, f64> Input>
    auto cross_correlation_peak(
        const Input& cross_correlation_map,
        const CrossCorrelationPeakOptions<N>& options = {}
    ) {
        using value_t = nt::mutable_value_type_t<Input>;
        using coord_t = Vec<f64, N>;
        using pair_t = Pair<coord_t, value_t>;
        if (cross_correlation_map.device().is_cpu()) {
            pair_t pair{};
            cross_correlation_peak<REMAP, N>(cross_correlation_map.view(), View(&pair.first), View(&pair.second), options);
            cross_correlation_map.eval();
            return pair;
        } else {
            const auto array_options = ArrayOption{cross_correlation_map.device(), Allocator::ASYNC};
            Array pair = noa::empty<pair_t>(1, array_options);
            cross_correlation_peak<REMAP, N>(
                cross_correlation_map.view(),
                View(&(pair.get()->first), 1, array_options),
                View(&(pair.get()->second), 1, array_options),
                options);
            return pair.first();
        }
    }

    template<Remap REMAP, typename Input,
             typename PeakCoord = View<Vec1<f32>>,
             typename PeakValue = View<nt::mutable_value_type_t<Input>>>
    void cross_correlation_peak_1d(
        const Input& xmap,
        const PeakCoord& peak_coordinates,
        const PeakValue& peak_values = {},
        const CrossCorrelationPeakOptions<1>& options = {}
    ) {
        cross_correlation_peak<REMAP>(xmap, peak_coordinates, peak_values, options);
    }
    template<Remap REMAP, typename Input,
             nt::varray_decay PeakCoord = View<Vec2<f32>>,
             nt::varray_decay PeakValue = View<nt::mutable_value_type_t<Input>>>
    void cross_correlation_peak_2d(
        const Input& xmap,
        const PeakCoord& peak_coordinates,
        const PeakValue& peak_values = {},
        const CrossCorrelationPeakOptions<2>& options = {}
    ) {
        cross_correlation_peak<REMAP>(xmap, peak_coordinates, peak_values, options);
    }
    template<Remap REMAP, typename Input,
             typename PeakCoord = View<Vec3<f32>>,
             typename PeakValue = View<nt::mutable_value_type_t<Input>>>
    void cross_correlation_peak_3d(
        const Input& xmap,
        const PeakCoord& peak_coordinates,
        const PeakValue& peak_values = {},
        const CrossCorrelationPeakOptions<3>& options = {}
    ) {
        cross_correlation_peak<REMAP>(xmap, peak_coordinates, peak_values, options);
    }

    template<Remap REMAP, typename Input>
    auto cross_correlation_peak_1d(const Input& xmap, const CrossCorrelationPeakOptions<1>& options = {}) {
        return cross_correlation_peak<REMAP>(xmap, options);
    }
    template<Remap REMAP, typename Input>
    auto cross_correlation_peak_2d(const Input& xmap, const CrossCorrelationPeakOptions<2>& options = {}) {
        return cross_correlation_peak<REMAP>(xmap, options);
    }
    template<Remap REMAP, typename Input>
    auto cross_correlation_peak_3d(const Input& xmap, const CrossCorrelationPeakOptions<3>& options = {}) {
        return cross_correlation_peak<REMAP>(xmap, options);
    }
}
