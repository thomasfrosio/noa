#pragma once

#include "noa/base/Complex.hpp"
#include "noa/base/Pair.hpp"
#include "noa/base/Vec.hpp"
#include "noa/fft/Transform.hpp"
#include "noa/runtime/Array.hpp"
#include "noa/runtime/ReduceAxesEwise.hpp"
#include "noa/runtime/ReduceAxesIwise.hpp"
#include "noa/runtime/ReduceEwise.hpp"
#include "noa/signal/core/Correlation.hpp"
#include "noa/signal/core/LeastSquare.hpp"
#include "noa/signal/PhaseShift.hpp"
#include "noa/xform/core/Draw.hpp"

namespace noa::signal::details {
    struct CrossCorrelationScore {
        using enable_vectorization = bool;

        template<nt::real_or_complex I, nt::real T>
        constexpr void operator()(const I& x, const I& y, T& sum_xy) {
            if constexpr (nt::complex<I>) {
                sum_xy += static_cast<T>(x * conj(y));
            } else {
                sum_xy += static_cast<T>(x * y);
            }
        }

        template<nt::real T>
        static constexpr void join(const T& isum_xy, T& sum_xy) {
            sum_xy += isum_xy;
        }
    };

    struct CrossCorrelationScoreNormalized {
        using enable_vectorization = bool;
        using remove_default_final = bool;

        template<typename I, typename U, nt::real T>
           requires (nt::real<I, U> or nt::complex<I, U>)
        constexpr void operator()(const I& x, const I& y, T& sum_xx, T& sum_yy, U& sum_xy) {
            sum_xx += static_cast<T>(abs_squared(x));
            sum_yy += static_cast<T>(abs_squared(y));
            if constexpr (nt::complex<I>) {
                sum_xy += static_cast<U>(x * conj(y));
            } else {
                sum_xy += static_cast<U>(x * y);
            }
        }

        template<nt::real T, nt::real_or_complex U>
        static constexpr void join(const T& isum_xx, const T& isum_yy, const U& isum_xy, T& sum_xx, T& sum_yy, U& sum_xy) {
            sum_xx += isum_xx;
            sum_yy += isum_yy;
            sum_xy += isum_xy;
        }

        template<nt::real T, nt::real_or_complex U, typename V>
        static constexpr void post(const T& sum_xx, const T& sum_yy, const U& sum_xy, V& ncc) {
            const auto denom = sum_xx * sum_yy;
            if (denom <= 0) {
                ncc = V{};
            } else {
                ncc = static_cast<V>(sum_xy / sqrt(denom));
            }
        }
    };

    template<typename S>
    struct CrossCorrelationScoreCentered {
        using enable_vectorization = bool;
        using remove_default_post = bool;
        S size;

        template<typename I, typename T>
            requires (nt::real<I, T> or nt::complex<I, T>)
        constexpr void operator()(const I& x, const I& y, T& sum_x, T& sum_y, T& sum_xy) {
            sum_x += static_cast<T>(x);
            sum_y += static_cast<T>(y);
            if constexpr (nt::complex<I>) {
                sum_xy += static_cast<T>(x * conj(y));
            } else {
                sum_xy += static_cast<T>(x * y);
            }
        }

        template<nt::real_or_complex T>
        static constexpr void join(const T& isum_x, const T& isum_y, const T& isum_xy, T& sum_x, T& sum_y, T& sum_xy) {
            sum_x += isum_x;
            sum_y += isum_y;
            sum_xy += isum_xy;
        }

        template<nt::real_or_complex T, typename U>
        constexpr void post(const T& sum_x, const T& sum_y, const T& sum_xy, U& zcc) {
            if constexpr (nt::complex<U>)
                zcc = static_cast<U>(sum_xy - (sum_x * conj(sum_y)) / size);
            else
                zcc = static_cast<U>(sum_xy - (sum_x * sum_y) / size);
        }
    };

    template<typename S>
    struct CrossCorrelationScoreCenteredNormalized {
        using enable_vectorization = bool;
        using remove_default_post = bool;
        S size;

        template<typename I, typename U, nt::real T>
           requires (nt::real<I, U> or nt::complex<I, U>)
        constexpr void operator()(const I& x, const I& y, T& sum_xx, T& sum_yy, U& sum_x, U& sum_y, U& sum_xy) {
            sum_xx += static_cast<T>(abs_squared(x));
            sum_yy += static_cast<T>(abs_squared(y));
            sum_x += static_cast<U>(x);
            sum_y += static_cast<U>(y);
            if constexpr (nt::complex<I>) {
                sum_xy += static_cast<U>(x * conj(y));
            } else {
                sum_xy += static_cast<U>(x * y);
            }
        }

        template<nt::real T, nt::real_or_complex U>
        static constexpr void join(
            const T& isum_xx, const T& isum_yy, const U& isum_x, const U& isum_y, const U& isum_xy,
            T& sum_xx, T& sum_yy, U& sum_x, U& sum_y, U& sum_xy
        ) {
            sum_xx += isum_xx;
            sum_yy += isum_yy;
            sum_x += isum_x;
            sum_y += isum_y;
            sum_xy += isum_xy;
        }

        template<nt::real T, nt::real_or_complex U, typename V>
        constexpr void post(
            const T& sum_xx, const T& sum_yy, const U& sum_x, const U& sum_y, const U& sum_xy, V& zncc
        ) {
            const auto denom_x = sum_xx - abs_squared(sum_x) / size;
            const auto denom_y = sum_yy - abs_squared(sum_y) / size;
            auto denom = denom_x * denom_y;
            if (denom <= 0) {
                zncc = V{};
            } else {
                U num;
                if constexpr (nt::complex<U>)
                    num = sum_xy - sum_x * conj(sum_y) / size;
                else
                    num = sum_xy - sum_x * sum_y / size;
                denom = sqrt(denom);
                zncc = static_cast<V>(num / denom);
            }
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
                static_assert(nt::always_false<R>);
            }
        }
    };

    template<bool IS_CENTERED, usize REGISTRATION_RADIUS_LIMIT, typename T, nt::sinteger I, usize R>
    constexpr auto subpixel_registration_using_1d_parabola(
        auto input,
        const Shape<I, R>& shape,
        const Vec<i32, R>& registration_radius,
        const Vec<I, R>& peak_indices,
        const T& original_value
    ) {
        Vec<T, REGISTRATION_RADIUS_LIMIT * 2 + 1> buffer;

            f64 peak_value{};
            Vec<f64, N> peak_coordinate;
            for (usize dim{}; dim < R; ++dim) {
                // Reduce the problem to 1d by offsetting to the peak location, except for the current dimension.
                const auto* input_line = input.get();
                for (usize i{}; i < R; ++i)
                    input_line += (peak_indices[i] * input.strides()[i]) * (dim != i);

                auto peak_radius = static_cast<I>(registration_radius[dim]);
                auto peak_window = Span(buffer.data(), peak_radius * 2 + 1);
                auto peak_index = peak_indices[dim];
                const auto input_size = shape[dim];
                const auto input_stride = static_cast<I>(input.strides()[dim]);

                // If non-centered, the peak window can be split across two separate quadrants.
                // As such, retrieve the frequency and if it is a valid frequency, convert back
                // to an index and compute the memory offset.
                const auto peak_frequency = nf::index2frequency<IS_CENTERED>(peak_index, input_size);
                for (I i = -peak_radius, c{}; i <= peak_radius; ++i, ++c) {
                    if (i == 0) {
                        peak_window[c] = original_value;
                        continue;
                    }
                    const auto frequency = peak_frequency + i;
                    if (-input_size / 2 <= frequency and frequency <= (input_size - 1) / 2) {
                        const auto index = nf::frequency2index<IS_CENTERED>(frequency, input_size);
                        peak_window[c] = input_line[index * input_stride];
                    }
                }

                if constexpr (not IS_CENTERED)
                    peak_index = nf::fftshift(peak_index, input_size);

                // Subpixel registration.
                if (peak_radius == 1) {
                    auto [x, y] = lstsq_fit_quadratic_vertex_3points(peak_window[0], peak_window[1], peak_window[2]);
                    // Add x directly, since it's relative to peak_index.
                    peak_coordinate[dim] = static_cast<f64>(peak_index) + static_cast<f64>(x);
                    peak_value += static_cast<f64>(y);
                } else if (peak_radius == 0) {
                    // No registration, just save the value and index.
                    peak_coordinate[dim] = static_cast<f64>(peak_index);
                    peak_value += static_cast<f64>(peak_window[0]);
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
            peak_value /= static_cast<f64>(R); // take the average
            return Pair{peak_value, peak_coordinate};
    }

    template<usize B, usize R, bool IS_CENTERED, usize REGISTRATION_RADIUS_LIMIT,
             nt::readable_nd<B + R> Input,
             nt::writable_nd_optional<R> PeakCoordinates,
             nt::writable_nd_optional<R> PeakValues>
    struct PeakRegistration {
        using input_type = Input;
        using peak_coordinates_type = PeakCoordinates;
        using peak_values_type = PeakValues;
        static_assert(nt::same_mutable_value_type<peak_values_type, input_type>);

        using index_type = nt::index_type_t<input_type>;
        using coord_type = nt::value_type_t<peak_coordinates_type>;
        using value_type = nt::mutable_value_type_t<input_type>;
        static_assert(nt::vec_real_size<coord_type, R>);

        using index_n_type = Vec<index_type, R>;
        using shape_type = Shape<index_type, R>;

    public:
        constexpr PeakRegistration(
            const input_type& input,
            const peak_coordinates_type& peak_coordinates,
            const peak_values_type& peak_values,
            const Shape<index_type, R>& shape,
            const Vec<i32, R>& registration_radius
        ) : m_input_bn(input),
            m_peak_coordinates(peak_coordinates),
            m_peak_values(peak_values),
            m_shape_n(shape),
            m_registration_radius(registration_radius)
        {
            NOA_ASSERT(registration_radius <= static_cast<index_type>(REGISTRATION_RADIUS_LIMIT));
        }

        constexpr void operator()(const Vec<index_type, B>& batches) const {
            const auto peak_indices = IS_CENTERED ? m_shape_n.vec / 2 : index_n_type{0};
            const auto input_n = m_input_bn[batches];
            const auto central_value = input_n(peak_indices);
            const auto [peak_value, peak_coordinate] =
                subpixel_registration_using_1d_parabola<IS_CENTERED, REGISTRATION_RADIUS_LIMIT>(
                    input_n, m_shape_n, m_registration_radius, peak_indices, central_value);

            if (m_peak_coordinates)
                m_peak_coordinates(batches) = static_cast<coord_type>(peak_coordinate);
            if (m_peak_values)
                m_peak_values(batches) = static_cast<value_type>(peak_value);
        }

    private:
        input_type m_input_bn;
        peak_coordinates_type m_peak_coordinates;
        peak_values_type m_peak_values;
        shape_type m_shape_n;
        Vec<i32, R> m_registration_radius;
    };

    template<usize B, usize R, bool IS_CENTERED, usize REGISTRATION_RADIUS_LIMIT,
             nt::readable_nd<B + R> Input,
             nt::writable_nd_optional<B> PeakCoordinates,
             nt::writable_nd_optional<B> PeakValues>
    struct ReducePeak {
        using input_type = Input;
        using peak_coordinates_type = PeakCoordinates;
        using peak_values_type = PeakValues;
        static_assert(nt::same_mutable_value_type<peak_values_type, input_type>);

        using index_type = nt::index_type_t<input_type>;
        using coord_type = nt::value_type_t<peak_coordinates_type>;
        using value_type = nt::mutable_value_type_t<input_type>;
        static_assert(nt::vec_real_size<coord_type, R>);

        using reduced_type = Pair<value_type, index_type>;
        using index_n_type = Vec<index_type, R>;
        using shape_type = Shape<index_type, R>;

        using subregion_offset_type = Vec<index_type, R>;
        using ellipse_type = nx::DrawEllipse<R, f32, false>;

    public:
        constexpr ReducePeak(
            const input_type& input,
            const peak_coordinates_type& peak_coordinates,
            const peak_values_type& peak_values,
            const Shape<index_type, R>& shape_b,
            const Shape<index_type, R>& shape_r,
            const Vec<i32, R>& registration_radius,
            const index_n_type& subregion_offset,
            const index_n_type& maximum_lag,
            bool apply_ellipse
        ) : m_input(input),
            m_peak_coordinates(peak_coordinates),
            m_peak_values(peak_values),
            m_shape(shape_r),
            m_batch(shape_b),
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
            NOA_ASSERT(registration_radius <= static_cast<index_type>(REGISTRATION_RADIUS_LIMIT));
        }

    public:
        constexpr void operator()(const Vec<index_type, B + R>& subregion_indices, reduced_type& reduced) const {
            auto [batches, indices] = subregion_indices.template split<B>();

            f32 mask{1};
            if (m_apply_ellipse) {
                indices += m_subregion_offset;
                mask = m_ellipse.draw_at(indices.template as<f32>());
            }

            if constexpr (not IS_CENTERED)
                indices = nf::ifftshift(indices, m_shape);
            const auto batched_indices = indices.push_front(batches);

            const auto value = m_input(batched_indices) * static_cast<value_type>(mask);
            const auto offset = m_input.offset_at(batched_indices);
            join(reduced_type{value, offset}, reduced);
        }

        static constexpr void join(const reduced_type& current, reduced_type& reduced) {
            // If equal peaks, select one of those with no guarantee about which one.
            if (current.first > reduced.first)
                reduced = current;
        }

        constexpr void post(const reduced_type& reduced) { // single-threaded, one thread per batch
            const auto shape = m_batch.push_back(m_shape);
            const auto peak_indices = noa::offset2index(reduced.second, m_input.strides(), shape);
            const auto [batches, indices] = peak_indices.template split<B>();

            auto [peak_value, peak_coordinate] =
                subpixel_registration_using_1d_parabola<IS_CENTERED, REGISTRATION_RADIUS_LIMIT>(
                    m_input[batches], m_shape, m_registration_radius, peak_indices, reduced.first);

            if (m_peak_coordinates)
                m_peak_coordinates(batches) = static_cast<coord_type>(peak_coordinate);
            if (m_peak_values)
                m_peak_values(batches) = static_cast<value_type>(peak_value);
        }

    private:
        input_type m_input;
        peak_coordinates_type m_peak_coordinates;
        peak_values_type m_peak_values;
        shape_type m_shape;
        index_type m_batch;
        Vec<i32, R> m_registration_radius;
        ellipse_type m_ellipse;
        subregion_offset_type m_subregion_offset;
        bool m_apply_ellipse;
    };

    template<typename Lhs, typename Rhs, typename Output>
    concept cross_correlation_scores =
        nt::writable_array_decay<Output> and
        (nt::array_decay_of_real<Lhs, Rhs, Output> or nt::array_decay_of_complex<Lhs, Rhs, Output>) and
        nt::array_decay_with_same_nd<Lhs, Rhs> and
        nt::array_decay_between_nd<Output, 1, nt::array_size_v<Lhs>>;

    template<nf::Layout REMAP, typename Lhs, typename Rhs, typename Output, typename Buffer = Empty,
             typename BufferOrEmpty = std::conditional_t<nt::empty<Empty>, Rhs, Buffer>>
    concept cross_correlation_mapable = nt::array_decay_of_complex<Lhs, Rhs, BufferOrEmpty> and
         nt::array_decay_of_almost_same_type<Lhs, Rhs, BufferOrEmpty> and
         nt::writable_array_decay<Rhs, BufferOrEmpty> and
         nt::writable_array_decay_of_any<Output, nt::value_type_twice_t<Rhs>> and
         nt::array_decay_with_same_nd<Lhs, Rhs, BufferOrEmpty, Output> and
         (REMAP == nf::Layout::H2F or REMAP == nf::Layout::H2FC);

    template<typename PeakCoord, typename PeakValue, usize B, usize R, typename InputValue>
    concept cross_correlation_peaks_value_or_coord =
        nt::array_size_v<PeakCoord> == nt::array_size_v<PeakValue> and (
        nt::empty<PeakCoord> or (
            nt::array_nd<PeakCoord, B> and
            nt::any_of<nt::value_type_t<PeakCoord>, Vec<f32, R>, Vec<f64, R>>
         )) and (
        nt::empty<PeakValue> or (
            nt::array_nd<PeakValue, B> and
            nt::any_of<nt::value_type_t<PeakValue>, InputValue>
        ));

    template<nf::Layout REMAP, usize R, typename Input, typename PeakCoord, typename PeakValue, usize N = nt::array_size_v<Input>>
    concept cross_correlation_peaks_able =
        (1 <= R and R <= 3) and
        (REMAP == nf::Layout::F2F or REMAP == nf::Layout::FC2FC) and
        nt::readable_array_decay_of_any<Input, f32, f64> and (N > R) and
        cross_correlation_peaks_value_or_coord<
            std::decay_t<PeakCoord>, std::decay_t<PeakValue>, N - R, R, nt::mutable_value_type_t<Input>>;
}

namespace noa::signal {
    struct CrossCorrelationScoreOptions {
        /// Whether the inputs should be zero-centered.
        bool center = false;

        /// Whether the inputs should be L2-normalized.
        bool normalize = false;

        // TODO Add accurate?
    };

    /// Computes the (zero-)(normalized-)cross-correlation scores.
    /// \param[in] lhs      (B..R..) Left-hand side.
    /// \param[in] rhs      (B..R..) Right-hand side.
    /// \param[out] scores  (B..) Cross-correlation scores.
    /// \param options      Options for the reduction.
    /// \note The reduction is done using double-precision.
    /// \note This uses reduce_axes_ewise which may enforce certain contiguity restrictions on the arrays.
    template<typename Lhs, typename Rhs, typename Output>
        requires details::cross_correlation_scores<Lhs, Rhs, Output>
    void cross_correlation_scores(
        Lhs&& lhs,
        Rhs&& rhs,
        Output&& scores,
        const CrossCorrelationScoreOptions& options = {}
    ) {
        check(nd::are_arrays_valid(lhs, rhs, scores), "Empty array detected");
        check(lhs.shape() == rhs.shape(),
              "Inputs should have the same shape, but got lhs:shape={}, rhs:shape={}",
              lhs.shape(), rhs.shape());
        check(lhs.device() == rhs.device() and rhs.device() == scores.device(),
              "The input arrays should be on the same device, but got lhs:device={}, rhs:device={} and scores:device={}",
              lhs.device(), rhs.device(), scores.device());

        constexpr auto N = nt::array_size_v<Lhs>;
        constexpr auto B = nt::array_size_v<Output>;
        const auto [shape_b, shape_n] = rhs.shape().template split<B>();
        check(scores.is_contiguous() and scores.shape() == shape_b,
              "The output scores, specified as a contiguous array, should match the batch dimensions of the inputs. Got scores:shape={}, scores:strides={}, and input:batches={}",
              scores.shape(), scores.strides(), shape_b);

        using value_t = nt::mutable_value_type_t<Lhs>;
        using sum_t = nt::double_precision_t<value_t>;
        using real_t = f64;
        if (options.center and options.normalize) {
            const auto n_elements = static_cast<real_t>(shape_n.n_elements());
            reduce_axes_ewise(
                wrap(std::forward<Lhs>(lhs), std::forward<Rhs>(rhs)),
                wrap(real_t{}, real_t{}, sum_t{}, sum_t{}, sum_t{}),
                std::forward<Output>(scores),
                details::CrossCorrelationScoreCenteredNormalized{n_elements}
            );
        } else if (options.normalize) {
            reduce_axes_ewise(
                wrap(std::forward<Lhs>(lhs), std::forward<Rhs>(rhs)),
                wrap(real_t{}, real_t{}, sum_t{}),
                std::forward<Output>(scores),
                details::CrossCorrelationScoreNormalized{}
            );
        } else if (options.center) {
            const auto n_elements = static_cast<real_t>(shape_n.n_elements());
            reduce_axes_ewise(
                wrap(std::forward<Lhs>(lhs), std::forward<Rhs>(rhs)),
                wrap(sum_t{}, sum_t{}, sum_t{}),
                std::forward<Output>(scores),
                details::CrossCorrelationScoreCentered{n_elements}
            );
        } else {
            reduce_axes_ewise(
                wrap(std::forward<Lhs>(lhs), std::forward<Rhs>(rhs)),
                sum_t{},
                std::forward<Output>(scores),
                details::CrossCorrelationScore{}
            );
        }
    }

    /// Computes the (zero-)(normalized-)cross-correlation score.
    /// \note The reduction is done using double-precision.
    /// \note This uses reduce_ewise which may enforce certain contiguity restrictions on the arrays.
    template<typename Lhs, typename Rhs>
        requires ((nt::array_decay_of_real<Lhs, Rhs> or nt::array_decay_of_complex<Lhs, Rhs>) and nt::array_decay_with_same_nd<Lhs, Rhs>)
    [[nodiscard]] auto cross_correlation_score(Lhs&& lhs, Rhs&& rhs, const CrossCorrelationScoreOptions& options = {}) {
        check(not lhs.is_empty() and not rhs.is_empty(), "Empty array detected");
        check(lhs.shape() == rhs.shape() and not rhs.shape().is_batched(),
              "Arrays should have the same shape and should not be batched, but got lhs:shape={}, rhs:shape={}",
              lhs.shape(), rhs.shape());
        check(lhs.device() == rhs.device(),
              "The lhs and rhs input arrays should be on the same device, but got lhs:device={} and rhs:device={}",
              lhs.device(), rhs.device());

        using value_t = nt::mutable_value_type_t<Lhs>;
        using sum_t = nt::double_precision_t<value_t>;
        using real_t = f64;
        value_t score;
        if (options.center and options.normalize) {
            const auto n_elements = static_cast<real_t>(lhs.shape().n_elements());
            reduce_ewise(
                wrap(std::forward<Lhs>(lhs), std::forward<Rhs>(rhs)),
                wrap(real_t{}, real_t{}, sum_t{}, sum_t{}, sum_t{}),
                score,
                details::CrossCorrelationScoreCenteredNormalized{n_elements}
            );
        } else if (options.normalize) {
            reduce_ewise(
                wrap(std::forward<Lhs>(lhs), std::forward<Rhs>(rhs)),
                wrap(real_t{}, real_t{}, sum_t{}),
                score,
                details::CrossCorrelationScoreNormalized{}
            );
        } else if (options.center) {
            const auto n_elements = static_cast<real_t>(lhs.shape().n_elements());
            reduce_ewise(
                wrap(std::forward<Lhs>(lhs), std::forward<Rhs>(rhs)),
                wrap(sum_t{}, sum_t{}, sum_t{}),
                score,
                details::CrossCorrelationScoreCentered{n_elements}
            );
        } else {
            reduce_ewise(
                wrap(std::forward<Lhs>(lhs), std::forward<Rhs>(rhs)),
                sum_t{},
                score,
                details::CrossCorrelationScore{}
            );
        }
        return score;
    }

    struct CrossCorrelationMapOptions {
        /// Correlation mode to use. DOUBLE_PHASE_CORRELATION doubles the lags/shifts.
        Correlation mode = Correlation::CONVENTIONAL;

        /// Rank of the transform.
        /// See Shape::rank_checked for more details.
        i32 rank = -1;

        /// Normalization mode to use for the C2R transform producing the final output.
        /// This should match the mode that was used to compute the input transforms.
        nf::Norm ifft_norm = nf::NORM_DEFAULT;

        /// Whether the C2R transform should be cached.
        bool ifft_cache_plan{true};
    };

    /// Computes the cross-correlation map(s).
    /// \tparam REMAP:
    ///     Whether the output map(s) should be centered.
    ///     Should be H2F or H2FC.
    /// \param[in] lhs:
    ///     ((B..,)N..) Non-centered rFFT of the signal to cross-correlate.
    /// \param[in,out] rhs
    ///     ((B..,)N..) Non-centered rFFT of the signal to cross-correlate.
    ///     The rank of the transform, thus which dimensions are the B.. batch dimensions, is set by options.rank.
    ///     Autocorrelation is allowed, so lhs can be equal to rhs.
    ///     Overwritten by default (see buffer).
    /// \param[out] output:
    ///     ((B..,)N..) Cross-correlation map.
    ///     The rank of the transform, thus which dimensions are the B.. batch dimensions, is set by options.rank.
    ///     If REMAP is H2F, the zero-lags are at (B..,0..).
    ///     If REMAP is H2FC, the zero-lags are at {B..,N/2..).
    ///     Can overlap with the inputs (see buffer).
    /// \param options:
    ///     Correlation mode and ifft options.
    ///     The rank of the transforms, thus which dimensions are the B.. batch dimensions, is set by options.rank.
    ///
    /// \param[out] buffer:
    ///     ((B..,)N..) buffer to hold the rFFT, no broadcasting allowed, or Empty.
    ///     Can be lhs or rhs. If empty, rhs is used instead (the default).
    ///     Can be an alias of output, in which case an in-place iFFT is computed.
    ///     This is passed to the c2r, thus should be reshapeable to 4D (batch dimensions should be collapsible).
    ///
    /// \note
    ///     As mentioned above, this function takes the rFFT of the real inputs to correlate. The score with zero lag
    ///     can be computed more efficiently with the cross_correlation_score function. If other lags are to be
    ///     selected (which is the entire point of this function), the inputs should likely be zero-padded before
    ///     taking the rFFT to cancel the circular convolution effect of the DFT. The amount of padding along a
    ///     dimension is equal to the maximum lag allowed.
    /// \note
    ///     The resulting cross-correlation map is relative to rhs, and the lags describe by how much rhs should
    ///     be translated to match lhs. For instance, if rhs is a [y=+20, x=+10] translated version of lhs, the
    ///     resulting peak in the map is at lag [y=-20, x=-10].
    /// \note
    ///     As opposed to the cross_correlation_score function, this function does not take normalization flags.
    ///     To get the (Z)(N)CC scores, normalize the inputs appropriately before computing their rFFT (for instance,
    ///     the ZNCC needs zero-centered and L2-normalized real-space inputs). Importantly, the FFT normalization
    ///     should be ORTHO or BACKWARD. FORWARD outputs scores divided by a scaling factor of 1/n_elements.
    /// \note
    ///     To save memory, this function can operate fully in-place by overwriting lhs and/or rhs. When buffer is
    ///     empty, the temporary pairwise multiplication is saved into rhs. Then it is iFFTed to compute the cross-
    ///     correlation map. Since output can alias with lhs or rhs, no temporary buffer is necessary. In the case of
    ///     auto-correlation, no temporaries are necessary and everything can run on the input array.
    template<nf::Layout REMAP, typename Lhs, typename Rhs, typename Output, typename Buffer = Empty>
        requires details::cross_correlation_mapable<REMAP, Lhs, Rhs, Output, Buffer>
    void cross_correlation_map(
        Lhs&& lhs, Rhs&& rhs, Output&& output,
        const CrossCorrelationMapOptions& options = {},
        Buffer&& buffer = {}
    ) {
        const auto device = output.device();
        check(nd::are_arrays_valid(lhs, rhs, output), "Empty array detected");
        check(device == lhs.device() and device == rhs.device(),
              "The lhs, rhs and output arrays must be on the same device, but got lhs:device={}, rhs:device={} and output:device={}",
              lhs.device(), rhs.device(), device);

        using complex_t = nt::mutable_value_type_t<std::conditional_t<nt::empty<Empty>, Rhs, Buffer>>;
        constexpr auto N = nt::array_size_v<Lhs>;
        const auto rank = output.shape().rank_checked(options.rank);

        const auto expected_shape = output.shape().rfft();
        Array<complex_t, N, ArrayOwnership::VIEW> tmp;
        bool buffer_is_empty{true};
        if constexpr (nt::array_decay<Buffer>) {
            buffer_is_empty = buffer.is_empty();
            if (not buffer_is_empty) {
                check(device == buffer.device(),
                  "The temporary and output arrays must be on the same device, buffer:device={} and output:device={}",
                  buffer.device(), device);
                check(buffer.shape() == expected_shape and
                      nd::are_elements_unique(buffer.strides(), buffer.shape()),
                      "Given an output map of shape {}, the buffer should be of shape {} and have unique elements, but got buffer:shape={}, buffer:strides={}",
                      output.shape(), expected_shape, buffer.shape(), buffer.strides());
                tmp = buffer.view();
            }
        }
        if (buffer_is_empty) {
            check(nd::are_elements_unique(rhs.strides(), expected_shape),
                  "Since no temporary buffer is passed, the rhs input is used as buffer, thus should have unique elements with a shape of {}, but got rhs:shape={}, rhs:strides={}",
                  expected_shape, rhs.shape(), rhs.strides());
            tmp = rhs.view();
        }

        switch (options.mode) {
            case Correlation::CONVENTIONAL:
                ewise(wrap(std::forward<Lhs>(lhs), rhs), tmp,
                      details::CrossCorrelationMap<Correlation::CONVENTIONAL>{});
                break;
            case Correlation::PHASE:
                ewise(wrap(std::forward<Lhs>(lhs), rhs), tmp,
                      details::CrossCorrelationMap<Correlation::PHASE>{});
                break;
            case Correlation::DOUBLE_PHASE:
                ewise(wrap(std::forward<Lhs>(lhs), rhs), tmp,
                      details::CrossCorrelationMap<Correlation::DOUBLE_PHASE>{});
                break;
            case Correlation::MUTUAL:
                ewise(wrap(std::forward<Lhs>(lhs), rhs), tmp,
                      details::CrossCorrelationMap<Correlation::MUTUAL>{});
                break;
        }

        if constexpr (REMAP == nf::Layout::H2FC) {
            using real_t = nt::value_type_t<complex_t>;
            const auto shape_r = output.shape().vec.template as_nd<4>();
            if (rank == 1) {
                auto half_shift = (shape_r.template pop_front<3>() / 2).template as<real_t>();
                phase_shift_1d<"h2h">(tmp, tmp, output.shape(), half_shift);
            }
            if constexpr (N >= 2) {
                if (rank == 2) {
                    auto half_shift = (shape_r.template pop_front<2>() / 2).template as<real_t>();
                    phase_shift_2d<"h2h">(tmp, tmp, output.shape(), half_shift);
                }
            }
            if constexpr (N >= 3) {
                if (rank == 3) {
                    auto half_shift = (shape_r.template pop_front<1>() / 2).template as<real_t>();
                    phase_shift_3d<"h2h">(tmp, tmp, output.shape(), half_shift);
                }
            }
        }

        auto c2r_options = nf::FFTOptions{
            .rank = rank,
            .norm = options.ifft_norm,
            .cache_plan = options.ifft_cache_plan,
        };
        if (buffer_is_empty)
            return nf::c2r(std::forward<Rhs>(rhs), std::forward<Output>(output), c2r_options);
        if constexpr (nt::array_decay<Buffer>)
            return nf::c2r(std::forward<Buffer>(buffer), std::forward<Output>(output), c2r_options);
    }

    template<usize N>
    struct CrossCorrelationPeakOptions {
        /// ((D)H)W radius of the registration window, centered on the peak.
        /// To get subpixel-accuracy of the peak position and value, a 1d parabola is fitted along each dimension.
        /// This parameter specifies the radius of these parabolas. Zero is valid and turns off the registration.
        Vec<i32, N> registration_radius{Vec<i32, N>::from_value(1)};

        /// ((D)H)W maximum lag allowed, i.e., the peak is selected within this elliptical radius.
        /// If negative, it is ignored and the entire map is searched. If zero, the central peak at lag zero is
        /// guaranteed to be selected. Otherwise, an elliptical mask is applied on the centered cross-correlation
        /// map before the search. Note that to maximize performance, the implementation will select the minimum
        /// subregion within the map and only search within that subregion.
        Vec<f64, N> maximum_lag{Vec<f64, N>::from_value(-1)};
    };

    /// Find the cross-correlation peaks of the cross-correlation maps.
    /// \tparam LAYOUT:
    ///     Whether the map is centered. Should be F2F or FC2FC.
    /// \param[in] cross_correlation_maps:
    ///     (B..R..) cross-correlation maps.
    /// \param[out] peak_coordinates:
    ///     (B..) output array of (R..) coordinates of the highest peaks.
    ///     Can be an empty array or Empty, in which case the peak coordinates are not returned.
    /// \param[out] peak_values:
    ///     (B..) output array of values of the highest peaks.
    ///     Can be an empty array or Empty, in which case the peak coordinates are not returned.
    /// \param options:
    ///     Picking and registration options.
    template<nf::Layout LAYOUT, usize R, typename Input, typename PeakCoord = Empty, typename PeakValue = Empty>
        requires details::cross_correlation_peaks_able<LAYOUT, R, Input, PeakCoord, PeakValue>
    void cross_correlation_peaks(
        Input&& cross_correlation_maps,
        PeakCoord&& peak_coordinates,
        PeakValue&& peak_values = {},
        const CrossCorrelationPeakOptions<R>& options = {}
    ) {
        constexpr bool HAS_COORDINATES = nt::array<PeakCoord>;
        constexpr bool HAS_VALUES = nt::array<PeakValue>;
        constexpr auto N = nt::array_size_v<Input>;
        constexpr auto B = std::max(nt::array_size_v<PeakCoord>, nt::array_size_v<PeakValue>);

        check(not cross_correlation_maps.is_empty(), "Empty array detected");
        check(not cross_correlation_maps.strides().is_broadcast(), "The cross-correlation map should not be broadcast");
        const auto& device = cross_correlation_maps.device();
        const auto& shape = cross_correlation_maps.shape();
        auto [shape_b, shape_n] = shape.template split<B>();

        if constexpr (HAS_COORDINATES) {
            if (not peak_coordinates.is_empty()) {
                check(peak_coordinates.is_contiguous() and
                      peak_coordinates.shape() == shape_b,
                      "The number of peak coordinates, specified as a contiguous array, should be equal to the batch shape of the cross-correlation map. Got peak_coordinates:shape={}, peak_coordinates:strides={} and cross_correlation_maps:batches:shape={}",
                      peak_coordinates.shape(), peak_coordinates.strides(), shape_b);
                check(device == peak_coordinates.device(),
                      "The cross-correlation map and output peak coordinates must be on the same device, but got cross_correlation_maps:device={} and peak_coordinates:device={}",
                      device, peak_coordinates.device());
            }
        }

        if constexpr (HAS_VALUES) {
            if (not peak_values.is_empty()) {
                check(peak_values.is_contiguous() and
                      peak_values.shape() == shape_b,
                      "The number of peak values, specified as a contiguous array, should be equal to the batch shape of the cross-correlation map. Got peak_values:shape={}, peak_values:strides={} and cross_correlation_maps:batches:shape={}",
                      peak_values.shape(), peak_values.strides(), shape_b);
                check(device == peak_values.device(),
                      "The cross-correlation map and output peak values must be on the same device, but got cross_correlation_maps:device={} and peak_values:device={}",
                      device, peak_values.device());
            }
        }

        constexpr usize REGISTRATION_RADIUS_LIMIT = 4;
        check(noa::is_within(options.registration_radius, 0, REGISTRATION_RADIUS_LIMIT),
              "The registration radius should be a small positive value (less than {}), but got {}",
              REGISTRATION_RADIUS_LIMIT, options.registration_radius);

        if constexpr (HAS_COORDINATES or HAS_VALUES) {
            using value_t = nt::mutable_value_type_t<Input>;
            using index_t = nt::index_type_t<Input>;
            using coord_t = std::conditional_t<nt::empty<PeakCoord>, f64, nt::value_type_t<PeakCoord>>;
            using input_accessor_t = AccessorRestrict<const value_t, N, isize>;
            using peak_coordinates_accessor_t = AccessorRestrictContiguous<coord_t, B, isize>;
            using peak_values_accessor_t = AccessorRestrictContiguous<value_t, B, isize>;

            auto input_accessor = input_accessor_t(cross_correlation_maps.get(), cross_correlation_maps.strides());
            auto peak_coordinates_accessor = peak_coordinates_accessor_t{};
            auto peak_values_accessor = peak_values_accessor_t{};
            if constexpr (HAS_COORDINATES)
                peak_coordinates_accessor = peak_coordinates_accessor_t(peak_coordinates.get(), peak_coordinates.strides());
            if constexpr (HAS_VALUES)
                peak_values_accessor = peak_values_accessor_t(peak_values.get(), peak_values.strides());

            // Special case that doesn't require a reduction since we know where is the peak.
            // Just do the subpixel registration.
            if (options.maximum_lag == 0) {
                auto op = details::PeakRegistration<
                    B, R, LAYOUT.is_xc2xx(), REGISTRATION_RADIUS_LIMIT,
                    input_accessor_t, peak_coordinates_accessor_t, peak_values_accessor_t>(
                    input_accessor, peak_coordinates_accessor, peak_values_accessor,
                    shape_n, options.registration_radius
                );
                iwise(
                    shape_b, device, op,
                    std::forward<Input>(cross_correlation_maps),
                    std::forward<PeakCoord>(peak_coordinates),
                    std::forward<PeakValue>(peak_values)
                );
                return;
            }

            // Restrict the search window within the maximum allowed lag.
            bool apply_ellipse{};
            auto maximum_allowed_lag = shape_n.vec.pop_front() / 2;
            Vec<index_t, R> maximum_lag;
            Vec<index_t, R> subregion_offset;
            Shape<index_t, R> subregion_shape;
            for (usize i{}; i < R; ++i) {
                if (options.maximum_lag[i] < 0) {
                    maximum_lag[i] = maximum_allowed_lag[i];
                } else {
                    auto lag_index = static_cast<index_t>(ceil(options.maximum_lag[i]));
                    maximum_lag[i] = min(maximum_allowed_lag[i], lag_index);
                    apply_ellipse = true;
                }

                // The reduction is done within the subregion that goes up to the maximum lag
                // (and centered on the original map center, of course).
                subregion_offset[i] = maximum_allowed_lag[i] - maximum_lag[i];
                subregion_shape[i] = min(maximum_lag[i] * 2 + 1, shape_n[i + 1]);
            }

            // Search for the peaks, and do the subpixel registration.
            using reducer_t = details::ReducePeak<
                B, R, LAYOUT.is_xc2xx(), REGISTRATION_RADIUS_LIMIT,
                input_accessor_t, peak_coordinates_accessor_t, peak_values_accessor_t>;

            auto reduce_axes = ReduceAxes<N>::all();
            for (usize i{}; i < B; ++i)
                reduce_axes[i] = false;

            auto initial_reduction_value = Pair{std::numeric_limits<value_t>::lowest(), index_t{}};
            reduce_axes_iwise(
                subregion_shape.push_front(shape_b), device, initial_reduction_value, reduce_axes,
                reducer_t(input_accessor, peak_coordinates_accessor, peak_values_accessor,
                          shape_n, options.registration_radius,
                          subregion_offset, maximum_lag, apply_ellipse),
                std::forward<Input>(cross_correlation_maps),
                std::forward<PeakCoord>(peak_coordinates),
                std::forward<PeakValue>(peak_values)
            );
        }
    }

    template<nf::Layout REMAP, typename Input, typename PeakCoord = Empty, typename PeakValue = Empty>
    void cross_correlation_peaks_1d(
        const Input& xmap,
        const PeakCoord& peak_coordinates,
        const PeakValue& peak_values = {},
        const CrossCorrelationPeakOptions<1>& options = {}
    ) {
        cross_correlation_peaks<REMAP>(xmap, peak_coordinates, peak_values, options);
    }
    template<nf::Layout REMAP, typename Input, typename PeakCoord = Empty, typename PeakValue = Empty>
    void cross_correlation_peaks_2d(
        const Input& xmap,
        const PeakCoord& peak_coordinates,
        const PeakValue& peak_values = {},
        const CrossCorrelationPeakOptions<2>& options = {}
    ) {
        cross_correlation_peaks<REMAP>(xmap, peak_coordinates, peak_values, options);
    }
    template<nf::Layout REMAP, typename Input, typename PeakCoord = Empty, typename PeakValue = Empty>
    void cross_correlation_peaks_3d(
        const Input& xmap,
        const PeakCoord& peak_coordinates,
        const PeakValue& peak_values = {},
        const CrossCorrelationPeakOptions<3>& options = {}
    ) {
        cross_correlation_peaks<REMAP>(xmap, peak_coordinates, peak_values, options);
    }

    template<nf::Layout REMAP, nt::readable_array_decay_of_almost_any<f32, f64> Input>
        requires (nt::array_nd<Input, 1, 2, 3>)
    auto cross_correlation_peak(
        const Input& cross_correlation_map,
        const CrossCorrelationPeakOptions<nt::array_size_v<Input>>& options = {}
    ) {
        constexpr auto R = nt::array_size_v<Input>;
        using value_t = nt::mutable_value_type_t<Input>;
        using coord_t = Vec<f64, R>;
        using pair_t = Pair<coord_t, value_t>;
        using coord_view_t = Array<coord_t, 1, ArrayOwnership::VIEW>;
        using value_view_t = Array<value_t, 1, ArrayOwnership::VIEW>;

        auto cross_correlation_map_batched = cross_correlation_map.view().template as_nd<R + 1>();
        if (cross_correlation_map.device().is_cpu()) {
            pair_t pair{};
            cross_correlation_peaks<REMAP>(
                cross_correlation_map_batched,
                coord_view_t(&pair.first, 1),
                value_view_t(&pair.second, 1),
                options
            );
            cross_correlation_map.eval();
            return pair;
        }

        const auto array_options = ArrayOption{cross_correlation_map.device(), Allocator::ASYNC};
        const auto pair = Array<pair_t, 1>(1, array_options);
        const auto coord = coord_view_t(&(pair.get()->first), pair.shape(), pair.strides(), array_options, Unchecked{});
        const auto value = value_view_t(&(pair.get()->second), pair.shape(), pair.strides(), array_options, Unchecked{});
        cross_correlation_peaks<REMAP>(
            cross_correlation_map_batched,
            coord_view_t(&(pair.get()->first), pair.shape(), pair.strides(), array_options, Unchecked{}),
            value_view_t(&(pair.get()->second), pair.shape(), pair.strides(), array_options, Unchecked{}),
            options
        );
        return pair.first();
    }

    template<nf::Layout REMAP, typename Input>
    auto cross_correlation_peak_1d(const Input& xmap, const CrossCorrelationPeakOptions<1>& options = {}) {
        return cross_correlation_peak<REMAP>(xmap, options);
    }
    template<nf::Layout REMAP, typename Input>
    auto cross_correlation_peak_2d(const Input& xmap, const CrossCorrelationPeakOptions<2>& options = {}) {
        return cross_correlation_peak<REMAP>(xmap, options);
    }
    template<nf::Layout REMAP, typename Input>
    auto cross_correlation_peak_3d(const Input& xmap, const CrossCorrelationPeakOptions<3>& options = {}) {
        return cross_correlation_peak<REMAP>(xmap, options);
    }
}
