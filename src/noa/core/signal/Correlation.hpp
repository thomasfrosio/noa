#pragma once

#include "noa/core/types/Complex.hpp"
#include "noa/core/types/Vec.hpp"
#include "noa/core/types/Pair.hpp"
#include "noa/core/math/LeastSquare.hpp"
#include "noa/core/geometry/DrawShape.hpp"

namespace noa::signal::guts {
    struct CrossCorrelationL2Norm {
        using allow_vectorization = bool;
        using remove_defaulted_final = bool;

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
        using allow_vectorization = bool;

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

    template<size_t N, bool IS_CENTERED, bool SUBREGION, size_t REGISTRATION_RADIUS_LIMIT,
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
        using f32_n_type = Vec<f32, N>;

        using draw_ellipse_type = noa::geometry::guts::DrawEllipse<N, f32, false>;
        using subregion_offset_type = std::conditional_t<SUBREGION, Vec<index_type, N>, Empty>;
        using ellipse_type = std::conditional_t<SUBREGION, draw_ellipse_type, Empty>;

    public:
        constexpr ReducePeak(
            const input_type& input,
            const peak_coordinates_type& peak_coordinates,
            const peak_values_type& peak_values,
            const Shape<index_type, N + 1>& shape,
            const index_n_type& registration_radius,
            const index_n_type& subregion_offset = {},
            const f32_n_type& maximum_lag = {}
        ) : m_input(input),
            m_peak_coordinates(peak_coordinates),
            m_peak_values(peak_values),
            m_shape(shape.pop_front()),
            m_batch(shape[0]),
            m_registration_radius(registration_radius)
        {
            if constexpr (SUBREGION) {
                m_subregion_offset = subregion_offset;
                const auto center = f32_n_type::from_vec(shape.vec.pop_front() / 2); // DC position
                constexpr auto cvalue = 1.f;
                constexpr auto is_smooth = false;
                m_ellipse = ellipse_type(center, maximum_lag, cvalue, is_smooth);
            }
            NOA_ASSERT(all(registration_radius <= static_cast<index_type>(REGISTRATION_RADIUS_LIMIT)));
        }

    public:
        constexpr void init(const nt::vec_of_size<N + 1> auto& subregion_indices, reduced_type& reduced) const {
            auto batch = subregion_indices[0];
            auto indices = subregion_indices.pop_front();

            f32 mask{1};
            if constexpr (SUBREGION) {
                indices += m_subregion_offset;
                mask = m_ellipse(Vec<f32, N>::from_vec(indices));
            }

            if constexpr (not IS_CENTERED)
                indices = noa::fft::ifftshift(indices, m_shape);
            const auto batched_indices = indices.push_front(batch);

            const auto value = m_input(batched_indices) * static_cast<value_type>(mask);
            const auto offset = m_input.offset_at(batched_indices); // the reduction is per batch, so this is okay
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
                m_input[batch], peak_indices.pop_front());
            if (m_peak_coordinates)
                m_peak_coordinates(batch) = static_cast<coord_type>(peak_coordinate);
            if (m_peak_values)
                m_peak_values(batch) = static_cast<value_type>(peak_value);
        }

    private:
        constexpr auto subpixel_registration_using_1d_parabola_(auto input, const index_n_type& peak_indices) {
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
        NOA_NO_UNIQUE_ADDRESS ellipse_type m_ellipse;
        NOA_NO_UNIQUE_ADDRESS subregion_offset_type m_subregion_offset;
    };
}
