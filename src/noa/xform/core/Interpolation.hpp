#pragma once

#include "noa/fft/core/Frequency.hpp"
#include "noa/fft/core/Layout.hpp"
#include "noa/runtime/core/Shape.hpp"
#include "noa/runtime/core/Border.hpp"
#include "noa/xform/core/Traits.hpp"
#include "noa/xform/core/Interp.hpp"

namespace noa::traits {
    namespace details {
        template<typename T, usize... N>
        struct fetchable_nd_t {
            template<usize S, usize... J>
            static consteval bool has_fetch(std::index_sequence<J...>) {
                using vec_t = Vec<f32, S>;
                using out_t = mutable_value_type_t<T>;
                return std::convertible_to<decltype(std::declval<const T&>().fetch(f32{J}...)), out_t> and
                       std::convertible_to<decltype(std::declval<const T&>().fetch(vec_t{})), out_t> and
                       std::convertible_to<decltype(std::declval<const T&>().fetch_raw(f32{J}...)), out_t> and
                       std::convertible_to<decltype(std::declval<const T&>().fetch_raw(vec_t{})), out_t> and
                       std::convertible_to<decltype(std::declval<const T&>().fetch_preprocess(vec_t{})), vec_t>;
            }

            static constexpr bool value = ((has_fetch<N>(std::make_index_sequence<N>{})) or ...);
        };
    }

    template<typename T, Border BORDER, usize... N>
    concept textureable_nd = std::copyable<std::remove_cv_t<T>> and T::BORDER == BORDER and requires {
        typename T::value_type;
        typename T::index_type;
    } and details::fetchable_nd_t<T, N...>::value;

    template<typename T, Border BORDER, usize... N>
    concept lerpable_nd = textureable_nd<T, BORDER, N...> and T::INTERP == nx::Interp::LINEAR_FAST;

    template<typename T, Border BORDER, usize... N>
    concept nearestable_nd = textureable_nd<T, BORDER, N...> and T::INTERP == nx::Interp::NEAREST_FAST;

    template<typename T, Border BORDER, usize... N>
    concept interpable_nd =
        std::copyable<std::remove_cv_t<T>> and
        nt::real_or_complex<typename T::value_type> and
        nt::integer<typename T::index_type> and sizeof(typename T::index_type) >= 4 and
        (readable_nd<std::remove_reference_t<T>, N...> or
         textureable_nd<std::remove_reference_t<T>, BORDER, N...>);

    template<typename T, usize N>
    concept shapeable_nd = requires {
        { std::declval<const std::decay_t<T>&>().shape() } -> almost_same_as<Shape<typename T::index_type, N>>;
    };
}

namespace noa::xform {
    /// Computes the interpolation weights.
    /// \tparam INTERP  Interpolation method.
    /// \tparam Weight  Weight value type.
    /// \tparam Coord   Coordinate value type. If it is a Vec, then Weight must be a Vec too.
    /// \param fraction Fraction(s), between 0 and 1.
    /// \returns Interpolation weights in the form of Vec<Weight, N>, where N is the size of the interpolation window.
    template<Interp INTERP, typename Weight, typename Coord> requires nt::vec_real<Weight, Coord>
    [[nodiscard]] NOA_FHD constexpr auto interpolation_weights(Coord fraction) {
        using real_t = nt::value_type_t<Weight>;
        const auto f = static_cast<Weight>(fraction);

        if constexpr (INTERP.is_almost_any(Interp::NEAREST)) {
            // This is not used by the interpolation functions.
            // In practice, the rounding is done by the interpolation function
            // and only the correct element (with weight=1) is read/fetched.
            Vec<Weight, 2> coefficients{};
            for (usize i{}; auto e: round(f))
                coefficients[i++][static_cast<i32>(e)] = 1;
            return coefficients;

        } else if constexpr (INTERP.is_almost_any(Interp::LINEAR)) {
            return Vec{1 - f, f};

        } else if constexpr (INTERP.is_almost_any(Interp::CUBIC)) {
            // https://stackoverflow.com/a/26828782, some use A=-0.75
            constexpr auto A = static_cast<real_t>(-0.5);
            Vec<Weight, 4> coefficients;
            coefficients[0] = ((A * (f + 1) - 5 * A) * (f + 1) + 8 * A) * (f + 1) - 4 * A;
            coefficients[1] = ((A + 2) * f - (A + 3)) * f * f + 1;
            coefficients[2] = ((A + 2) * (1 - f) - (A + 3)) * (1 - f) * (1 - f) + 1;
            coefficients[3] = 1 - coefficients[0] - coefficients[1] - coefficients[2];
            return coefficients;

        } else if constexpr (INTERP.is_almost_any(Interp::CUBIC_BSPLINE)) {
            constexpr auto ONE_SIXTH = static_cast<real_t>(1) / static_cast<real_t>(6);
            constexpr auto TWO_THIRD = static_cast<real_t>(2) / static_cast<real_t>(3);
            const auto one_minus = 1 - f;
            const auto one_squared = one_minus * one_minus;
            const auto squared = f * f;
            return Vec{ONE_SIXTH * one_squared * one_minus,
                       TWO_THIRD - static_cast<real_t>(0.5) * squared * (2 - f),
                       TWO_THIRD - static_cast<real_t>(0.5) * one_squared * (2 - one_minus),
                       ONE_SIXTH * squared * f};

        } else if constexpr (INTERP.is_almost_any(Interp::LANCZOS4, Interp::LANCZOS6, Interp::LANCZOS8)) {
            constexpr usize SIZE = INTERP.window_size();
            constexpr usize CENTER = SIZE / 2 - 1;

            // Instead of computing the windowed-sinc for every point in the nd-window, use this trick from OpenCV
            // to only compute one sin and cos per dimension, regardless of the window size. See:
            // https://github.com/opencv/opencv/blob/master/modules/imgproc/src/imgwarp.cpp#L162
            // I think this relies on the Chebyshev polynomials, but I'm not sure.
            // Regardless, it gives the expected windowed-sinc, and it works for any N (N=[4,6,8] in our case).
            constexpr auto PI = Constant<real_t>::PI;
            constexpr auto s45 = static_cast<real_t>(0.7071067811865475); // sin(Constant<real_t>::PI / 4);
            constexpr real_t cs[][2] =
                {{1, 0}, {-s45, -s45}, {0, 1}, {s45, -s45}, {-1, 0}, {s45, s45}, {0, -1}, {-s45, s45}};

            Weight sum{};
            Vec<Weight, SIZE> coefficients{};
            for (usize j{}; j < Weight::SIZE; ++j) {
                if (f[j] <= std::numeric_limits<real_t>::epsilon()) {
                    coefficients[CENTER][j] = 1;
                    sum[j] = 1;
                } else if (f[j] >= 1 - std::numeric_limits<real_t>::epsilon()) {
                    coefficients[CENTER + 1][j] = 1;
                    sum[j] = 1;
                } else {
                    const real_t x0 = -(f[j] + static_cast<real_t>(CENTER));
                    const real_t y0 = x0 * PI * static_cast<real_t>(0.25);
                    const auto sc0 = sincos(y0);

                    for (usize i{}; i < SIZE; i++) {
                        const auto fij = -(f[j] + CENTER - static_cast<real_t>(i));
                        const auto yj = fij * PI * static_cast<real_t>(0.25);
                        coefficients[i][j] = (cs[i][0] * sc0[0] + cs[i][1] * sc0[1]) / (yj * yj);
                        sum[j] += coefficients[i][j];
                    }
                }
            }

            // Normalize the convolution kernel.
            coefficients *= 1 / sum;
            return coefficients;

        } else {
            static_assert(nt::always_false<Weight>);
        }
    }

    /// Returns whether the interpolation window around the given coordinates/indices is fully inbound.
    /// If true, the interpolation functions can be called with ASSUME_INBOUND=true.
    /// If false, the interpolation window is either entirely out-of-bounds OR
    /// at the edge of the array boundary with a least one element/axis inbound.
    template<Interp INTERP, usize N, nt::sinteger SInt, nt::any_of<f32, f64, SInt> Coord,  usize A0, usize A1>
        requires (1 <= N and N <= 3)
    [[nodiscard]] NOA_FHD constexpr auto is_interpolation_window_inbound(
        const Shape<SInt, N, A1>& shape,
        const Vec<Coord, N, A0>& coordinates
    ) noexcept -> bool {
        constexpr SInt SIZE = INTERP.window_size();
        constexpr SInt START = -(SIZE - 1) / 2;
        constexpr SInt END = SIZE / 2;
        for (usize i{}; i < N; ++i) {
            SInt index;
            if constexpr (nt::same_as<Coord, SInt>)
                index = coordinates[i];
            else
                index = static_cast<SInt>(floor(coordinates[i]));
            if (index + START < 0 or index + END >= shape[i])
                return false;
        }
        return true;
    }

    /// Returns whether the interpolation window around the given coordinates/indices is fully out-of-bound.
    template<Interp INTERP, usize N, nt::sinteger SInt, nt::any_of<f32, f64, SInt> Coord,  usize A0, usize A1>
        requires (1 <= N and N <= 3)
    [[nodiscard]] NOA_FHD constexpr auto is_interpolation_window_outbound(
        const Shape<SInt, N, A1>& shape,
        const Vec<Coord, N, A0>& coordinates
    ) noexcept -> bool {
        constexpr SInt SIZE = INTERP.window_size();
        constexpr SInt START = -(SIZE - 1) / 2;
        constexpr SInt END = SIZE / 2;
        for (usize i{}; i < N; ++i) {
            SInt index;
            if constexpr (nt::same_as<Coord, SInt>)
                index = coordinates[i];
            else
                index = static_cast<SInt>(floor(coordinates[i]));
            if (index + END >= 0 or index + START < shape[i])
                return true;
        }
        return false;
    }

    /// Interpolates the 1d|2d|3d input data at the given coordinates.
    /// \tparam INTERP:
    ///     Interpolation method.
    /// \tparam BORDER:
    ///     Border type.
    /// \tparam ASSUME_INBOUND:
    ///     Assume is_interpolation_window_inbound(shape, coordinate) == true.
    ///     If true, the shape is ignored and no bound checking is done.
    /// \tparam OPTIMISTIC_INBOUND_CHECK:
    ///     Check for is_interpolation_window_inbound(shape, coordinate) once and if true, disable the bound checks
    ///     for all elements in the interpolation window. If ASSUME_INBOUND=true, this option is ignored.
    ///     This leads to better performance, except for borderless addressing (Border::CLAMP, PERIODIC, MIRROR)
    ///     in CUDA. If the coordinates are often inbound, enabling this option should lead to better performance.
    /// \tparam OPTIMISTIC_CHECK_OOBOUND:
    ///     Similar to OPTIMISTIC_INBOUND_CHECK, but checking for out-of-bounds using is_interpolation_window_outbound.
    ///     If ASSUME_INBOUND=true or Border is not ZERO or VALUE, this option is ignored.
    ///     Benchmarks show this option is best disabled, but in scenarios where most coordinates are fully
    ///     out-of-bounds, enabling it could lead to better performance.
    /// \param input:
    ///     Readable type, usually an Accessor(Value) or Span, mapping the input array.
    /// \param coordinate:
    ///     ((D)H)W coordinates to interpolate at.
    /// \param shape:
    ///     ((D)H)W shape of the input array.
    /// \param cvalue:
    ///     Constant value for out-of-bounds conditions.
    ///     Only used for BORDER == Border::VALUE.
    template<Interp INTERP, Border BORDER,
             bool ASSUME_INBOUND = false,
             bool OPTIMISTIC_INBOUND_CHECK = true,
             bool OPTIMISTIC_CHECK_OOBOUND = false,
             usize N,
             nt::any_of<f32, f64> Coord,
             nt::readable_nd<N> T,
             nt::sinteger SInt,
             usize A0, usize A1, typename C>
    requires (1 <= N and N <= 3 and (BORDER != Border::VALUE or nt::same_as_mutable_value_type_of<C, T>))
    [[nodiscard]] NOA_IHD constexpr auto interpolate(
        const T& input,
        const Vec<Coord, N, A0>& coordinate,
        const Shape<SInt, N, A1>& shape,
        const C& cvalue
    ) noexcept -> nt::mutable_value_type_t<T> {
        // Utility to read from the input while accounting for the border mode.
        using value_t = nt::mutable_value_type_t<T>;
        auto value_at = [&]<bool INBOUND = false, typename... S>(S... indices) {
            if constexpr (ASSUME_INBOUND or INBOUND) {
                return input(indices...);
            } else {
                if constexpr (BORDER == Border::ZERO or BORDER == Border::VALUE) {
                    if (is_inbound(shape, indices...))
                        return input(indices...);
                    if constexpr (BORDER == Border::ZERO)
                        return value_t{};
                    else
                        return cvalue;
                } else {
                    return input(index_at<BORDER>(shape, indices...));
                }
            }
        };

        if constexpr (INTERP.is_almost_any(Interp::NEAREST)) {
            // For nearest, the compiler cannot optimize away the second read with weight=0,
            // so keep a compile-time branch for this case reading only one value from the input.
            return value_at(round(coordinate).template as<SInt, A1>());
        } else {
            // N=2:           0, 1
            // N=4:       -1, 0, 1, 2
            // N=6:    -2,-1, 0, 1, 2, 3
            // N=8: -3,-2,-1, 0, 1, 2, 3, 4
            constexpr SInt SIZE = INTERP.window_size();
            constexpr SInt START = -(SIZE - 1) / 2;
            constexpr SInt END = SIZE / 2;

            const auto floored = floor(coordinate);
            const auto indices = floored.template as<SInt, A1>();
            const auto fraction = coordinate - floored;

            // If completely out-of-bounds, return the output value if possible.
            if constexpr (OPTIMISTIC_CHECK_OOBOUND and not ASSUME_INBOUND and BORDER.is_any(Border::ZERO, Border::VALUE)) {
                if (indices + END < 0 or indices + START >= shape.vec) { // if any
                    if constexpr (BORDER == Border::ZERO)
                        return value_t{};
                    else
                        return cvalue;
                }
            }

            // Optimistically check for inbound once. If so, no other bound checking will be necessary.
            bool is_window_inbound{};
            if constexpr (ASSUME_INBOUND)
                is_window_inbound = true;
            else if constexpr (OPTIMISTIC_INBOUND_CHECK)
                is_window_inbound = is_interpolation_window_inbound<INTERP>(shape, indices);
            else
                is_window_inbound = false;

            // If indices are inbound, no need to interpolate in the case of AccessorValue.
            if constexpr (nt::accessor_value<T>)
                if (is_window_inbound)
                    return input();

            // Interpolate.
            // Explicitly write the interpolation for SIZE 2 or 4 windows:
            //  - GCC/Clang may generate better code as the benchmarks show an improvement.
            //  - Benchmarks in nvcc show no significant improvements.
            using real_t = nt::value_type_t<value_t>;
            using weight_t = Vec<real_t, N>;
            Vec<weight_t, SIZE> weights = interpolation_weights<INTERP, weight_t>(fraction);

            value_t interpolant{};
            if constexpr (N == 1) {
                if (ASSUME_INBOUND or is_window_inbound) {
                    for (SInt x{START}; x <= END; ++x)
                        interpolant += input(indices[0] + x) * weights[x - START][0];
                } else {
                    for (SInt x{START}; x <= END; ++x)
                        interpolant += value_at(indices[0] + x) * weights[x - START][0];
                }
            } else if constexpr (N == 2) {
                if (ASSUME_INBOUND or is_window_inbound) {
                    for (SInt y{START}; y <= END; ++y) {
                        value_t interpolant_y{};
                        auto iy = indices[0] + y;
                        for (SInt x{START}; x <= END; ++x)
                            interpolant_y += input(iy, indices[1] + x) * weights[x - START][1];
                        interpolant += interpolant_y * weights[y - START][0];
                    }
                } else {
                    for (SInt y{START}; y <= END; ++y) {
                        value_t interpolant_y{};
                        auto iy = indices[0] + y;
                        for (SInt x{START}; x <= END; ++x)
                            interpolant_y += value_at(iy, indices[1] + x) * weights[x - START][1];
                        interpolant += interpolant_y * weights[y - START][0];
                    }
                }
            } else if constexpr (N == 3) {
                if (ASSUME_INBOUND or is_window_inbound) {
                    for (SInt z{START}; z <= END; ++z) {
                        value_t interpolant_z{};
                        auto iz = indices[0] + z;
                        for (SInt y{START}; y <= END; ++y) {
                            value_t interpolant_y{};
                            auto iy = indices[1] + y;
                            for (SInt x{START}; x <= END; ++x)
                                interpolant_y += input(iz, iy, indices[2] + x) * weights[x - START][2];
                            interpolant_z += interpolant_y * weights[y - START][1];
                        }
                        interpolant += interpolant_z * weights[z - START][0];
                    }
                } else {
                    for (SInt z{START}; z <= END; ++z) {
                        value_t interpolant_z{};
                        auto iz = indices[0] + z;
                        for (SInt y{START}; y <= END; ++y) {
                            value_t interpolant_y{};
                            auto iy = indices[1] + y;
                            for (SInt x{START}; x <= END; ++x)
                                interpolant_y += value_at(iz, iy, indices[2] + x) * weights[x - START][2];
                            interpolant_z += interpolant_y * weights[y - START][1];
                        }
                        interpolant += interpolant_z * weights[z - START][0];
                    }
                }
            } else {
                static_assert(nt::always_false<SInt>);
            }
            return interpolant;
        }
    }

    /// Interpolates the 1d|2d|3d input data at the given coordinates.
    /// \tparam INTERP:
    ///     Interpolation method.
    ///     Only fast modes can lead to hardware accelerated interpolation
    ///     (at the cost of a lower precision).
    /// \tparam BORDER:
    ///     Border mode. Must be equal to the input border mode, since out-of-bound
    ///     coordinates are entirely handled by the input texture.
    /// \tparam N:
    ///     Number of dimensions.
    /// \param input:
    ///     Accessor texture object mapping a N-d input array.
    /// \param coordinate:
    ///     ((D)H)W coordinates to interpolate at.
    template<Interp INTERP, Border BORDER, usize N, size_t A,
             nt::textureable_nd<BORDER, N> T,
             nt::any_of<f32, f64> Coord>
    requires (1 <= N and N <= 3)
    [[nodiscard]] NOA_FHD constexpr auto interpolate_using_texture(
        const T& input,
        const Vec<Coord, N, A>& coordinate
    ) noexcept -> nt::mutable_value_type_t<T> {
        if constexpr (INTERP == T::INTERP) {
            return input.fetch(coordinate); // the texture can handle both the interpolation and addressing

        } else if constexpr (INTERP.is_almost_any(Interp::NEAREST)) {
            // Special case for nearest-neighbor interpolation in accurate mode. Here the "interpolation"
            // is done in software and the texture, regardless of its mode, fetches at the closest integer location.
            return input.fetch(round(coordinate));

        } else {
            using value_t = nt::mutable_value_type_t<T>;
            using real_t = nt::value_type_t<value_t>;
            using coordn_t = Vec<Coord, N, A>;
            using weight_t = Vec<real_t, N>;

            constexpr i32 SIZE = INTERP.window_size();
            const coordn_t floored = floor(coordinate);
            const coordn_t fraction = coordinate - floored;
            const Vec<weight_t, static_cast<usize>(SIZE)> weights = interpolation_weights<INTERP, weight_t>(fraction);

            if constexpr (INTERP == Interp::CUBIC_BSPLINE_FAST and nt::lerpable_nd<T, BORDER, N>) {
                // Use hardware accelerated lerp(s) to compute the interpolated value. This works for any
                // interpolation method in fast mode (allowing lower precision).

                // This was expanded from the original implementation of Daniel Ruijters
                // for Cubic B-spline interpolation in CUDA: http://www.dannyruijters.nl/cubicinterpolation/
                // FIXME Unfortunately, this only works for interpolation functions with no negative parts,
                //       otherwise lerp_coord may end up outside of the [0,1] range, which doesn't work if
                //       we want to pass it to the texture. I couldn't find a way to make it work, so only
                //       do this for Cubic B-spline (which doesn't have negative weights), for now...

                // Compute the adjusted coordinates for the lerp and the weights.
                constexpr i32 HALF = SIZE / 2;
                Vec<weight_t, static_cast<usize>(HALF)> lerp_weight;
                Vec<coordn_t, static_cast<usize>(HALF)> lerp_coord;
                for (i32 i{}; i < HALF; ++i) {
                    const i32 index = (i * 2);
                    const i32 offset = -(HALF - 1 - index);
                    lerp_weight[i] = weights[index] + weights[index + 1];
                    lerp_coord[i] = input.fetch_preprocess(
                        (weights[index + 1] / lerp_weight[i]).template as<Coord>() +
                        floored + static_cast<Coord>(offset));
                }

                // Lerp at the adjusted coordinates and correct for the interpolation weights.
                value_t value{};
                if constexpr (N == 1) {
                    for (i32 x{}; x < HALF; ++x)
                        value += input.fetch_raw(lerp_coord[x][0]) * lerp_weight[x][0];
                } else if constexpr (N == 2) {
                    for (i32 y{}; y < HALF; ++y) {
                        value_t value_y{};
                        for (i32 x{}; x < HALF; ++x)
                            value_y += input.fetch_raw(lerp_coord[y][0], lerp_coord[x][1]) * lerp_weight[x][1];
                        value += value_y * lerp_weight[y][0];
                    }
                } else if constexpr (N == 3) {
                    for (i32 z{}; z < HALF; ++z) {
                        value_t value_z{};
                        for (i32 y{}; y < HALF; ++y) {
                            value_t value_y{};
                            for (i32 x{}; x < HALF; ++x)
                                value_y += input.fetch_raw(lerp_coord[z][0], lerp_coord[y][1], lerp_coord[x][2]) * lerp_weight[x][2];
                            value_z += value_y * lerp_weight[y][1];
                        }
                        value += value_z * lerp_weight[z][0];
                    }
                }
                return value;

            } else {
                // Hardware interpolation is not allowed, but we can still use the texture addressing
                // and do the interpolation in software.

                // N=2:           0, 1
                // N=4:       -1, 0, 1, 2
                // N=6:    -2,-1, 0, 1, 2, 3
                // N=8: -3,-2,-1, 0, 1, 2, 3, 4
                constexpr i32 START = -(SIZE - 1) / 2;
                constexpr i32 END = SIZE / 2;

                value_t value{};
                if constexpr (N == 1) {
                    for (i32 x{START}; x <= END; ++x)
                        value += input.fetch(floored + x) * weights[x - START][0];

                } else if constexpr (N == 2) {
                    for (i32 y{START}; y <= END; ++y) {
                        value_t value_y{};
                        for (i32 x{START}; x <= END; ++x)
                            value_y += input.fetch(floored + coordn_t::from_values(y, x)) * weights[x - START][1];
                        value += value_y * weights[y - START][0];
                    }
                } else if constexpr (N == 3) {
                    for (i32 z{START}; z <= END; ++z) {
                        value_t value_z{};
                        for (i32 y{START}; y <= END; ++y) {
                            value_t value_y{};
                            for (i32 x{START}; x <= END; ++x)
                                value_y += input.fetch(floored + coordn_t::from_values(z, y, x)) * weights[x - START][2];
                            value_z += value_y * weights[y - START][1];
                        }
                        value += value_z * weights[z - START][0];
                    }
                }
                return value;
            }
        }
    }

    namespace details {
        template<typename T, typename CoordOrSInt, size_t N, size_t A>
        NOA_FHD constexpr auto flip_frequency(Vec<CoordOrSInt, N, A>& frequency) -> T {
            if (frequency[N - 1] < 0) {
                frequency *= -1;
                return T{-1};
            }
            return T{1};
        }
    }

    /// Returns whether the interpolation spectrum window around the given flipped (integer) frequency is fully inbound.
    /// If true, the interpolation functions can be called with ASSUME_INBOUND=true.
    /// If false, the interpolation window is either entirely out-of-bounds OR
    /// at the edge of the array boundary with a least one element/axis inbound.
    template<Interp INTERP, bool IS_FLIPPED = false,
             usize N, nt::sinteger SInt, nt::any_of<f32, f64, SInt> Coord, usize A0, usize A1>
        requires (1 <= N and N <= 3)
    [[nodiscard]] NOA_FHD constexpr auto is_interpolation_spectrum_window_inbound(
        const Pair<Vec<SInt, N, A0>, Vec<SInt, N, A0>>& bounds,
        const Vec<Coord, N, A1>& frequency
    ) noexcept -> bool {
        constexpr SInt SIZE = INTERP.window_size();
        constexpr SInt START = -(SIZE - 1) / 2;
        constexpr SInt END = SIZE / 2;

        SInt sign;
        if constexpr (not IS_FLIPPED) {
            if (frequency[N - 1] < 0)
                sign = -1;
            else
                sign = 1;
        } else {
            (void) sign;
        }

        for (usize i{}; i < N; ++i) {
            SInt int_frequency;
            if constexpr (nt::same_as<Coord, SInt>)
                int_frequency = frequency[i];
            else
                int_frequency = static_cast<SInt>(floor(frequency[i]));

            if constexpr (not IS_FLIPPED)
                int_frequency *= sign;

            if (int_frequency + START < bounds.first[i] or int_frequency + END >= bounds.second[i])
                return false;
        }
        return true;
    }

    /// Interpolates the 1d|2d|3d input spectrum at the given coordinates.
    /// \tparam ASSUME_INBOUND:
    ///     Assume is_interpolation_spectrum_window_inbound(bounds, frequency) == true.
    ///     If true, it assumes the flipped frequencies are always inbound and no bound checking is done.
    /// \tparam ALWAYS_FLIP_PER_INDEX:
    ///     To handle non-redundant inputs (rFFT), the frequencies are switched to the complex conjugate if it falls
    ///     at x<0. For interpolation windows of size 2 (LINEAR), we can flip early the input frequency once because
    ///     the window doesn't have negative offsets, so if the frequency is flipped, we know that all indices within
    ///     the interpolation window will be >= 0. For larger windows, this is not the case and flipping needs to be
    ///     done for each index in the window. Flipping early is therefore more efficient.
    ///     However, this early flip results in slight error with rFFTs with even sizes. This affects only a few
    ///     elements at the Nyquist frequencies (the ones on the central axes, e.g. x=0) and incorrectly weights
    ///     the interpolated values towards zero. This option is here to turn off early flipping and therefore removes
    ///     this (tiny) error.
    /// \tparam OPTIMISTIC_INBOUND_CHECK:
    ///     Check for is_interpolation_spectrum_window_inbound(bounds, frequency) once and if true, disable the bound
    ///     checks for all elements in the interpolation window. If ASSUME_INBOUND=true, this option is ignored.
    ///     This leads to better performance, except for borderless addressing (Border::CLAMP, PERIODIC, MIRROR)
    ///     in CUDA. If the coordinates are often inbound, enabling this option should lead to better performance.
    /// \tparam REMAP:
    ///     Remap operator. The output layout is ignored.
    /// \tparam INTERP:
    ///     Interpolation method.
    /// \param input:
    ///     Readable type, usually an Accessor(Value) or Span, mapping the input spectrum.
    /// \param frequency:
    ///     Centered ((D)H)W frequency, in samples (not normalized).
    /// \param shape:
    ///     ((D)H)W (logical) shape of the input spectrum.
    template<nf::Layout REMAP, Interp INTERP,
             bool ASSUME_INBOUND = false,
             bool ALWAYS_FLIP_PER_INDEX = false,
             bool OPTIMISTIC_INBOUND_CHECK = true,
             usize N,
             nt::readable_nd<N> T,
             nt::any_of<f32, f64> Coord,
             nt::sinteger Int,
             usize A0, usize A1, usize A2>
    [[nodiscard]] NOA_FHD constexpr auto interpolate_spectrum(
        const T& input,
        Vec<Coord, N, A0> frequency,
        const Shape<Int, N, A1>& shape,
        const Pair<Vec<Int, N, A2>, Vec<Int, N, A2>>& bounds
    ) noexcept -> nt::mutable_value_type_t<T> {
        using value_t = nt::mutable_value_type_t<T>;
        using real_t = nt::value_type_t<value_t>;
        using indices_t = Vec<Int, N, A1>;
        using weight_t = Vec<real_t, N>;

        constexpr bool IS_RFFT = REMAP.is_hx2xx();
        constexpr bool IS_CENTERED = REMAP.is_xc2xx();

        // Frequency bounds.
        if constexpr (INTERP.is_almost_any(Interp::NEAREST)) {
            // For nearest, the compiler doesn't seem able to optimize away the second read with weight=0,
            // so keep a compile-time branch for this case so that only one value is read from the accessor.
            value_t value{};
            auto int_frequency = static_cast<indices_t>(round(frequency));
            real_t conjugate_sign = details::flip_frequency<real_t>(int_frequency);
            if (nf::is_inbound<IS_RFFT, true>(bounds, int_frequency)) {
                auto indices = nf::frequency2index<IS_CENTERED, IS_RFFT>(int_frequency, shape);
                auto v = input[indices];
                if constexpr (nt::complex<value_t>)
                    v.imag *= conjugate_sign;
                value = v;
            }
            return value;
        } else {
            constexpr Int SIZE = INTERP.window_size();
            auto update_at = [&]<bool INBOUND = false, typename W, typename O>(
                Vec<Int, N> int_frequency, const W& weight, O& output
            ) {
                constexpr bool FLIP_PER_INDEX = IS_RFFT and (ALWAYS_FLIP_PER_INDEX or SIZE > 2);
                real_t conjugate_sign;
                if constexpr (FLIP_PER_INDEX)
                    conjugate_sign = details::flip_frequency<real_t>(int_frequency);
                if (ASSUME_INBOUND or INBOUND or nf::is_inbound<IS_RFFT, true>(bounds, int_frequency)) {
                    auto indices = nf::frequency2index<IS_CENTERED, IS_RFFT>(int_frequency, shape);
                    auto value = input[indices];
                    NOA_NV_DIAG_SUPPRESS(549);
                    if constexpr (FLIP_PER_INDEX and nt::complex<value_t>)
                        value.imag *= conjugate_sign;
                    NOA_NV_DIAG_DEFAULT(549);
                    output += value * weight;
                }
            };

            // return value_t{};
            // Handle non-redundant inputs by switching to the complex conjugate.
            constexpr bool FLIP_EARLY = IS_RFFT and (not ALWAYS_FLIP_PER_INDEX and SIZE <= 2);
            real_t early_conjugate_sign;
            if constexpr (FLIP_EARLY)
                early_conjugate_sign = details::flip_frequency<real_t>(frequency);

            // N=2:           0, 1
            // N=4:       -1, 0, 1, 2
            // N=6:    -2,-1, 0, 1, 2, 3
            // N=8: -3,-2,-1, 0, 1, 2, 3, 4
            constexpr Int START = -(SIZE - 1) / 2;
            constexpr Int END = SIZE / 2;

            const auto floored = floor(frequency);
            const auto indices = static_cast<indices_t>(floored);
            const auto fraction = frequency - floored;
            const Vec<weight_t, SIZE> weights = interpolation_weights<INTERP, weight_t>(fraction);

            // Optimistically check for inbound once.
            // If the frequency is flipped early, checking the bounds is slightly less work.
            // If inbound, no further bound checking is required.
            bool is_window_inbound{};
            if constexpr (ASSUME_INBOUND)
                is_window_inbound = true;
            else if constexpr (OPTIMISTIC_INBOUND_CHECK)
                is_window_inbound = is_interpolation_spectrum_window_inbound<INTERP, FLIP_EARLY>(bounds, indices);
            else
                is_window_inbound = false;

            value_t value{};
            if constexpr (N == 1) {
                if (ASSUME_INBOUND or is_window_inbound) {
                    for (Int x{START}; x <= END; ++x)
                        update_at.template operator()<true>(Vec{indices[0] + x}, weights[x - START][0], value);
                } else {
                    for (Int x{START}; x <= END; ++x)
                        update_at(Vec{indices[0] + x}, weights[x - START][0], value);
                }
            } else if constexpr (N == 2) {
                if (ASSUME_INBOUND or is_window_inbound) {
                    for (Int y{START}; y <= END; ++y) {
                        value_t value_y{};
                        auto iy = indices[0] + y;
                        for (Int x{START}; x <= END; ++x)
                            update_at.template operator()<true>(Vec{iy, indices[1] + x}, weights[x - START][1], value_y);
                        value += value_y * weights[y - START][0];
                    }
                } else {
                    for (Int y{START}; y <= END; ++y) {
                        value_t value_y{};
                        auto iy = indices[0] + y;
                        for (Int x{START}; x <= END; ++x)
                            update_at(Vec{iy, indices[1] + x}, weights[x - START][1], value_y);
                        value += value_y * weights[y - START][0];
                    }
                }
            } else if constexpr (N == 3) {
                if (ASSUME_INBOUND or is_window_inbound) {
                    for (Int z{START}; z <= END; ++z) {
                        value_t value_z{};
                        auto iz = indices[0] + z;
                        for (Int y{START}; y <= END; ++y) {
                            value_t value_y{};
                            auto iy = indices[1] + y;
                            for (Int x{START}; x <= END; ++x)
                                update_at.template operator()<true>(Vec{iz, iy, indices[2] + x}, weights[x - START][2], value_y);
                            value_z += value_y * weights[y - START][1];
                        }
                        value += value_z * weights[z - START][0];
                    }
                } else {
                    for (Int z{START}; z <= END; ++z) {
                        value_t value_z{};
                        auto iz = indices[0] + z;
                        for (Int y{START}; y <= END; ++y) {
                            value_t value_y{};
                            auto iy = indices[1] + y;
                            for (Int x{START}; x <= END; ++x)
                                update_at(Vec{iz, iy, indices[2] + x}, weights[x - START][2], value_y);
                            value_z += value_y * weights[y - START][1];
                        }
                        value += value_z * weights[z - START][0];
                    }
                }
            }
            if constexpr (FLIP_EARLY and nt::complex<value_t>)
                value.imag *= early_conjugate_sign;
            return value;
        }
    }

    /// Interpolates the 1d|2d|3d input centered spectrum at the given coordinates.
    /// \tparam REMAP:
    ///     Remap operator.
    ///     The output layout is ignored.
    ///     The input layout must be centered for hardware interpolation.
    /// \tparam ALWAYS_FLIP_PER_INDEX:
    ///     See interpolate_spectrum.
    /// \tparam INTERP:
    ///     Interpolation method.
    /// \param frequency:
    ///     Centered ((D)H)W frequency, in samples (i.e. not normalized).
    /// \param input:
    ///     Textureable type mapping the input spectrum, using Border::ZERO addressing.
    /// \param shape:
    ///     ((D)H)W (logical) shape of the input spectrum.
    template<nf::Layout REMAP, Interp INTERP,
             bool ALWAYS_FLIP_PER_INDEX = false,
             size_t N,
             nt::textureable_nd<Border::ZERO, N> T,
             nt::any_of<f32, f64> Coord,
             nt::sinteger Int,
             size_t A0, size_t A1>
    [[nodiscard]] NOA_FHD constexpr auto interpolate_spectrum_using_texture(
        const T& input,
        Vec<Coord, N, A0> frequency,
        const Shape<Int, N, A1>& shape
    ) noexcept -> nt::mutable_value_type_t<T> {
        using value_t = nt::mutable_value_type_t<T>;
        using real_t = nt::value_type_t<value_t>;
        using coordn_t = Vec<Coord, N, A0>;
        using indices_t = Vec<Int, N, A1>;

        constexpr Int SIZE = INTERP.window_size();
        constexpr bool IS_RFFT = REMAP.is_hx2xx();
        constexpr bool IS_CENTERED = REMAP.is_xc2xx();

        // Handle non-redundant inputs by switching to the complex conjugate.
        constexpr bool FLIP_EARLY = IS_RFFT and (not ALWAYS_FLIP_PER_INDEX and SIZE <= 2);
        real_t early_conjugate_sign;
        if constexpr (FLIP_EARLY)
            early_conjugate_sign = details::flip_frequency<real_t>(frequency);

        value_t value{};
        if constexpr (INTERP == T::INTERP) {
            // The texture can handle both the interpolation and the addressing.
            value = input.fetch(nf::frequency2index<IS_CENTERED, IS_RFFT>(frequency, shape));

        } else if constexpr (INTERP.is_almost_any(Interp::NEAREST)) {
            // Special case for nearest-neighbor interpolation in accurate mode. Here the "interpolation"
            // is done in software and the texture, regardless of its mode, fetches at the closest integer location.
            value = input.fetch(nf::frequency2index<IS_CENTERED, IS_RFFT>(round(frequency), shape));

        } else {
            auto value_at = [&shape, &input](auto freq) {
                constexpr bool FLIP_PER_INDEX = IS_RFFT and (ALWAYS_FLIP_PER_INDEX or SIZE > 2);
                real_t conjugate_sign;
                if constexpr (FLIP_PER_INDEX)
                    conjugate_sign = details::flip_frequency<real_t>(freq);
                freq = nf::frequency2index<IS_CENTERED, IS_RFFT>(freq, shape);
                value_t fetched = input.fetch(static_cast<coordn_t>(freq));
                NOA_NV_DIAG_SUPPRESS(549);
                if constexpr (FLIP_PER_INDEX and nt::complex<value_t>)
                    fetched.imag *= conjugate_sign;
                NOA_NV_DIAG_DEFAULT(549);
                return fetched;
            };

            const coordn_t floored = floor(frequency);
            const coordn_t fraction = frequency - floored;
            using weight_t = Vec<real_t, N>;
            const Vec<weight_t, SIZE> weights = interpolation_weights<INTERP, weight_t>(fraction);

            if constexpr (REMAP.is_hc2xx() and INTERP == Interp::CUBIC_BSPLINE_FAST and nt::lerpable_nd<T, Border::ZERO, N>) {
                // Use hardware accelerated lerp(s) to compute the interpolated value.
                // TODO Limited to CUBIC_BSPLINE_FAST for now. See interpolate_using_texture.

                // Compute the adjusted coordinates for the lerp.
                constexpr Int HALF = SIZE / 2;
                Vec<weight_t, HALF> lw; // lerp weight
                Vec<coordn_t, HALF> lc; // lerp coordinate
                for (Int i{}; i < HALF; ++i) {
                    const auto index = (i * 2);
                    const auto offset = static_cast<Coord>(-(HALF - 1 - index));
                    lw[i] = weights[index] + weights[index + 1];
                    lc[i] = static_cast<coordn_t>(weights[index + 1] / lw[i]) + floored + offset;
                }

                // Lerp at the adjusted coordinates and correct for the interpolation weights.
                if constexpr (N == 1) {
                    for (Int x{}; x < HALF; ++x)
                        value += value_at(coordn_t{lc[x][0]}) * lw[x][0];

                } else if constexpr (N == 2) {
                    for (Int y{}; y < HALF; ++y) {
                        value_t value_y{};
                        for (Int x{}; x < HALF; ++x)
                            value_y += value_at(coordn_t{lc[y][0], lc[x][1]}) * lw[x][1];
                        value += value_y * lw[y][0];
                    }
                } else if constexpr (N == 3) {
                    for (Int z{}; z < HALF; ++z) {
                        value_t value_z{};
                        for (Int y{}; y < HALF; ++y) {
                            value_t value_y{};
                            for (Int x{}; x < HALF; ++x)
                                value_y += value_at(coordn_t{lc[z][0], lc[y][1], lc[x][2]}) * lw[x][2];
                            value_z += value_y * lw[y][1];
                        }
                        value += value_z * lw[z][0];
                    }
                }
            } else {
                // Hardware interpolation is not allowed or possible, but we can still use the texture addressing
                // and do the interpolation in software. Note that this mode also allows for non-centered spectra.
                constexpr Int START = -(SIZE - 1) / 2;
                constexpr Int END = SIZE / 2;
                const auto indices = static_cast<indices_t>(floored);
                if constexpr (N == 1) {
                    for (Int x{START}; x <= END; ++x)
                        value += value_at(Vec{indices[0] + x}) * weights[x - START][0];

                } else if constexpr (N == 2) {
                    for (Int y{START}; y <= END; ++y) {
                        value_t value_y{};
                        auto iy = indices[0] + y;
                        for (Int x{START}; x <= END; ++x)
                            value_y += value_at(Vec{iy, indices[1] + x}) * weights[x - START][1];
                        value += value_y * weights[y - START][0];
                    }
                } else if constexpr (N == 3) {
                    for (Int z{START}; z <= END; ++z) {
                        value_t value_z{};
                        auto iz = indices[0] + z;
                        for (Int y{START}; y <= END; ++y) {
                            value_t value_y{};
                            auto iy = indices[1] + y;
                            for (Int x{START}; x <= END; ++x)
                                value_y += value_at(Vec{iz, iy, indices[2] + x}) * weights[x - START][2];
                            value_z += value_y * weights[y - START][1];
                        }
                        value += value_z * weights[z - START][0];
                    }
                }
            }
        }
        if constexpr (FLIP_EARLY and nt::complex<value_t>)
            value.imag *= early_conjugate_sign;
        return value;
    }

    /// Interpolates {1|2|3}d (complex) floating-point data, using a given interpolation and border mode.
    /// \tparam N       Number of dimensions of the data. Between 1 and 3.
    /// \tparam INTERP_ Interpolation method.
    /// \tparam BORDER_ Border mode, aka addressing mode.
    /// \tparam NO_SHAPE:
    ///     Whether the nd-shape should be stored inside the class.
    ///     If true, the interpolated input should be texture-able or shapeable.
    /// \tparam TEXTURE_ONLY:
    ///     Whether only texture-able inputs are expected. If true, implies NO_SHAPE=true.
    ///     This allowed, both the shape and cvalue are ingored and the class is empty.
    ///
    /// \note Out-of-bounds coordinates:
    ///     One of the main differences between these interpolations and what we can find in other cryoEM packages,
    ///     is that the interpolation window can be partially out-of-bound (OOB), that is, elements that are OOB
    ///     are replaced according to a Border. cryoEM packages usually check that all elements are inbound,
    ///     and if there's even one element OOB, they don't interpolate.
    ///     In texture mode, the texture is responsible for the addressing, so some Border modes may not be supported.
    ///     See interpolate_using_texture for more details. If the texture does not have the required mode, i.e.,
    ///     BORDER != Input::BORDER, the interpolator tries to fall back on the `interpolate` function.
    ///
    /// \note Coordinate system:
    ///     The coordinate system matches the indexing, as expected.
    ///     For instance, the first data sample at index 0 is located at the coordinate 0 and the coordinate 0.5
    ///     is exactly in between the first and second element. As such, the fractional part of the coordinate
    ///     corresponds to the ratio/weight used by the interpolation functions. In other words, the coordinate system
    ///     locates the data between -0.5 and N-1 + 0.5.
    template<Interp INTERP_, Border BORDER_, usize N, nt::integer I, typename V = Empty, bool NO_SHAPE = false, bool TEXTURE_ONLY = false>
    class Interpolator {
    public:
        using cvalue_type = V;
        using mutable_cvalue_type = std::remove_const_t<cvalue_type>;
        using offset_type = I;
        using index_type = std::make_signed_t<offset_type>;

        static constexpr Interp INTERP = INTERP_;
        static constexpr Border BORDER = BORDER_;
        static constexpr size_t SIZE = N;
        static_assert(N == 1 or N == 2 or N == 3);

        #if defined(__CUDA_ARCH__)
        static constexpr bool DEFAULT_OPTIMISTIC_INBOUND_CHECK = BORDER == Border::ZERO or BORDER == Border::CLAMP;
        #else
        static constexpr bool DEFAULT_OPTIMISTIC_INBOUND_CHECK = true;
        #endif
        static constexpr bool DEFAULT_OPTIMISTIC_CHECK_OOBOUND = false;

        using shape_nd_type = Shape<index_type, N>;
        using shape_nd_or_empty_type = std::conditional_t<NO_SHAPE or TEXTURE_ONLY, Empty, shape_nd_type>;
        using cvalue_or_empty_type = std::conditional_t<not TEXTURE_ONLY and BORDER == Border::VALUE, mutable_cvalue_type, Empty>;

    public:
        /// Unsafe default construction...
        constexpr Interpolator() = default;

        /// Creates a new interpolator by storing a copy of the shape and cvalue.
        /// If NO_SHAPE=true, the shape is ignored.
        /// If TEXTURE_ONLY=true, both the shape and cvalue are ignored.
        template<usize A>
        NOA_HD constexpr Interpolator(
            const Shape<index_type, N, A>& shape,
            mutable_cvalue_type cvalue = mutable_cvalue_type{}
        ) noexcept {
            if constexpr (not TEXTURE_ONLY) {
                if constexpr (not NO_SHAPE)
                    m_shape = shape_nd_type::from_shape(shape);
                if constexpr (not std::is_empty_v<cvalue_or_empty_type>)
                    m_cvalue = cvalue;
            }
        }

    public:
        /// N-d interpolation of the input data at a given coordinates.
        /// \param input        Unbatched ND input.
        /// \param coordinates  Un-normalized ND coordinates.
        template<bool ASSUME_INBOUND = false,
                 bool OPTIMISTIC_INBOUND_CHECK = DEFAULT_OPTIMISTIC_INBOUND_CHECK,
                 bool OPTIMISTIC_CHECK_OOBOUND = DEFAULT_OPTIMISTIC_CHECK_OOBOUND,
                 nt::interpable_nd<BORDER_, N> Input, nt::any_of<f32, f64> C, usize A>
        [[nodiscard]] NOA_HD constexpr auto get(
            const Input& input,
            const Vec<C, N, A>& coordinates
        ) const -> nt::mutable_value_type_t<Input> {
            if constexpr (nt::textureable_nd<Input, BORDER, N>) {
                return nx::interpolate_using_texture<INTERP, BORDER>(input, coordinates);
            } else if constexpr (not TEXTURE_ONLY) {
                if constexpr (nt::shapeable_nd<Input, N>)
                    return nx::interpolate<
                        INTERP, BORDER, ASSUME_INBOUND, OPTIMISTIC_INBOUND_CHECK, OPTIMISTIC_CHECK_OOBOUND
                    >(input, coordinates, input.shape(), m_cvalue);
                else if constexpr (not NO_SHAPE)
                    return nx::interpolate<
                        INTERP, BORDER, ASSUME_INBOUND, OPTIMISTIC_INBOUND_CHECK, OPTIMISTIC_CHECK_OOBOUND
                    >(input, coordinates, m_shape, m_cvalue);
                else if constexpr (ASSUME_INBOUND)
                    return nx::interpolate<
                        INTERP, BORDER, ASSUME_INBOUND, OPTIMISTIC_INBOUND_CHECK, OPTIMISTIC_CHECK_OOBOUND
                    >(input, coordinates, shape_nd_type{}, m_cvalue);
                else
                    static_assert(nt::always_false<Input>, "A valid shape cannot be retrieved from the input and the shape was not stored (NO_SHAPE=true)");
            } else {
                static_assert(nt::always_false<Input>, "The input is not texture-able (given the BORDER and number of axes N) and the interpolator is TEXTURE_ONLY=true");
            }
            unreachable();
        }

        /// N-d interpolation of the input data at a given coordinates.
        /// Overload specifying the shape passed to the interpolation function.
        template<bool ASSUME_INBOUND = false,
                 bool OPTIMISTIC_INBOUND_CHECK = DEFAULT_OPTIMISTIC_INBOUND_CHECK,
                 bool OPTIMISTIC_CHECK_OOBOUND = DEFAULT_OPTIMISTIC_CHECK_OOBOUND,
                 nt::interpable_nd<BORDER_, N> Input, nt::any_of<f32, f64> C, usize A1, usize A2>
        [[nodiscard]] NOA_HD constexpr auto get(
            const Input& input,
            const Vec<C, N, A1>& coordinates,
            const Shape<index_type, N, A2>& shape
        ) const -> nt::mutable_value_type_t<Input> {
            if constexpr (nt::textureable_nd<Input, BORDER, N>) {
                return nx::interpolate_using_texture<INTERP, BORDER>(input, coordinates);
            } else if constexpr (not TEXTURE_ONLY) {
                return nx::interpolate<
                    INTERP, BORDER, ASSUME_INBOUND, OPTIMISTIC_INBOUND_CHECK, OPTIMISTIC_CHECK_OOBOUND
                >(input, coordinates, shape, m_cvalue);
            } else {
                static_assert(nt::always_false<Input>, "The input is not texture-able (given the BORDER and number of axes N) and the interpolator is TEXTURE_ONLY=true");
            }
            unreachable();
        }

        [[nodiscard]] NOA_HD constexpr auto shape() const noexcept -> const auto& { return m_shape; }
        [[nodiscard]] NOA_HD constexpr auto cvalue() const noexcept -> const auto& { return m_cvalue; }

    private:
        NOA_NO_UNIQUE_ADDRESS shape_nd_or_empty_type m_shape{};
        NOA_NO_UNIQUE_ADDRESS cvalue_or_empty_type m_cvalue{};
    };

    /// Interpolates {1|2|3}d (power) spectra with a given FFT layout, using a given interpolation.
    /// \tparam REMAP: FFT layout of the input. The output layout is ignored.
    /// See Interpolator for more details.
    template<nf::Layout REMAP, Interp INTERP_, usize N, nt::integer I, bool NO_SHAPE = false, bool TEXTURE_ONLY = false>
    class InterpolatorSpectrum {
    public:
        using offset_type = I;
        using index_type = std::make_signed_t<offset_type>;
        using shape_nd_type = Shape<index_type, N>;
        using bound_nd_type = Vec<index_type, N>;
        using shape_nd_or_empty_type = std::conditional_t<NO_SHAPE, Empty, shape_nd_type>;
        using bounds_nd_or_empty_type = std::conditional_t<NO_SHAPE or TEXTURE_ONLY, Empty, Pair<bound_nd_type, bound_nd_type>>;

        static constexpr Interp INTERP = INTERP_;
        static constexpr Border BORDER = Border::ZERO;
        static constexpr size_t SIZE = N;
        static_assert(N == 1 or N == 2 or N == 3);

        static constexpr bool DEFAULT_ALWAYS_FLIP_PER_INDEX = false;
        static constexpr bool DEFAULT_OPTIMISTIC_INBOUND_CHECK = true;

    public:
        /// Unsafe default construction...
        constexpr InterpolatorSpectrum() = default;

        /// Constructs an interpolator.
        NOA_HD constexpr InterpolatorSpectrum(shape_nd_type shape) noexcept {
            if constexpr (not NO_SHAPE) {
                m_shape = shape;
                if constexpr (not TEXTURE_ONLY)
                    m_bounds = nf::frequency_bounds<REMAP.is_hx2xx()>(shape);
            }
        }

    public:
        /// N-d interpolation of the input spectrum at a given frequency.
        /// \param input        Unbatched ND input.
        /// \param frequency    Unnormalized and centered frequency.
        template<bool ASSUME_INBOUND = false,
                 bool ALWAYS_FLIP_PER_INDEX = DEFAULT_ALWAYS_FLIP_PER_INDEX,
                 bool OPTIMISTIC_INBOUND_CHECK = DEFAULT_OPTIMISTIC_INBOUND_CHECK,
                 nt::interpable_nd<Border::ZERO, N> Input,
                 nt::any_of<f32, f64> C, usize A> requires (not NO_SHAPE)
        [[nodiscard]] NOA_HD constexpr auto get(
            const Input& input,
            const Vec<C, N, A>& frequency
        ) const -> nt::mutable_value_type_t<Input> {
            if constexpr (nt::textureable_nd<Input, BORDER, N>) {
                return nx::interpolate_spectrum_using_texture<
                    REMAP, INTERP, ALWAYS_FLIP_PER_INDEX
                >(input, frequency, m_shape);
            } else if constexpr (not TEXTURE_ONLY) {
                return nx::interpolate_spectrum<
                    REMAP, INTERP, ASSUME_INBOUND, ALWAYS_FLIP_PER_INDEX, OPTIMISTIC_INBOUND_CHECK
                >(input, frequency, m_shape, m_bounds);
            } else {
                static_assert(nt::always_false<Input>, "The input is not texture-able (given the BORDER and number of axes N) and the interpolator is TEXTURE_ONLY=true");
            }
            unreachable();
        }

        /// N-d interpolation of the input data at a given frequency.
        /// Overload specifying the shape and bounds.
        /// If the input is a texture, the bounds are not used and empty can be entered.
        template<bool ASSUME_INBOUND = false,
                 bool ALWAYS_FLIP_PER_INDEX = DEFAULT_ALWAYS_FLIP_PER_INDEX,
                 bool OPTIMISTIC_INBOUND_CHECK = DEFAULT_OPTIMISTIC_INBOUND_CHECK,
                 nt::interpable_nd<Border::ZERO, N> Input,
                 nt::any_of<f32, f64> C, usize A1, usize A2,
                 typename Bounds = Empty>
        [[nodiscard]] NOA_HD constexpr auto get(
            const Input& input,
            const Vec<C, N, A1>& frequency,
            const Shape<index_type, N, A2>& shape,
            const Bounds& bounds
        ) const -> nt::mutable_value_type_t<Input> {
            if constexpr (nt::textureable_nd<Input, BORDER, N>) {
                return nx::interpolate_spectrum_using_texture<
                    REMAP, INTERP, ALWAYS_FLIP_PER_INDEX
                >(input, frequency, shape);
            } else if constexpr (not TEXTURE_ONLY) {
                return nx::interpolate_spectrum<
                    REMAP, INTERP, ASSUME_INBOUND, ALWAYS_FLIP_PER_INDEX, OPTIMISTIC_INBOUND_CHECK
                >(input, frequency, shape, bounds);
            } else {
                static_assert(nt::always_false<Input>, "The input is not texture-able (given the BORDER and number of axes N) and the interpolator is TEXTURE_ONLY=true");
            }
            unreachable();
        }

        [[nodiscard]] NOA_HD constexpr auto shape() const noexcept -> const auto& { return m_shape; }
        [[nodiscard]] NOA_HD constexpr auto bounds() const noexcept -> const auto& { return m_bounds; }

    private:
        NOA_NO_UNIQUE_ADDRESS shape_nd_or_empty_type m_shape{};
        NOA_NO_UNIQUE_ADDRESS bounds_nd_or_empty_type m_bounds{};
    };
}

namespace noa::traits {
    template<nx::Interp INTERP, Border BORDER, usize N, typename I,  typename V, bool A, bool B>
    struct proclaim_is_interpolator<nx::Interpolator<INTERP, BORDER, N, I, V, A, B>> : std::true_type {};

    template<nx::Interp INTERP, Border BORDER, usize N, typename I,  typename V, bool A, bool B, usize S>
    struct proclaim_is_interpolator_nd<nx::Interpolator<INTERP, BORDER, N, I, V, A, B>, S> : std::bool_constant<N == S> {};

    template<nf::Layout REMAP, nx::Interp INTERP, usize N, typename I, bool A, bool B>
    struct proclaim_is_interpolator_spectrum<nx::InterpolatorSpectrum<REMAP, INTERP, N, I, A, B>> : std::true_type {};

    template<nf::Layout REMAP, nx::Interp INTERP, usize N, typename I, bool A, bool B, usize S>
    struct proclaim_is_interpolator_spectrum_nd<nx::InterpolatorSpectrum<REMAP, INTERP, N, I, A, B>, S> : std::bool_constant<N == S> {};
}
