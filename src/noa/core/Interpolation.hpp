#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/indexing/Offset.hpp"
#include "noa/core/math/Constant.hpp"
#include "noa/core/math/Generic.hpp"
#include "noa/core/types/Vec.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/Enums.hpp"
#include "noa/core/Remap.hpp"
#include "noa/core/fft/Frequency.hpp"
#include "noa/core/utils/Misc.hpp"

namespace noa::guts {
    template<Remap REMAP, typename T, nt::integer I, size_t N, size_t A0, size_t A1>
    requires (REMAP.is_xc2xx() or nt::integer<T>)
    constexpr auto interp_frequency_to_index(Vec<T, N, A0> frequency, const Shape<I, N, A1>& shape) {
        auto to_index = [](const auto& f, const auto& s) {
            auto i = f + static_cast<T>(s / 2); // frequency [left,right] -> freq [0,s)
            if constexpr (not REMAP.is_xc2xx() and nt::integer<T>)
                i = noa::fft::ifftshift(i, s); // convert to non-centered input
            return i;
        };
        if constexpr (N >= 2) // 3d=z, 2d=y
            frequency[0] = to_index(frequency[0], shape[0]);
        if constexpr (N == 3) // 3d=y
            frequency[1] = to_index(frequency[1], shape[1]);
        if constexpr (REMAP.is_fx2xx()) // x
            frequency[N - 1] = to_index(frequency[N - 1], shape[N - 1]);
        return frequency;
    }
}

namespace noa {
    /// Enum-class-like object encoding an interpolation method.
    /// \note "_FAST" methods allow the use of lerp fetches (e.g. CUDA textures in linear mode) to accelerate the
    ///       interpolation. If textures are not provided to the Interpolator (see below), these methods are equivalent to the non-
    ///       "_FAST" methods. Textures provide multidimensional caching, hardware interpolation (nearest or lerp) and
    ///       addressing (see Border). While it may result in faster computation, textures usually encodes the
    ///       floating-point coordinates (usually f32 or f64 values) at which to interpolate using low precision
    ///       representations (e.g. CUDA's textures use 8 bits decimals), thus leading to an overall lower precision
    ///       operation than software interpolation.
    struct Interp {
        enum class Method : i32 {
            /// Nearest neighbour interpolation.
            NEAREST = 0,
            NEAREST_FAST = 100,

            /// Linear interpolation (lerp).
            LINEAR = 1,
            LINEAR_FAST = 101,

            /// Linear interpolation with cosine smoothing.
            COSINE = 2,
            COSINE_FAST = 102,

            /// Cubic interpolation.
            CUBIC = 3,
            CUBIC_FAST = 103,

            /// Cubic B-spline interpolation.
            CUBIC_BSPLINE = 4,
            CUBIC_BSPLINE_FAST = 104,

            /// Windowed-sinc interpolation, with Lanczos window of size 4, 6, or 8.
            LANCZOS4 = 5,
            LANCZOS6 = 6,
            LANCZOS8 = 7,
            LANCZOS4_FAST = 105,
            LANCZOS6_FAST = 106,
            LANCZOS8_FAST = 107,
        } value;

    public: // simplify Interp::Method into Interp
        using enum Method;
        NOA_HD constexpr /*implicit*/ Interp(Method value_) noexcept: value(value_) {}
        NOA_HD constexpr /*implicit*/ operator Method() const noexcept { return value; }

    public: // additional methods
        /// Whether the interpolation method is equal to any of the entered values.
        [[nodiscard]] NOA_HD constexpr bool is_any(auto... values) const noexcept {
            return ((value == values) or ...);
        }

        /// Whether the interpolation method, or its (non-)fast alternative, is equal to any of the entered values.
        [[nodiscard]] NOA_HD constexpr bool is_almost_any(auto... values) const noexcept {
            auto underlying = to_underlying(value);
            if (underlying >= 100)
                underlying -= 100;
            auto v = static_cast<Method>(underlying);
            return ((v == values) or ...);
        }

        /// Whether the interpolation method allows fast computation using texture-lerp.
        [[nodiscard]] NOA_HD constexpr bool is_fast() const noexcept {
            return to_underlying(value) >= 100;
        }

        /// Get the size of the interpolation window.
        [[nodiscard]] NOA_HD constexpr auto window_size() const noexcept -> i32 {
            switch (value) {
                case NEAREST:
                case LINEAR:
                case COSINE:
                case NEAREST_FAST:
                case LINEAR_FAST:
                case COSINE_FAST:
                    return 2;
                case CUBIC:
                case CUBIC_FAST:
                case CUBIC_BSPLINE:
                case CUBIC_BSPLINE_FAST:
                case LANCZOS4:
                case LANCZOS4_FAST:
                    return 4;
                case LANCZOS6:
                case LANCZOS6_FAST:
                    return 6;
                case LANCZOS8:
                case LANCZOS8_FAST:
                    return 8;
            }
            return 0; // unreachable
        }
    };

    inline std::ostream& operator<<(std::ostream& os, Interp interp) {
        switch (interp) {
            case Interp::NEAREST:
                return os << "Interp::NEAREST";
            case Interp::NEAREST_FAST:
                return os << "Interp::NEAREST_FAST";
            case Interp::LINEAR:
                return os << "Interp::LINEAR";
            case Interp::LINEAR_FAST:
                return os << "Interp::LINEAR_FAST";
            case Interp::COSINE:
                return os << "Interp::COSINE";
            case Interp::COSINE_FAST:
                return os << "Interp::COSINE_FAST";
            case Interp::CUBIC:
                return os << "Interp::CUBIC";
            case Interp::CUBIC_FAST:
                return os << "Interp::CUBIC_FAST";
            case Interp::CUBIC_BSPLINE:
                return os << "Interp::CUBIC_BSPLINE";
            case Interp::CUBIC_BSPLINE_FAST:
                return os << "Interp::CUBIC_BSPLINE_FAST";
            case Interp::LANCZOS4:
                return os << "Interp::LANCZOS4";
            case Interp::LANCZOS6:
                return os << "Interp::LANCZOS6";
            case Interp::LANCZOS8:
                return os << "Interp::LANCZOS8";
            case Interp::LANCZOS4_FAST:
                return os << "Interp::LANCZOS4_FAST";
            case Interp::LANCZOS6_FAST:
                return os << "Interp::LANCZOS6_FAST";
            case Interp::LANCZOS8_FAST:
                return os << "Interp::LANCZOS8_FAST";
        }
        return os;
    }
}

namespace fmt {
    template<> struct formatter<noa::Interp> : ostream_formatter {};
}

namespace noa::traits {
    namespace guts {
        template<typename T, size_t... N>
        struct fetchable_nd_t {
            template<size_t S, size_t... J>
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

    template<typename T, Border BORDER, size_t... N>
    concept textureable_nd = std::copyable<std::remove_cv_t<T>> and T::BORDER == BORDER and requires {
        typename T::value_type;
        typename T::index_type;
    } and guts::fetchable_nd_t<T, N...>::value;

    template<typename T, Remap REMAP, Interp INTERP, size_t... N>
    concept textureable_spectrum_nd =
        textureable_nd<T, Border::ZERO, N...> and
        (REMAP.is_xc2xx() or not INTERP.is_fast());

    template<typename T, Border BORDER, size_t... N>
    concept lerpable_nd = textureable_nd<T, BORDER, N...> and T::INTERP == Interp::LINEAR_FAST;

    template<typename T, Border BORDER, size_t... N>
    concept nearestable_nd = textureable_nd<T, BORDER, N...> and T::INTERP == Interp::NEAREST_FAST;

    template<typename T, Border BORDER, size_t... N>
    concept interpable_nd =
        std::copyable<std::remove_cv_t<T>> and
        nt::real_or_complex<typename T::value_type> and
        nt::any_of<typename T::index_type, i32, u32, i64, u64> and
        (readable_nd<std::remove_reference_t<decltype(std::declval<const T&>()[0])>, N...> or
         textureable_nd<std::remove_reference_t<decltype(std::declval<const T&>()[0])>, BORDER, N...>);

    template<typename T, Remap REMAP, Interp INTERP, size_t... N>
    concept interpable_spectrum_nd = std::copyable<std::remove_cv_t<T>> and
        nt::real_or_complex<typename T::value_type> and
        nt::any_of<typename T::index_type, i32, u32, i64, u64> and
        (readable_nd<std::remove_reference_t<decltype(std::declval<const T&>()[0])>, N...> or
         textureable_spectrum_nd<std::remove_reference_t<decltype(std::declval<const T&>()[0])>, REMAP, INTERP, N...>);
}

namespace noa {
    /// Computes the interpolation weights.
    /// \tparam INTERP  Interpolation method.
    /// \tparam Weight  Weight value type.
    /// \tparam Coord   Coordinate value type. If it is a Vec, then Weight must be a Vec too.
    /// \param fraction Fraction(s), between 0 and 1.
    /// \returns Interpolation weights in the form of Vec<Weight, N>, where N is the size of the interpolation window.
    template<Interp INTERP, typename Weight, typename Coord>
    requires (nt::vec_real<Weight, Coord> or nt::real<Weight, Coord>)
    NOA_IHD constexpr auto interpolation_weights(Coord fraction) {
        using real_t = nt::value_type_t<Weight>;
        const auto f = static_cast<Weight>(fraction);

        if constexpr (INTERP.is_almost_any(Interp::NEAREST)) {
            // This is not used by the Interpolator.
            // In practice, the rounding is done by the interpolation function
            // and only the correct element (with weight=1) is read/fetched.
            Vec<Weight, 2> coefficients{};
            if constexpr (nt::vec<Weight>) {
                for (size_t i{}; auto e: round(f))
                    coefficients[i++][static_cast<i32>(e)] = 1;
            } else {
                coefficients[static_cast<i32>(round(f))] = 1;
            }
            return coefficients;

        } else if constexpr (INTERP.is_almost_any(Interp::LINEAR)) {
            return Vec{1 - f, f};

        } else if constexpr (INTERP.is_almost_any(Interp::COSINE)) {
            const auto t = (1 - cos(fraction * Constant<real_t>::PI)) / static_cast<real_t>(2);
            return Vec{1 - t, t};

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
            constexpr size_t SIZE = INTERP.window_size();
            constexpr size_t CENTER = SIZE / 2 - 1;

            // Shortcut for identity transform, i.e. x ~= 0.
            // FIXME
            if (vall([](auto fi){ return abs(fi) < std::numeric_limits<real_t>::epsilon(); }, f)) {
                Vec<Weight, SIZE> coefficients{};
                coefficients[CENTER] = 1;
                return coefficients;
            }

            // Instead of computing the windowed-sinc for every point in the nd-window, use this trick from OpenCV
            // to only compute one sin and cos per dimension, regardless of the window size. See:
            // https://github.com/opencv/opencv/blob/master/modules/imgproc/src/imgwarp.cpp#L162
            // I think this relies on the Chebyshev polynomials, but I'm not sure.
            // Regardless, it gives the expected windowed-sinc, and it works for any N (N=[4,6,8] in our case).
            const auto x0 = -(f + CENTER);
            const auto y0 = x0 * Constant<real_t>::PI * static_cast<real_t>(0.25);
            const auto s0 = sin(y0);
            const auto c0 = cos(y0);

            constexpr auto s45 = static_cast<real_t>(0.7071067811865475); // sin(Constant<real_t>::PI / 4);
            constexpr real_t cs[][2]=
                    {{1, 0}, {-s45, -s45}, {0, 1}, {s45, -s45}, {-1, 0}, {s45, s45}, {0, -1}, {-s45, s45}};

            Weight sum{};
            Vec<Weight, SIZE> coefficients{};
            for (size_t i{}; i < SIZE; i++) {
                const auto fi = -(f + CENTER - static_cast<real_t>(i));
                const auto y = fi * Constant<real_t>::PI * static_cast<real_t>(0.25);
                coefficients[i] = (cs[i][0] * s0 + cs[i][1] * c0) / (y * y);
                sum += coefficients[i];
            }

            // Normalize the convolution kernel.
            coefficients *= 1 / sum;
            return coefficients;

        } else {
            static_assert(nt::always_false<Weight>);
        }
    }

    /// Interpolates the 1d|2d|3d input data at the given coordinates.
    /// \tparam INTERP      Interpolation method.
    /// \tparam BORDER      Border type.
    /// \param input        Readable type, usually an Accessor(Value) or Span, mapping the input array.
    /// \param coordinate   ((D)H)W coordinates to interpolate at.
    /// \param shape        ((D)H)W shape of the input array.
    /// \param cvalue       Constant value, only used for BORDER == Border::VALUE.
    template<Interp INTERP, Border BORDER, size_t N,
             nt::any_of<f32, f64> Coord,
             nt::readable_nd<N> T,
             nt::sinteger SInt,
             size_t A0, size_t A1, typename C>
    requires (1 <= N and N <= 3 and (BORDER != Border::VALUE or nt::same_as_mutable_value_type_of<C, T>))
    NOA_HD constexpr auto interpolate(
            const T& input,
            const Vec<Coord, N, A0>& coordinate,
            const Shape<SInt, N, A1>& shape,
            const C& cvalue
    ) noexcept -> nt::mutable_value_type_t<T> {
        using value_t = nt::mutable_value_type_t<T>;
        using real_t = nt::value_type_t<value_t>;
        using indices_t = Vec<SInt, N, A1>;
        using weight_t = Vec<real_t, N, next_power_of_2(alignof(real_t) * N)>;

        // Utility to read from the input while accounting for the border mode.
        auto value_at = [&](const auto& indices) {
            if constexpr (BORDER == Border::ZERO or BORDER == Border::VALUE) {
                if (ni::is_inbound(shape, indices))
                    return input(indices);
                if constexpr (BORDER == Border::ZERO)
                    return value_t{};
                else
                    return cvalue;
            } else {
                return input(ni::index_at<BORDER>(indices, shape));
            }
        };

        if constexpr (INTERP == Interp::NEAREST) {
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
            const auto fraction = coordinate - floored; // TODO increase alignment

            // If indices are inbound, no need to interpolate in the case of AccessorValue.
            if constexpr (nt::accessor_value<T>) {
                if (vall([](auto i, auto s) { return i - START >= 0 and i + END < s; }, indices, shape))
                    return input();
            }

            // Interpolate.
            Vec<weight_t, SIZE> weights = interpolation_weights<INTERP, weight_t>(fraction);
            value_t interpolant{};
            if constexpr (N == 1) {
                for (SInt x{START}; x <= END; ++x)
                    interpolant += value_at(indices + x) * weights[x - START][0];

            } else if constexpr (N == 2) {
                for (SInt y{START}; y <= END; ++y) {
                    value_t interpolant_y{};
                    for (SInt x{START}; x <= END; ++x)
                        interpolant_y += value_at(indices + indices_t{y, x}) * weights[x - START][1];
                    interpolant += interpolant_y * weights[y - START][0];
                }
            } else if constexpr (N == 3) {
                for (SInt z{START}; z <= END; ++z) {
                    value_t interpolant_z{};
                    for (SInt y{START}; y <= END; ++y) {
                        value_t interpolant_y{};
                        for (SInt x{START}; x <= END; ++x)
                            interpolant_y += value_at(indices + indices_t{z, y, x}) * weights[x - START][2];
                        interpolant_z += interpolant_y * weights[y - START][1];
                    }
                    interpolant += interpolant_z * weights[z - START][0];
                }
            } else {
                static_assert(nt::always_false<SInt>);
            }
            return interpolant;
        }
    }

    /// Interpolates the 1d|2d|3d input data at the given coordinates.
    /// \details This function computes the interpolated value by computing a series of intermediate lerps. If these
    ///          lerps are hardware accelerated (e.g. CUDA textures), this reduces the number of reads by a factor of 2
    ///          in each dimension. For instance, for 4x4x4 cubic 3d windows, instead of needing 64 reads, only 8 lerp
    ///          fetches are needed. If these fetches are hardware accelerated, it can speedup the overall computation.
    ///          In the case of Interp::NEAREST_FAST, nearest-neighbor fetches are used instead, if available.
    ///
    /// \tparam INTERP      Interpolation method. Only fast modes are supported, indicating that a low precision is
    ///                     tolerated.
    /// \tparam BORDER      Border mode. Must be equal to the input border mode, since out-of-bound coordinates are
    ///                     entirely handled by the lerpable input.
    /// \tparam N           Number of dimensions.
    /// \param input        Texture-like object mapping a N-d input array.
    /// \param coordinate   ((D)H)W coordinates to interpolate at.
    template<Interp INTERP, Border BORDER, size_t N, size_t A,
             nt::textureable_nd<BORDER, N> T,
             nt::any_of<f32, f64> R>
    requires (1 <= N and N <= 3 and INTERP.is_fast())
    NOA_HD constexpr auto interpolate_using_texture(
            const T& input,
            const Vec<R, N, A>& coordinate
    ) noexcept -> nt::mutable_value_type_t<T> {
        if constexpr (INTERP == Interp::NEAREST_FAST and nt::nearestable_nd<T, BORDER, N>) {
            // Special case for nearest-neighbor interpolation in fast mode. Let the texture do everything.
            return input.fetch(coordinate);

        } else if constexpr (INTERP.is_almost_any(Interp::NEAREST)) {
            // Special case for nearest-neighbor interpolation in accurate mode. Here the "interpolation"
            // is done in software and the texture, regardless of its mode, fetches at the closest integer location.
            return input.fetch(round(coordinate));

        } else {
            using value_t = nt::mutable_value_type_t<T>;
            using real_t = nt::value_type_t<value_t>;
            using coord_t = R;
            using coordn_t = Vec<coord_t, N, A>;
            using weight_t = Vec<real_t, N, next_power_of_2(alignof(real_t) * N)>;

            constexpr i32 SIZE = INTERP.window_size();
            const coordn_t indices = floor(coordinate);
            const coordn_t fraction = coordinate - indices;
            const Vec<weight_t, static_cast<size_t>(SIZE)> weights = interpolation_weights<INTERP, weight_t>(fraction);

            if constexpr (INTERP.is_fast() and nt::lerpable_nd<T, BORDER, N>) {
                // Use hardware accelerated lerp(s) to compute the interpolated value. This works for any
                // interpolation method in fast mode (allowing lower precision).

                // This was expanded from the original implementation of Daniel Ruijters
                // for Cubic B-spline interpolation in CUDA: http://www.dannyruijters.nl/cubicinterpolation/
                // Compute the adjusted coordinates for the lerp and the weights.
                constexpr i32 HALF = SIZE / 2;
                Vec<weight_t, static_cast<size_t>(HALF)> lerp_weight;
                Vec<coordn_t, static_cast<size_t>(HALF)> lerp_coord;
                for (i32 i{}; i < HALF; ++i) {
                    const i32 index = (i * 2);
                    const i32 offset = -(HALF - 1 - index);
                    lerp_weight[i] = weights[index] + weights[index + 1];
                    lerp_coord[i] = input.fetch_preprocess(
                            static_cast<coord_t>(weights[index + 1] / lerp_weight[i]) +
                            indices + static_cast<coord_t>(offset));
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
                        value += input.fetch(indices + x) * weights[x - START][0];

                } else if constexpr (N == 2) {
                    for (i32 y{START}; y <= END; ++y) {
                        value_t value_y{};
                        for (i32 x{START}; x <= END; ++x)
                            value_y += input.fetch(indices + coordn_t::from_values(y, x)) * weights[x - START][1];
                        value += value_y * weights[y - START][0];
                    }
                } else if constexpr (N == 3) {
                    for (i32 z{START}; z <= END; ++z) {
                        value_t value_z{};
                        for (i32 y{START}; y <= END; ++y) {
                            value_t value_y{};
                            for (i32 x{START}; x <= END; ++x)
                                value_y += input.fetch(indices + coordn_t::from_values(z, y, x)) * weights[x - START][2];
                            value_z += value_y * weights[y - START][1];
                        }
                        value += value_z * weights[z - START][0];
                    }
                }
                return value;
            }
        }
    }

    /// Interpolates the 1d|2d|3d input spectrum at the given coordinates.
    /// \tparam REMAP       Remap operator. The output layout is ignored.
    /// \tparam INTERP      Interpolation method.
    /// \param input        Readable type, usually an Accessor(Value) or Span, mapping the input spectrum.
    /// \param frequency    Centered ((D)H)W frequency, in samples (not normalized).
    /// \param shape        ((D)H)W (logical) shape of the input spectrum.
    template<Remap REMAP, Interp INTERP, size_t N,
             nt::readable_nd<N> T,
             nt::any_of<f32, f64> Coord,
             nt::sinteger Int,
             size_t A0, size_t A1>
    NOA_HD constexpr auto interpolate_spectrum(
            const T& input,
            const Vec<Coord, N, A0>& frequency,
            const Shape<Int, N, A1>& shape
    ) noexcept -> nt::mutable_value_type_t<T> {
        using value_t = nt::mutable_value_type_t<T>;
        using real_t = nt::value_type_t<value_t>;
        using indices_t = Vec<Int, N, A1>;
        using weight_t = Vec<value_t, N, next_power_of_2(alignof(real_t) * N)>;

        // Handle non-redundant inputs by switching to the complex conjugate.
        // For windows of size 2, we can flip the frequency at this point because the window doesn't have negative
        // offsets, so if the frequency is flipped, we know that all indices will be on the right side of the DC.
        // For larger windows, this is not the case and flipping needs to be done for each index in the window.
        constexpr Int SIZE = INTERP.window_size();
        constexpr bool FLIP = REMAP.is_hx2xx();
        constexpr bool FLIP_PER_INDEX = SIZE > 2;
        real_t conjugate{};
        if constexpr (FLIP and not FLIP_PER_INDEX) {
            if (frequency[N - 1] < 0) {
                frequency *= -1;
                if constexpr (nt::complex<value_t>)
                    conjugate = -1;
            }
        }

        const auto left = -shape.vec / 2;
        const auto right = (shape.vec - 1) / 2;
        auto update_at = [&](auto freq, auto weight, auto& output) {
            real_t conj{1};
            if constexpr (FLIP and FLIP_PER_INDEX) {
                if (freq[N - 1] < 0) {
                    freq *= -1;
                    if constexpr (nt::complex<value_t>)
                        conj = -1;
                }
            }

            const bool is_inside_spectrum = vall_enumerate([]<size_t J>(auto l, auto r, auto i) {
                if constexpr (REMAP.is_hx2xx() and J == N - 1)
                    return i <= r; // for non-redundant spectra, we know i >= 0 along the width
                return l <= i and i <= r;
            }, left, right, freq);

            if (is_inside_spectrum) {
                auto value = input(ng::interp_frequency_to_index<REMAP>(freq));
                if constexpr (FLIP and FLIP_PER_INDEX and nt::complex<value_t>)
                    value.imag *= conj;
                output += value * weight;
            }
        };

        if constexpr (INTERP.is_almost_any(Interp::NEAREST)) {
            // For nearest, the compiler doesn't seem able to optimize away the second read with weight=0,
            // so keep a compile-time branch for this case so that only one value is read from the accessor.
            value_t interpolant{};
            update_at(static_cast<indices_t>(round(frequency)), real_t{1}, interpolant);
            return interpolant;
        } else {
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

            value_t value{};
            if constexpr (N == 1) {
                for (Int x{START}; x <= END; ++x)
                    update_at(indices + x, weights[x - START][0], value);

            } else if constexpr (N == 2) {
                for (Int y{START}; y <= END; ++y) {
                    value_t value_y{};
                    for (Int x{START}; x <= END; ++x)
                        update_at(indices + indices_t{y, x}, weights[x - START][1], value_y);
                    value += value_y * weights[y - START][0];
                }
            } else if constexpr (N == 3) {
                for (Int z{START}; z <= END; ++z) {
                    value_t value_z{};
                    for (Int y{START}; y <= END; ++y) {
                        value_t value_y{};
                        for (Int x{START}; x <= END; ++x)
                            update_at(indices + indices_t{z, y, x}, weights[x - START][2], value_y);
                        value_z += value_y * weights[y - START][1];
                    }
                    value += value_z * weights[z - START][0];
                }
            }
            if constexpr (FLIP and not FLIP_PER_INDEX and nt::complex<value_t>)
                value.imag *= conjugate;
            return value;
        }
    }

    /// Interpolates the 1d|2d|3d input centered spectrum at the given coordinates.
    /// \tparam REMAP       Remap operator. The output layout is ignored. The input layout must be centered
    ///                     for fast interpolation modes; non-centered inputs are only supported with the
    ///                     accurate interpolation modes.
    /// \tparam INTERP      Interpolation method.
    /// \param frequency    Centered ((D)H)W frequency, in samples (i.e. not normalized).
    /// \param texture      Textureable type mapping the input spectrum, using Border::ZERO addressing.
    /// \param shape        ((D)H)W (logical) shape of the input spectrum.
    template<Remap REMAP, Interp INTERP, size_t N,
             nt::textureable_nd<Border::ZERO, N> T,
             nt::any_of<f32, f64> Coord,
             nt::sinteger Int,
             size_t A0, size_t A1>
    requires (REMAP.is_xc2xx() or not INTERP.is_fast())
    NOA_HD constexpr auto interpolate_spectrum_using_texture(
            const T& input,
            const Vec<Coord, N, A0>& frequency,
            const Shape<Int, N, A1>& shape
    ) noexcept -> nt::mutable_value_type_t<T> {
        using value_t = nt::mutable_value_type_t<T>;
        using real_t = nt::value_type_t<value_t>;
        using coordn_t = Vec<Coord, N, A0>;
        using indices_t = Vec<Int, N, A1>;

        // Handle non-redundant inputs by switching to the complex conjugate.
        // For windows of size 2, we can flip the frequency at this point because the window doesn't have negative
        // offsets, so if the frequency is flipped, we know that all indices will be on the right side of the DC.
        // For larger windows, this is not the case and flipping needs to be done for each index in the window.
        constexpr Int SIZE = INTERP.window_size();
        constexpr bool FLIP = REMAP.is_hx2xx();
        constexpr bool FLIP_PER_INDEX = SIZE > 2;
        real_t conjugate{};
        if constexpr (FLIP and not FLIP_PER_INDEX) {
            if (frequency[N - 1] < 0) {
                frequency *= -1;
                if constexpr (nt::complex<value_t>)
                    conjugate = -1;
            }
        }

        value_t value;
        if constexpr (INTERP == Interp::NEAREST_FAST and nt::nearestable_nd<T, Border::ZERO, N>) {
            // Special case for nearest-neighbor interpolation in fast mode. Let the texture do everything.
            value = input.fetch(ng::interp_frequency_to_index<REMAP>(frequency, shape));

        } else if constexpr (INTERP.is_almost_any(Interp::NEAREST)) {
            // Special case for nearest-neighbor interpolation in accurate mode. Here the "interpolation"
            // is done in software and the texture, regardless of its mode, fetches at the closest integer location.
            auto indices = ng::interp_frequency_to_index<REMAP>(static_cast<indices_t>(round(frequency)), shape);
            value = input.fetch(static_cast<coordn_t>(indices));

        } else {
            auto value_at = [&shape, &input](auto freq) {
                real_t conj{1};
                if constexpr (FLIP and FLIP_PER_INDEX) {
                    if (freq[N - 1] < 0) {
                        freq *= -1;
                        if constexpr (nt::complex<value_t>)
                            conj = -1;
                    }
                }
                freq = ng::interp_frequency_to_index<REMAP>(freq, shape);
                value_t fetched = input.fetch(static_cast<coordn_t>(freq));
                if constexpr (FLIP and FLIP_PER_INDEX and nt::complex<value_t>)
                    fetched.imag = conj;
                return fetched;
            };

            const coordn_t floored = floor(frequency);
            const coordn_t fraction = frequency - floored;
            using weight_t = Vec<real_t, N, next_power_of_2(alignof(real_t) * N)>;
            const Vec<weight_t, SIZE> weights = interpolation_weights<INTERP, weight_t>(fraction);

            if constexpr (INTERP.is_fast() and nt::lerpable_nd<T, Border::ZERO>) {
                // This was expanded from the original implementation of Daniel Ruijters
                // for Cubic B-spline interpolation in CUDA: http://www.dannyruijters.nl/cubicinterpolation/
                // Compute the adjusted coordinates for the lerp.
                constexpr Int HALF = SIZE / 2;
                Vec<weight_t, HALF> lw; // lerp weight
                Vec<coordn_t, HALF> lc; // lerp coordinate
                for (Int i{}; i < HALF; ++i) {
                    const auto index = (i * 2);
                    const auto offset = static_cast<Coord>(-(HALF - 1 - index));
                    lw[i] = weights[index] + weights[index + 1];
                    lc[i] = static_cast<Coord>(weights[index + 1] / lw[i]) + floored + offset;
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
                // Hardware interpolation is not allowed, but we can still use the texture addressing
                // and do the interpolation in software. This mode allows for non-redundant spectra.
                constexpr Int START = -(SIZE - 1) / 2;
                constexpr Int END = SIZE / 2;
                const auto indices = static_cast<indices_t>(floored);
                if constexpr (N == 1) {
                    for (Int x{START}; x <= END; ++x)
                        value += value_at(indices + x) * weights[x - START][0];

                } else if constexpr (N == 2) {
                    for (Int y{START}; y <= END; ++y) {
                        value_t value_y{};
                        for (Int x{START}; x <= END; ++x)
                            value_y += value_at(indices + indices_t{y, x}) * weights[x - START][1];
                        value += value_y * weights[y - START][0];
                    }
                } else if constexpr (N == 3) {
                    for (Int z{START}; z <= END; ++z) {
                        value_t value_z{};
                        for (Int y{START}; y <= END; ++y) {
                            value_t value_y{};
                            for (Int x{START}; x <= END; ++x)
                                value_y += value_at(indices + indices_t{z, y, x}) * weights[x - START][2];
                            value_z += value_y * weights[y - START][1];
                        }
                        value += value_z * weights[z - START][0];
                    }
                }
            }
        }
        if constexpr (FLIP and not FLIP_PER_INDEX and nt::complex<value_t>)
            value.imag *= conjugate;
        return value;
    }

    /// Interpolates {1|2|3}d (complex) floating-point data, using a given interpolation and border mode.
    ///
    /// \details
    /// Input trait:
    ///     - The input data is abstracted behind the readable and textureable concepts, allowing to support different
    ///       memory layouts and pointer traits (all accessor types are supported). Hardware interpolation and
    ///       addressing is supported via the textureable concept. See the `interpolate_using_texture` function for
    ///       more details. See the `interpable_nd` concept for more details about the requirements on the input type.
    ///     - The interpolator propagates the readable trait, i.e. if the input type is readable, so is the interpolator.
    ///
    /// Out-of-bounds:
    ///     - One of the main differences between these interpolations and what we can find in other cryoEM packages,
    ///       is that the interpolation window can be partially out-of-bound (OOB), that is, elements that are OOB
    ///       are replaced according to a Border. cryoEM packages usually check that all elements are in-bound
    ///       and if there's even one element OOB, they don't interpolate.
    ///     - In texture mode, the texture is responsible for the addressing, so some Border modes may not be supported.
    ///       See `interpolate_using_texture` for more details. If the texture does not have the required mode, i.e.
    ///       BORDER != Input::BORDER, the interpolator tries to fall back on the `interpolate` function (which
    ///       requires the readable_nd trait. If the texture is not readable, a compile time error will be given.
    ///
    /// Coordinate system:
    ///     The coordinate system matches the indexing, as expected.
    ///     For instance, the first data sample at index 0 is located at the coordinate 0 and the coordinate 0.5
    ///     is exactly in between the first and second element. As such, the fractional part of the coordinate
    ///     corresponds to the ratio/weight used by the interpolation functions. In other words, the coordinate system
    ///     locates the data between -0.5 and N-1 + 0.5.
    ///
    /// \tparam N       Number of dimensions of the data. Between 1 and 3.
    /// \tparam INTERP  Interpolation method.
    /// \tparam BORDER  Border mode, aka addressing mode.
    /// \tparam Input   Batched input data. Readable or textureable.
    /// \example
    /// \code
    /// auto data_2d = Span<f32, 4, i64>{...}; // input data, using the BDHW convention (like Array/View)
    /// auto accessor = Accessor<const f32, 3, i64>{data_2d.get(), data_2d.strides().filter(0, 2, 3)}; // 2d batch data, depth=1
    /// auto op = Interpolator<2, Interp::CUBIC, Border::ZERO, decltype(accessor)>(accessor, data_2d.shape().filter(2, 3));
    /// auto coordinate = Vec<f64, 2>{...};
    /// auto interpolated_value_batch0 = op.interpolate(coordinate);
    /// auto interpolated_value_batch2 = op.interpolate(coordinate, 2);
    /// auto value = op(6, 7); // batch=0, height=6, width=7
    /// auto value = op(2, 6, 7); // batch=2, height=6, width=7
    /// \endcode
    template<size_t N, Interp INTERP, Border BORDER, nt::interpable_nd<BORDER, N> Input>
    class Interpolator {
    public:
        using input_type = Input;
        using value_type = input_type::value_type;
        using offset_type = input_type::index_type;
        using mutable_value_type = nt::mutable_value_type_t<value_type>;
        using index_type = std::make_signed_t<offset_type>;

        static constexpr size_t SIZE = N;
        static constexpr bool is_textureable = requires (const input_type& t) {
            { t[0] } -> nt::textureable_nd<BORDER, N>;
        };

        using shape_nd_type = Shape<index_type, N, next_power_of_2(sizeof(index_type) * N)>;
        using shape_nd_or_empty_type = std::conditional_t<is_textureable, Empty, shape_nd_type>;
        using value_or_empty_type = std::conditional_t<BORDER == Border::VALUE, mutable_value_type, Empty>;

    public:
        constexpr Interpolator() = default;

        /// Constructs an interpolator from an accessor-like object.
        /// This stores a copy of the input accessor, shape and cvalue. The created instance
        /// can then be used to interpolate the nd-data (as described in the interpolate function).
        template<size_t A> requires (not is_textureable)
        NOA_HD constexpr Interpolator(
                const input_type& input,
                const Shape<index_type, N, A>& shape,
                mutable_value_type cvalue = mutable_value_type{}
        ) noexcept:
                m_input(input),
                m_shape(shape_nd_type::from_shape(shape))
        {
            if constexpr (not std::is_empty_v<value_or_empty_type>)
                m_cvalue = cvalue;
        }

        /// Constructs an interpolator from a texture-like object.
        /// This stores a copy of the input texture. Note that the addressing is handled by the texture (as described
        /// in the interpolate_using_texture function), so the shape and cvalue are ignored and only provided here
        /// to match the constructor taking accessors.
        template<size_t A> requires is_textureable
        NOA_HD constexpr explicit Interpolator(
                const input_type& input,
                const Shape<index_type, N, A>& = {},
                mutable_value_type = mutable_value_type{}
        ) noexcept:
                m_input(input) {}

    public:
        /// N-d interpolation of the data at a given batch.
        /// \param coordinates  Un-normalized coordinates.
        /// \param batch        Optional batch index. The input object is allowed to ignore the batch
        ///                     and can thus effectively broadcast the input along the batch dimension.
        template<nt::any_of<f32, f64> T, size_t A, nt::integer I = index_type>
        NOA_HD constexpr auto interpolate_at(const Vec<T, N, A>& coordinates, I batch = I{}) const -> mutable_value_type {
            using vec_t = Vec<T, N, next_power_of_2(alignof(T) * N)>;
            if constexpr (is_textureable) {
                return noa::interpolate_using_texture<INTERP, BORDER>(m_input[batch], vec_t::from_vec(coordinates));
            } else { // readable
                return noa::interpolate<INTERP, BORDER>(m_input[batch], vec_t::from_vec(coordinates), m_shape, m_cvalue);
            }
        }

    public: // independently, make it readable if input supports it
        template<nt::integer... I>
        requires (N + 1 == sizeof...(I) and nt::readable_nd<input_type, N + 1>)
        NOA_HD constexpr auto operator()(I... indices) const -> mutable_value_type {
            return m_input(indices...);
        }

        template<nt::integer... I>
        requires (N == sizeof...(I) and nt::readable_nd<input_type, N + 1>)
        NOA_HD constexpr auto operator()(I... indices) const -> mutable_value_type {
            return m_input(0, indices...);
        }

        template<nt::integer I, size_t S, size_t A>
        requires (N == S and nt::readable_nd<input_type, N + 1>)
        NOA_HD constexpr auto operator()(const Vec<I, S, A>& indices) const -> mutable_value_type {
            return m_input(indices.push_front(1));
        }

        template<nt::integer I, size_t S, size_t A>
        requires (N + 1 == S and nt::readable_nd<input_type, N + 1>)
        NOA_HD constexpr auto operator()(const Vec<I, S, A>& indices) const -> mutable_value_type {
            return m_input(indices);
        }

    private:
        input_type m_input{};
        NOA_NO_UNIQUE_ADDRESS shape_nd_or_empty_type m_shape{};
        NOA_NO_UNIQUE_ADDRESS value_or_empty_type m_cvalue{};
    };

    template<size_t N, Remap REMAP, Interp INTERP, nt::interpable_spectrum_nd<REMAP, INTERP, N> Input>
    class InterpolatorSpectrum {
    public:
        using input_type = Input;
        using value_type = input_type::value_type;
        using offset_type = input_type::index_type;
        using mutable_value_type = nt::mutable_value_type_t<value_type>;
        using index_type = std::make_signed_t<offset_type>;
        using shape_nd_type = Shape<index_type, N>;

        static constexpr size_t SIZE = N;
        static constexpr bool is_textureable = requires(const input_type& t) {
            { t[0] } -> nt::textureable_spectrum_nd<REMAP, INTERP, N>;
        };

    public:
        constexpr InterpolatorSpectrum() = default;

        /// Constructs an interpolator.
        /// The created instance can then be used to interpolate the nd-data (as described in the interpolate function).
        NOA_HD constexpr InterpolatorSpectrum(
                const input_type& input,
                shape_nd_type shape
        ) noexcept:
                m_input(input),
                m_shape(shape) {}

    public:
        /// N-d interpolation of the data at a given batch.
        /// \param frequency    Un-normalized centered frequency.
        /// \param batch        Optional batch index. The input object is allowed to ignore the batch
        ///                     and can thus effectively broadcast the input along the batch dimension.
        template<nt::any_of<f32, f64> T, size_t A, nt::integer I = index_type>
        NOA_HD constexpr auto interpolate_spectrum_at(const Vec<T, N, A>& frequency, I batch = I{}) const -> mutable_value_type {
            if constexpr (is_textureable) {
                return noa::interpolate_spectrum_using_texture<REMAP, INTERP>(m_input[batch], frequency, m_shape);
            } else { // readable
                return noa::interpolate_spectrum<REMAP, INTERP>(m_input[batch], frequency, m_shape);
            }
        }

    public: // independently, make it readable if input supports it
        template<nt::integer... I>
        requires (nt::readable_nd<input_type, sizeof...(I)> and (N == sizeof...(I) or N + 1 == sizeof...(I)))
        NOA_HD constexpr auto operator()(I... indices) const -> mutable_value_type {
            return m_input(indices...);
        }

        template<nt::integer I, size_t S, size_t A>
        requires (nt::readable_nd<input_type, N> and (N == S or N + 1 == S))
        NOA_HD constexpr auto operator()(const Vec<I, S, A>& indices) const -> mutable_value_type {
            return m_input(indices);
        }

    private:
        input_type m_input{};
        NOA_NO_UNIQUE_ADDRESS shape_nd_type m_shape{};
    };
}

namespace noa::traits {
    template<size_t N, Interp INTERP, Border BORDER, typename Input>
    struct proclaim_is_interpolator<noa::Interpolator<N, INTERP, BORDER, Input>> : std::true_type {};

    template<size_t N, Interp INTERP, Border BORDER, typename Input, size_t S>
    struct proclaim_is_interpolator_nd<noa::Interpolator<N, INTERP, BORDER, Input>, S> : std::bool_constant<N == S> {};

    template<size_t N, Remap REMAP, Interp INTERP, typename Input>
    struct proclaim_is_interpolator_spectrum<noa::InterpolatorSpectrum<N, REMAP, INTERP, Input>> : std::true_type {};

    template<size_t N, Remap REMAP, Interp INTERP, typename Input, size_t S>
    struct proclaim_is_interpolator_spectrum_nd<noa::InterpolatorSpectrum<N, REMAP, INTERP, Input>, S> : std::bool_constant<N == S> {};
}
