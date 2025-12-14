#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <random>

#include <noa/core/Traits.hpp>
#include <noa/core/types/Half.hpp>
#include <noa/core/types/Shape.hpp>
#include <noa/core/types/Span.hpp>
#include <noa/core/indexing/Offset.hpp>
#include <noa/core/io/IO.hpp>

namespace test {
    extern noa::Path NOA_DATA_PATH; // defined at runtime by main.

    using namespace noa::types;
    namespace nt = noa::traits;

    template<typename T>
    class Randomizer {
    public:
        using value_type = T;
        using scalar_type = nt::value_type_t<value_type>;
        using gen_scalar_type = std::conditional_t<std::is_same_v<scalar_type, noa::f16>, noa::f32, scalar_type>;
        using distributor_type = std::conditional_t<std::is_integral_v<value_type>,
                                                    std::uniform_int_distribution<value_type>,
                                                    std::uniform_real_distribution<gen_scalar_type>>;

    private:
        std::random_device rand_dev;
        std::mt19937 generator;
        distributor_type distribution;

    public:
        template<typename U, typename V>
        constexpr Randomizer(U range_from, V range_to)
                : generator(rand_dev()),
                  distribution(static_cast<gen_scalar_type>(range_from),
                               static_cast<gen_scalar_type>(range_to)) {}

        constexpr auto get() -> value_type {
            if constexpr(nt::complex<T>) {
                return {static_cast<scalar_type>(distribution(generator)),
                        static_cast<scalar_type>(distribution(generator))};
            } else {
                return static_cast<value_type>(distribution(generator));
            }
        }
    };

    template<typename T>
    [[nodiscard]] constexpr auto random_value(T min, T max) -> T {
        return Randomizer<T>(min, max).get();
    }

    template<typename T> requires std::is_array_v<T>
    [[nodiscard]] constexpr auto make_unique(nt::integer auto n) {
        return std::make_unique<T>(noa::safe_cast<size_t>(n));
    }

    template<typename T>
    constexpr void randomize(T* data, nt::integer auto n_elements, auto&& randomizer) {
        using random_t = std::remove_reference_t<decltype(randomizer)>::value_type;
        if constexpr (nt::same_as<T, random_t>) {
            for (i64 idx = 0; idx < static_cast<i64>(n_elements); ++idx)
                data[idx] = randomizer.get();
        } else if constexpr (nt::complex<T>) {
            using value_t = nt::value_type_t<T>;
            randomize(reinterpret_cast<value_t*>(data), n_elements * 2, randomizer);
        } else {
            for (i64 idx = 0; idx < static_cast<i64>(n_elements); ++idx)
                data[idx] = static_cast<T>(randomizer.get());
        }
    }

    template<typename T>
    constexpr auto random(nt::integer auto n_elements, auto&& randomizer) {
        auto data = std::make_unique<T[]>(static_cast<size_t>(n_elements));
        randomize(data.get(), n_elements, randomizer);
        return data;
    }

    struct RandomShapeOptions {
        /// Randomize the batch dimension within this range.
        Pair<i32, i32> batch_range{1, 1};

        /// Whether the DHW dimensions should have only even sizes.
        bool only_even_sizes{false};

        /// Whether the DHW dimensions should have only odd sizes.
        bool only_odd_sizes{false};
    };

    /// Generates a random shape.
    template<std::integral T = isize, size_t N = 4> requires (N >= 1)
    constexpr auto random_shape(
            std::integral auto ndim,
            RandomShapeOptions options = {}
    ) -> Shape<T, N> {
        noa::check(1 <= ndim and ndim <= 4);
        Vec min{32, 32, 32, 32};
        Vec max{1024, 512, 128, 128};

        constexpr auto n_ = static_cast<i32>(N);
        const auto ndim_ = static_cast<i32>(ndim);
        auto randomizer = Randomizer<T>(min[ndim_ - 1], max[ndim_ - 1]);
        const i32 n_iter = std::min(ndim_, n_);
        const i32 offset = std::max(0, n_ - ndim_);

        auto shape = Shape<T, N>::from_value(1);
        for (i32 i{}; i < n_iter; ++i) {
            auto size = randomizer.get();
            if ((options.only_even_sizes and not noa::is_even(size)) or
                (options.only_odd_sizes and noa::is_even(size)))
                size += 1;
            shape[offset + i] = size;
        }

        if (N == 4) {
            shape[0] = Randomizer<T>( // FIXME
                options.batch_range.first,
                options.batch_range.second).get();
        }
        return shape;
    }
    template<std::integral T = isize, size_t N = 4> requires (N >= 1)
    constexpr auto random_shape_batched(
            std::integral auto ndim,
            RandomShapeOptions options = RandomShapeOptions{.batch_range={1, 10}}
    ) -> Shape<T, N> {
        if (options.batch_range.first == 1 and options.batch_range.second == 1)
            options.batch_range = {1, 10};
        return random_shape<T, N>(ndim, options);
    }

    template<typename T, typename U>
    constexpr void fill(T* data, nt::integer auto n_elements, U value) {
        std::fill(data, data + n_elements, static_cast<T>(value));
    }
    template<typename T, typename U>
    constexpr auto fill(nt::integer auto n_elements, U value) {
        auto data = std::make_unique<T[]>(static_cast<size_t>(n_elements));
        fill(data.get(), n_elements, value);
        return data;
    }

    template<typename T>
    constexpr void zero(T* data, nt::integer auto n_elements) {
        std::fill(data, data + n_elements, T{});
    }
    template<typename T>
    constexpr auto zero(nt::integer auto n_elements) {
        auto data = std::make_unique<T[]>(static_cast<size_t>(n_elements));
        zero(data.get(), n_elements);
        return data;
    }

    template<typename T, typename U = i64>
    constexpr void arange(T* data, nt::integer auto n_elements, U start = 0, U step = 1) {
        for (i64 i = 0; i < static_cast<i64>(n_elements); ++i, start += step)
            data[i] = static_cast<T>(start);
    }
    template<typename T, typename U = i64>
    constexpr auto arange(nt::integer auto n_elements, U start = 0, U step = 1) {
        auto data = std::make_unique<T[]>(static_cast<size_t>(n_elements));
        arange(data.get(), n_elements, start, step);
        return data;
    }

    template<typename T>
    constexpr void copy(const T* src, T* dst, nt::integer auto n_elements) {
        std::copy(src, src + n_elements, dst);
    }

    template<typename T, typename U>
    constexpr void scale(T* array, nt::integer auto size, U scale) {
        std::transform(array, array + size, array, [scale](const T& a) { return a * scale; });
    }

    enum class MatchMode {
        Absolute, AbsoluteSafe, Relative
    };

    template<typename T, size_t N>
    class MatchResult {
    public:
        using value_type = T;
        using index_type = isize;
        using indices_type = Vec<index_type, N>;
        using shape_type = Shape<index_type, N>;

    public:
        constexpr explicit operator bool() const {
            return match;
        }

    public:
        value_type total_abs_diff{};
        value_type max_abs_diff{};
        value_type epsilon{};
        value_type lhs_value{};
        value_type rhs_value{};
        shape_type shape{};
        indices_type indices_failed{};
        MatchMode mode{};
        bool match{};
    };

    template<typename T, size_t N>
    class Match {
    public:
        using value_type = T;
        using index_type = isize;
        using strides_type = Strides<index_type, N>;
        using shape_type = Shape<index_type, N>;
        using indices_type = Vec<index_type, N>;
        using result_type = MatchResult<value_type, N>;

    public:
        constexpr Match(
            const value_type* lhs, const strides_type& lhs_strides,
            const value_type* rhs, const strides_type& rhs_strides,
            const shape_type& shape
        ) : m_shape{shape},
            m_lhs_strides{lhs_strides},
            m_rhs_strides{rhs_strides},
            m_lhs{lhs},
            m_rhs{rhs} {}

        constexpr auto check(MatchMode mode, auto epsilon) -> result_type {
            value_type epsilon_ = get_epsilon_(epsilon);
            if (mode == MatchMode::Absolute)
                return check_(Match::check_abs_<value_type>, epsilon_);
            else if (mode == MatchMode::AbsoluteSafe)
                return check_(Match::check_abs_safe_<value_type>, epsilon_);
            else if (mode == MatchMode::Relative)
                return check_(Match::check_rel_<value_type>, epsilon_);
            else
                noa::panic("MathMode not supported");
        }

    private:
        constexpr auto get_epsilon_(auto epsilon) -> value_type {
            value_type out;
            if constexpr (nt::complex<value_type> and not nt::complex<decltype(epsilon)>) {
                using real_t = value_type::value_type;
                out.real = static_cast<real_t>(epsilon);
                out.imag = static_cast<real_t>(epsilon);
            } else {
                out = static_cast<value_type>(epsilon);
            }
            return out;
        }

        template<typename F>
        auto check_(F&& value_checker, value_type epsilon) {
            result_type result{.shape=m_shape, .match=true};

            auto compare_values = [&, this](auto... indices) {
                value_type lhs = m_lhs[noa::indexing::offset_at(m_lhs_strides, indices...)];
                value_type rhs = m_rhs[noa::indexing::offset_at(m_rhs_strides, indices...)];
                const auto [passed, abs_diff] = value_checker(lhs, rhs, epsilon);

                result.total_abs_diff += abs_diff;
                if constexpr (nt::is_complex_v<value_type>) {
                    result.max_abs_diff.real = noa::max(result.max_abs_diff.real, abs_diff.real);
                    result.max_abs_diff.imag = noa::max(result.max_abs_diff.imag, abs_diff.imag);
                } else {
                    result.max_abs_diff = noa::max(result.max_abs_diff, abs_diff);
                }
                if (result.match and not passed) {
                    result.lhs_value = lhs;
                    result.rhs_value = rhs;
                    result.indices_failed = {indices...};
                    result.match = false;
                }
            };

            if constexpr (N == 1) {
                for (index_type i{}; i < m_shape[0]; ++i)
                    compare_values(i);
            } else if constexpr (N == 2) {
                for (index_type i{}; i < m_shape[0]; ++i)
                    for (index_type j{}; j < m_shape[1]; ++j)
                        compare_values(i, j);
            } else if constexpr (N == 3) {
                for (index_type i{}; i < m_shape[0]; ++i)
                    for (index_type j{}; j < m_shape[1]; ++j)
                        for (index_type k{}; k < m_shape[2]; ++k)
                            compare_values(i, j, k);
            } else if constexpr (N == 4) {
                for (index_type i{}; i < m_shape[0]; ++i)
                    for (index_type j{}; j < m_shape[1]; ++j)
                        for (index_type k{}; k < m_shape[2]; ++k)
                            for (index_type l{}; l < m_shape[3]; ++l)
                                compare_values(i, j, k, l);
            } else {
                static_assert(nt::always_false<F>);
            }

            return result;
        }

        template<typename U>
        static constexpr auto check_abs_(
            U input,
            U expected,
            U epsilon
        ) -> std::pair<bool, U> {
            if constexpr (nt::complex<U>) {
                const auto [real_passed, real_abs_diff] = check_abs_(input.real, expected.real, epsilon.real);
                const auto [imag_passed, imag_abs_diff] = check_abs_(input.imag, expected.imag, epsilon.imag);
                return {real_passed and imag_passed, U{real_abs_diff, imag_abs_diff}};

            } else if constexpr (std::is_integral_v<U>) {
                U diff = static_cast<U>(noa::abs(input - expected));
                const bool is_passed = diff <= epsilon;
                return {is_passed, diff};

            } else {
                U diff = noa::abs(input - expected);
                return {diff <= epsilon, diff};
            }
        }

        template<typename U>
        static constexpr auto check_abs_safe_(
            U input,
            U expected,
            U epsilon
        ) -> std::pair<bool, U> {
            if constexpr (nt::complex<U>) {
                const auto [real_passed, real_abs_diff] = check_abs_safe_(input.real, expected.real, epsilon.real);
                const auto [imag_passed, imag_abs_diff] = check_abs_safe_(input.imag, expected.imag, epsilon.imag);
                return {real_passed and imag_passed, U{real_abs_diff, imag_abs_diff}};

            } else if constexpr (std::is_integral_v<U>) {
                U diff = static_cast<U>(noa::abs(input - expected));
                const bool is_passed = diff <= epsilon;
                return {is_passed, diff};

            } else {
                // Relative epsilons comparisons are usually meaningless for close-to-zero numbers,
                // hence the absolute comparison first, acting as a safety net.
                U diff = noa::abs(input - expected);
                if (not noa::is_finite(diff))
                    return {false, diff};
                const bool is_passed =
                    diff <= epsilon or
                    diff <= noa::max(noa::abs(input), noa::abs(expected)) * epsilon;
                return {is_passed, diff};
            }
        }

        template<typename U>
        static constexpr auto check_rel_(
            U input,
            U expected,
            U epsilon
        ) -> std::pair<bool, U> {
            if constexpr (nt::complex<U>) {
                const auto [real_passed, real_abs_diff] = check_rel_(input.real, expected.real, epsilon.real);
                const auto [imag_passed, imag_abs_diff] = check_rel_(input.imag, expected.imag, epsilon.imag);
                return {real_passed and imag_passed, U{real_abs_diff, imag_abs_diff}};

            } else if constexpr (std::is_integral_v<U>) {
                U diff = static_cast<U>(noa::abs(input - expected));
                const bool is_passed = diff <= epsilon;
                return {is_passed, diff};

            } else {
                auto margin = epsilon * noa::max(noa::abs(input), noa::abs(expected));
                if (noa::is_inf(margin))
                    margin = U{0};
                const bool is_passed =
                    (input + margin >= expected) and
                    (expected + margin >= input); // abs(a-b) <= epsilon
                return {is_passed, noa::abs(input - expected)};
            }
        }

    private:
        shape_type m_shape{};
        strides_type m_lhs_strides{};
        strides_type m_rhs_strides{};
        const value_type* m_lhs{};
        const value_type* m_rhs{};
    };

    template<typename T>
    concept span_like = requires (T v) {
        typename T::value_type;
        typename T::index_type;
        { T::SIZE } -> std::convertible_to<size_t>;
        { v.get() } -> std::convertible_to<typename T::value_type*>;
        requires noa::traits::strides<std::remove_reference_t<decltype(v.strides_full())>>;
        requires noa::traits::shape<std::remove_reference_t<decltype(v.shape())>>;
    };

    template<span_like Lhs, span_like Rhs, typename T = nt::mutable_value_type_t<Lhs>, typename Epsilon = T>
    auto allclose(MatchMode mode, const Lhs& lhs, const Rhs& rhs, Epsilon epsilon = Epsilon{}) {
        constexpr size_t N = Lhs::SIZE;
        static_assert(Rhs::SIZE == N and nt::almost_same_as<nt::mutable_value_type_t<Lhs>, nt::mutable_value_type_t<Rhs>>);
        noa::check(lhs.shape() == rhs.shape(),
                   "The shapes do not match, lhs:shape={}, rhs:shape={}",
                   lhs.shape(), rhs.shape());

        // Sync if possible.
        if constexpr (requires { lhs.eval(); })
            lhs.eval();
        if constexpr (requires { rhs.eval(); })
            rhs.eval();

        if constexpr (requires { lhs.is_dereferenceable(); })
            noa::check(lhs.is_dereferenceable());
        if constexpr (requires { rhs.is_dereferenceable(); })
            noa::check(rhs.is_dereferenceable());

        return Match<T, N>(lhs.get(), lhs.strides_full(), rhs.get(), rhs.strides_full(), lhs.shape()).check(mode, epsilon);
    }

    template<span_like Lhs, nt::numeric Rhs, typename T = nt::mutable_value_type_t<Lhs>, typename Epsilon = T>
    auto allclose(MatchMode mode, const Lhs& lhs, const Rhs& rhs, Epsilon epsilon = Epsilon{}) {
        constexpr size_t N = Lhs::SIZE;
        if constexpr (requires { lhs.eval(); })
            lhs.eval();
        if constexpr (requires { lhs.is_dereferenceable(); })
            noa::check(lhs.is_dereferenceable());
        const auto val = static_cast<nt::mutable_value_type_t<Lhs>>(rhs);
        return Match<T, N>(lhs.get(), lhs.strides_full(), &val, Strides<isize, N>{}, lhs.shape()).check(mode, epsilon);
    }

    template<nt::numeric T, typename U = nt::mutable_value_type_t<T>, typename Epsilon = U>
    auto allclose(MatchMode mode, const T* lhs, const T* rhs, auto n_elements, Epsilon epsilon = Epsilon{}) {
        auto shape = Shape<isize, 1>::from_value(n_elements);
        auto stride = Strides<isize, 1>{1};
        return Match<T, 1>(lhs, stride, rhs, stride, shape).check(mode, epsilon);
    }

    template<nt::numeric Lhs, nt::numeric Rhs, typename T = nt::mutable_value_type_t<Lhs>, typename Epsilon = T>
    auto allclose(MatchMode mode, const Lhs* lhs, const Rhs& rhs, auto n_elements, Epsilon epsilon = Epsilon{}) {
        auto shape = Shape<isize, 1>::from_value(n_elements);
        const auto val = static_cast<Lhs>(rhs);
        return Match<T, 1>(lhs, Strides<isize, 1>{1}, &val, Strides<isize, 1>{0}, shape).check(mode, epsilon);
    }

    #define TEST_GENERATE_ALLCLOSE(short_name, long_name)                                                             \
    template<span_like Lhs, span_like Rhs, typename T = nt::value_type_t<Lhs>, typename Epsilon = T>        \
    auto allclose_##short_name(const Lhs& lhs, const Rhs& rhs, Epsilon epsilon = Epsilon{}) {                    \
        return allclose(MatchMode::long_name, lhs, rhs, epsilon);                                                \
    }                                                                                                       \
    template<span_like Lhs, nt::numeric Rhs, typename T = nt::value_type_t<Lhs>, typename Epsilon = T>      \
    auto allclose_##short_name(const Lhs& lhs, const Rhs& rhs, Epsilon epsilon = Epsilon{}) {                    \
        return allclose(MatchMode::long_name, lhs, rhs, epsilon);                                                \
    }                                                                                                       \
    template<nt::numeric T, typename U = nt::value_type_t<T>, typename Epsilon = U>                         \
    auto allclose_##short_name(const T* lhs, const T* rhs, auto n_elements, Epsilon epsilon = Epsilon{}) {       \
        return allclose(MatchMode::long_name, lhs, rhs, n_elements, epsilon);                                    \
    }                                                                                                       \
    template<nt::numeric Lhs, nt::numeric Rhs, typename T = nt::value_type_t<Lhs>, typename Epsilon = T>    \
    auto allclose_##short_name(const Lhs* lhs, const Rhs& rhs, auto n_elements, Epsilon epsilon = Epsilon{}) {   \
        return allclose(MatchMode::long_name, lhs, rhs, n_elements, epsilon);                                    \
    }

    TEST_GENERATE_ALLCLOSE(abs, Absolute);
    TEST_GENERATE_ALLCLOSE(abs_safe, AbsoluteSafe);
    TEST_GENERATE_ALLCLOSE(rel, Relative);
}

namespace test {
    inline std::ostream& operator<<(std::ostream& os, const MatchMode mode) {
        switch (mode) {
            case MatchMode::Absolute:
                return os << "Absolute";
            case MatchMode::AbsoluteSafe:
                return os << "AbsoluteSafe";
            case MatchMode::Relative:
                return os << "Relative";
        }
        return os;
    }

    template<typename T, size_t N>
    std::ostream& operator<<(std::ostream& os, const MatchResult<T, N>& result) {
        if (result)
            return os << "Match: lhs and rhs match within the provided mode and epsilon";
        else {
            if constexpr (std::is_integral_v<T>) {
                os << fmt::format(
                        "Match: failed at indices={}, lhs={}, rhs={}, shape={}, "
                        "mode={}, epsilon={}, total_abs_diff={}, max_abs_diff={}\n",
                        result.indices_failed, result.lhs_value, result.rhs_value, result.shape,
                        result.mode, result.epsilon, result.total_abs_diff, result.max_abs_diff);
            } else {
                os << fmt::format(
                        "Match: failed at indices={}, lhs={}, rhs={}, shape={}, "
                        "mode={}, epsilon={}, total_abs_diff={}, max_abs_diff={}\n",
                        result.indices_failed, result.lhs_value, result.rhs_value,
                        result.shape, result.mode, result.epsilon,
                        result.total_abs_diff, result.max_abs_diff);
            }
            return os;
        }
    }
}

namespace fmt {
    template<> struct formatter<::test::MatchMode> : ostream_formatter {};
}
