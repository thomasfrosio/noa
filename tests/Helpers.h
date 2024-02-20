#pragma once

#include <noa/core/Types.hpp>
#include <noa/core/Math.hpp>
#include <noa/core/io/IO.hpp>
//#include <noa/unified/Array.hpp>

#include <random>
#include <cstdlib>
#include <cstring>
#include <cmath>

namespace test {
    extern noa::Path NOA_DATA_PATH; // defined at runtime by main.
}

namespace test {
    template<typename T, size_t N>
    inline std::vector<T> array2vector(const std::array<T, N>& array) {
        return std::vector<T>(array.cbegin(), array.cend());
    }
}

namespace test {
    /// Randomizer for integers, floating-points and noa::Complex types.
    template<typename T>
    class Randomizer {
    private:
        using value_type = T;
        using scalar_type = noa::traits::value_type_t<value_type>;
        using gen_scalar_type = std::conditional_t<
                std::is_same_v<scalar_type, noa::f16>, noa::f32, scalar_type>;
        using distributor_type = std::conditional_t<
                std::is_integral_v<value_type>,
                std::uniform_int_distribution<value_type>,
                std::uniform_real_distribution<gen_scalar_type>>;
        std::random_device rand_dev;
        std::mt19937 generator;
        distributor_type distribution;

    public:
        template<typename U>
        Randomizer(U range_from, U range_to)
                : generator(rand_dev()),
                  distribution(static_cast<gen_scalar_type>(range_from),
                               static_cast<gen_scalar_type>(range_to)) {}

        inline value_type get() {
            if constexpr(noa::traits::is_complex_v<T>) {
                return {static_cast<scalar_type>(distribution(generator)),
                        static_cast<scalar_type>(distribution(generator))};
            } else {
                return static_cast<value_type>(distribution(generator));
            }
        }
    };

    inline noa::Shape4<noa::i64> get_random_shape4(noa::i64 ndim) {
        if (ndim == 2) {
            test::Randomizer<noa::i64> rand(32, 512);
            return noa::Shape4<noa::i64>{1, 1, rand.get(), rand.get()};
        } else if (ndim == 3) {
            test::Randomizer<noa::i64> rand(32, 128);
            return noa::Shape4<noa::i64>{1, rand.get(), rand.get(), rand.get()};
        } else if (ndim == 4) {
            test::Randomizer<noa::i64> rand(32, 128);
            return noa::Shape4<noa::i64>{test::Randomizer<noa::i64>(1, 4).get(),
                                         rand.get(), rand.get(), rand.get()};
        } else {
            test::Randomizer<noa::i64> rand(32, 1024);
            return noa::Shape4<noa::i64>{1, 1, 1, rand.get()};
        }
    }

    inline noa::Shape4<noa::i64> get_random_shape4_batched(noa::i64 ndim) {
        test::Randomizer<noa::i64> rand_batch(1, 4);
        if (ndim == 2) {
            test::Randomizer<noa::i64> randomizer(32, 512);
            return noa::Shape4<noa::i64>{rand_batch.get(), 1,
                                         randomizer.get(), randomizer.get()};
        } else if (ndim >= 3) {
            test::Randomizer<noa::i64> randomizer(32, 128);
            return noa::Shape4<noa::i64>{rand_batch.get(), randomizer.get(),
                                         randomizer.get(), randomizer.get()};
        } else {
            test::Randomizer<noa::i64> randomizer(32, 1024);
            return noa::Shape4<noa::i64>{rand_batch.get(), 1, 1, randomizer.get()};
        }
    }

    inline noa::Shape4<noa::i64> get_random_shape4(noa::i64 ndim, bool even) { // if even is false, return odd only
        noa::Shape4<noa::i64> shape = get_random_shape4(ndim);
        shape += noa::Shape4<noa::i64>::from_vec(shape != 1) * // ignore empty dimensions
                 ((shape % 2) * even + !even * noa::Shape4<noa::i64>::from_vec((shape % 2) == 0));
        return shape;
    }

    inline noa::Shape4<noa::i64> get_random_shape4_batched(noa::i64 ndim, bool even) {
        noa::Shape4<noa::i64> shape = get_random_shape4_batched(ndim);
        shape += noa::Shape4<noa::i64>::from_vec(shape != 1) * // ignore empty dimensions
                 ((shape % 2) * even + !even * noa::Shape4<noa::i64>::from_vec((shape % 2) == 0));
        return shape;
    }
}

namespace test {
    template<typename T, typename U>
    inline void randomize(T* data, noa::i64 n_elements, test::Randomizer<U>& randomizer) {
        if constexpr (std::is_same_v<T, U>) {
            for (noa::i64 idx = 0; idx < n_elements; ++idx)
                data[idx] = randomizer.get();
        } else if constexpr (noa::traits::is_complex_v<T>) {
            using value_t = noa::traits::value_type_t<T>;
            randomize(reinterpret_cast<value_t*>(data), n_elements * 2, randomizer);
        } else {
            for (noa::i64 idx = 0; idx < n_elements; ++idx)
                data[idx] = static_cast<T>(randomizer.get());
        }
    }

    template<typename T, typename U>
    inline void randomize(T* data, noa::i64 n_elements, test::Randomizer<U>&& randomizer) {
        randomize(data, n_elements, randomizer);
    }

    template<typename T, typename U>
    inline void memset(T* data, noa::i64 elements, U value) {
        std::fill(data, data + elements, static_cast<T>(value));
    }

    template<typename T, typename U = noa::i64>
    inline void arange(T* data, noa::i64 elements, U start = 0) {
        for (noa::i64 i = 0; i < elements; ++i, ++start)
            data[i] = static_cast<T>(start);
    }

    template<typename T>
    inline void copy(const T* src, T* dst, noa::i64 elements) {
        std::copy(src, src + elements, dst);
    }

    template<typename T, typename U>
    inline void scale(T* array, noa::i64 size, U scale) {
        std::transform(array, array + size, array, [scale](const T& a) { return a * scale; });
    }
}

namespace test {
    template<typename T>
    inline T get_difference(const T* in, const T* out, noa::i64 elements) {
        if constexpr (noa::traits::is_complex_v<T>) {
            using real_t = noa::traits::value_type_t<T>;
            T diff{0}, tmp;
            for (noa::i64 idx{0}; idx < elements; ++idx) {
                tmp = out[idx] - in[idx];
                diff += T{tmp.real > real_t{0} ? tmp.real : -tmp.real,
                          tmp.imag > real_t{0} ? tmp.imag : -tmp.imag};
            }
            return diff;
        } else if constexpr (std::is_integral_v<T>) {
            int64_t diff{0};
            for (noa::i64 idx{0}; idx < elements; ++idx)
                diff += noa::abs(static_cast<int64_t>(out[idx] - in[idx]));
            return noa::clamp_cast<T>(diff);
        } else {
            T diff{0}, tmp;
            for (noa::i64 idx{0}; idx < elements; ++idx) {
                tmp = out[idx] - in[idx];
                diff += tmp > T{0} ? tmp : -tmp;
            }
            return diff;
        }
    }
}

namespace test {
    enum CompType {
        MATCH_ABS,
        MATCH_ABS_SAFE,
        MATCH_REL,
    };

    /// On construction, each element of the input array is checked.
    /// The operator bool of this class returns true if all elements have passed the check.
    template<typename T>
    class Matcher {
    public:
        using value_type = T;
        using index_type = noa::i64;
        using strides_type = noa::Strides4<index_type>;
        using shape_type = noa::Shape4<index_type>;
        using index4_type = noa::Vec4<index_type>;
//        using array_type = noa::Array<value_type>;

    public:
        Matcher() = default;

        template<typename Epsilon, typename = std::enable_if_t<noa::traits::is_numeric_v<Epsilon>>>
        Matcher(CompType comparison,
                const value_type* lhs, const strides_type& lhs_strides,
                const value_type* rhs, const strides_type& rhs_strides,
                const shape_type& shape, Epsilon epsilon) noexcept
                : m_shape(shape),
                  m_lhs_strides(lhs_strides),
                  m_rhs_strides(rhs_strides),
                  m_lhs(lhs),
                  m_rhs(rhs),
                  m_comparison(comparison) {
            set_epsilon_(epsilon);
            check_();
        }

        template<typename Epsilon, typename = std::enable_if_t<noa::traits::is_numeric_v<Epsilon>>>
        Matcher(CompType comparison,
                const value_type* lhs, const strides_type& lhs_strides,
                const value_type& rhs,
                shape_type shape, Epsilon epsilon) noexcept
                : m_shape(shape),
                  m_lhs_strides(lhs_strides),
                  m_lhs(lhs),
                  m_rhs(&rhs),
                  m_comparison(comparison) {
            set_epsilon_(epsilon);
            check_();
        }

        template<typename Epsilon, typename = std::enable_if_t<noa::traits::is_numeric_v<Epsilon>>>
        Matcher(CompType comparison, const value_type* lhs, const value_type* rhs,
                index_type elements, Epsilon epsilon) noexcept
            : Matcher(comparison,
                      lhs, shape_type{1, 1, 1, elements}.strides(),
                      rhs, shape_type{1, 1, 1, elements}.strides(),
                      shape_type{1, 1, 1, elements}, epsilon) {}

        template<typename Epsilon, typename = std::enable_if_t<noa::traits::is_numeric_v<Epsilon>>>
        Matcher(CompType comparison, const value_type* lhs, const value_type& rhs,
                index_type elements, Epsilon epsilon) noexcept
                : Matcher(comparison,
                          lhs, shape_type{1, 1, 1, elements}.strides(),
                          rhs,
                          shape_type{1, 1, 1, elements}, epsilon) {}

//        template<typename Epsilon, typename = std::enable_if_t<noa::traits::is_numeric_v<Epsilon>>>
//        Matcher(CompType comparison, const array_type& lhs, const array_type& rhs, Epsilon epsilon) noexcept
//                : m_shape(lhs.shape()),
//                  m_lhs_strides(lhs.strides()),
//                  m_rhs_strides(rhs.strides()),
//                  m_lhs(lhs.eval().get()),
//                  m_rhs(rhs.eval().get()),
//                  m_comparison(comparison) {
//            NOA_ASSERT(noa::all(lhs.shape() == rhs.shape()));
//            NOA_ASSERT(lhs.is_dereferenceable() && rhs.is_dereferenceable());
//            set_epsilon_(epsilon);
//            check_();
//        }
//
//        template<typename Epsilon, typename = std::enable_if_t<noa::traits::is_numeric_v<Epsilon>>>
//        Matcher(CompType comparison, const array_type& lhs, const value_type& rhs, Epsilon epsilon) noexcept
//                : m_shape(lhs.shape()),
//                  m_lhs_strides(lhs.strides()),
//                  m_lhs(lhs.eval().get()),
//                  m_rhs(&rhs),
//                  m_comparison(comparison) {
//            NOA_ASSERT(lhs.is_dereferenceable());
//            set_epsilon_(epsilon);
//            check_();
//        }

        explicit operator bool() const noexcept {
            return m_match;
        }

        friend std::ostream& operator<<(std::ostream& os, const Matcher& matcher) {
            if (matcher)
                return os << "Matcher: all checks are within the expected value(s)";
            else {
                os << fmt::format("Matcher: check failed at index={}", matcher.m_index_failed);
                if (matcher.m_shape.ndim() > 1)
                    os << fmt::format(", shape={}", matcher.m_index_failed, matcher.m_shape);

                T lhs_value = matcher.m_lhs[noa::indexing::offset_at(matcher.m_index_failed, matcher.m_lhs_strides)];
                T rhs_value = matcher.m_rhs[noa::indexing::offset_at(matcher.m_index_failed, matcher.m_rhs_strides)];
                if constexpr (std::is_integral_v<T>) {
                    os << fmt::format(
                            ", lhs={}, rhs={}, epsilon={}, total_abs_diff={}, max_abs_diff={}",
                            lhs_value, rhs_value, matcher.m_epsilon,
                            matcher.m_total_abs_diff, matcher.m_max_abs_diff);
                } else {
                    int dyn_precision{};
                    if constexpr (noa::traits::is_complex_v<T>) {
                        dyn_precision = noa::max(
                                static_cast<int>(-noa::log10(noa::abs(matcher.m_epsilon.real))),
                                static_cast<int>(-noa::log10(noa::abs(matcher.m_epsilon.imag)))) + 2;
                    } else {
                        dyn_precision = static_cast<int>(-noa::log10(noa::abs(matcher.m_epsilon))) + 2;
                    }

                    os << fmt::format(
                            ", lhs={:.{}}, rhs={:.{}}, epsilon={:.{}}, total_abs_diff={}, max_abs_diff={}",
                            lhs_value, dyn_precision, rhs_value, dyn_precision,
                            matcher.m_epsilon, dyn_precision, matcher.m_total_abs_diff, matcher.m_max_abs_diff);
                }

                if (matcher.m_comparison == MATCH_ABS)
                    os << ", comparison=MATCH_ABS";
                else if (matcher.m_comparison == MATCH_REL)
                    os << ", comparison=MATCH_REL";
                else if (matcher.m_comparison == MATCH_ABS_SAFE)
                    os << ", comparison=MATCH_ABS_SAFE";
                return os;
            }
        }

    private:
        template<typename Epsilon>
        void set_epsilon_(Epsilon epsilon) {
            if constexpr (noa::traits::is_complex_v<value_type> && !noa::traits::is_complex_v<Epsilon>) {
                using real_t = typename value_type::value_type;
                m_epsilon.real = static_cast<real_t>(epsilon);
                m_epsilon.imag = static_cast<real_t>(epsilon);
            } else {
                m_epsilon = static_cast<value_type>(epsilon);
            }
        }

        void check_() noexcept {
            if (m_comparison == MATCH_ABS)
                check_(Matcher::check_abs_<T>);
            else if (m_comparison == MATCH_ABS_SAFE)
                check_(Matcher::check_abs_safe_<T>);
            else if (m_comparison == MATCH_REL)
                check_(Matcher::check_rel_<T>);
        }

        template<typename F>
        void check_(F&& value_checker) noexcept {
            m_total_abs_diff = value_type{};
            m_max_abs_diff = value_type{};
            m_match = true;
            for (index_type i = 0; i < m_shape[0]; ++i) {
                for (index_type j = 0; j < m_shape[1]; ++j) {
                    for (index_type k = 0; k < m_shape[2]; ++k) {
                        for (index_type l = 0; l < m_shape[3]; ++l) {
                            const auto [passed, abs_diff] = value_checker(
                                    m_lhs[noa::indexing::offset_at(i, j, k, l, m_lhs_strides)],
                                    m_rhs[noa::indexing::offset_at(i, j, k, l, m_rhs_strides)],
                                    m_epsilon);

                            m_total_abs_diff += abs_diff;
                            if constexpr (noa::traits::is_complex_v<value_type>) {
                                m_max_abs_diff.real = noa::max(m_max_abs_diff.real, abs_diff.real);
                                m_max_abs_diff.imag = noa::max(m_max_abs_diff.imag, abs_diff.imag);
                            } else {
                                m_max_abs_diff = noa::max(m_max_abs_diff, abs_diff);
                            }
                            if (m_match && !passed) {
                                m_index_failed = {i, j, k, l};
                                m_match = false;
                            }
                        }
                    }
                }
            }
        }

        template<typename value_type>
        static std::pair<bool, value_type> check_abs_(value_type input, value_type expected, value_type epsilon) noexcept {
            if constexpr (noa::traits::is_complex_v<value_type>) {
                using real_t = typename value_type::value_type;
                const auto [real_passed, real_abs_diff] = Matcher<T>::check_abs_<real_t>(
                        input.real, expected.real, epsilon.real);
                const auto [imag_passed, imag_abs_diff] = Matcher<T>::check_abs_<real_t>(
                        input.imag, expected.imag, epsilon.imag);
                return {real_passed && imag_passed, value_type{real_abs_diff, imag_abs_diff}};

            } else if constexpr (std::is_integral_v<value_type>) {
                value_type diff = noa::abs(input - expected);
                const bool is_passed = diff <= epsilon;
                return {is_passed, diff};

            } else {
                value_type diff = noa::abs(input - expected);
                return {diff <= epsilon, diff};
            }
        }

        template<typename value_type>
        static std::pair<bool, value_type> check_abs_safe_(value_type input, value_type expected, value_type epsilon) noexcept {
            if constexpr (noa::traits::is_complex_v<value_type>) {
                using real_t = typename value_type::value_type;
                const auto [real_passed, real_abs_diff] = Matcher<T>::check_abs_safe_<real_t>(
                        input.real, expected.real, epsilon.real);
                const auto [imag_passed, imag_abs_diff] = Matcher<T>::check_abs_safe_<real_t>(
                        input.imag, expected.imag, epsilon.imag);
                return {real_passed && imag_passed, value_type{real_abs_diff, imag_abs_diff}};

            } else if constexpr (std::is_integral_v<value_type>) {
                value_type diff = noa::abs(input - expected);
                const bool is_passed = diff <= epsilon;
                return {is_passed, diff};

            } else {
                // Relative epsilons comparisons are usually meaningless for close-to-zero numbers,
                // hence the absolute comparison first, acting as a safety net.
                value_type diff = noa::abs(input - expected);
                if (!noa::is_finite(diff))
                    return {false, diff};
                const bool is_passed =
                        diff <= epsilon ||
                        diff <= noa::max(noa::abs(input), noa::abs(expected)) * epsilon;
                return {is_passed, diff};
            }
        }

        template<typename value_type>
        static std::pair<bool, value_type> check_rel_(value_type input, value_type expected, value_type epsilon) noexcept {
            if constexpr (noa::traits::is_complex_v<value_type>) {
                using real_t = typename value_type::value_type;
                const auto [real_passed, real_abs_diff] = Matcher<T>::check_rel_<real_t>(
                        input.real, expected.real, epsilon.real);
                const auto [imag_passed, imag_abs_diff] = Matcher<T>::check_rel_<real_t>(
                        input.imag, expected.imag, epsilon.imag);
                return {real_passed && imag_passed, value_type{real_abs_diff, imag_abs_diff}};


            } else if constexpr (std::is_integral_v<value_type>) {
                value_type diff = noa::abs(input - expected);
                const bool is_passed = diff <= epsilon;
                return {is_passed, diff};

            } else {
                auto margin = epsilon * noa::max(noa::abs(input), noa::abs(expected));
                if (noa::is_inf(margin))
                    margin = value_type{0};
                const bool is_passed =
                        (input + margin >= expected) &&
                        (expected + margin >= input); // abs(a-b) <= epsilon
                return {is_passed, noa::abs(input - expected)};
            }
        }

    private:
        shape_type m_shape{};
        strides_type m_lhs_strides{};
        strides_type m_rhs_strides{};
        index4_type m_index_failed{};
        const value_type* m_lhs{};
        const value_type* m_rhs{};
        value_type m_epsilon{};
        value_type m_total_abs_diff{};
        value_type m_max_abs_diff{};
        CompType m_comparison{};
        bool m_match{false};
    };
}
