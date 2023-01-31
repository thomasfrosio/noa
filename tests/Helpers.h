#pragma once

#include <noa/core/Types.hpp>
#include <noa/core/Math.hpp>
#include <noa/core/traits/Numerics.hpp>

//#ifdef NOA_ENABLE_UNIFIED
//#include <noa/Array.h>
//#endif

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
        using value_type = noa::traits::value_type_t<T>;
        using gen_value_type = std::conditional_t<
                std::is_same_v<value_type, noa::f16>, noa::f32, value_type>;
        using distributor_type = std::conditional_t<
                std::is_integral_v<T>,
                std::uniform_int_distribution<T>,
                std::uniform_real_distribution<gen_value_type>>;
        std::random_device rand_dev;
        std::mt19937 generator;
        distributor_type distribution;

    public:
        template<typename U>
        Randomizer(U range_from, U range_to)
                : generator(rand_dev()),
                  distribution(static_cast<gen_value_type>(range_from),
                               static_cast<gen_value_type>(range_to)) {}

        inline T get() {
            if constexpr(noa::traits::is_complex_v<T>) {
                return {static_cast<value_type>(distribution(generator)),
                        static_cast<value_type>(distribution(generator))};
            } else {
                return static_cast<T>(distribution(generator));
            }
        }
    };

    inline noa::Shape4<noa::u64> get_random_shape4(size_t ndim) {
        if (ndim == 2) {
            test::Randomizer<noa::u64> rand(32, 512);
            return noa::Shape4<noa::u64>{1, 1, rand.get(), rand.get()};
        } else if (ndim == 3) {
            test::Randomizer<noa::u64> rand(32, 128);
            return noa::Shape4<noa::u64>{1, rand.get(), rand.get(), rand.get()};
        } else if (ndim == 4) {
            test::Randomizer<noa::u64> rand(32, 128);
            return noa::Shape4<noa::u64>{test::Randomizer<noa::u64>(1, 3).get(),
                               rand.get(), rand.get(), rand.get()};
        } else {
            test::Randomizer<noa::u64> rand(32, 1024);
            return noa::Shape4<noa::u64>{1, 1, 1, rand.get()};
        }
    }

    inline noa::Shape4<noa::u64> get_random_shape4_batched(size_t ndim) {
        test::Randomizer<noa::u64> rand_batch(1, 3);
        if (ndim == 2) {
            test::Randomizer<noa::u64> randomizer(32, 512);
            return noa::Shape4<noa::u64>{rand_batch.get(), 1,
                               randomizer.get(), randomizer.get()};
        } else if (ndim >= 3) {
            test::Randomizer<noa::u64> randomizer(32, 128);
            return noa::Shape4<noa::u64>{rand_batch.get(), randomizer.get(),
                               randomizer.get(), randomizer.get()};
        } else {
            test::Randomizer<noa::u64> randomizer(32, 1024);
            return noa::Shape4<noa::u64>{rand_batch.get(), 1, 1, randomizer.get()};
        }
    }

    inline noa::Shape4<noa::u64> get_random_shape4(size_t ndim, bool even) { // if even is false, return odd only
        noa::Shape4<noa::u64> shape = get_random_shape4(ndim);
        shape += noa::Shape4<noa::u64>{shape != 1} * // ignore empty dimensions
                 ((shape % 2) * even + !even * noa::Shape4<noa::u64>{(shape % 2) == 0});
        return shape;
    }

    inline noa::Shape4<noa::u64> get_random_shape4_batched(size_t ndim, bool even) {
        noa::Shape4<noa::u64> shape = get_random_shape4_batched(ndim);
        shape += noa::Shape4<noa::u64>{shape != 1} * // ignore empty dimensions
                 ((shape % 2) * even + !even * noa::Shape4<noa::u64>{(shape % 2) == 0});
        return shape;
    }
}

namespace test {
    template<typename T, typename U>
    inline void randomize(T* data, size_t elements, test::Randomizer<U>& randomizer) {
        if constexpr (std::is_same_v<T, U>) {
            for (size_t idx = 0; idx < elements; ++idx)
                data[idx] = randomizer.get();
        } else if constexpr (noa::traits::is_complex_v<T>) {
            using value_t = noa::traits::value_type_t<T>;
            randomize(reinterpret_cast<value_t*>(data), elements * 2, randomizer);
        } else {
            for (size_t idx = 0; idx < elements; ++idx)
                data[idx] = static_cast<T>(randomizer.get());
        }
    }

    template<typename T, typename U>
    inline void memset(T* data, size_t elements, U value) {
        std::fill(data, data + elements, static_cast<T>(value));
    }

    template<typename T, typename U = size_t>
    inline void arange(T* data, size_t elements, U start = 0) {
        for (size_t i = 0; i < elements; ++i, ++start)
            data[i] = static_cast<T>(start);
    }

    template<typename T>
    inline void copy(const T* src, T* dst, size_t elements) {
        std::copy(src, src + elements, dst);
    }

    template<typename T, typename U>
    inline void scale(T* array, size_t size, U scale) {
        std::transform(array, array + size, array, [scale](const T& a) { return a * scale; });
    }
}

namespace test {
    template<typename T>
    inline T get_difference(const T* in, const T* out, size_t elements) {
        if constexpr (noa::traits::is_complex_v<T>) {
            using real_t = noa::traits::value_type_t<T>;
            T diff{0}, tmp;
            for (size_t idx{0}; idx < elements; ++idx) {
                tmp = out[idx] - in[idx];
                diff += T{tmp.real > real_t{0} ? tmp.real : -tmp.real,
                          tmp.imag > real_t{0} ? tmp.imag : -tmp.imag};
            }
            return diff;
        } else if constexpr (std::is_integral_v<T>) {
            int64_t diff{0};
            for (size_t idx{0}; idx < elements; ++idx)
                diff += std::abs(static_cast<int64_t>(out[idx] - in[idx]));
            return noa::clamp_cast<T>(diff);
        } else {
            T diff{0}, tmp;
            for (size_t idx{0}; idx < elements; ++idx) {
                tmp = out[idx] - in[idx];
                diff += tmp > T{0} ? tmp : -tmp;
            }
            return diff;
        }
    }
}

namespace test {
    using namespace ::noa;
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
        Matcher() = default;

        template<typename U>
        Matcher(CompType comparison,
                const T* lhs, noa::Strides4<noa::u64> lhs_strides,
                const T* rhs, noa::Strides4<noa::u64> rhs_strides,
                noa::Shape4<noa::u64> shape, U epsilon) noexcept
                : m_shape(shape),
                  m_lhs_strides(lhs_strides),
                  m_rhs_strides(rhs_strides),
                  m_lhs(lhs),
                  m_rhs(rhs),
                  m_comparison(comparison) {
            if constexpr (noa::traits::is_complex_v<T> && !noa::traits::is_complex_v<U>) {
                using value_t = noa::traits::value_type_t<T>;
                m_epsilon.real = static_cast<value_t>(epsilon);
                m_epsilon.imag = static_cast<value_t>(epsilon);
            } else {
                m_epsilon = static_cast<T>(epsilon);
            }
            check_();
        }

        template<typename U>
        Matcher(CompType comparison,
                const T* lhs, noa::Strides4<noa::u64> lhs_strides,
                const T rhs,
                noa::Shape4<noa::u64> shape, U epsilon) noexcept
                : m_shape(shape),
                  m_lhs_strides(lhs_strides),
                  m_lhs(lhs),
                  m_rhs(&m_rhs_value),
                  m_rhs_value(rhs),
                  m_comparison(comparison) {
            if constexpr (noa::traits::is_complex_v<T> && !noa::traits::is_complex_v<U>) {
                using value_t = noa::traits::value_type_t<T>;
                m_epsilon.real = static_cast<value_t>(epsilon);
                m_epsilon.imag = static_cast<value_t>(epsilon);
            } else {
                m_epsilon = static_cast<T>(epsilon);
            }
            check_();
        }

        template<typename U>
        Matcher(CompType comparison, const T* lhs, const T* rhs, size_t elements, U epsilon) noexcept
            : Matcher(comparison,
                      lhs, noa::Shape4<noa::u64>{1, 1, 1, elements}.strides(),
                      rhs, noa::Shape4<noa::u64>{1, 1, 1, elements}.strides(),
                      noa::Shape4<noa::u64>{1, 1, 1, elements}, epsilon) {}

        template<typename U>
        Matcher(CompType comparison, const T* lhs, T rhs, size_t elements, U epsilon) noexcept
                : Matcher(comparison,
                          lhs, noa::Shape4<noa::u64>{1, 1, 1, elements}.strides(),
                          rhs,
                          noa::Shape4<noa::u64>{1, 1, 1, elements}, epsilon) {}

//        #ifdef NOA_ENABLE_UNIFIED
//        template<typename U>
//        Matcher(CompType comparison, const noa::Array<T>& lhs, const noa::Array<T>& rhs, U epsilon) noexcept
//                : Matcher(comparison,
//                          lhs.eval().get(), lhs.strides(),
//                          rhs.eval().get(), rhs.strides(),
//                          lhs.shape(), epsilon) {
//            NOA_ASSERT(all(lhs.shape() == rhs.shape()));
//        }
//
//        template<typename U>
//        Matcher(CompType comparison, const noa::Array<T>& lhs, T rhs, U epsilon) noexcept
//                : Matcher(comparison, lhs.eval().get(), lhs.strides(), rhs, lhs.shape(), epsilon) {}
//        #endif

        explicit operator bool() const noexcept {
            return m_match;
        }

        friend std::ostream& operator<<(std::ostream& os, const Matcher<T>& matcher) {
            if (matcher)
                return os << "Matcher: all checks are within the expected value(s)";
            else {
                using namespace noa;
                using namespace noa::indexing;

                Vec4<u64> idx = matcher.m_index_failed;
                if (matcher.m_shape.ndim() == 1)
                    os << "Matcher: check failed at index=" << idx;
                else
                    os << "Matcher: check failed at index=" << idx << ", shape=" << matcher.m_shape;

                T lhs_value = matcher.m_lhs[at(idx, matcher.m_lhs_strides)];
                T rhs_value = matcher.m_rhs[at(idx, matcher.m_rhs_strides)];
                if constexpr (std::is_integral_v<T>) {
                    os << noa::string::format(", lhs={}, rhs={}, epsilon={}, total_abs_diff={}",
                                              lhs_value, rhs_value, matcher.m_epsilon, matcher.m_total_abs_diff);
                } else {
                    int dyn_precision{};
                    if constexpr (noa::traits::is_complex_v<T>)
                        dyn_precision =
                                noa::math::max(int(-noa::math::log10(noa::math::abs(matcher.m_epsilon.real))),
                                               int(-noa::math::log10(noa::math::abs(matcher.m_epsilon.imag)))) + 2;
                    else
                        dyn_precision = int(-noa::math::log10(noa::math::abs(matcher.m_epsilon))) + 2;

                    os << noa::string::format(", lhs={:.{}}, rhs={:.{}}, epsilon={:.{}}, total_abs_diff={}",
                                              lhs_value, dyn_precision,
                                              rhs_value, dyn_precision,
                                              matcher.m_epsilon, dyn_precision,
                                              matcher.m_total_abs_diff);
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
        void check_() noexcept {
            if (m_comparison == MATCH_ABS)
                check_(Matcher<T>::check_abs_<T>);
            else if (m_comparison == MATCH_ABS_SAFE)
                check_(Matcher<T>::check_abs_safe_<T>);
            else if (m_comparison == MATCH_REL)
                check_(Matcher<T>::check_rel_<T>);
        }

        template<typename F>
        void check_(F&& value_checker) noexcept {
            m_total_abs_diff = T{0};
            m_match = true;
            for (u64 i = 0; i < m_shape[0]; ++i) {
                for (u64 j = 0; j < m_shape[1]; ++j) {
                    for (u64 k = 0; k < m_shape[2]; ++k) {
                        for (u64 l = 0; l < m_shape[3]; ++l) {
                            const bool passed = value_checker(
                                    m_lhs[noa::indexing::at(i, j, k, l, m_lhs_strides)],
                                    m_rhs[noa::indexing::at(i, j, k, l, m_rhs_strides)],
                                    m_epsilon,
                                    m_total_abs_diff);
                            if (m_match && !passed) {
                                m_index_failed = {i, j, k, l};
                                m_match = false;
                            }
                        }
                    }
                }
            }
        }

        template<typename U>
        static bool check_abs_(U input, U expected, U epsilon, U& total_abs_diff) noexcept {
            if constexpr (noa::traits::is_complex_v<U>) {
                using real_t = noa::traits::value_type_t<U>;
                return Matcher<T>::check_abs_<real_t>(input.real, expected.real, epsilon.real, total_abs_diff.real) &&
                       Matcher<T>::check_abs_<real_t>(input.imag, expected.imag, epsilon.imag, total_abs_diff.imag);
            } else if constexpr (std::is_integral_v<U>) {
                U diff = input - expected;
                if constexpr (traits::is_uint_v<U>)
                    total_abs_diff += diff;
                else
                    total_abs_diff += noa::math::abs(diff);
                return (input - expected) <= epsilon;
            } else {
                U diff = noa::math::abs(input - expected);
                total_abs_diff += diff;
                return diff <= epsilon;
            }
        }

        template<typename U>
        static bool check_abs_safe_(U input, U expected, U epsilon, U& total_abs_diff) noexcept {
            if constexpr (noa::traits::is_complex_v<U>) {
                using real_t = noa::traits::value_type_t<U>;
                return Matcher<T>::check_abs_safe_<real_t>(input.real, expected.real, epsilon.real, total_abs_diff.real) &&
                       Matcher<T>::check_abs_safe_<real_t>(input.imag, expected.imag, epsilon.imag, total_abs_diff.imag);
            } else if constexpr (std::is_integral_v<U>) {
                U diff = input - expected;
                if constexpr (traits::is_uint_v<U>)
                    total_abs_diff += diff;
                else
                    total_abs_diff += noa::math::abs(diff);
                return (input - expected) <= epsilon;
            } else {
                // Relative epsilons comparisons are usually meaningless for close-to-zero numbers,
                // hence the absolute comparison first, acting as a safety net.
                using namespace ::noa;
                U diff = math::abs(input - expected);
                total_abs_diff += diff;
                if (!math::is_finite(diff))
                    return false;
                return diff <= epsilon || diff <= math::max(math::abs(input), math::abs(expected)) * epsilon;
            }
        }

        template<typename U>
        static bool check_rel_(U input, U expected, U epsilon, U& total_abs_diff) noexcept {
            if constexpr (noa::traits::is_complex_v<U>) {
                using real_t = noa::traits::value_type_t<U>;
                return Matcher<T>::check_rel_<real_t>(input.real, expected.real, epsilon.real, total_abs_diff.real) &&
                       Matcher<T>::check_rel_<real_t>(input.imag, expected.imag, epsilon.imag, total_abs_diff.imag);
            } else if constexpr (std::is_integral_v<U>) {
                U diff = input - expected;
                if constexpr (traits::is_uint_v<U>)
                    total_abs_diff += diff;
                else
                    total_abs_diff += noa::math::abs(diff);
                return (input - expected) <= epsilon;
            } else {
                total_abs_diff += math::abs(input - expected);
                auto margin = epsilon * noa::math::max(noa::math::abs(input), noa::math::abs(expected));
                if (noa::math::is_inf(margin))
                    margin = U(0);
                return (input + margin >= expected) && (expected + margin >= input); // abs(a-b) <= epsilon
            }
        }

    private:
        noa::Shape4<noa::u64> m_shape{};
        noa::Strides4<noa::u64> m_lhs_strides{};
        noa::Strides4<noa::u64> m_rhs_strides{};
        noa::Vec4<noa::u64> m_index_failed{};
        const T* m_lhs{};
        const T* m_rhs{};
        T m_rhs_value{};
        T m_epsilon{};
        T m_total_abs_diff{};
        CompType m_comparison{};
        bool m_match{true};
    };
}
