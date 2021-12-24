#pragma once

#include <noa/common/Types.h>
#include <noa/common/Math.h>
#include <noa/common/traits/BaseTypes.h>

#include <random>
#include <cstdlib>
#include <cstring>
#include <cmath>

namespace test {
    extern noa::path_t PATH_NOA_DATA; // defined at runtime by main.
}

namespace test {
    template<typename T, size_t N>
    inline std::vector<T> toVector(const std::array<T, N>& array) {
        return std::vector<T>(array.cbegin(), array.cend());
    }
}

namespace test {
    /// Randomizer for integers, floating-points and noa::Complex types.
    template<typename T>
    class Randomizer {
    private:
        using value_t = noa::traits::value_type_t<T>;
        using gen_value_t = std::conditional_t<std::is_same_v<T, noa::half_t>, float, noa::traits::value_type_t<T>>;
        using distributor_t = std::conditional_t<std::is_integral_v<T>,
                                                 std::uniform_int_distribution<T>,
                                                 std::uniform_real_distribution<gen_value_t>>;
        std::random_device rand_dev;
        std::mt19937 generator;
        distributor_t distribution;
    public:
        template<typename U>
        Randomizer(U range_from, U range_to)
                : generator(rand_dev()),
                  distribution(static_cast<gen_value_t>(range_from), static_cast<gen_value_t>(range_to)) {}
        inline T get() {
            if constexpr(noa::traits::is_complex_v<T>)
                return {static_cast<value_t>(distribution(generator)), static_cast<value_t>(distribution(generator))};
            else
                return static_cast<T>(distribution(generator));
        }
    };

    inline noa::size3_t getRandomShape(size_t ndim) {
        if (ndim == 2) {
            test::Randomizer<size_t> randomizer(32, 512);
            return noa::size3_t{randomizer.get(), randomizer.get(), 1};
        } else if (ndim == 3) {
            test::Randomizer<size_t> randomizer(32, 128);
            return noa::size3_t{randomizer.get(), randomizer.get(), randomizer.get()};
        } else {
            test::Randomizer<size_t> randomizer(32, 1024);
            return noa::size3_t{randomizer.get(), 1, 1};
        }
    }

    inline noa::size3_t getRandomShape(size_t ndim, bool even) {
        noa::size3_t shape = getRandomShape(ndim);
        if (even) {
            shape.x += shape.x % 2;
            if (ndim >= 2)
                shape.y += shape.y % 2;
            if (ndim == 3)
                shape.z += shape.z % 2;
        } else {
            shape.x += !(shape.x % 2);
            if (ndim >= 2)
                shape.y += !(shape.y % 2);
            if (ndim == 3)
                shape.z += !(shape.z % 2);
        }
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
        for (size_t idx = 0; idx < elements; ++idx)
            data[idx] = static_cast<T>(value);
    }

    template<typename T, typename U>
    inline void normalize(T* array, size_t size, U scale) {
        for (size_t idx{0}; idx < size; ++idx) {
            array[idx] *= scale;
        }
    }
}

namespace test {
    template<typename T>
    inline T getDifference(const T* in, const T* out, size_t elements) {
        if constexpr (std::is_same_v<T, noa::cfloat_t> || std::is_same_v<T, noa::cdouble_t>) {
            T diff{0}, tmp;
            for (size_t idx{0}; idx < elements; ++idx) {
                tmp = out[idx] - in[idx];
                diff += T{tmp.real > 0 ? tmp.real : -tmp.real,
                          tmp.imag > 0 ? tmp.imag : -tmp.imag};
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
                diff += tmp > 0 ? tmp : -tmp;
            }
            return diff;
        }
    }

    template<typename T>
    inline T getAverageDifference(const T* in, const T* out, size_t elements) {
        T diff = getDifference(in, out, elements);
        if constexpr (std::is_same_v<T, noa::cfloat_t>) {
            return diff / static_cast<float>(elements);
        } else if constexpr (std::is_same_v<T, noa::cdouble_t>) {
            return diff / static_cast<double>(elements);
        } else {
            return diff / static_cast<T>(elements);
        }
    }

    // The differences are normalized by the magnitude of the values being compared. This is important since depending
    // on the magnitude of the floating-point being compared, the expected error will vary. This is exactly what
    // isWithinRel (or noa::math::isEqual) does but reduces the array to one value so that it can then be asserted by
    // REQUIRE isWithinAbs.
    // Use this version, as opposed to getDifference, when the values are expected to have a large magnitude.
    template<typename T>
    inline T getNormalizedDifference(const T* in, const T* out, size_t elements) {
        if constexpr (std::is_same_v<T, noa::cfloat_t> || std::is_same_v<T, noa::cdouble_t>) {
            using real_t = noa::traits::value_type_t<T>;
            return getNormalizedDifference(reinterpret_cast<const real_t*>(in),
                                           reinterpret_cast<const real_t*>(out),
                                           elements * 2);
        } else {
            T diff{0}, tmp, mag;
            for (size_t idx{0}; idx < elements; ++idx) {
                mag = noa::math::max(noa::math::abs(out[idx]), noa::math::abs(in[idx]));
                tmp = (out[idx] - in[idx]);
                if (mag != 0)
                    tmp /= mag;
                diff += tmp > 0 ? tmp : -tmp;
            }
            return diff;
        }
    }

    template<typename T>
    inline T getAverageNormalizedDifference(const T* in, const T* out, size_t elements) {
        T diff = getNormalizedDifference(in, out, elements);
        if constexpr (std::is_same_v<T, noa::cfloat_t>) {
            return diff / static_cast<float>(elements);
        } else if constexpr (std::is_same_v<T, noa::cdouble_t>) {
            return diff / static_cast<double>(elements);
        } else {
            return diff / static_cast<T>(elements);
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
        Matcher() = default;

        /// Check that \p input matches \p expected.
        /// \details An element-wise comparison is performed between \p input and \p expected.
        ///          There's a match if input values are equal to the expected values +/- \p epsilon.
        /// \note If \p T is complex, epsilon can be complex or real. In the later, the same real-valued epsilon
        ///       is checked against the real and imaginary part.
        template<typename U>
        Matcher(CompType comparison, const T* input, const T* expected, size_t elements, U epsilon) noexcept
                : m_input(input), m_expected_ptr(expected), m_elements(elements), m_comparison(comparison) {
            if constexpr (noa::traits::is_complex_v<T> && !noa::traits::is_complex_v<U>) {
                using value_t = noa::traits::value_type_t<T>;
                m_epsilon.real = static_cast<value_t>(epsilon);
                m_epsilon.imag = static_cast<value_t>(epsilon);
            } else {
                m_epsilon = static_cast<T>(epsilon);
            }
            check_();
        }

        /// Check that \p input matches \p expected.
        /// \details An element-wise comparison is performed between \p input and \p expected.
        ///          There's a match if input values are equal to the expected value +/- \p epsilon.
        /// \note If \p T is complex, epsilon can be complex or real. In the later, the same real-valued epsilon
        ///       is checked against the real and imaginary part.
        template<typename U>
        Matcher(CompType comparison, const T* input, T expected, size_t elements, U epsilon) noexcept
                : m_input(input), m_elements(elements), m_expected_value(expected), m_comparison(comparison) {
            if constexpr (noa::traits::is_complex_v<T> && !noa::traits::is_complex_v<U>) {
                using value_t = noa::traits::value_type_t<T>;
                m_epsilon.real = static_cast<value_t>(epsilon);
                m_epsilon.imag = static_cast<value_t>(epsilon);
            } else {
                m_epsilon = static_cast<T>(epsilon);
            }
            check_();
        }

        explicit operator bool() const noexcept {
            return m_match;
        }

        friend std::ostream& operator<<(std::ostream& os, const Matcher<T>& matcher) {
            if (matcher)
                return os << "Matcher: all checks are within the expected value(s)";
            else {
                size_t idx = matcher.m_index_failed;
                os << "Matcher: check failed at index=" << idx;

                T expected = matcher.m_expected_ptr ? matcher.m_expected_ptr[idx] : matcher.m_expected_value;
                if constexpr (std::is_integral_v<T>) {
                    os << noa::string::format(", value={}, expected={}, epsilon={}",
                                              matcher.m_input[idx], expected, matcher.m_epsilon);
                } else {
                    int dyn_precision;
                    if constexpr (noa::traits::is_complex_v<T>)
                        dyn_precision = std::max(int(-std::log10(std::abs(matcher.m_epsilon.real))),
                                                 int(-std::log10(std::abs(matcher.m_epsilon.imag)))) + 2;
                    else
                        dyn_precision = int(-std::log10(std::abs(matcher.m_epsilon))) + 2;

                    os << noa::string::format(", value={:.{}}, expected={:.{}}, epsilon={:.{}}",
                                              matcher.m_input[idx], dyn_precision,
                                              expected, dyn_precision,
                                              matcher.m_epsilon, dyn_precision);
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
                check_(Matcher<T>::checkAbs_<T>);
            else if (m_comparison == MATCH_ABS_SAFE)
                check_(Matcher<T>::checkAbsSafe_<T>);
            else if (m_comparison == MATCH_REL)
                check_(Matcher<T>::checkRel_<T>);
        }

        template<typename F>
        void check_(F&& value_checker) noexcept {
            if (m_expected_ptr) {
                for (size_t i = 0; i < m_elements; ++i) {
                    if (!value_checker(m_input[i], m_expected_ptr[i], m_epsilon)) {
                        m_index_failed = i;
                        m_match = false;
                        return;
                    }
                }
            } else {
                for (size_t i = 0; i < m_elements; ++i) {
                    if (!value_checker(m_input[i], m_expected_value, m_epsilon)) {
                        m_index_failed = i;
                        m_match = false;
                        return;
                    }
                }
            }
            m_match = true;
        }

        template<typename U>
        static bool checkAbs_(U input, U expected, U epsilon) noexcept {
            if constexpr (noa::traits::is_complex_v<U>) {
                using real_t = noa::traits::value_type_t<U>;
                return Matcher<T>::checkAbs_<real_t>(input.real, expected.real, epsilon.real) &&
                       Matcher<T>::checkAbs_<real_t>(input.imag, expected.imag, epsilon.imag);
            } else if constexpr (std::is_integral_v<U>) {
                return (input - expected) <= epsilon;
            } else {
                return noa::math::abs(input - expected) <= epsilon;
            }
        }

        template<typename U>
        static bool checkAbsSafe_(U input, U expected, U epsilon) noexcept {
            if constexpr (noa::traits::is_complex_v<U>) {
                using real_t = noa::traits::value_type_t<U>;
                return Matcher<T>::checkAbsSafe_<real_t>(input.real, expected.real, epsilon.real) &&
                       Matcher<T>::checkAbsSafe_<real_t>(input.imag, expected.imag, epsilon.imag);
            } else if constexpr (std::is_integral_v<U>) {
                return (input - expected) <= epsilon;
            } else {
                // Relative epsilons comparisons are usually meaningless for close-to-zero numbers,
                // hence the absolute comparison first, acting as a safety net.
                using namespace ::noa;
                U diff = math::abs(input - expected);
                if (!math::isFinite(diff))
                    return false;
                return diff <= epsilon || diff <= math::max(math::abs(input), math::abs(expected)) * epsilon;
            }
        }

        template<typename U>
        static bool checkRel_(U input, U expected, U epsilon) noexcept {
            if constexpr (noa::traits::is_complex_v<U>) {
                using real_t = noa::traits::value_type_t<U>;
                return Matcher<T>::checkRel_<real_t>(input.real, expected.real, epsilon.real) &&
                       Matcher<T>::checkRel_<real_t>(input.imag, expected.imag, epsilon.imag);
            } else if constexpr (std::is_integral_v<U>) {
                return (input - expected) <= epsilon;
            } else {
                auto margin = epsilon * noa::math::max(noa::math::abs(input), noa::math::abs(expected));
                if (noa::math::isInf(margin))
                    margin = U(0);
                return (input + margin >= expected) && (expected + margin >= input); // abs(a-b) <= epsilon
            }
        }

    private:
        const T* m_input{};
        const T* m_expected_ptr{};
        size_t m_elements{};
        size_t m_index_failed{};
        T m_expected_value{};
        T m_epsilon{};
        CompType m_comparison{};
        bool m_match{true};
    };
}
