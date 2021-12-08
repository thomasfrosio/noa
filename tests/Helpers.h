#pragma once

#include <noa/common/Types.h>
#include <noa/common/Math.h>
#include <noa/common/traits/BaseTypes.h>

#include <random>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include <catch2/catch.hpp>

namespace test {
    extern noa::path_t PATH_TEST_DATA; // defined at runtime by main.
}

#define REQUIRE_FOR_ALL(range, predicate) for (auto& e: (range)) REQUIRE((predicate(e)))

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
        using distributor_t = std::conditional_t<std::is_integral_v<T>,
                                                 std::uniform_int_distribution<T>,
                                                 std::uniform_real_distribution<value_t>>;
        std::random_device rand_dev;
        std::mt19937 generator;
        distributor_t distribution;
    public:
        template<typename U>
        Randomizer(U range_from, U range_to)
                : generator(rand_dev()),
                  distribution(static_cast<value_t>(range_from), static_cast<value_t>(range_to)) {}
        inline T get() {
            if constexpr(noa::traits::is_complex_v<T>)
                return {distribution(generator), distribution(generator)};
            else
                return distribution(generator);
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

    template<typename T, typename U>
    inline void normalize(T* array, size_t size, U scale) {
        for (size_t idx{0}; idx < size; ++idx) {
            array[idx] *= scale;
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
        MATCH_REL,
    };

    /// On construction, each element of the input array is checked.
    /// The operator bool of this class returns true if all elements have passed the check.
    template<typename T>
    class Matcher {
    public:
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
                // Get meaningful precision:
                int dyn_precision;
                if constexpr (noa::traits::is_complex_v<T>)
                    dyn_precision = std::max(int(-std::log10(std::abs(matcher.m_epsilon.real))),
                                             int(-std::log10(std::abs(matcher.m_epsilon.imag)))) + 2;
                else
                    dyn_precision = int(-std::log10(std::abs(matcher.m_epsilon))) + 2;

                size_t idx = matcher.m_index_failed;
                os << "Matcher: check failed at index=" << idx;

                if (matcher.m_expected_ptr) {
                    os << noa::string::format(", value={:.{}}, expected={:.{}}, epsilon={:.{}}",
                                              matcher.m_input[idx], dyn_precision,
                                              matcher.m_expected_ptr[idx], dyn_precision,
                                              matcher.m_epsilon, dyn_precision);
                } else {
                    os << noa::string::format(", value={:.{}}, expected={:.{}}, epsilon={:.{}}",
                                              matcher.m_input[idx], dyn_precision,
                                              matcher.m_expected_value, dyn_precision,
                                              matcher.m_epsilon, dyn_precision);
                }

                if (matcher.m_comparison == MATCH_ABS)
                    os << ", comparison=MATCH_ABS";
                else if (matcher.m_comparison == MATCH_REL)
                    os << ", comparison=MATCH_REL";
                return os;
            }
        }

    private:
        void check_() noexcept {
            if (m_comparison == MATCH_ABS)
                check_(Matcher<T>::checkAbs_<T>);
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
                return (input - expected) == 0;
            } else {
                return std::abs(input - expected) <= epsilon;
            }
        }

        template<typename U>
        static bool checkRel_(U input, U expected, U epsilon) noexcept {
            if constexpr (noa::traits::is_complex_v<U>) {
                using real_t = noa::traits::value_type_t<U>;
                return Matcher<T>::checkRel_<real_t>(input.real, expected.real, epsilon.real) &&
                       Matcher<T>::checkRel_<real_t>(input.imag, expected.imag, epsilon.imag);
            } else {
                auto margin = epsilon * noa::math::max(std::abs(input), noa::math::abs(expected));
                if (std::isinf(margin))
                    margin = 0;
                return (input + margin >= expected) && (expected + margin >= input); // abs(a-b) <= epsilon
            }
        }

    private:
        const T* m_input;
        const T* m_expected_ptr{};
        size_t m_elements;
        size_t m_index_failed{};
        T m_expected_value{};
        T m_epsilon;
        CompType m_comparison;
        bool m_match{};
    };
}

// MATCHERS:
namespace test {
    template<typename T, typename U>
    class WithinAbs : public Catch::MatcherBase<T> {
    private:
        T m_expected;
        U m_epsilon;
    public:
        WithinAbs(T expected, U epsilon) : m_expected(expected), m_epsilon(epsilon) {}

        bool match(const T& value) const override {
            if constexpr (std::is_same_v<T, noa::cfloat_t>) {
                return noa::math::abs(value.real - m_expected.real) <= static_cast<float>(m_epsilon) &&
                       noa::math::abs(value.imag - m_expected.imag) <= static_cast<float>(m_epsilon);
            } else if constexpr (std::is_same_v<T, noa::cdouble_t>) {
                return noa::math::abs(value.real - m_expected.real) <= static_cast<double>(m_epsilon) &&
                       noa::math::abs(value.imag - m_expected.imag) <= static_cast<double>(m_epsilon);
            } else if constexpr (std::is_integral_v<T>) {
                return value - m_expected == 0;
            } else {
                return noa::math::abs(value - m_expected) <= static_cast<T>(m_epsilon);
            }
        }

        std::string describe() const override {
            std::ostringstream ss;
            if constexpr (std::is_integral_v<T>)
                ss << "is equal to " << m_expected;
            else
                ss << "is equal to " << m_expected << " +/- abs epsilon of " << m_epsilon;
            return ss.str();
        }
    };

    // Whether or not the tested value is equal to \a expected_value +/- \a epsilon.
    // \note For complex types, the same epsilon is applied to the real and imaginary part.
    // \note For integral types, \a epsilon is ignored.
    template<typename T, typename U, typename = std::enable_if_t<std::is_floating_point_v<U>>>
    inline WithinAbs<T, U> isWithinAbs(T expected_value, U epsilon) {
        return WithinAbs(expected_value, epsilon);
    }

    template<typename T, typename U>
    class WithinRel : public Catch::MatcherBase<T> {
    private:
        using real_t = noa::traits::value_type_t<T>;
        T m_expected;
        real_t m_epsilon;

        static bool isWithin_(real_t expected, real_t result, real_t epsilon) {
            auto margin = epsilon * noa::math::max(noa::math::abs(result), noa::math::abs(expected));
            if (std::isinf(margin))
                margin = 0;
            return (result + margin >= expected) && (expected + margin >= result); // abs(a-b) <= epsilon
        }

    public:
        WithinRel(T expected, U epsilon) : m_expected(expected), m_epsilon(static_cast<real_t>(epsilon)) {}

        bool match(const T& value) const override {
            if constexpr (noa::traits::is_complex_v<T>)
                return isWithin_(m_expected.real, value.real, m_epsilon) &&
                       isWithin_(m_expected.imag, value.imag, m_epsilon);
            else
                return isWithin_(m_expected, value, m_epsilon);
        }

        std::string describe() const override {
            std::ostringstream ss;
            ss << "and " << m_expected << " are within " << m_epsilon * 100 << "% of each other";
            return ss.str();
        }
    };

    // Whether or not the tested value and \a expected_value are within \a epsilon % of each other.
    // \note For complex types, the same epsilon is applied to the real and imaginary part.
    // \warning For close to zero or zeros, it might be necessary to have an absolute check since the epsilon is
    //          scaled by the value, resulting in an extremely small epsilon...
    template<typename T, typename U = noa::traits::value_type_t<T>>
    inline WithinRel<T, U> isWithinRel(T expected_value,
                                       U epsilon = noa::math::Limits<noa::traits::value_type_t<T>>::epsilon() * 100) {
        static_assert(std::is_floating_point_v<U> && (noa::traits::is_float_v<T> || noa::traits::is_complex_v<T>));
        return WithinRel(expected_value, epsilon);
    }
}
