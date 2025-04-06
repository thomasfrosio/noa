#include <noa/core/types/Complex.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

#define REQUIRE_COMPLEX_EQUALS_ABS(x, y, abs)                                         \
REQUIRE_THAT(f64(noa::real(x)), Catch::Matchers::WithinAbs(f64(noa::real(y)), abs));  \
REQUIRE_THAT(f64(noa::imag(x)), Catch::Matchers::WithinAbs(f64(noa::imag(y)), abs))

using namespace ::noa::types;
static_assert(sizeof(c16) == sizeof(f16) * 2); // no padding
static_assert(sizeof(c32) == sizeof(f32) * 2);
static_assert(sizeof(c64) == sizeof(f64) * 2);
static_assert(alignof(c16) == 4);
static_assert(alignof(c32) == 8);
static_assert(alignof(c64) == 16);

TEMPLATE_TEST_CASE("core::Complex", "", f16, f32, f64) {
    using real_t = TestType;
    using c0 = noa::Complex<real_t>;
    using c1 = std::complex<real_t>;

    AND_THEN("Basic aggregate initialization") {
        static_assert(std::is_aggregate_v<c0>);
        c0 a{};
        c0 b; // zero initialized
        REQUIRE(a == b);

        a = c0{real_t{-2}, real_t{-1}};
        REQUIRE((a.real == real_t{-2} and a.imag == real_t{-1}));
        a = c0::from_values(1, 2);
        REQUIRE((a.real == real_t{1} and a.imag == real_t{2}));
        a = c0::from_real(4);
        REQUIRE((a.real == real_t{4} and a.imag == real_t{}));
        real_t c[2]{real_t{6}, real_t{5}};
        a = c0::from_pointer(c);
        REQUIRE((a.real == real_t{6} and a.imag == real_t{5}));
        REQUIRE(a.template as<f64>() == c64{6, 5});
        REQUIRE(static_cast<c64>(a) == c64{6, 5});
    }

    AND_THEN("Check array-like access") {
        // This is guaranteed by the C++11 standard for std::complex.
        // I would be very surprised if this doesn't apply to noa::Complex
        // since it is just a struct with 2 floats/doubles...
        // I'm not entirely sure that this is UB.
        REQUIRE(sizeof(c0) == sizeof(c1));
        auto* arr_complex = new c0[3];
        arr_complex[0] = c0{real_t{1.5}, real_t{2.5}};
        arr_complex[1] = c0{real_t{3.5}, real_t{4.5}};
        arr_complex[2] = c0{real_t{5.5}, real_t{6.5}};

        auto* arr_char = reinterpret_cast<real_t*>(arr_complex);
        REQUIRE(c0{arr_char[0], arr_char[1]} == arr_complex[0]);
        REQUIRE(c0{arr_char[2], arr_char[3]} == arr_complex[1]);
        REQUIRE(c0{arr_char[4], arr_char[5]} == arr_complex[2]);

        // This is equivalent...
        auto* a1 = reinterpret_cast<char*>(arr_char) + sizeof(real_t) * 2;
        auto* a2 = a1 + sizeof(real_t) * 2;

        REQUIRE((reinterpret_cast<char*>(arr_complex + 1) == a1));
        REQUIRE((reinterpret_cast<char*>(arr_complex + 2) == a2));
        delete[] arr_complex;
    }

    AND_THEN("Check 'equivalence' with std::complex") {
        c0 noa_complex{};
        c1 std_complex{};

        // Generating some random numbers
        test::Randomizer<real_t> randomizer1(real_t(-5), real_t(5));
        test::Randomizer<real_t> randomizer2(real_t(.5), real_t(1));

        real_t scalar1 = randomizer1.get();
        real_t scalar2 = randomizer1.get();
        real_t scalar3 = randomizer1.get();
        real_t scalar4_nonzero = randomizer2.get();
        real_t scalar5_nonzero = randomizer2.get();
        real_t scalar6_nonzero = randomizer2.get();

        f64 epsilon = std::is_same_v<Half, real_t> ? 5e-3 : 1e-6;

        AND_THEN("Operators: '+=', '-=', '*=', '/='") {
            std_complex += scalar1;
            noa_complex += scalar1;
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);

            std_complex = real_t{0}; std_complex -= scalar2;
            noa_complex = real_t{0}; noa_complex -= scalar2;
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);

            std_complex = scalar1; std_complex *= scalar3;
            noa_complex = scalar1; noa_complex *= scalar3;
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);

            std_complex = scalar2; std_complex /= scalar4_nonzero;
            noa_complex = scalar2; noa_complex /= scalar4_nonzero;
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);

            std_complex = scalar3; std_complex += c1(scalar5_nonzero, scalar2);
            noa_complex = scalar3; noa_complex += c0{scalar5_nonzero, scalar2};
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);

            std_complex = scalar3; std_complex -= c1(scalar1, scalar6_nonzero);
            noa_complex = scalar3; noa_complex -= c0{scalar1, scalar6_nonzero};
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);

            std_complex = scalar2; std_complex *= c1(scalar3, scalar1);
            noa_complex = scalar2; noa_complex *= c0{scalar3, scalar1};
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);

            std_complex = scalar1; std_complex /= c1(scalar6_nonzero, scalar4_nonzero);
            noa_complex = scalar1; noa_complex /= c0{scalar6_nonzero, scalar4_nonzero};
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);
        }

        AND_THEN("Operators: '+', '-', '*', '/'") {
            std_complex = c1(scalar3, scalar1) + c1(scalar4_nonzero, scalar5_nonzero);
            noa_complex = c0{scalar3, scalar1} + c0{scalar4_nonzero, scalar5_nonzero};
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);
            std_complex = c1(scalar2, scalar3) + scalar4_nonzero;
            noa_complex = c0{scalar2, scalar3} + scalar4_nonzero;
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);
            std_complex = scalar1 + c1(scalar5_nonzero, scalar6_nonzero);
            noa_complex = scalar1 + c0{scalar5_nonzero, scalar6_nonzero};
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);

            std_complex = c1(scalar3, scalar1) - c1(scalar4_nonzero, scalar5_nonzero);
            noa_complex = c0{scalar3, scalar1} - c0{scalar4_nonzero, scalar5_nonzero};
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);
            std_complex = c1(scalar2, scalar3) - scalar4_nonzero;
            noa_complex = c0{scalar2, scalar3} - scalar4_nonzero;
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);
            std_complex = scalar1 - c1(scalar5_nonzero, scalar6_nonzero);
            noa_complex = scalar1 - c0{scalar5_nonzero, scalar6_nonzero};
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);

            std_complex = c1(scalar3, scalar1) * c1(scalar4_nonzero, scalar5_nonzero);
            noa_complex = c0{scalar3, scalar1} * c0{scalar4_nonzero, scalar5_nonzero};
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);
            std_complex = c1(scalar2, scalar3) * scalar4_nonzero;
            noa_complex = c0{scalar2, scalar3} * scalar4_nonzero;
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);
            std_complex = scalar1 * c1(scalar5_nonzero, scalar6_nonzero);
            noa_complex = scalar1 * c0{scalar5_nonzero, scalar6_nonzero};
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);

            std_complex = c1(scalar3, scalar1) / c1(scalar4_nonzero, scalar5_nonzero);
            noa_complex = c0{scalar3, scalar1} / c0{scalar4_nonzero, scalar5_nonzero};
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);
            std_complex = c1(scalar2, scalar3) / scalar4_nonzero;
            noa_complex = c0{scalar2, scalar3} / scalar4_nonzero;
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);
            std_complex = scalar1 / c1(scalar5_nonzero, scalar6_nonzero);
            noa_complex = scalar1 / c0{scalar5_nonzero, scalar6_nonzero};
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);
        }

        AND_THEN("Other non-member functions") {
            if constexpr (std::is_same_v<real_t, Half>) {
                auto f1 = static_cast<f32>(scalar1);
                auto f2 = static_cast<f32>(scalar2);
                auto f3 = static_cast<f32>(scalar3);
                auto f4 = static_cast<f32>(scalar4_nonzero);
                c16 nc{scalar1, scalar3};
                std::complex<f32> sc(f1, f3);

                nc = c16{scalar1, scalar3};
                sc = std::complex<f32>(f1,f3);
                REQUIRE(nc == sc);

                nc = noa::abs(nc);
                sc = std::abs(sc);
                REQUIRE_COMPLEX_EQUALS_ABS(nc, sc, epsilon);

                nc = noa::arg(c16{scalar3, scalar1});
                sc = std::arg(std::complex<f32>(f3, f1));
                REQUIRE_COMPLEX_EQUALS_ABS(nc, sc, epsilon);

                // norm is too ambiguous...
                nc = noa::abs_squared(c16{scalar3, scalar1});
                sc = std::norm(std::complex<f32>(f3, f1));
                REQUIRE_COMPLEX_EQUALS_ABS(nc, sc, epsilon * 10);

                nc = noa::conj(c16{scalar2, scalar1});
                sc = std::conj(std::complex<f32>(f2, f1));
                REQUIRE_COMPLEX_EQUALS_ABS(nc, sc, epsilon);

                nc = noa::polar(scalar4_nonzero, scalar1);
                sc = std::polar(f4, f1);
                REQUIRE_COMPLEX_EQUALS_ABS(nc, sc, epsilon);

            } else {
                noa_complex = c0{scalar1, scalar3};
                std_complex = c1(scalar1, scalar3);
                REQUIRE(noa_complex == std_complex);

                noa_complex = noa::abs(noa_complex);
                std_complex = std::abs(std_complex);
                REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);

                noa_complex = noa::arg(c0{scalar3, scalar1});
                std_complex = std::arg(c1(scalar3, scalar1));
                REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);

                noa_complex = noa::abs_squared(c0{scalar3, scalar1});
                std_complex = std::norm(c1(scalar3, scalar1));
                REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);

                noa_complex = noa::conj(c0{scalar2, scalar1});
                std_complex = std::conj(c1(scalar2, scalar1));
                REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);

                noa_complex = noa::polar(scalar4_nonzero, scalar1);
                std_complex = std::polar(scalar4_nonzero, scalar1);
                REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);
            }
        }
    }

    AND_THEN("to string") {
        noa::Complex<TestType> z{};
        REQUIRE((fmt::format("{:.3f}", z) == std::string{"(0.000,0.000)"}));
        REQUIRE(fmt::format("{::.2f}", std::array<c32, 2>{}) == "[(0.00,0.00), (0.00,0.00)]");
        REQUIRE((noa::string::stringify<c16>() == "c16" and
                 noa::string::stringify<c32>() == "c32" and
                 noa::string::stringify<c64>() == "c64"));
    }
}
