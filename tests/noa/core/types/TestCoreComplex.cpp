#include <noa/core/types/Complex.hpp>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

#define REQUIRE_COMPLEX_EQUALS_ABS(x, y, abs)                                       \
REQUIRE_THAT(double(noa::real(x)), Catch::WithinAbs(double(noa::real(y)), abs));    \
REQUIRE_THAT(double(noa::imag(x)), Catch::WithinAbs(double(noa::imag(y)), abs))

TEMPLATE_TEST_CASE("core::Complex", "[noa][complex]", Half, float, double) {
    using noaComplex = Complex<TestType>;
    using stdComplex = std::complex<TestType>;

    AND_THEN("Check array-like access") {
        // This is guaranteed by the C++11 standard for std::complex.
        // I would be very surprised if this doesn't apply to noa::Complex
        // since it is just a struct with 2 floats/doubles...
        REQUIRE(sizeof(noaComplex) == sizeof(stdComplex));
        auto* arr_complex = new noaComplex[3];
        arr_complex[0] = noaComplex{TestType{1.5}, TestType{2.5}};
        arr_complex[1] = noaComplex{TestType{3.5}, TestType{4.5}};
        arr_complex[2] = noaComplex{TestType{5.5}, TestType{6.5}};

        auto* arr_char = reinterpret_cast<TestType*>(arr_complex);
        REQUIRE(noaComplex{arr_char[0], arr_char[1]} == arr_complex[0]);
        REQUIRE(noaComplex{arr_char[2], arr_char[3]} == arr_complex[1]);
        REQUIRE(noaComplex{arr_char[4], arr_char[5]} == arr_complex[2]);

        // This is equivalent...
        auto* a1 = reinterpret_cast<char*>(arr_char) + sizeof(TestType) * 2;
        auto* a2 = a1 + sizeof(TestType) * 2;

        REQUIRE((reinterpret_cast<char*>(arr_complex + 1) == a1));
        REQUIRE((reinterpret_cast<char*>(arr_complex + 2) == a2));
        delete[] arr_complex;
    }

    AND_THEN("Check 'equivalence' with std::complex") {
        noaComplex noa_complex{};
        stdComplex std_complex{};

        // Generating some random numbers
        test::Randomizer<TestType> randomizer1(TestType(-5), TestType(5));
        test::Randomizer<TestType> randomizer2(TestType(.5), TestType(1));

        TestType scalar1 = randomizer1.get();
        TestType scalar2 = randomizer1.get();
        TestType scalar3 = randomizer1.get();
        TestType scalar4_nonzero = randomizer2.get();
        TestType scalar5_nonzero = randomizer2.get();
        TestType scalar6_nonzero = randomizer2.get();

        double epsilon = std::is_same_v<Half, TestType> ? 5e-3 : 1e-6;

        AND_THEN("Operators: '+=', '-=', '*=', '/='") {
            std_complex += scalar1;
            noa_complex += scalar1;
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);

            std_complex = TestType{0}; std_complex -= scalar2;
            noa_complex = TestType{0}; noa_complex -= scalar2;
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);

            std_complex = scalar1; std_complex *= scalar3;
            noa_complex = scalar1; noa_complex *= scalar3;
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);

            std_complex = scalar2; std_complex /= scalar4_nonzero;
            noa_complex = scalar2; noa_complex /= scalar4_nonzero;
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);

            std_complex = scalar3; std_complex += stdComplex(scalar5_nonzero, scalar2);
            noa_complex = scalar3; noa_complex += noaComplex{scalar5_nonzero, scalar2};
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);

            std_complex = scalar3; std_complex -= stdComplex(scalar1, scalar6_nonzero);
            noa_complex = scalar3; noa_complex -= noaComplex{scalar1, scalar6_nonzero};
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);

            std_complex = scalar2; std_complex *= stdComplex(scalar3, scalar1);
            noa_complex = scalar2; noa_complex *= noaComplex{scalar3, scalar1};
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);

            std_complex = scalar1; std_complex /= stdComplex(scalar6_nonzero, scalar4_nonzero);
            noa_complex = scalar1; noa_complex /= noaComplex{scalar6_nonzero, scalar4_nonzero};
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);
        }

        AND_THEN("Operators: '+', '-', '*', '/'") {
            std_complex = stdComplex(scalar3, scalar1) + stdComplex(scalar4_nonzero, scalar5_nonzero);
            noa_complex = noaComplex{scalar3, scalar1} + noaComplex{scalar4_nonzero, scalar5_nonzero};
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);
            std_complex = stdComplex(scalar2, scalar3) + scalar4_nonzero;
            noa_complex = noaComplex{scalar2, scalar3} + scalar4_nonzero;
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);
            std_complex = scalar1 + stdComplex(scalar5_nonzero, scalar6_nonzero);
            noa_complex = scalar1 + noaComplex{scalar5_nonzero, scalar6_nonzero};
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);

            std_complex = stdComplex(scalar3, scalar1) - stdComplex(scalar4_nonzero, scalar5_nonzero);
            noa_complex = noaComplex{scalar3, scalar1} - noaComplex{scalar4_nonzero, scalar5_nonzero};
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);
            std_complex = stdComplex(scalar2, scalar3) - scalar4_nonzero;
            noa_complex = noaComplex{scalar2, scalar3} - scalar4_nonzero;
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);
            std_complex = scalar1 - stdComplex(scalar5_nonzero, scalar6_nonzero);
            noa_complex = scalar1 - noaComplex{scalar5_nonzero, scalar6_nonzero};
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);

            std_complex = stdComplex(scalar3, scalar1) * stdComplex(scalar4_nonzero, scalar5_nonzero);
            noa_complex = noaComplex{scalar3, scalar1} * noaComplex{scalar4_nonzero, scalar5_nonzero};
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);
            std_complex = stdComplex(scalar2, scalar3) * scalar4_nonzero;
            noa_complex = noaComplex{scalar2, scalar3} * scalar4_nonzero;
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);
            std_complex = scalar1 * stdComplex(scalar5_nonzero, scalar6_nonzero);
            noa_complex = scalar1 * noaComplex{scalar5_nonzero, scalar6_nonzero};
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);

            std_complex = stdComplex(scalar3, scalar1) / stdComplex(scalar4_nonzero, scalar5_nonzero);
            noa_complex = noaComplex{scalar3, scalar1} / noaComplex{scalar4_nonzero, scalar5_nonzero};
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);
            std_complex = stdComplex(scalar2, scalar3) / scalar4_nonzero;
            noa_complex = noaComplex{scalar2, scalar3} / scalar4_nonzero;
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);
            std_complex = scalar1 / stdComplex(scalar5_nonzero, scalar6_nonzero);
            noa_complex = scalar1 / noaComplex{scalar5_nonzero, scalar6_nonzero};
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);
        }

        AND_THEN("Other non-member functions") {
            if constexpr (std::is_same_v<TestType, Half>) {
                auto f1 = static_cast<float>(scalar1);
                auto f2 = static_cast<float>(scalar2);
                auto f3 = static_cast<float>(scalar3);
                auto f4 = static_cast<float>(scalar4_nonzero);
                noa::Complex<Half> nc{scalar1, scalar3};
                std::complex<float> sc(f1, f3);

                nc = noa::Complex<Half>{scalar1, scalar3};
                sc = std::complex<float>(f1,f3);
                REQUIRE(nc == sc);

                nc = noa::abs(nc);
                sc = std::abs(sc);
                REQUIRE_COMPLEX_EQUALS_ABS(nc, sc, epsilon);

                nc = noa::arg(noa::Complex<Half>{scalar3, scalar1});
                sc = std::arg(std::complex<float>(f3, f1));
                REQUIRE_COMPLEX_EQUALS_ABS(nc, sc, epsilon);

                // norm is too ambiguous...
                nc = noa::abs_squared(noa::Complex<Half>{scalar3, scalar1});
                sc = std::norm(std::complex<float>(f3, f1));
                REQUIRE_COMPLEX_EQUALS_ABS(nc, sc, epsilon * 10);

                nc = noa::conj(noa::Complex<Half>{scalar2, scalar1});
                sc = std::conj(std::complex<float>(f2, f1));
                REQUIRE_COMPLEX_EQUALS_ABS(nc, sc, epsilon);

                nc = noa::polar(scalar4_nonzero, scalar1);
                sc = std::polar(f4, f1);
                REQUIRE_COMPLEX_EQUALS_ABS(nc, sc, epsilon);

            } else {
                noa_complex = noaComplex{scalar1, scalar3};
                std_complex = stdComplex(scalar1, scalar3);
                REQUIRE(noa_complex == std_complex);

                noa_complex = noa::abs(noa_complex);
                std_complex = std::abs(std_complex);
                REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);

                noa_complex = noa::arg(noaComplex{scalar3, scalar1});
                std_complex = std::arg(stdComplex(scalar3, scalar1));
                REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);

                noa_complex = noa::abs_squared(noaComplex{scalar3, scalar1});
                std_complex = std::norm(stdComplex(scalar3, scalar1));
                REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, epsilon);

                noa_complex = noa::conj(noaComplex{scalar2, scalar1});
                std_complex = std::conj(stdComplex(scalar2, scalar1));
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
    }
}
