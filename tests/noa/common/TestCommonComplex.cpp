#include <noa/common/types/Complex.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

#define REQUIRE_COMPLEX_EQUALS_ABS(x, y, abs)                               \
REQUIRE_THAT(math::real(x), Catch::WithinAbs(double(math::real(y)), abs));  \
REQUIRE_THAT(math::imag(x), Catch::WithinAbs(double(math::imag(y)), abs))

TEMPLATE_TEST_CASE("Complex", "[noa][complex]", float, double) {
    using noaComplex = Complex<TestType>;
    using stdComplex = std::complex<TestType>;

    AND_THEN("Check array-like access") {
        // This is guaranteed by the C++11 standard for std::complex.
        // I would be very surprised if this doesn't apply to noa::Complex
        // since it is just a struct with 2 floats/doubles...
        REQUIRE(sizeof(noaComplex) == sizeof(stdComplex));
        auto* arr_complex = new noaComplex[3];
        arr_complex[0] = noaComplex(1.5, 2.5);
        arr_complex[1] = noaComplex(3.5, 4.5);
        arr_complex[2] = noaComplex(5.5, 6.5);

        auto* arr_char = reinterpret_cast<TestType*>(arr_complex);
        REQUIRE(noaComplex(arr_char[0], arr_char[1]) == arr_complex[0]);
        REQUIRE(noaComplex(arr_char[2], arr_char[3]) == arr_complex[1]);
        REQUIRE(noaComplex(arr_char[4], arr_char[5]) == arr_complex[2]);

        // This is equivalent...
        auto* a1 = reinterpret_cast<char*>(arr_char) + sizeof(TestType) * 2;
        auto* a2 = a1 + sizeof(TestType) * 2;

        REQUIRE((reinterpret_cast<char*>(arr_complex + 1) == a1));
        REQUIRE((reinterpret_cast<char*>(arr_complex + 2) == a2));
        delete[] arr_complex;
    }

    AND_THEN("Check 'equivalence' with std::complex") {
        noaComplex noa_complex;
        stdComplex std_complex;

        // Generating some random numbers
        test::Randomizer<TestType> randomizer1(TestType(-5), TestType(5));
        test::Randomizer<TestType> randomizer2(TestType(.5), TestType(1));

        TestType scalar1 = randomizer1.get();
        TestType scalar2 = randomizer1.get();
        TestType scalar3 = randomizer1.get();
        TestType scalar4_nonzero = randomizer2.get();
        TestType scalar5_nonzero = randomizer2.get();
        TestType scalar6_nonzero = randomizer2.get();

        AND_THEN("Operators: '+=', '-=', '*=', '/='") {
            std_complex += scalar1;
            noa_complex += scalar1;
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);

            std_complex = 0; std_complex -= scalar2;
            noa_complex = 0; noa_complex -= scalar2;
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);

            std_complex = scalar1; std_complex *= scalar3;
            noa_complex = scalar1; noa_complex *= scalar3;
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);

            std_complex = scalar2; std_complex /= scalar4_nonzero;
            noa_complex = scalar2; noa_complex /= scalar4_nonzero;
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);

            std_complex = scalar3; std_complex += stdComplex(scalar5_nonzero, scalar2);
            noa_complex = scalar3; noa_complex += noaComplex(scalar5_nonzero, scalar2);
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);

            std_complex = scalar3; std_complex -= stdComplex(scalar1, scalar6_nonzero);
            noa_complex = scalar3; noa_complex -= noaComplex(scalar1, scalar6_nonzero);
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);

            std_complex = scalar2; std_complex *= stdComplex(scalar3, scalar1);
            noa_complex = scalar2; noa_complex *= noaComplex(scalar3, scalar1);
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);

            std_complex = scalar1; std_complex /= stdComplex(scalar6_nonzero, scalar4_nonzero);
            noa_complex = scalar1; noa_complex /= noaComplex(scalar6_nonzero, scalar4_nonzero);
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);
        }

        AND_THEN("Operators: '+', '-', '*', '/'") {
            std_complex = stdComplex(scalar3, scalar1) + stdComplex(scalar4_nonzero, scalar5_nonzero);
            noa_complex = noaComplex(scalar3, scalar1) + noaComplex(scalar4_nonzero, scalar5_nonzero);
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);
            std_complex = stdComplex(scalar2, scalar3) + scalar4_nonzero;
            noa_complex = noaComplex(scalar2, scalar3) + scalar4_nonzero;
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);
            std_complex = scalar1 + stdComplex(scalar5_nonzero, scalar6_nonzero);
            noa_complex = scalar1 + noaComplex(scalar5_nonzero, scalar6_nonzero);
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);

            std_complex = stdComplex(scalar3, scalar1) - stdComplex(scalar4_nonzero, scalar5_nonzero);
            noa_complex = noaComplex(scalar3, scalar1) - noaComplex(scalar4_nonzero, scalar5_nonzero);
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);
            std_complex = stdComplex(scalar2, scalar3) - scalar4_nonzero;
            noa_complex = noaComplex(scalar2, scalar3) - scalar4_nonzero;
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);
            std_complex = scalar1 - stdComplex(scalar5_nonzero, scalar6_nonzero);
            noa_complex = scalar1 - noaComplex(scalar5_nonzero, scalar6_nonzero);
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);

            std_complex = stdComplex(scalar3, scalar1) * stdComplex(scalar4_nonzero, scalar5_nonzero);
            noa_complex = noaComplex(scalar3, scalar1) * noaComplex(scalar4_nonzero, scalar5_nonzero);
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);
            std_complex = stdComplex(scalar2, scalar3) * scalar4_nonzero;
            noa_complex = noaComplex(scalar2, scalar3) * scalar4_nonzero;
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);
            std_complex = scalar1 * stdComplex(scalar5_nonzero, scalar6_nonzero);
            noa_complex = scalar1 * noaComplex(scalar5_nonzero, scalar6_nonzero);
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);

            std_complex = stdComplex(scalar3, scalar1) / stdComplex(scalar4_nonzero, scalar5_nonzero);
            noa_complex = noaComplex(scalar3, scalar1) / noaComplex(scalar4_nonzero, scalar5_nonzero);
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);
            std_complex = stdComplex(scalar2, scalar3) / scalar4_nonzero;
            noa_complex = noaComplex(scalar2, scalar3) / scalar4_nonzero;
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);
            std_complex = scalar1 / stdComplex(scalar5_nonzero, scalar6_nonzero);
            noa_complex = scalar1 / noaComplex(scalar5_nonzero, scalar6_nonzero);
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);
        }

        AND_THEN("Other non-member functions") {
            noa_complex = noaComplex(scalar1, scalar3);
            std_complex = stdComplex(scalar1, scalar3);
            REQUIRE(noa_complex == std_complex);

            noa_complex = math::abs(noa_complex);
            std_complex = std::abs(std_complex);
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);

            noa_complex = math::arg(noaComplex(scalar3, scalar1));
            std_complex = std::arg(stdComplex(scalar3, scalar1));
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);

            noa_complex = math::norm(noaComplex(scalar3, scalar1));
            std_complex = std::norm(stdComplex(scalar3, scalar1));
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);

            noa_complex = math::conj(noaComplex(scalar2, scalar1));
            std_complex = std::conj(stdComplex(scalar2, scalar1));
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);

            noa_complex = math::polar(scalar4_nonzero, scalar1);
            std_complex = std::polar(scalar4_nonzero, scalar1);
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);
        }
    }

    AND_THEN("to string") {
        noa::Complex<TestType> z;
        REQUIRE((string::format("{:.4}", z) == std::string{"(0.000,0.000)"}));
    }
}
