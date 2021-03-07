#include <catch2/catch.hpp>

#include <noa/util/Complex.h>
#include <noa/util/OS.h>

#include "../../Helpers.h"

using namespace ::Noa;

#define REQUIRE_COMPLEX_EQUALS_ABS(x, y, abs)                       \
REQUIRE_THAT(x.real(), Catch::WithinAbs(double(y.real()), abs));    \
REQUIRE_THAT(x.imag(), Catch::WithinAbs(double(y.imag()), abs))

TEMPLATE_TEST_CASE("Complex numbers", "[noa][complex]", float, double) {
    using NoaComplex = Complex<TestType>;
    using StdComplex = std::complex<TestType>;

    AND_THEN("Architecture requirement") {
        INFO("The implementation (adapted from nvidia/thrust) assumes int is 32bits and little-endian");
        REQUIRE(sizeof(int) == 4);
        REQUIRE(!OS::isBigEndian());
    }

    AND_THEN("Check array-like access") {
        // This is guaranteed by the C++11 standard for std::complex.
        // I would be very surprised if this doesn't apply to Noa::Complex
        // since it is just a struct with 2 floats/doubles...
        REQUIRE(sizeof(NoaComplex) == sizeof(StdComplex));
        auto* arr_complex = new NoaComplex[3];
        arr_complex[0] = NoaComplex(1.5, 2.5);
        arr_complex[1] = NoaComplex(3.5, 4.5);
        arr_complex[2] = NoaComplex(5.5, 6.5);

        auto* arr_char = reinterpret_cast<TestType*>(arr_complex);
        REQUIRE(NoaComplex(arr_char[0], arr_char[1]) == arr_complex[0]);
        REQUIRE(NoaComplex(arr_char[2], arr_char[3]) == arr_complex[1]);
        REQUIRE(NoaComplex(arr_char[4], arr_char[5]) == arr_complex[2]);

        // This is equivalent, but I'd rather make sure the addresses match.
        auto* a1 = reinterpret_cast<char*>(arr_char) + sizeof(TestType) * 2;
        auto* a2 = a1 + sizeof(TestType) * 2;

        REQUIRE((reinterpret_cast<char*>(arr_complex + 1) == a1));
        REQUIRE((reinterpret_cast<char*>(arr_complex + 2) == a2));
        delete[] arr_complex;
    }

    AND_THEN("Check 'equivalence' with std::complex") {
        NoaComplex noa_complex;
        StdComplex std_complex;

        // Generating some random numbers
        Test::RealRandomizer<TestType> randomizer1(TestType(-5), TestType(5));
        Test::RealRandomizer<TestType> randomizer2(TestType(.5), TestType(1));

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

            std_complex = scalar3; std_complex += StdComplex(scalar5_nonzero, scalar2);
            noa_complex = scalar3; noa_complex += NoaComplex(scalar5_nonzero, scalar2);
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);

            std_complex = scalar3; std_complex -= StdComplex(scalar1, scalar6_nonzero);
            noa_complex = scalar3; noa_complex -= NoaComplex(scalar1, scalar6_nonzero);
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);

            std_complex = scalar2; std_complex *= StdComplex(scalar3, scalar1);
            noa_complex = scalar2; noa_complex *= NoaComplex(scalar3, scalar1);
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);

            std_complex = scalar1; std_complex /= StdComplex(scalar6_nonzero, scalar4_nonzero);
            noa_complex = scalar1; noa_complex /= NoaComplex(scalar6_nonzero, scalar4_nonzero);
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);
        }

        AND_THEN("Operators: '+', '-', '*', '/'") {
            std_complex = StdComplex(scalar3, scalar1) + StdComplex(scalar4_nonzero, scalar5_nonzero);
            noa_complex = NoaComplex(scalar3, scalar1) + NoaComplex(scalar4_nonzero, scalar5_nonzero);
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);
            std_complex = StdComplex(scalar2, scalar3) + scalar4_nonzero;
            noa_complex = NoaComplex(scalar2, scalar3) + scalar4_nonzero;
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);
            std_complex = scalar1 + StdComplex(scalar5_nonzero, scalar6_nonzero);
            noa_complex = scalar1 + NoaComplex(scalar5_nonzero, scalar6_nonzero);
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);

            std_complex = StdComplex(scalar3, scalar1) - StdComplex(scalar4_nonzero, scalar5_nonzero);
            noa_complex = NoaComplex(scalar3, scalar1) - NoaComplex(scalar4_nonzero, scalar5_nonzero);
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);
            std_complex = StdComplex(scalar2, scalar3) - scalar4_nonzero;
            noa_complex = NoaComplex(scalar2, scalar3) - scalar4_nonzero;
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);
            std_complex = scalar1 - StdComplex(scalar5_nonzero, scalar6_nonzero);
            noa_complex = scalar1 - NoaComplex(scalar5_nonzero, scalar6_nonzero);
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);

            std_complex = StdComplex(scalar3, scalar1) * StdComplex(scalar4_nonzero, scalar5_nonzero);
            noa_complex = NoaComplex(scalar3, scalar1) * NoaComplex(scalar4_nonzero, scalar5_nonzero);
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);
            std_complex = StdComplex(scalar2, scalar3) * scalar4_nonzero;
            noa_complex = NoaComplex(scalar2, scalar3) * scalar4_nonzero;
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);
            std_complex = scalar1 * StdComplex(scalar5_nonzero, scalar6_nonzero);
            noa_complex = scalar1 * NoaComplex(scalar5_nonzero, scalar6_nonzero);
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);

            std_complex = StdComplex(scalar3, scalar1) / StdComplex(scalar4_nonzero, scalar5_nonzero);
            noa_complex = NoaComplex(scalar3, scalar1) / NoaComplex(scalar4_nonzero, scalar5_nonzero);
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);
            std_complex = StdComplex(scalar2, scalar3) / scalar4_nonzero;
            noa_complex = NoaComplex(scalar2, scalar3) / scalar4_nonzero;
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);
            std_complex = scalar1 / StdComplex(scalar5_nonzero, scalar6_nonzero);
            noa_complex = scalar1 / NoaComplex(scalar5_nonzero, scalar6_nonzero);
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);
        }

        AND_THEN("Other non-member functions") {
            noa_complex = NoaComplex(scalar1, scalar3);
            std_complex = StdComplex(scalar1, scalar3);
            REQUIRE(noa_complex == std_complex);

            noa_complex = Math::abs(noa_complex);
            std_complex = std::abs(std_complex);
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);

            noa_complex = Math::arg(NoaComplex(scalar3, scalar1));
            std_complex = std::arg(StdComplex(scalar3, scalar1));
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);

            noa_complex = Math::norm(NoaComplex(scalar3, scalar1));
            std_complex = std::norm(StdComplex(scalar3, scalar1));
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);

            noa_complex = Math::conj(NoaComplex(scalar2, scalar1));
            std_complex = std::conj(StdComplex(scalar2, scalar1));
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);

            noa_complex = Math::polar(scalar4_nonzero, scalar1);
            std_complex = std::polar(scalar4_nonzero, scalar1);
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);

            noa_complex = Math::exp(NoaComplex(scalar3, scalar1));
            std_complex = std::exp(StdComplex(scalar3, scalar1));
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);

            noa_complex = Math::log(NoaComplex(scalar3, scalar2));
            std_complex = std::log(StdComplex(scalar3, scalar2));
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);

            noa_complex = Math::log10(NoaComplex(scalar4_nonzero, scalar1));
            std_complex = std::log10(StdComplex(scalar4_nonzero, scalar1));
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);

            noa_complex = Math::sqrt(NoaComplex(scalar4_nonzero, scalar2));
            std_complex = std::sqrt(StdComplex(scalar4_nonzero, scalar2));
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);

            noa_complex = Math::pow(NoaComplex(scalar3, scalar1), NoaComplex(scalar2, scalar6_nonzero));
            std_complex = std::pow(StdComplex(scalar3, scalar1), StdComplex(scalar2, scalar6_nonzero));
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-4); // exp, log...

            noa_complex = Math::pow(NoaComplex(scalar2, scalar6_nonzero), scalar4_nonzero);
            std_complex = std::pow(StdComplex(scalar2, scalar6_nonzero), scalar4_nonzero);
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-6);

            noa_complex = Math::pow(scalar4_nonzero, NoaComplex(scalar2, scalar6_nonzero));
            std_complex = std::pow(scalar4_nonzero, StdComplex(scalar2, scalar6_nonzero));
            REQUIRE_COMPLEX_EQUALS_ABS(noa_complex, std_complex, 1e-4);
        }
    }

    AND_THEN("to string") {
        Noa::Complex<TestType> z;
        REQUIRE((toString(z) == std::string{"(0.0,0.0)"}));
        REQUIRE((String::format("{}", z) == std::string{"(0.0,0.0)"}));
    }
}
