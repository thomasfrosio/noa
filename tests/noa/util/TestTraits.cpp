/*
 * Test noa/util/Traits.h
 */

#include <catch2/catch.hpp>
#include "noa/util/Traits.h"


TEMPLATE_TEST_CASE("Traits:: integers and sequence of integers", "[noa][traits]",
                   short, int, long, long long, float, double, long double,
                   std::complex<float>, std::complex<double>, std::complex<long double>,
                   std::string, std::string_view, bool) {

    using namespace ::Noa::Traits;

    GIVEN("a correct match") {
        if constexpr (std::is_same_v<TestType, short> ||
                      std::is_same_v<TestType, int> ||
                      std::is_same_v<TestType, long> ||
                      std::is_same_v<TestType, long long>) {

            // single value
            REQUIRE(is_int_v<TestType>);
            REQUIRE(is_int_v<std::add_const_t<TestType>>);
            REQUIRE(is_int_v<std::add_volatile_t<TestType>>);
            REQUIRE(is_int_v<std::add_cv_t<TestType>>);
            REQUIRE(is_int_v<std::add_lvalue_reference_t<TestType>>);
            REQUIRE(is_int_v<std::add_rvalue_reference_t<TestType>>);

            // sequence of values
            REQUIRE(is_sequence_of_int_v<std::vector<TestType>>);
            REQUIRE(is_sequence_of_int_v<std::vector<std::add_const_t<TestType>>>);
            REQUIRE(is_sequence_of_int_v<std::vector<std::add_volatile_t<TestType>>>);
            REQUIRE(is_sequence_of_int_v<std::vector<std::add_cv_t<TestType>>>);
            REQUIRE(is_sequence_of_int_v<std::vector<std::add_lvalue_reference_t<TestType>>>);
            REQUIRE(is_sequence_of_int_v<std::vector<std::add_rvalue_reference_t<TestType>>>);
            REQUIRE(is_sequence_of_int_v<std::array<TestType, 1>>);
            REQUIRE(is_sequence_of_int_v<std::array<std::add_const_t<TestType>, 1>>);
            REQUIRE(is_sequence_of_int_v<std::array<std::add_volatile_t<TestType>, 1>>);
            REQUIRE(is_sequence_of_int_v<std::array<std::add_cv_t<TestType>, 1>>);
            REQUIRE(is_sequence_of_int_v<std::array<std::add_lvalue_reference_t<TestType>, 1>>);
            REQUIRE(is_sequence_of_int_v<std::array<std::add_rvalue_reference_t<TestType>, 1>>);

            REQUIRE(is_vector_of_int_v<std::vector<TestType>>);
            REQUIRE(is_vector_of_int_v<std::vector<std::add_const_t<TestType>>>);
            REQUIRE(is_vector_of_int_v<std::vector<std::add_volatile_t<TestType>>>);
            REQUIRE(is_vector_of_int_v<std::vector<std::add_cv_t<TestType>>>);
            REQUIRE(is_vector_of_int_v<std::vector<std::add_lvalue_reference_t<TestType>>>);
            REQUIRE(is_vector_of_int_v<std::vector<std::add_rvalue_reference_t<TestType>>>);

            REQUIRE(is_array_of_int_v<std::array<TestType, 1>>);
            REQUIRE(is_array_of_int_v<std::array<std::add_const_t<TestType>, 1>>);
            REQUIRE(is_array_of_int_v<std::array<std::add_volatile_t<TestType>, 1>>);
            REQUIRE(is_array_of_int_v<std::array<std::add_cv_t<TestType>, 1>>);
            REQUIRE(is_array_of_int_v<std::array<std::add_lvalue_reference_t<TestType>, 1>>);
            REQUIRE(is_array_of_int_v<std::array<std::add_rvalue_reference_t<TestType>, 1>>);
        }
    }

    GIVEN("an incorrect match") {
        REQUIRE(!is_vector_of_int_v<std::array<TestType, 1>>);
        REQUIRE(!is_array_of_int_v<std::vector<TestType>>);

        using TestTypePpointer = std::add_pointer<TestType>;
        REQUIRE(!is_int_v<TestTypePpointer>);
        REQUIRE(!is_vector_of_int_v<std::vector<TestTypePpointer>>);
        REQUIRE(!is_array_of_int_v<std::array<TestTypePpointer, 1>>);
        REQUIRE(!is_sequence_of_int_v<std::vector<TestTypePpointer>>);

        if constexpr (!(std::is_same_v<TestType, short> ||
                        std::is_same_v<TestType, int> ||
                        std::is_same_v<TestType, long> ||
                        std::is_same_v<TestType, long long>)) {

            // single value
            REQUIRE(!is_int_v<TestType>);
            REQUIRE(!is_int_v<std::add_const_t<TestType>>);
            REQUIRE(!is_int_v<std::add_volatile_t<TestType>>);
            REQUIRE(!is_int_v<std::add_cv_t<TestType>>);
            REQUIRE(!is_int_v<std::add_lvalue_reference_t<TestType>>);
            REQUIRE(!is_int_v<std::add_rvalue_reference_t<TestType>>);

            // sequence of values
            REQUIRE(!is_sequence_of_int_v<std::vector<TestType>>);
            REQUIRE(!is_sequence_of_int_v<std::vector<std::add_const_t<TestType>>>);
            REQUIRE(!is_sequence_of_int_v<std::vector<std::add_volatile_t<TestType>>>);
            REQUIRE(!is_sequence_of_int_v<std::vector<std::add_cv_t<TestType>>>);
            REQUIRE(!is_sequence_of_int_v<std::vector<std::add_lvalue_reference_t<TestType>>>);
            REQUIRE(!is_sequence_of_int_v<std::vector<std::add_rvalue_reference_t<TestType>>>);
            REQUIRE(!is_sequence_of_int_v<std::array<TestType, 1>>);
            REQUIRE(!is_sequence_of_int_v<std::array<std::add_const_t<TestType>, 1>>);
            REQUIRE(!is_sequence_of_int_v<std::array<std::add_volatile_t<TestType>, 1>>);
            REQUIRE(!is_sequence_of_int_v<std::array<std::add_cv_t<TestType>, 1>>);
            REQUIRE(!is_sequence_of_int_v<std::array<std::add_lvalue_reference_t<TestType>, 1>>);
            REQUIRE(!is_sequence_of_int_v<std::array<std::add_rvalue_reference_t<TestType>, 1>>);

            REQUIRE(!is_vector_of_int_v<std::vector<TestType>>);
            REQUIRE(!is_vector_of_int_v<std::vector<std::add_const_t<TestType>>>);
            REQUIRE(!is_vector_of_int_v<std::vector<std::add_volatile_t<TestType>>>);
            REQUIRE(!is_vector_of_int_v<std::vector<std::add_cv_t<TestType>>>);
            REQUIRE(!is_vector_of_int_v<std::vector<std::add_lvalue_reference_t<TestType>>>);
            REQUIRE(!is_vector_of_int_v<std::vector<std::add_rvalue_reference_t<TestType>>>);

            REQUIRE(!is_array_of_int_v<std::array<TestType, 1>>);
            REQUIRE(!is_array_of_int_v<std::array<std::add_const_t<TestType>, 1>>);
            REQUIRE(!is_array_of_int_v<std::array<std::add_volatile_t<TestType>, 1>>);
            REQUIRE(!is_array_of_int_v<std::array<std::add_cv_t<TestType>, 1>>);
            REQUIRE(!is_array_of_int_v<std::array<std::add_lvalue_reference_t<TestType>, 1>>);
            REQUIRE(!is_array_of_int_v<std::array<std::add_rvalue_reference_t<TestType>, 1>>);
        }
    }
}


TEMPLATE_TEST_CASE("Traits:: floating-points and sequence of floating-points", "[noa][traits]",
                   short, int, long, long long, float, double, long double,
                   std::complex<float>, std::complex<double>, std::complex<long double>,
                   std::string, std::string_view, bool) {

    using namespace ::Noa::Traits;

    GIVEN("a correct match") {
        if constexpr (std::is_same_v<TestType, float> ||
                      std::is_same_v<TestType, double> ||
                      std::is_same_v<TestType, long double>) {

            // single value
            REQUIRE(is_float_v<TestType>);
            REQUIRE(is_float_v<std::add_const_t<TestType>>);
            REQUIRE(is_float_v<std::add_volatile_t<TestType>>);
            REQUIRE(is_float_v<std::add_cv_t<TestType>>);
            REQUIRE(is_float_v<std::add_lvalue_reference_t<TestType>>);
            REQUIRE(is_float_v<std::add_rvalue_reference_t<TestType>>);

            // sequence of values
            REQUIRE(is_sequence_of_float_v<std::vector<TestType>>);
            REQUIRE(is_sequence_of_float_v<std::vector<std::add_const_t<TestType>>>);
            REQUIRE(is_sequence_of_float_v<std::vector<std::add_volatile_t<TestType>>>);
            REQUIRE(is_sequence_of_float_v<std::vector<std::add_cv_t<TestType>>>);
            REQUIRE(is_sequence_of_float_v<std::vector<std::add_lvalue_reference_t<TestType>>>);
            REQUIRE(is_sequence_of_float_v<std::vector<std::add_rvalue_reference_t<TestType>>>);
            REQUIRE(is_sequence_of_float_v<std::array<TestType, 1>>);
            REQUIRE(is_sequence_of_float_v<std::array<std::add_const_t<TestType>, 1>>);
            REQUIRE(is_sequence_of_float_v<std::array<std::add_volatile_t<TestType>, 1>>);
            REQUIRE(is_sequence_of_float_v<std::array<std::add_cv_t<TestType>, 1>>);
            REQUIRE(is_sequence_of_float_v<std::array<std::add_lvalue_reference_t<TestType>, 1>>);
            REQUIRE(is_sequence_of_float_v<std::array<std::add_rvalue_reference_t<TestType>, 1>>);

            REQUIRE(is_vector_of_float_v<std::vector<TestType>>);
            REQUIRE(is_vector_of_float_v<std::vector<std::add_const_t<TestType>>>);
            REQUIRE(is_vector_of_float_v<std::vector<std::add_volatile_t<TestType>>>);
            REQUIRE(is_vector_of_float_v<std::vector<std::add_cv_t<TestType>>>);
            REQUIRE(is_vector_of_float_v<std::vector<std::add_lvalue_reference_t<TestType>>>);
            REQUIRE(is_vector_of_float_v<std::vector<std::add_rvalue_reference_t<TestType>>>);

            REQUIRE(is_array_of_float_v<std::array<TestType, 1>>);
            REQUIRE(is_array_of_float_v<std::array<std::add_const_t<TestType>, 1>>);
            REQUIRE(is_array_of_float_v<std::array<std::add_volatile_t<TestType>, 1>>);
            REQUIRE(is_array_of_float_v<std::array<std::add_cv_t<TestType>, 1>>);
            REQUIRE(is_array_of_float_v<std::array<std::add_lvalue_reference_t<TestType>, 1>>);
            REQUIRE(is_array_of_float_v<std::array<std::add_rvalue_reference_t<TestType>, 1>>);
        }
    }

    GIVEN("an incorrect match") {
        REQUIRE(!is_vector_of_float_v<std::array<TestType, 1>>);
        REQUIRE(!is_array_of_float_v<std::vector<TestType>>);

        using TestTypePpointer = std::add_pointer<TestType>;
        REQUIRE(!is_float_v<TestTypePpointer>);
        REQUIRE(!is_vector_of_float_v<std::vector<TestTypePpointer>>);
        REQUIRE(!is_array_of_float_v<std::array<TestTypePpointer, 1>>);
        REQUIRE(!is_sequence_of_float_v<std::vector<TestTypePpointer>>);

        if constexpr (!(std::is_same_v<TestType, float> ||
                        std::is_same_v<TestType, double> ||
                        std::is_same_v<TestType, long double>)) {

            // single value
            REQUIRE(!is_float_v<TestType>);
            REQUIRE(!is_float_v<std::add_const_t<TestType>>);
            REQUIRE(!is_float_v<std::add_volatile_t<TestType>>);
            REQUIRE(!is_float_v<std::add_cv_t<TestType>>);
            REQUIRE(!is_float_v<std::add_lvalue_reference_t<TestType>>);
            REQUIRE(!is_float_v<std::add_rvalue_reference_t<TestType>>);

            // sequence of values
            REQUIRE(!is_sequence_of_float_v<std::vector<TestType>>);
            REQUIRE(!is_sequence_of_float_v<std::vector<std::add_const_t<TestType>>>);
            REQUIRE(!is_sequence_of_float_v<std::vector<std::add_volatile_t<TestType>>>);
            REQUIRE(!is_sequence_of_float_v<std::vector<std::add_cv_t<TestType>>>);
            REQUIRE(!is_sequence_of_float_v<std::vector<std::add_lvalue_reference_t<TestType>>>);
            REQUIRE(!is_sequence_of_float_v<std::vector<std::add_rvalue_reference_t<TestType>>>);
            REQUIRE(!is_sequence_of_float_v<std::array<TestType, 1>>);
            REQUIRE(!is_sequence_of_float_v<std::array<std::add_const_t<TestType>, 1>>);
            REQUIRE(!is_sequence_of_float_v<std::array<std::add_volatile_t<TestType>, 1>>);
            REQUIRE(!is_sequence_of_float_v<std::array<std::add_cv_t<TestType>, 1>>);
            REQUIRE(!is_sequence_of_float_v<std::array<std::add_lvalue_reference_t<TestType>, 1>>);
            REQUIRE(!is_sequence_of_float_v<std::array<std::add_rvalue_reference_t<TestType>, 1>>);

            REQUIRE(!is_vector_of_float_v<std::vector<TestType>>);
            REQUIRE(!is_vector_of_float_v<std::vector<std::add_const_t<TestType>>>);
            REQUIRE(!is_vector_of_float_v<std::vector<std::add_volatile_t<TestType>>>);
            REQUIRE(!is_vector_of_float_v<std::vector<std::add_cv_t<TestType>>>);
            REQUIRE(!is_vector_of_float_v<std::vector<std::add_lvalue_reference_t<TestType>>>);
            REQUIRE(!is_vector_of_float_v<std::vector<std::add_rvalue_reference_t<TestType>>>);

            REQUIRE(!is_array_of_float_v<std::array<TestType, 1>>);
            REQUIRE(!is_array_of_float_v<std::array<std::add_const_t<TestType>, 1>>);
            REQUIRE(!is_array_of_float_v<std::array<std::add_volatile_t<TestType>, 1>>);
            REQUIRE(!is_array_of_float_v<std::array<std::add_cv_t<TestType>, 1>>);
            REQUIRE(!is_array_of_float_v<std::array<std::add_lvalue_reference_t<TestType>, 1>>);
            REQUIRE(!is_array_of_float_v<std::array<std::add_rvalue_reference_t<TestType>, 1>>);
        }
    }
}


TEMPLATE_TEST_CASE("Traits:: complex and sequence of complex", "[noa][traits]",
                   short, int, long, long long, float, double, long double,
                   std::complex<float>, std::complex<double>, std::complex<long double>,
                   std::string, std::string_view, bool) {

    using namespace ::Noa::Traits;

    GIVEN("a correct match") {
        if constexpr (std::is_same_v<TestType, std::complex<float>> ||
                      std::is_same_v<TestType, std::complex<double>> ||
                      std::is_same_v<TestType, std::complex<long double>>) {

            // single value
            REQUIRE(is_complex_v<TestType>);
            REQUIRE(is_complex_v<std::add_const_t<TestType>>);
            REQUIRE(is_complex_v<std::add_volatile_t<TestType>>);
            REQUIRE(is_complex_v<std::add_cv_t<TestType>>);
            REQUIRE(is_complex_v<std::add_lvalue_reference_t<TestType>>);
            REQUIRE(is_complex_v<std::add_rvalue_reference_t<TestType>>);

            // sequence of values
            REQUIRE(is_sequence_of_complex_v<std::vector<TestType>>);
            REQUIRE(is_sequence_of_complex_v<std::vector<std::add_const_t<TestType>>>);
            REQUIRE(is_sequence_of_complex_v<std::vector<std::add_volatile_t<TestType>>>);
            REQUIRE(is_sequence_of_complex_v<std::vector<std::add_cv_t<TestType>>>);
            REQUIRE(is_sequence_of_complex_v<std::vector<std::add_lvalue_reference_t<TestType>>>);
            REQUIRE(is_sequence_of_complex_v<std::vector<std::add_rvalue_reference_t<TestType>>>);
            REQUIRE(is_sequence_of_complex_v<std::array<TestType, 1>>);
            REQUIRE(is_sequence_of_complex_v<std::array<std::add_const_t<TestType>, 1>>);
            REQUIRE(is_sequence_of_complex_v<std::array<std::add_volatile_t<TestType>, 1>>);
            REQUIRE(is_sequence_of_complex_v<std::array<std::add_cv_t<TestType>, 1>>);
            REQUIRE(is_sequence_of_complex_v<std::array<std::add_lvalue_reference_t<TestType>, 1>>);
            REQUIRE(is_sequence_of_complex_v<std::array<std::add_rvalue_reference_t<TestType>, 1>>);

            REQUIRE(is_vector_of_complex_v<std::vector<TestType>>);
            REQUIRE(is_vector_of_complex_v<std::vector<std::add_const_t<TestType>>>);
            REQUIRE(is_vector_of_complex_v<std::vector<std::add_volatile_t<TestType>>>);
            REQUIRE(is_vector_of_complex_v<std::vector<std::add_cv_t<TestType>>>);
            REQUIRE(is_vector_of_complex_v<std::vector<std::add_lvalue_reference_t<TestType>>>);
            REQUIRE(is_vector_of_complex_v<std::vector<std::add_rvalue_reference_t<TestType>>>);

            REQUIRE(is_array_of_complex_v<std::array<TestType, 1>>);
            REQUIRE(is_array_of_complex_v<std::array<std::add_const_t<TestType>, 1>>);
            REQUIRE(is_array_of_complex_v<std::array<std::add_volatile_t<TestType>, 1>>);
            REQUIRE(is_array_of_complex_v<std::array<std::add_cv_t<TestType>, 1>>);
            REQUIRE(is_array_of_complex_v<std::array<std::add_lvalue_reference_t<TestType>, 1>>);
            REQUIRE(is_array_of_complex_v<std::array<std::add_rvalue_reference_t<TestType>, 1>>);
        }
    }

    GIVEN("an incorrect match") {
        REQUIRE(!is_vector_of_complex_v<std::array<TestType, 1>>);
        REQUIRE(!is_array_of_complex_v<std::vector<TestType>>);

        using TestTypePpointer = std::add_pointer<TestType>;
        REQUIRE(!is_complex_v<TestTypePpointer>);
        REQUIRE(!is_vector_of_complex_v<std::vector<TestTypePpointer>>);
        REQUIRE(!is_array_of_complex_v<std::array<TestTypePpointer, 1>>);
        REQUIRE(!is_sequence_of_complex_v<std::vector<TestTypePpointer>>);

        if constexpr (!(std::is_same_v<TestType, std::complex<float>> ||
                        std::is_same_v<TestType, std::complex<double>> ||
                        std::is_same_v<TestType, std::complex<long double>>)) {

            // single value
            REQUIRE(!is_complex_v<TestType>);
            REQUIRE(!is_complex_v<std::add_const_t<TestType>>);
            REQUIRE(!is_complex_v<std::add_volatile_t<TestType>>);
            REQUIRE(!is_complex_v<std::add_cv_t<TestType>>);
            REQUIRE(!is_complex_v<std::add_lvalue_reference_t<TestType>>);
            REQUIRE(!is_complex_v<std::add_rvalue_reference_t<TestType>>);

            // sequence of values
            REQUIRE(!is_sequence_of_complex_v<std::vector<TestType>>);
            REQUIRE(!is_sequence_of_complex_v<std::vector<std::add_const_t<TestType>>>);
            REQUIRE(!is_sequence_of_complex_v<std::vector<std::add_volatile_t<TestType>>>);
            REQUIRE(!is_sequence_of_complex_v<std::vector<std::add_cv_t<TestType>>>);
            REQUIRE(!is_sequence_of_complex_v<std::vector<std::add_lvalue_reference_t<TestType>>>);
            REQUIRE(!is_sequence_of_complex_v<std::vector<std::add_rvalue_reference_t<TestType>>>);
            REQUIRE(!is_sequence_of_complex_v<std::array<TestType, 1>>);
            REQUIRE(!is_sequence_of_complex_v<std::array<std::add_const_t<TestType>, 1>>);
            REQUIRE(!is_sequence_of_complex_v<std::array<std::add_volatile_t<TestType>, 1>>);
            REQUIRE(!is_sequence_of_complex_v<std::array<std::add_cv_t<TestType>, 1>>);
            REQUIRE(!is_sequence_of_complex_v<std::array<std::add_lvalue_reference_t<TestType>, 1>>);
            REQUIRE(!is_sequence_of_complex_v<std::array<std::add_rvalue_reference_t<TestType>, 1>>);

            REQUIRE(!is_vector_of_complex_v<std::vector<TestType>>);
            REQUIRE(!is_vector_of_complex_v<std::vector<std::add_const_t<TestType>>>);
            REQUIRE(!is_vector_of_complex_v<std::vector<std::add_volatile_t<TestType>>>);
            REQUIRE(!is_vector_of_complex_v<std::vector<std::add_cv_t<TestType>>>);
            REQUIRE(!is_vector_of_complex_v<std::vector<std::add_lvalue_reference_t<TestType>>>);
            REQUIRE(!is_vector_of_complex_v<std::vector<std::add_rvalue_reference_t<TestType>>>);

            REQUIRE(!is_array_of_complex_v<std::array<TestType, 1>>);
            REQUIRE(!is_array_of_complex_v<std::array<std::add_const_t<TestType>, 1>>);
            REQUIRE(!is_array_of_complex_v<std::array<std::add_volatile_t<TestType>, 1>>);
            REQUIRE(!is_array_of_complex_v<std::array<std::add_cv_t<TestType>, 1>>);
            REQUIRE(!is_array_of_complex_v<std::array<std::add_lvalue_reference_t<TestType>, 1>>);
            REQUIRE(!is_array_of_complex_v<std::array<std::add_rvalue_reference_t<TestType>, 1>>);
        }
    }
}


TEMPLATE_TEST_CASE("Traits:: string(_view) and sequence of string(_view)", "[noa][traits]",
                   short, int, long, long long, float, double, long double,
                   std::complex<float>, std::complex<double>, std::complex<long double>,
                   std::string, std::string_view, bool) {

    using namespace ::Noa::Traits;

    GIVEN("a correct match") {
        if constexpr (std::is_same_v<TestType, std::string> ||
                      std::is_same_v<TestType, std::string_view>) {

            // single value
            REQUIRE(is_string_v<TestType>);
            REQUIRE(is_string_v<std::add_const_t<TestType>>);
            REQUIRE(is_string_v<std::add_volatile_t<TestType>>);
            REQUIRE(is_string_v<std::add_cv_t<TestType>>);
            REQUIRE(is_string_v<std::add_lvalue_reference_t<TestType>>);
            REQUIRE(is_string_v<std::add_rvalue_reference_t<TestType>>);

            // sequence of values
            REQUIRE(is_sequence_of_string_v<std::vector<TestType>>);
            REQUIRE(is_sequence_of_string_v<std::vector<std::add_const_t<TestType>>>);
            REQUIRE(is_sequence_of_string_v<std::vector<std::add_volatile_t<TestType>>>);
            REQUIRE(is_sequence_of_string_v<std::vector<std::add_cv_t<TestType>>>);
            REQUIRE(is_sequence_of_string_v<std::vector<std::add_lvalue_reference_t<TestType>>>);
            REQUIRE(is_sequence_of_string_v<std::vector<std::add_rvalue_reference_t<TestType>>>);
            REQUIRE(is_sequence_of_string_v<std::array<TestType, 1>>);
            REQUIRE(is_sequence_of_string_v<std::array<std::add_const_t<TestType>, 1>>);
            REQUIRE(is_sequence_of_string_v<std::array<std::add_volatile_t<TestType>, 1>>);
            REQUIRE(is_sequence_of_string_v<std::array<std::add_cv_t<TestType>, 1>>);
            REQUIRE(is_sequence_of_string_v<std::array<std::add_lvalue_reference_t<TestType>, 1>>);
            REQUIRE(is_sequence_of_string_v<std::array<std::add_rvalue_reference_t<TestType>, 1>>);

            REQUIRE(is_vector_of_string_v<std::vector<TestType>>);
            REQUIRE(is_vector_of_string_v<std::vector<std::add_const_t<TestType>>>);
            REQUIRE(is_vector_of_string_v<std::vector<std::add_volatile_t<TestType>>>);
            REQUIRE(is_vector_of_string_v<std::vector<std::add_cv_t<TestType>>>);
            REQUIRE(is_vector_of_string_v<std::vector<std::add_lvalue_reference_t<TestType>>>);
            REQUIRE(is_vector_of_string_v<std::vector<std::add_rvalue_reference_t<TestType>>>);

            REQUIRE(is_array_of_string_v<std::array<TestType, 1>>);
            REQUIRE(is_array_of_string_v<std::array<std::add_const_t<TestType>, 1>>);
            REQUIRE(is_array_of_string_v<std::array<std::add_volatile_t<TestType>, 1>>);
            REQUIRE(is_array_of_string_v<std::array<std::add_cv_t<TestType>, 1>>);
            REQUIRE(is_array_of_string_v<std::array<std::add_lvalue_reference_t<TestType>, 1>>);
            REQUIRE(is_array_of_string_v<std::array<std::add_rvalue_reference_t<TestType>, 1>>);
        }
    }

    GIVEN("an incorrect match") {
        REQUIRE(!is_vector_of_string_v<std::array<TestType, 1>>);
        REQUIRE(!is_array_of_string_v<std::vector<TestType>>);

        using TestTypePpointer = std::add_pointer<TestType>;
        REQUIRE(!is_string_v<TestTypePpointer>);
        REQUIRE(!is_vector_of_string_v<std::vector<TestTypePpointer>>);
        REQUIRE(!is_array_of_string_v<std::array<TestTypePpointer, 1>>);
        REQUIRE(!is_sequence_of_string_v<std::vector<TestTypePpointer>>);

        if constexpr (!(std::is_same_v<TestType, std::string> ||
                        std::is_same_v<TestType, std::string_view>)) {

            // single value
            REQUIRE(!is_string_v<TestType>);
            REQUIRE(!is_string_v<std::add_const_t<TestType>>);
            REQUIRE(!is_string_v<std::add_volatile_t<TestType>>);
            REQUIRE(!is_string_v<std::add_cv_t<TestType>>);
            REQUIRE(!is_string_v<std::add_lvalue_reference_t<TestType>>);
            REQUIRE(!is_string_v<std::add_rvalue_reference_t<TestType>>);

            // sequence of values
            REQUIRE(!is_sequence_of_string_v<std::vector<TestType>>);
            REQUIRE(!is_sequence_of_string_v<std::vector<std::add_const_t<TestType>>>);
            REQUIRE(!is_sequence_of_string_v<std::vector<std::add_volatile_t<TestType>>>);
            REQUIRE(!is_sequence_of_string_v<std::vector<std::add_cv_t<TestType>>>);
            REQUIRE(!is_sequence_of_string_v<std::vector<std::add_lvalue_reference_t<TestType>>>);
            REQUIRE(!is_sequence_of_string_v<std::vector<std::add_rvalue_reference_t<TestType>>>);
            REQUIRE(!is_sequence_of_string_v<std::array<TestType, 1>>);
            REQUIRE(!is_sequence_of_string_v<std::array<std::add_const_t<TestType>, 1>>);
            REQUIRE(!is_sequence_of_string_v<std::array<std::add_volatile_t<TestType>, 1>>);
            REQUIRE(!is_sequence_of_string_v<std::array<std::add_cv_t<TestType>, 1>>);
            REQUIRE(!is_sequence_of_string_v<std::array<std::add_lvalue_reference_t<TestType>, 1>>);
            REQUIRE(!is_sequence_of_string_v<std::array<std::add_rvalue_reference_t<TestType>, 1>>);

            REQUIRE(!is_vector_of_string_v<std::vector<TestType>>);
            REQUIRE(!is_vector_of_string_v<std::vector<std::add_const_t<TestType>>>);
            REQUIRE(!is_vector_of_string_v<std::vector<std::add_volatile_t<TestType>>>);
            REQUIRE(!is_vector_of_string_v<std::vector<std::add_cv_t<TestType>>>);
            REQUIRE(!is_vector_of_string_v<std::vector<std::add_lvalue_reference_t<TestType>>>);
            REQUIRE(!is_vector_of_string_v<std::vector<std::add_rvalue_reference_t<TestType>>>);

            REQUIRE(!is_array_of_string_v<std::array<TestType, 1>>);
            REQUIRE(!is_array_of_string_v<std::array<std::add_const_t<TestType>, 1>>);
            REQUIRE(!is_array_of_string_v<std::array<std::add_volatile_t<TestType>, 1>>);
            REQUIRE(!is_array_of_string_v<std::array<std::add_cv_t<TestType>, 1>>);
            REQUIRE(!is_array_of_string_v<std::array<std::add_lvalue_reference_t<TestType>, 1>>);
            REQUIRE(!is_array_of_string_v<std::array<std::add_rvalue_reference_t<TestType>, 1>>);
        }
    }
}


TEMPLATE_TEST_CASE("Traits:: boolean and sequence of booleans", "[noa][traits]",
                   short, int, long, long long, float, double, long double,
                   std::complex<float>, std::complex<double>, std::complex<long double>,
                   std::string, std::string_view, bool) {

    using namespace ::Noa::Traits;

    GIVEN("a correct match") {
        if constexpr (std::is_same_v<TestType, bool>) {

            // single value
            REQUIRE(is_bool_v<TestType>);
            REQUIRE(is_bool_v<std::add_const_t<TestType>>);
            REQUIRE(is_bool_v<std::add_volatile_t<TestType>>);
            REQUIRE(is_bool_v<std::add_cv_t<TestType>>);
            REQUIRE(is_bool_v<std::add_lvalue_reference_t<TestType>>);
            REQUIRE(is_bool_v<std::add_rvalue_reference_t<TestType>>);

            // sequence of values
            REQUIRE(is_sequence_of_bool_v<std::vector<TestType>>);
            REQUIRE(is_sequence_of_bool_v<std::vector<std::add_const_t<TestType>>>);
            REQUIRE(is_sequence_of_bool_v<std::vector<std::add_volatile_t<TestType>>>);
            REQUIRE(is_sequence_of_bool_v<std::vector<std::add_cv_t<TestType>>>);
            REQUIRE(is_sequence_of_bool_v<std::vector<std::add_lvalue_reference_t<TestType>>>);
            REQUIRE(is_sequence_of_bool_v<std::vector<std::add_rvalue_reference_t<TestType>>>);
            REQUIRE(is_sequence_of_bool_v<std::array<TestType, 1>>);
            REQUIRE(is_sequence_of_bool_v<std::array<std::add_const_t<TestType>, 1>>);
            REQUIRE(is_sequence_of_bool_v<std::array<std::add_volatile_t<TestType>, 1>>);
            REQUIRE(is_sequence_of_bool_v<std::array<std::add_cv_t<TestType>, 1>>);
            REQUIRE(is_sequence_of_bool_v<std::array<std::add_lvalue_reference_t<TestType>, 1>>);
            REQUIRE(is_sequence_of_bool_v<std::array<std::add_rvalue_reference_t<TestType>, 1>>);

            REQUIRE(is_vector_of_bool_v<std::vector<TestType>>);
            REQUIRE(is_vector_of_bool_v<std::vector<std::add_const_t<TestType>>>);
            REQUIRE(is_vector_of_bool_v<std::vector<std::add_volatile_t<TestType>>>);
            REQUIRE(is_vector_of_bool_v<std::vector<std::add_cv_t<TestType>>>);
            REQUIRE(is_vector_of_bool_v<std::vector<std::add_lvalue_reference_t<TestType>>>);
            REQUIRE(is_vector_of_bool_v<std::vector<std::add_rvalue_reference_t<TestType>>>);

            REQUIRE(is_array_of_bool_v<std::array<TestType, 1>>);
            REQUIRE(is_array_of_bool_v<std::array<std::add_const_t<TestType>, 1>>);
            REQUIRE(is_array_of_bool_v<std::array<std::add_volatile_t<TestType>, 1>>);
            REQUIRE(is_array_of_bool_v<std::array<std::add_cv_t<TestType>, 1>>);
            REQUIRE(is_array_of_bool_v<std::array<std::add_lvalue_reference_t<TestType>, 1>>);
            REQUIRE(is_array_of_bool_v<std::array<std::add_rvalue_reference_t<TestType>, 1>>);
        }
    }

    GIVEN("an incorrect match") {
        REQUIRE(!is_vector_of_bool_v<std::array<TestType, 1>>);
        REQUIRE(!is_array_of_bool_v<std::vector<TestType>>);

        using TestTypePpointer = std::add_pointer<TestType>;
        REQUIRE(!is_bool_v<TestTypePpointer>);
        REQUIRE(!is_vector_of_bool_v<std::vector<TestTypePpointer>>);
        REQUIRE(!is_array_of_bool_v<std::array<TestTypePpointer, 1>>);
        REQUIRE(!is_sequence_of_bool_v<std::vector<TestTypePpointer>>);

        if constexpr (!(std::is_same_v<TestType, bool>)) {

            // single value
            REQUIRE(!is_bool_v<TestType>);
            REQUIRE(!is_bool_v<std::add_const_t<TestType>>);
            REQUIRE(!is_bool_v<std::add_volatile_t<TestType>>);
            REQUIRE(!is_bool_v<std::add_cv_t<TestType>>);
            REQUIRE(!is_bool_v<std::add_lvalue_reference_t<TestType>>);
            REQUIRE(!is_bool_v<std::add_rvalue_reference_t<TestType>>);

            // sequence of values
            REQUIRE(!is_sequence_of_bool_v<std::vector<TestType>>);
            REQUIRE(!is_sequence_of_bool_v<std::vector<std::add_const_t<TestType>>>);
            REQUIRE(!is_sequence_of_bool_v<std::vector<std::add_volatile_t<TestType>>>);
            REQUIRE(!is_sequence_of_bool_v<std::vector<std::add_cv_t<TestType>>>);
            REQUIRE(!is_sequence_of_bool_v<std::vector<std::add_lvalue_reference_t<TestType>>>);
            REQUIRE(!is_sequence_of_bool_v<std::vector<std::add_rvalue_reference_t<TestType>>>);
            REQUIRE(!is_sequence_of_bool_v<std::array<TestType, 1>>);
            REQUIRE(!is_sequence_of_bool_v<std::array<std::add_const_t<TestType>, 1>>);
            REQUIRE(!is_sequence_of_bool_v<std::array<std::add_volatile_t<TestType>, 1>>);
            REQUIRE(!is_sequence_of_bool_v<std::array<std::add_cv_t<TestType>, 1>>);
            REQUIRE(!is_sequence_of_bool_v<std::array<std::add_lvalue_reference_t<TestType>, 1>>);
            REQUIRE(!is_sequence_of_bool_v<std::array<std::add_rvalue_reference_t<TestType>, 1>>);

            REQUIRE(!is_vector_of_bool_v<std::vector<TestType>>);
            REQUIRE(!is_vector_of_bool_v<std::vector<std::add_const_t<TestType>>>);
            REQUIRE(!is_vector_of_bool_v<std::vector<std::add_volatile_t<TestType>>>);
            REQUIRE(!is_vector_of_bool_v<std::vector<std::add_cv_t<TestType>>>);
            REQUIRE(!is_vector_of_bool_v<std::vector<std::add_lvalue_reference_t<TestType>>>);
            REQUIRE(!is_vector_of_bool_v<std::vector<std::add_rvalue_reference_t<TestType>>>);

            REQUIRE(!is_array_of_bool_v<std::array<TestType, 1>>);
            REQUIRE(!is_array_of_bool_v<std::array<std::add_const_t<TestType>, 1>>);
            REQUIRE(!is_array_of_bool_v<std::array<std::add_volatile_t<TestType>, 1>>);
            REQUIRE(!is_array_of_bool_v<std::array<std::add_cv_t<TestType>, 1>>);
            REQUIRE(!is_array_of_bool_v<std::array<std::add_lvalue_reference_t<TestType>, 1>>);
            REQUIRE(!is_array_of_bool_v<std::array<std::add_rvalue_reference_t<TestType>, 1>>);
        }
    }
}


TEMPLATE_TEST_CASE("Traits:: scalar and sequence of scalars", "[noa][traits]",
                   short, int, long, long long, float, double, long double,
                   std::complex<float>, std::complex<double>, std::complex<long double>,
                   std::string, std::string_view, bool) {

    using namespace ::Noa::Traits;

    GIVEN("a correct match") {
        if constexpr (std::is_same_v<TestType, short> ||
                      std::is_same_v<TestType, int> ||
                      std::is_same_v<TestType, long> ||
                      std::is_same_v<TestType, long long> ||
                      std::is_same_v<TestType, float> ||
                      std::is_same_v<TestType, double> ||
                      std::is_same_v<TestType, long double>) {

            // single value
            REQUIRE(is_scalar_v<TestType>);
            REQUIRE(is_scalar_v<std::add_const_t<TestType>>);
            REQUIRE(is_scalar_v<std::add_volatile_t<TestType>>);
            REQUIRE(is_scalar_v<std::add_cv_t<TestType>>);
            REQUIRE(is_scalar_v<std::add_lvalue_reference_t<TestType>>);
            REQUIRE(is_scalar_v<std::add_rvalue_reference_t<TestType>>);

            // sequence of values
            REQUIRE(is_sequence_of_scalar_v<std::vector<TestType>>);
            REQUIRE(is_sequence_of_scalar_v<std::vector<std::add_const_t<TestType>>>);
            REQUIRE(is_sequence_of_scalar_v<std::vector<std::add_volatile_t<TestType>>>);
            REQUIRE(is_sequence_of_scalar_v<std::vector<std::add_cv_t<TestType>>>);
            REQUIRE(is_sequence_of_scalar_v<std::vector<std::add_lvalue_reference_t<TestType>>>);
            REQUIRE(is_sequence_of_scalar_v<std::vector<std::add_rvalue_reference_t<TestType>>>);
            REQUIRE(is_sequence_of_scalar_v<std::array<TestType, 1>>);
            REQUIRE(is_sequence_of_scalar_v<std::array<std::add_const_t<TestType>, 1>>);
            REQUIRE(is_sequence_of_scalar_v<std::array<std::add_volatile_t<TestType>, 1>>);
            REQUIRE(is_sequence_of_scalar_v<std::array<std::add_cv_t<TestType>, 1>>);
            REQUIRE(is_sequence_of_scalar_v<std::array<std::add_lvalue_reference_t<TestType>, 1>>);
            REQUIRE(is_sequence_of_scalar_v<std::array<std::add_rvalue_reference_t<TestType>, 1>>);

            REQUIRE(is_vector_of_scalar_v<std::vector<TestType>>);
            REQUIRE(is_vector_of_scalar_v<std::vector<std::add_const_t<TestType>>>);
            REQUIRE(is_vector_of_scalar_v<std::vector<std::add_volatile_t<TestType>>>);
            REQUIRE(is_vector_of_scalar_v<std::vector<std::add_cv_t<TestType>>>);
            REQUIRE(is_vector_of_scalar_v<std::vector<std::add_lvalue_reference_t<TestType>>>);
            REQUIRE(is_vector_of_scalar_v<std::vector<std::add_rvalue_reference_t<TestType>>>);

            REQUIRE(is_array_of_scalar_v<std::array<TestType, 1>>);
            REQUIRE(is_array_of_scalar_v<std::array<std::add_const_t<TestType>, 1>>);
            REQUIRE(is_array_of_scalar_v<std::array<std::add_volatile_t<TestType>, 1>>);
            REQUIRE(is_array_of_scalar_v<std::array<std::add_cv_t<TestType>, 1>>);
            REQUIRE(is_array_of_scalar_v<std::array<std::add_lvalue_reference_t<TestType>, 1>>);
            REQUIRE(is_array_of_scalar_v<std::array<std::add_rvalue_reference_t<TestType>, 1>>);
        }
    }

    GIVEN("an incorrect match") {
        REQUIRE(!is_vector_of_scalar_v<std::array<TestType, 1>>);
        REQUIRE(!is_array_of_scalar_v<std::vector<TestType>>);

        using TestTypePpointer = std::add_pointer<TestType>;
        REQUIRE(!is_scalar_v<TestTypePpointer>);
        REQUIRE(!is_vector_of_scalar_v<std::vector<TestTypePpointer>>);
        REQUIRE(!is_array_of_scalar_v<std::array<TestTypePpointer, 1>>);
        REQUIRE(!is_sequence_of_scalar_v<std::vector<TestTypePpointer>>);

        if constexpr (!(std::is_same_v<TestType, short> ||
                        std::is_same_v<TestType, int> ||
                        std::is_same_v<TestType, long> ||
                        std::is_same_v<TestType, long long> ||
                        std::is_same_v<TestType, float> ||
                        std::is_same_v<TestType, double> ||
                        std::is_same_v<TestType, long double>)) {

            // single value
            REQUIRE(!is_scalar_v<TestType>);
            REQUIRE(!is_scalar_v<std::add_const_t<TestType>>);
            REQUIRE(!is_scalar_v<std::add_volatile_t<TestType>>);
            REQUIRE(!is_scalar_v<std::add_cv_t<TestType>>);
            REQUIRE(!is_scalar_v<std::add_lvalue_reference_t<TestType>>);
            REQUIRE(!is_scalar_v<std::add_rvalue_reference_t<TestType>>);

            // sequence of values
            REQUIRE(!is_sequence_of_scalar_v<std::vector<TestType>>);
            REQUIRE(!is_sequence_of_scalar_v<std::vector<std::add_const_t<TestType>>>);
            REQUIRE(!is_sequence_of_scalar_v<std::vector<std::add_volatile_t<TestType>>>);
            REQUIRE(!is_sequence_of_scalar_v<std::vector<std::add_cv_t<TestType>>>);
            REQUIRE(!is_sequence_of_scalar_v<std::vector<std::add_lvalue_reference_t<TestType>>>);
            REQUIRE(!is_sequence_of_scalar_v<std::vector<std::add_rvalue_reference_t<TestType>>>);
            REQUIRE(!is_sequence_of_scalar_v<std::array<TestType, 1>>);
            REQUIRE(!is_sequence_of_scalar_v<std::array<std::add_const_t<TestType>, 1>>);
            REQUIRE(!is_sequence_of_scalar_v<std::array<std::add_volatile_t<TestType>, 1>>);
            REQUIRE(!is_sequence_of_scalar_v<std::array<std::add_cv_t<TestType>, 1>>);
            REQUIRE(!is_sequence_of_scalar_v<std::array<std::add_lvalue_reference_t<TestType>, 1>>);
            REQUIRE(!is_sequence_of_scalar_v<std::array<std::add_rvalue_reference_t<TestType>, 1>>);

            REQUIRE(!is_vector_of_scalar_v<std::vector<TestType>>);
            REQUIRE(!is_vector_of_scalar_v<std::vector<std::add_const_t<TestType>>>);
            REQUIRE(!is_vector_of_scalar_v<std::vector<std::add_volatile_t<TestType>>>);
            REQUIRE(!is_vector_of_scalar_v<std::vector<std::add_cv_t<TestType>>>);
            REQUIRE(!is_vector_of_scalar_v<std::vector<std::add_lvalue_reference_t<TestType>>>);
            REQUIRE(!is_vector_of_scalar_v<std::vector<std::add_rvalue_reference_t<TestType>>>);

            REQUIRE(!is_array_of_scalar_v<std::array<TestType, 1>>);
            REQUIRE(!is_array_of_scalar_v<std::array<std::add_const_t<TestType>, 1>>);
            REQUIRE(!is_array_of_scalar_v<std::array<std::add_volatile_t<TestType>, 1>>);
            REQUIRE(!is_array_of_scalar_v<std::array<std::add_cv_t<TestType>, 1>>);
            REQUIRE(!is_array_of_scalar_v<std::array<std::add_lvalue_reference_t<TestType>, 1>>);
            REQUIRE(!is_array_of_scalar_v<std::array<std::add_rvalue_reference_t<TestType>, 1>>);
        }
    }
}


TEMPLATE_TEST_CASE("Traits:: arith and sequence of ariths", "[noa][traits]",
                   short, int, long, long long, float, double, long double,
                   std::complex<float>, std::complex<double>, std::complex<long double>,
                   std::string, std::string_view, bool) {

    using namespace ::Noa::Traits;

    GIVEN("a correct match") {
        if constexpr (std::is_same_v<TestType, short> ||
                      std::is_same_v<TestType, int> ||
                      std::is_same_v<TestType, long> ||
                      std::is_same_v<TestType, long long> ||
                      std::is_same_v<TestType, float> ||
                      std::is_same_v<TestType, double> ||
                      std::is_same_v<TestType, long double> ||
                      std::is_same_v<TestType, std::complex<float>> ||
                      std::is_same_v<TestType, std::complex<double>> ||
                      std::is_same_v<TestType, std::complex<long double>>) {

            // single value
            REQUIRE(is_arith_v<TestType>);
            REQUIRE(is_arith_v<std::add_const_t<TestType>>);
            REQUIRE(is_arith_v<std::add_volatile_t<TestType>>);
            REQUIRE(is_arith_v<std::add_cv_t<TestType>>);
            REQUIRE(is_arith_v<std::add_lvalue_reference_t<TestType>>);
            REQUIRE(is_arith_v<std::add_rvalue_reference_t<TestType>>);

            // sequence of values
            REQUIRE(is_sequence_of_arith_v<std::vector<TestType>>);
            REQUIRE(is_sequence_of_arith_v<std::vector<std::add_const_t<TestType>>>);
            REQUIRE(is_sequence_of_arith_v<std::vector<std::add_volatile_t<TestType>>>);
            REQUIRE(is_sequence_of_arith_v<std::vector<std::add_cv_t<TestType>>>);
            REQUIRE(is_sequence_of_arith_v<std::vector<std::add_lvalue_reference_t<TestType>>>);
            REQUIRE(is_sequence_of_arith_v<std::vector<std::add_rvalue_reference_t<TestType>>>);
            REQUIRE(is_sequence_of_arith_v<std::array<TestType, 1>>);
            REQUIRE(is_sequence_of_arith_v<std::array<std::add_const_t<TestType>, 1>>);
            REQUIRE(is_sequence_of_arith_v<std::array<std::add_volatile_t<TestType>, 1>>);
            REQUIRE(is_sequence_of_arith_v<std::array<std::add_cv_t<TestType>, 1>>);
            REQUIRE(is_sequence_of_arith_v<std::array<std::add_lvalue_reference_t<TestType>, 1>>);
            REQUIRE(is_sequence_of_arith_v<std::array<std::add_rvalue_reference_t<TestType>, 1>>);

            REQUIRE(is_vector_of_arith_v<std::vector<TestType>>);
            REQUIRE(is_vector_of_arith_v<std::vector<std::add_const_t<TestType>>>);
            REQUIRE(is_vector_of_arith_v<std::vector<std::add_volatile_t<TestType>>>);
            REQUIRE(is_vector_of_arith_v<std::vector<std::add_cv_t<TestType>>>);
            REQUIRE(is_vector_of_arith_v<std::vector<std::add_lvalue_reference_t<TestType>>>);
            REQUIRE(is_vector_of_arith_v<std::vector<std::add_rvalue_reference_t<TestType>>>);

            REQUIRE(is_array_of_arith_v<std::array<TestType, 1>>);
            REQUIRE(is_array_of_arith_v<std::array<std::add_const_t<TestType>, 1>>);
            REQUIRE(is_array_of_arith_v<std::array<std::add_volatile_t<TestType>, 1>>);
            REQUIRE(is_array_of_arith_v<std::array<std::add_cv_t<TestType>, 1>>);
            REQUIRE(is_array_of_arith_v<std::array<std::add_lvalue_reference_t<TestType>, 1>>);
            REQUIRE(is_array_of_arith_v<std::array<std::add_rvalue_reference_t<TestType>, 1>>);
        }
    }

    GIVEN("an incorrect match") {
        REQUIRE(!is_vector_of_arith_v<std::array<TestType, 1>>);
        REQUIRE(!is_array_of_arith_v<std::vector<TestType>>);

        using TestTypePpointer = std::add_pointer<TestType>;
        REQUIRE(!is_arith_v<TestTypePpointer>);
        REQUIRE(!is_vector_of_arith_v<std::vector<TestTypePpointer>>);
        REQUIRE(!is_array_of_arith_v<std::array<TestTypePpointer, 1>>);
        REQUIRE(!is_sequence_of_arith_v<std::vector<TestTypePpointer>>);

        if constexpr (!(std::is_same_v<TestType, short> ||
                        std::is_same_v<TestType, int> ||
                        std::is_same_v<TestType, long> ||
                        std::is_same_v<TestType, long long> ||
                        std::is_same_v<TestType, float> ||
                        std::is_same_v<TestType, double> ||
                        std::is_same_v<TestType, long double> ||
                        std::is_same_v<TestType, std::complex<float>> ||
                        std::is_same_v<TestType, std::complex<double>> ||
                        std::is_same_v<TestType, std::complex<long double>>)) {

            // single value
            REQUIRE(!is_arith_v<TestType>);
            REQUIRE(!is_arith_v<std::add_const_t<TestType>>);
            REQUIRE(!is_arith_v<std::add_volatile_t<TestType>>);
            REQUIRE(!is_arith_v<std::add_cv_t<TestType>>);
            REQUIRE(!is_arith_v<std::add_lvalue_reference_t<TestType>>);
            REQUIRE(!is_arith_v<std::add_rvalue_reference_t<TestType>>);

            // sequence of values
            REQUIRE(!is_sequence_of_arith_v<std::vector<TestType>>);
            REQUIRE(!is_sequence_of_arith_v<std::vector<std::add_const_t<TestType>>>);
            REQUIRE(!is_sequence_of_arith_v<std::vector<std::add_volatile_t<TestType>>>);
            REQUIRE(!is_sequence_of_arith_v<std::vector<std::add_cv_t<TestType>>>);
            REQUIRE(!is_sequence_of_arith_v<std::vector<std::add_lvalue_reference_t<TestType>>>);
            REQUIRE(!is_sequence_of_arith_v<std::vector<std::add_rvalue_reference_t<TestType>>>);
            REQUIRE(!is_sequence_of_arith_v<std::array<TestType, 1>>);
            REQUIRE(!is_sequence_of_arith_v<std::array<std::add_const_t<TestType>, 1>>);
            REQUIRE(!is_sequence_of_arith_v<std::array<std::add_volatile_t<TestType>, 1>>);
            REQUIRE(!is_sequence_of_arith_v<std::array<std::add_cv_t<TestType>, 1>>);
            REQUIRE(!is_sequence_of_arith_v<std::array<std::add_lvalue_reference_t<TestType>, 1>>);
            REQUIRE(!is_sequence_of_arith_v<std::array<std::add_rvalue_reference_t<TestType>, 1>>);

            REQUIRE(!is_vector_of_arith_v<std::vector<TestType>>);
            REQUIRE(!is_vector_of_arith_v<std::vector<std::add_const_t<TestType>>>);
            REQUIRE(!is_vector_of_arith_v<std::vector<std::add_volatile_t<TestType>>>);
            REQUIRE(!is_vector_of_arith_v<std::vector<std::add_cv_t<TestType>>>);
            REQUIRE(!is_vector_of_arith_v<std::vector<std::add_lvalue_reference_t<TestType>>>);
            REQUIRE(!is_vector_of_arith_v<std::vector<std::add_rvalue_reference_t<TestType>>>);

            REQUIRE(!is_array_of_arith_v<std::array<TestType, 1>>);
            REQUIRE(!is_array_of_arith_v<std::array<std::add_const_t<TestType>, 1>>);
            REQUIRE(!is_array_of_arith_v<std::array<std::add_volatile_t<TestType>, 1>>);
            REQUIRE(!is_array_of_arith_v<std::array<std::add_cv_t<TestType>, 1>>);
            REQUIRE(!is_array_of_arith_v<std::array<std::add_lvalue_reference_t<TestType>, 1>>);
            REQUIRE(!is_array_of_arith_v<std::array<std::add_rvalue_reference_t<TestType>, 1>>);
        }
    }
}


TEMPLATE_TEST_CASE("Traits: generic sequence", "[noa][traits]",
                   short, int, long, long long, float, double, long double,
                   std::complex<float>, std::complex<double>, std::complex<long double>,
                   std::string, std::string_view, bool) {

    using namespace ::Noa::Traits;

    GIVEN("a correct match") {
        REQUIRE(is_sequence_v<std::vector<TestType>>);
        REQUIRE(is_sequence_v<std::vector<std::add_const_t<TestType>>>);
        REQUIRE(is_sequence_v<std::vector<std::add_volatile_t<TestType>>>);
        REQUIRE(is_sequence_v<std::vector<std::add_cv_t<TestType>>>);
        REQUIRE(is_sequence_v<std::vector<std::add_lvalue_reference_t<TestType>>>);
        REQUIRE(is_sequence_v<std::vector<std::add_rvalue_reference_t<TestType>>>);
        REQUIRE(is_sequence_v<std::array<TestType, 1>>);
        REQUIRE(is_sequence_v<std::array<std::add_const_t<TestType>, 1>>);
        REQUIRE(is_sequence_v<std::array<std::add_volatile_t<TestType>, 1>>);
        REQUIRE(is_sequence_v<std::array<std::add_cv_t<TestType>, 1>>);
        REQUIRE(is_sequence_v<std::array<std::add_lvalue_reference_t<TestType>, 1>>);
        REQUIRE(is_sequence_v<std::array<std::add_rvalue_reference_t<TestType>, 1>>);

        REQUIRE(is_vector_v<std::vector<TestType>>);
        REQUIRE(is_vector_v<std::vector<std::add_const_t<TestType>>>);
        REQUIRE(is_vector_v<std::vector<std::add_volatile_t<TestType>>>);
        REQUIRE(is_vector_v<std::vector<std::add_cv_t<TestType>>>);
        REQUIRE(is_vector_v<std::vector<std::add_lvalue_reference_t<TestType>>>);
        REQUIRE(is_vector_v<std::vector<std::add_rvalue_reference_t<TestType>>>);

        REQUIRE(is_array_v<std::array<TestType, 1>>);
        REQUIRE(is_array_v<std::array<std::add_const_t<TestType>, 1>>);
        REQUIRE(is_array_v<std::array<std::add_volatile_t<TestType>, 1>>);
        REQUIRE(is_array_v<std::array<std::add_cv_t<TestType>, 1>>);
        REQUIRE(is_array_v<std::array<std::add_lvalue_reference_t<TestType>, 1>>);
        REQUIRE(is_array_v<std::array<std::add_rvalue_reference_t<TestType>, 1>>);

        using TestTypePointer = std::add_pointer<TestType>;
        REQUIRE(is_sequence_v<std::vector<TestTypePointer>>);
        REQUIRE(is_sequence_v<std::array<TestTypePointer, 1>>);
        REQUIRE(is_vector_v<std::vector<TestTypePointer>>);
        REQUIRE(is_array_v<std::array<TestTypePointer, 1>>);
    }

    GIVEN("an incorrect match") {
        REQUIRE(!is_sequence_v<TestType>);
        REQUIRE(!is_sequence_v<std::add_const_t<TestType>>);
        REQUIRE(!is_sequence_v<std::add_volatile_t<TestType>>);
        REQUIRE(!is_sequence_v<std::add_cv_t<TestType>>);
        REQUIRE(!is_sequence_v<std::add_lvalue_reference_t<TestType>>);
        REQUIRE(!is_sequence_v<std::add_rvalue_reference_t<TestType>>);

        REQUIRE(!is_array_v<std::vector<TestType>>);
        REQUIRE(!is_array_v<std::vector<std::add_const_t<TestType>>>);
        REQUIRE(!is_array_v<std::vector<std::add_volatile_t<TestType>>>);
        REQUIRE(!is_array_v<std::vector<std::add_cv_t<TestType>>>);
        REQUIRE(!is_array_v<std::vector<std::add_lvalue_reference_t<TestType>>>);
        REQUIRE(!is_array_v<std::vector<std::add_rvalue_reference_t<TestType>>>);

        REQUIRE(!is_vector_v<std::array<TestType, 1>>);
        REQUIRE(!is_vector_v<std::array<std::add_const_t<TestType>, 1>>);
        REQUIRE(!is_vector_v<std::array<std::add_volatile_t<TestType>, 1>>);
        REQUIRE(!is_vector_v<std::array<std::add_cv_t<TestType>, 1>>);
        REQUIRE(!is_vector_v<std::array<std::add_lvalue_reference_t<TestType>, 1>>);
        REQUIRE(!is_vector_v<std::array<std::add_rvalue_reference_t<TestType>, 1>>);
    }
}


TEMPLATE_TEST_CASE("Traits: sequence of same type", "[noa][traits]",
                   short, int, long, long long, float, double, long double,
                   std::complex<float>, std::complex<double>, std::complex<long double>,
                   std::string, std::string_view, bool) {
    using namespace Noa::Traits;
    using TestTypePointer = std::add_pointer<TestType>;

    GIVEN("a correct match") {
        REQUIRE(are_sequence_of_same_type_v<std::vector<TestType>, std::vector<TestType>>);
        REQUIRE(are_sequence_of_same_type_v<std::vector<TestType>, std::array<TestType, 1>>);
        REQUIRE(are_sequence_of_same_type_v<std::array<TestType, 1>, std::array<TestType, 1>>);
        REQUIRE(are_sequence_of_same_type_v<std::array<TestType, 1>, std::vector<TestType>>);
        REQUIRE(are_sequence_of_same_type_v<std::vector<TestTypePointer>, std::array<TestTypePointer, 1>>);
    }

    GIVEN("an incorrect match") {
        REQUIRE(!are_sequence_of_same_type_v<std::vector<std::add_const_t<TestType>>, std::vector<TestType>>);
        REQUIRE(!are_sequence_of_same_type_v<std::array<std::add_lvalue_reference_t<TestType>, 1>, std::vector<TestType>>);
        REQUIRE(!are_sequence_of_same_type_v<std::vector<int>, std::vector<unsigned int>>);
        REQUIRE(!are_sequence_of_same_type_v<std::vector<float>, std::vector<double>>);
        REQUIRE(!are_sequence_of_same_type_v<std::vector<float>, std::vector<const float>>);
        REQUIRE(!are_sequence_of_same_type_v<std::vector<double>, std::vector<int>>);
        REQUIRE(!are_sequence_of_same_type_v<std::vector<std::complex<float>>, std::vector<std::complex<double>>>);
        REQUIRE(!are_sequence_of_same_type_v<std::vector<std::string>, std::vector<std::string_view>>);
    }
}


TEMPLATE_TEST_CASE("Traits: sequence of type", "[noa][traits]",
                   short, int, long, long long, float, double, long double,
                   std::complex<float>, std::complex<double>, std::complex<long double>,
                   std::string, std::string_view, bool) {
    using namespace Noa::Traits;
    using TestTypePointer = std::add_pointer<TestType>;

    GIVEN("a correct match") {
        REQUIRE(is_sequence_of_type_v<std::vector<TestType>, TestType>);
        REQUIRE(is_sequence_of_type_v<std::array<TestType, 1>, TestType>);
        REQUIRE(is_sequence_of_type_v<std::vector<TestTypePointer>, TestTypePointer>);
    }

    GIVEN("an incorrect match") {
        REQUIRE(!is_sequence_of_type_v<std::vector<std::add_const_t<TestType>>, TestType>);
        REQUIRE(!is_sequence_of_type_v<std::array<std::add_lvalue_reference_t<TestType>, 1>, TestType>);
        REQUIRE(!is_sequence_of_type_v<std::vector<int>, unsigned int>);
        REQUIRE(!is_sequence_of_type_v<std::vector<float>, double>);
        REQUIRE(!is_sequence_of_type_v<std::vector<float>, const float>);
        REQUIRE(!is_sequence_of_type_v<std::vector<double>, int>);
        REQUIRE(!is_sequence_of_type_v<std::vector<std::complex<float>>, std::complex<double>>);
        REQUIRE(!is_sequence_of_type_v<std::vector<std::string>, std::string_view>);
    }
}


TEMPLATE_TEST_CASE("Traits: is_same", "[noa][traits]",
                   short, int, long, long long, float, double, long double,
                   std::complex<float>, std::complex<double>, std::complex<long double>,
                   std::string, std::string_view, bool) {
    using namespace Noa::Traits;

    GIVEN("a correct match") {
        REQUIRE(is_same_v<TestType, TestType>);
        REQUIRE(is_same_v<std::add_const_t<TestType>, TestType>);
        REQUIRE(is_same_v<std::add_volatile_t<TestType>, TestType>);
        REQUIRE(is_same_v<std::add_cv_t<TestType>, TestType>);
        REQUIRE(is_same_v<std::add_lvalue_reference_t<TestType>, TestType>);
        REQUIRE(is_same_v<std::add_rvalue_reference_t<TestType>, TestType>);
        REQUIRE(is_same_v<std::add_pointer<TestType>, std::add_pointer<TestType>>);
    }

    GIVEN("an incorrect match") {
        REQUIRE(!is_same_v<int, unsigned int>);
        REQUIRE(!is_same_v<float, double>);
        REQUIRE(!is_same_v<double, int>);
        REQUIRE(!is_same_v<std::complex<float>, std::complex<double>>);
        REQUIRE(!is_same_v<std::string, std::string_view>);
        REQUIRE(!is_same_v<std::add_pointer<TestType>, TestType>);
    }
}


TEMPLATE_TEST_CASE("Traits: always_false", "[noa][traits]",
                   short, int, long, long long, float, double, long double,
                   std::complex<float>, std::complex<double>, std::complex<long double>,
                   std::string, std::string_view, bool) {
    using namespace Noa::Traits;
    REQUIRE(!always_false_v<TestType>);
    REQUIRE(!always_false<std::array<TestType, 3>>::value);
    REQUIRE(!always_false<std::vector<TestType>>::value);
}
