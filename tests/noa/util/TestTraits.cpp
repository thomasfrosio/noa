#include <catch2/catch.hpp>
#include "noa/util/Traits.h"


#define REQUIRE_FOR_ALL_TYPES(type_trait)                       \
REQUIRE(type_trait<TestType>);                                  \
REQUIRE(type_trait<std::add_const_t<TestType>>);                \
REQUIRE(type_trait<std::add_volatile_t<TestType>>);             \
REQUIRE(type_trait<std::add_cv_t<TestType>>);                   \
REQUIRE(type_trait<std::add_lvalue_reference_t<TestType>>);     \
REQUIRE(type_trait<std::add_rvalue_reference_t<TestType>>)

#define REQUIRE_FOR_ALL_TYPES_VECTOR(type_trait)                            \
REQUIRE(type_trait<std::vector<TestType>>);                                 \
REQUIRE(type_trait<std::add_const_t<std::vector<TestType>>>);               \
REQUIRE(type_trait<std::add_volatile_t<std::vector<TestType>>>);            \
REQUIRE(type_trait<std::add_cv_t<std::vector<TestType>>>);                  \
REQUIRE(type_trait<std::add_lvalue_reference_t<std::vector<TestType>>>);    \
REQUIRE(type_trait<std::add_rvalue_reference_t<std::vector<TestType>>>)

#define REQUIRE_FOR_ALL_TYPES_ARRAY(type_trait)                             \
REQUIRE(type_trait<std::array<TestType, 1>>);                               \
REQUIRE(type_trait<std::add_const_t<std::array<TestType, 1>>>);             \
REQUIRE(type_trait<std::add_volatile_t<std::array<TestType, 1>>>);          \
REQUIRE(type_trait<std::add_cv_t<std::array<TestType, 1>>>);                \
REQUIRE(type_trait<std::add_lvalue_reference_t<std::array<TestType, 1>>>);  \
REQUIRE(type_trait<std::add_rvalue_reference_t<std::array<TestType, 1>>>)

#define REQUIRE_FALSE_FOR_ALL_TYPES(type_trait)                       \
REQUIRE_FALSE(type_trait<TestType>);                                  \
REQUIRE_FALSE(type_trait<std::add_const_t<TestType>>);                \
REQUIRE_FALSE(type_trait<std::add_volatile_t<TestType>>);             \
REQUIRE_FALSE(type_trait<std::add_cv_t<TestType>>);                   \
REQUIRE_FALSE(type_trait<std::add_lvalue_reference_t<TestType>>);     \
REQUIRE_FALSE(type_trait<std::add_rvalue_reference_t<TestType>>)

#define REQUIRE_FALSE_FOR_ALL_TYPES_VECTOR(type_trait)                            \
REQUIRE_FALSE(type_trait<std::vector<TestType>>);                                 \
REQUIRE_FALSE(type_trait<std::add_const_t<std::vector<TestType>>>);               \
REQUIRE_FALSE(type_trait<std::add_volatile_t<std::vector<TestType>>>);            \
REQUIRE_FALSE(type_trait<std::add_cv_t<std::vector<TestType>>>);                  \
REQUIRE_FALSE(type_trait<std::add_lvalue_reference_t<std::vector<TestType>>>);    \
REQUIRE_FALSE(type_trait<std::add_rvalue_reference_t<std::vector<TestType>>>)

#define REQUIRE_FALSE_FOR_ALL_TYPES_ARRAY(type_trait)                             \
REQUIRE_FALSE(type_trait<std::array<TestType, 1>>);                               \
REQUIRE_FALSE(type_trait<std::add_const_t<std::array<TestType, 1>>>);             \
REQUIRE_FALSE(type_trait<std::add_volatile_t<std::array<TestType, 1>>>);          \
REQUIRE_FALSE(type_trait<std::add_cv_t<std::array<TestType, 1>>>);                \
REQUIRE_FALSE(type_trait<std::add_lvalue_reference_t<std::array<TestType, 1>>>);  \
REQUIRE_FALSE(type_trait<std::add_rvalue_reference_t<std::array<TestType, 1>>>)

using namespace ::Noa::Traits;

TEMPLATE_TEST_CASE("Traits:: (sequence of) integers", "[noa][traits]",
                   short, int, long, long long, float, double, long double,
                   std::complex<float>, std::complex<double>, std::complex<long double>,
                   std::string, std::string_view, bool) {

    if constexpr (std::is_same_v<TestType, short> ||
                  std::is_same_v<TestType, int> ||
                  std::is_same_v<TestType, long> ||
                  std::is_same_v<TestType, long long>) {

        REQUIRE_FOR_ALL_TYPES(is_int_v);
        REQUIRE_FOR_ALL_TYPES_VECTOR(is_sequence_of_int_v);
        REQUIRE_FOR_ALL_TYPES_ARRAY(is_sequence_of_int_v);
        REQUIRE_FOR_ALL_TYPES_VECTOR(is_vector_of_int_v);
        REQUIRE_FOR_ALL_TYPES_ARRAY(is_array_of_int_v);

    } else {
        REQUIRE_FALSE_FOR_ALL_TYPES(is_int_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_VECTOR(is_sequence_of_int_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_ARRAY(is_sequence_of_int_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_VECTOR(is_vector_of_int_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_ARRAY(is_array_of_int_v);
    }

    using TestTypePpointer = std::add_pointer<TestType>;
    REQUIRE_FALSE(is_int_v<TestTypePpointer>);
    REQUIRE_FALSE(is_vector_of_int_v<std::vector<TestTypePpointer>>);
    REQUIRE_FALSE(is_array_of_int_v<std::array<TestTypePpointer, 1>>);
    REQUIRE_FALSE(is_sequence_of_int_v<std::vector<TestTypePpointer>>);
}


TEMPLATE_TEST_CASE("Traits:: (sequence of) floating-points", "[noa][traits]",
                   short, int, long, long long, float, double, long double,
                   std::complex<float>, std::complex<double>, std::complex<long double>,
                   std::string, std::string_view, bool) {

    if constexpr (std::is_same_v<TestType, float> ||
                  std::is_same_v<TestType, double> ||
                  std::is_same_v<TestType, long double>) {

        REQUIRE_FOR_ALL_TYPES(is_float_v);
        REQUIRE_FOR_ALL_TYPES_VECTOR(is_sequence_of_float_v);
        REQUIRE_FOR_ALL_TYPES_ARRAY(is_sequence_of_float_v);
        REQUIRE_FOR_ALL_TYPES_VECTOR(is_vector_of_float_v);
        REQUIRE_FOR_ALL_TYPES_ARRAY(is_array_of_float_v);

    } else {
        REQUIRE_FALSE_FOR_ALL_TYPES(is_float_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_VECTOR(is_sequence_of_float_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_ARRAY(is_sequence_of_float_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_VECTOR(is_vector_of_float_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_ARRAY(is_array_of_float_v);
    }

    using TestTypePpointer = std::add_pointer<TestType>;
    REQUIRE_FALSE(is_float_v<TestTypePpointer>);
    REQUIRE_FALSE(is_vector_of_float_v<std::vector<TestTypePpointer>>);
    REQUIRE_FALSE(is_array_of_float_v<std::array<TestTypePpointer, 1>>);
    REQUIRE_FALSE(is_sequence_of_float_v<std::vector<TestTypePpointer>>);
}


TEMPLATE_TEST_CASE("Traits:: (sequence of) complex", "[noa][traits]",
                   short, int, long, long long, float, double, long double,
                   std::complex<float>, std::complex<double>, std::complex<long double>,
                   std::string, std::string_view, bool) {

    if constexpr (std::is_same_v<TestType, std::complex<float>> ||
                  std::is_same_v<TestType, std::complex<double>> ||
                  std::is_same_v<TestType, std::complex<long double>>) {

        REQUIRE_FOR_ALL_TYPES(is_complex_v);
        REQUIRE_FOR_ALL_TYPES_VECTOR(is_sequence_of_complex_v);
        REQUIRE_FOR_ALL_TYPES_ARRAY(is_sequence_of_complex_v);
        REQUIRE_FOR_ALL_TYPES_VECTOR(is_vector_of_complex_v);
        REQUIRE_FOR_ALL_TYPES_ARRAY(is_array_of_complex_v);

    } else {
        REQUIRE_FALSE_FOR_ALL_TYPES(is_complex_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_VECTOR(is_sequence_of_complex_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_ARRAY(is_sequence_of_complex_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_VECTOR(is_vector_of_complex_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_ARRAY(is_array_of_complex_v);
    }

    using TestTypePpointer = std::add_pointer<TestType>;
    REQUIRE_FALSE(is_complex_v<TestTypePpointer>);
    REQUIRE_FALSE(is_vector_of_complex_v<std::vector<TestTypePpointer>>);
    REQUIRE_FALSE(is_array_of_complex_v<std::array<TestTypePpointer, 1>>);
    REQUIRE_FALSE(is_sequence_of_complex_v<std::vector<TestTypePpointer>>);
}


TEMPLATE_TEST_CASE("Traits:: (sequence of) string(_view)", "[noa][traits]",
                   short, int, long, long long, float, double, long double,
                   std::complex<float>, std::complex<double>, std::complex<long double>,
                   std::string, std::string_view, bool) {

    if constexpr (std::is_same_v<TestType, std::string> ||
                  std::is_same_v<TestType, std::string_view>) {

        REQUIRE_FOR_ALL_TYPES(is_string_v);
        REQUIRE_FOR_ALL_TYPES_VECTOR(is_sequence_of_string_v);
        REQUIRE_FOR_ALL_TYPES_ARRAY(is_sequence_of_string_v);
        REQUIRE_FOR_ALL_TYPES_VECTOR(is_vector_of_string_v);
        REQUIRE_FOR_ALL_TYPES_ARRAY(is_array_of_string_v);

    } else {
        REQUIRE_FALSE_FOR_ALL_TYPES(is_string_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_VECTOR(is_sequence_of_string_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_ARRAY(is_sequence_of_string_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_VECTOR(is_vector_of_string_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_ARRAY(is_array_of_string_v);
    }

    using TestTypePpointer = std::add_pointer<TestType>;
    REQUIRE_FALSE(is_string_v<TestTypePpointer>);
    REQUIRE_FALSE(is_vector_of_string_v<std::vector<TestTypePpointer>>);
    REQUIRE_FALSE(is_array_of_string_v<std::array<TestTypePpointer, 1>>);
    REQUIRE_FALSE(is_sequence_of_string_v<std::vector<TestTypePpointer>>);
}


TEMPLATE_TEST_CASE("Traits:: (sequence of) bool", "[noa][traits]",
                   short, int, long, long long, float, double, long double,
                   std::complex<float>, std::complex<double>, std::complex<long double>,
                   std::string, std::string_view, bool) {

    if constexpr (std::is_same_v<TestType, bool>) {
        REQUIRE_FOR_ALL_TYPES(is_bool_v);
        REQUIRE_FOR_ALL_TYPES_VECTOR(is_sequence_of_bool_v);
        REQUIRE_FOR_ALL_TYPES_ARRAY(is_sequence_of_bool_v);
        REQUIRE_FOR_ALL_TYPES_VECTOR(is_vector_of_bool_v);
        REQUIRE_FOR_ALL_TYPES_ARRAY(is_array_of_bool_v);

    } else {
        REQUIRE_FALSE_FOR_ALL_TYPES(is_bool_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_VECTOR(is_sequence_of_bool_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_ARRAY(is_sequence_of_bool_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_VECTOR(is_vector_of_bool_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_ARRAY(is_array_of_bool_v);
    }

    using TestTypePpointer = std::add_pointer<TestType>;
    REQUIRE_FALSE(is_bool_v<TestTypePpointer>);
    REQUIRE_FALSE(is_vector_of_bool_v<std::vector<TestTypePpointer>>);
    REQUIRE_FALSE(is_array_of_bool_v<std::array<TestTypePpointer, 1>>);
    REQUIRE_FALSE(is_sequence_of_bool_v<std::vector<TestTypePpointer>>);
}


TEMPLATE_TEST_CASE("Traits:: (sequence of) scalar", "[noa][traits]",
                   short, int, long, long long, float, double, long double,
                   std::complex<float>, std::complex<double>, std::complex<long double>,
                   std::string, std::string_view, bool) {

    if constexpr (std::is_same_v<TestType, short> ||
                  std::is_same_v<TestType, int> ||
                  std::is_same_v<TestType, long> ||
                  std::is_same_v<TestType, long long> ||
                  std::is_same_v<TestType, float> ||
                  std::is_same_v<TestType, double> ||
                  std::is_same_v<TestType, long double>) {

        REQUIRE_FOR_ALL_TYPES(is_scalar_v);
        REQUIRE_FOR_ALL_TYPES_VECTOR(is_sequence_of_scalar_v);
        REQUIRE_FOR_ALL_TYPES_ARRAY(is_sequence_of_scalar_v);
        REQUIRE_FOR_ALL_TYPES_VECTOR(is_vector_of_scalar_v);
        REQUIRE_FOR_ALL_TYPES_ARRAY(is_array_of_scalar_v);

    } else {
        REQUIRE_FALSE_FOR_ALL_TYPES(is_scalar_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_VECTOR(is_sequence_of_scalar_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_ARRAY(is_sequence_of_scalar_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_VECTOR(is_vector_of_scalar_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_ARRAY(is_array_of_scalar_v);
    }

    using TestTypePpointer = std::add_pointer<TestType>;
    REQUIRE_FALSE(is_scalar_v<TestTypePpointer>);
    REQUIRE_FALSE(is_vector_of_scalar_v<std::vector<TestTypePpointer>>);
    REQUIRE_FALSE(is_array_of_scalar_v<std::array<TestTypePpointer, 1>>);
    REQUIRE_FALSE(is_sequence_of_scalar_v<std::vector<TestTypePpointer>>);
}


TEMPLATE_TEST_CASE("Traits:: (sequence of) arith", "[noa][traits]",
                   short, int, long, long long, float, double, long double,
                   std::complex<float>, std::complex<double>, std::complex<long double>,
                   std::string, std::string_view, bool) {

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

        REQUIRE_FOR_ALL_TYPES(is_arith_v);
        REQUIRE_FOR_ALL_TYPES_VECTOR(is_sequence_of_arith_v);
        REQUIRE_FOR_ALL_TYPES_ARRAY(is_sequence_of_arith_v);
        REQUIRE_FOR_ALL_TYPES_VECTOR(is_vector_of_arith_v);
        REQUIRE_FOR_ALL_TYPES_ARRAY(is_array_of_arith_v);

    } else {
        REQUIRE_FALSE_FOR_ALL_TYPES(is_arith_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_VECTOR(is_sequence_of_arith_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_ARRAY(is_sequence_of_arith_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_VECTOR(is_vector_of_arith_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_ARRAY(is_array_of_arith_v);
    }

    using TestTypePpointer = std::add_pointer<TestType>;
    REQUIRE_FALSE(is_arith_v<TestTypePpointer>);
    REQUIRE_FALSE(is_vector_of_arith_v<std::vector<TestTypePpointer>>);
    REQUIRE_FALSE(is_array_of_arith_v<std::array<TestTypePpointer, 1>>);
    REQUIRE_FALSE(is_sequence_of_arith_v<std::vector<TestTypePpointer>>);
}


TEMPLATE_TEST_CASE("Traits: sequence", "[noa][traits]",
                   short, int, long, long long, float, double, long double,
                   std::complex<float>, std::complex<double>, std::complex<long double>,
                   std::string, std::string_view, bool) {

    REQUIRE_FOR_ALL_TYPES_VECTOR(is_sequence_v);
    REQUIRE_FOR_ALL_TYPES_ARRAY(is_sequence_v);
    REQUIRE_FOR_ALL_TYPES_VECTOR(is_vector_v);
    REQUIRE_FOR_ALL_TYPES_ARRAY(is_array_v);

    using TestTypePointer = std::add_pointer<TestType>;
    REQUIRE(is_sequence_v<std::vector<TestTypePointer>>);
    REQUIRE(is_sequence_v<std::array<TestTypePointer, 1>>);
    REQUIRE(is_vector_v<std::vector<TestTypePointer>>);
    REQUIRE(is_array_v<std::array<TestTypePointer, 1>>);

    REQUIRE_FALSE_FOR_ALL_TYPES(is_sequence_v);
    REQUIRE_FALSE_FOR_ALL_TYPES_VECTOR(is_array_v);
    REQUIRE_FALSE_FOR_ALL_TYPES_ARRAY(is_vector_v);
}


TEMPLATE_TEST_CASE("Traits: sequence of same type", "[noa][traits]",
                   short, int, long, long long, float, double, long double,
                   std::complex<float>, std::complex<double>, std::complex<long double>,
                   std::string, std::string_view, bool) {

    using TestTypePointer = std::add_pointer<TestType>;

    REQUIRE(are_sequence_of_same_type_v<std::vector<TestType>, std::vector<TestType>>);
    REQUIRE(are_sequence_of_same_type_v<std::vector<TestType>, std::array<TestType, 1>>);
    REQUIRE(are_sequence_of_same_type_v<std::array<TestType, 1>, std::array<TestType, 1>>);
    REQUIRE(are_sequence_of_same_type_v<std::array<TestType, 1>, std::vector<TestType>>);
    REQUIRE(are_sequence_of_same_type_v<std::vector<TestTypePointer>, std::array<TestTypePointer, 1>>);

    REQUIRE_FALSE(are_sequence_of_same_type_v<std::vector<std::add_const_t<TestType>>, std::vector<TestType>>);
    REQUIRE_FALSE(are_sequence_of_same_type_v<std::array<std::add_lvalue_reference_t<TestType>, 1>, std::vector<TestType>>);
    REQUIRE_FALSE(are_sequence_of_same_type_v<std::vector<int>, std::vector<unsigned int>>);
    REQUIRE_FALSE(are_sequence_of_same_type_v<std::vector<float>, std::vector<double>>);
    REQUIRE_FALSE(are_sequence_of_same_type_v<std::vector<float>, std::vector<const float>>);
    REQUIRE_FALSE(are_sequence_of_same_type_v<std::vector<double>, std::vector<int>>);
    REQUIRE_FALSE(are_sequence_of_same_type_v<std::vector<std::complex<float>>, std::vector<std::complex<double>>>);
    REQUIRE_FALSE(are_sequence_of_same_type_v<std::vector<std::string>, std::vector<std::string_view>>);
}


TEMPLATE_TEST_CASE("Traits: sequence of type", "[noa][traits]",
                   short, int, long, long long, float, double, long double,
                   std::complex<float>, std::complex<double>, std::complex<long double>,
                   std::string, std::string_view, bool) {
    using TestTypePointer = std::add_pointer<TestType>;

    REQUIRE(is_sequence_of_type_v<std::vector<TestType>, TestType>);
    REQUIRE(is_sequence_of_type_v<std::array<TestType, 1>, TestType>);
    REQUIRE(is_sequence_of_type_v<std::vector<TestTypePointer>, TestTypePointer>);

    REQUIRE_FALSE(is_sequence_of_type_v<std::vector<std::add_const_t<TestType>>, TestType>);
    REQUIRE_FALSE(is_sequence_of_type_v<std::array<std::add_lvalue_reference_t<TestType>, 1>, TestType>);
    REQUIRE_FALSE(is_sequence_of_type_v<std::vector<int>, unsigned int>);
    REQUIRE_FALSE(is_sequence_of_type_v<std::vector<float>, double>);
    REQUIRE_FALSE(is_sequence_of_type_v<std::vector<float>, const float>);
    REQUIRE_FALSE(is_sequence_of_type_v<std::vector<double>, int>);
    REQUIRE_FALSE(is_sequence_of_type_v<std::vector<std::complex<float>>, std::complex<double>>);
    REQUIRE_FALSE(is_sequence_of_type_v<std::vector<std::string>, std::string_view>);
}


TEMPLATE_TEST_CASE("Traits: is_same", "[noa][traits]",
                   short, int, long, long long, float, double, long double,
                   std::complex<float>, std::complex<double>, std::complex<long double>,
                   std::string, std::string_view, bool) {

    REQUIRE(is_same_v<TestType, TestType>);
    REQUIRE(is_same_v<std::add_const_t<TestType>, TestType>);
    REQUIRE(is_same_v<std::add_volatile_t<TestType>, TestType>);
    REQUIRE(is_same_v<std::add_cv_t<TestType>, TestType>);
    REQUIRE(is_same_v<std::add_lvalue_reference_t<TestType>, TestType>);
    REQUIRE(is_same_v<std::add_rvalue_reference_t<TestType>, TestType>);
    REQUIRE(is_same_v<std::add_pointer<TestType>, std::add_pointer<TestType>>);

    REQUIRE_FALSE(is_same_v<int, unsigned int>);
    REQUIRE_FALSE(is_same_v<float, double>);
    REQUIRE_FALSE(is_same_v<double, int>);
    REQUIRE_FALSE(is_same_v<std::complex<float>, std::complex<double>>);
    REQUIRE_FALSE(is_same_v<std::string, std::string_view>);
    REQUIRE_FALSE(is_same_v<std::add_pointer<TestType>, TestType>);
}


TEMPLATE_TEST_CASE("Traits: always_false", "[noa][traits]",
                   short, int, long, long long, float, double, long double,
                   std::complex<float>, std::complex<double>, std::complex<long double>,
                   std::string, std::string_view, bool) {
    REQUIRE_FALSE(always_false_v<TestType>);
    REQUIRE_FALSE(always_false<std::array<TestType, 3>>::value);
    REQUIRE_FALSE(always_false<std::vector<TestType>>::value);
}
