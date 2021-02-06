#include <catch2/catch.hpp>
#include "noa/util/Traits.h"
#include "noa/Types.h"


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

using namespace ::Noa;

TEMPLATE_TEST_CASE("Traits:: (sequence of) integers", "[noa][traits]",
                   uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t,
                   float, double, cfloat_t, cdouble_t,
                   std::complex<float>, std::complex<double>, std::complex<long double>,
                   std::string, std::string_view, bool) {

    if constexpr (std::is_same_v<TestType, int8_t> ||
                  std::is_same_v<TestType, int16_t> ||
                  std::is_same_v<TestType, int32_t> ||
                  std::is_same_v<TestType, int64_t> ||
                  std::is_same_v<TestType, uint8_t> ||
                  std::is_same_v<TestType, uint16_t> ||
                  std::is_same_v<TestType, uint32_t> ||
                  std::is_same_v<TestType, uint64_t>) {
        REQUIRE_FOR_ALL_TYPES(Traits::is_int_v);
        REQUIRE_FOR_ALL_TYPES_VECTOR(Traits::is_std_sequence_int_v);
        REQUIRE_FOR_ALL_TYPES_ARRAY(Traits::is_std_sequence_int_v);
        REQUIRE_FOR_ALL_TYPES_VECTOR(Traits::is_std_vector_int_v);
        REQUIRE_FOR_ALL_TYPES_ARRAY(Traits::is_std_array_int_v);

    } else {
        REQUIRE_FALSE_FOR_ALL_TYPES(Traits::is_int_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_VECTOR(Traits::is_std_sequence_int_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_ARRAY(Traits::is_std_sequence_int_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_VECTOR(Traits::is_std_vector_int_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_ARRAY(Traits::is_std_array_int_v);
    }

    if constexpr (std::is_same_v<TestType, uint16_t> ||
                  std::is_same_v<TestType, uint32_t> ||
                  std::is_same_v<TestType, uint64_t>) {
        REQUIRE_FOR_ALL_TYPES(Traits::is_int_v);
        REQUIRE_FOR_ALL_TYPES_VECTOR(Traits::is_std_sequence_uint_v);
        REQUIRE_FOR_ALL_TYPES_ARRAY(Traits::is_std_sequence_uint_v);
        REQUIRE_FOR_ALL_TYPES_VECTOR(Traits::is_std_vector_uint_v);
        REQUIRE_FOR_ALL_TYPES_ARRAY(Traits::is_std_array_uint_v);
    }

    using TestTypePpointer = std::add_pointer<TestType>;
    REQUIRE_FALSE(Traits::is_int_v<TestTypePpointer>);
    REQUIRE_FALSE(Traits::is_std_vector_int_v<std::vector<TestTypePpointer>>);
    REQUIRE_FALSE(Traits::is_std_array_int_v<std::array<TestTypePpointer, 1>>);
    REQUIRE_FALSE(Traits::is_std_sequence_int_v<std::vector<TestTypePpointer>>);
}


TEMPLATE_TEST_CASE("Traits:: (sequence of) floating-points", "[noa][traits]",
                   uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t,
                   float, double, cfloat_t, cdouble_t,
                   std::complex<float>, std::complex<double>,
                   std::string, std::string_view, bool) {

    if constexpr (std::is_same_v<TestType, float> || std::is_same_v<TestType, double>) {
        REQUIRE_FOR_ALL_TYPES(Traits::is_float_v);
        REQUIRE_FOR_ALL_TYPES_VECTOR(Traits::is_std_sequence_float_v);
        REQUIRE_FOR_ALL_TYPES_ARRAY(Traits::is_std_sequence_float_v);
        REQUIRE_FOR_ALL_TYPES_VECTOR(Traits::is_std_vector_float_v);
        REQUIRE_FOR_ALL_TYPES_ARRAY(Traits::is_std_array_float_v);

    } else {
        REQUIRE_FALSE_FOR_ALL_TYPES(Traits::is_float_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_VECTOR(Traits::is_std_sequence_float_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_ARRAY(Traits::is_std_sequence_float_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_VECTOR(Traits::is_std_vector_float_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_ARRAY(Traits::is_std_array_float_v);
    }

    using TestTypePpointer = std::add_pointer<TestType>;
    REQUIRE_FALSE(Traits::is_float_v<TestTypePpointer>);
    REQUIRE_FALSE(Traits::is_std_vector_float_v<std::vector<TestTypePpointer>>);
    REQUIRE_FALSE(Traits::is_std_array_float_v<std::array<TestTypePpointer, 1>>);
    REQUIRE_FALSE(Traits::is_std_sequence_float_v<std::vector<TestTypePpointer>>);
}


TEMPLATE_TEST_CASE("Traits:: (sequence of) complex", "[noa][traits]",
                   uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t,
                   float, double, cfloat_t, cdouble_t,
                   std::complex<float>, std::complex<double>,
                   std::string, std::string_view, bool) {

    if constexpr (std::is_same_v<TestType, std::complex<float>> ||
                  std::is_same_v<TestType, std::complex<double>>) {

        REQUIRE_FOR_ALL_TYPES(Traits::is_std_complex_v);
        REQUIRE_FOR_ALL_TYPES_VECTOR(Traits::is_std_sequence_std_complex_v);
        REQUIRE_FOR_ALL_TYPES_ARRAY(Traits::is_std_sequence_std_complex_v);
        REQUIRE_FOR_ALL_TYPES_VECTOR(Traits::is_std_vector_std_complex_v);
        REQUIRE_FOR_ALL_TYPES_ARRAY(Traits::is_std_array_std_complex_v);

    } else if constexpr (std::is_same_v<TestType, cfloat_t> ||
                         std::is_same_v<TestType, cdouble_t>) {

        REQUIRE_FOR_ALL_TYPES(Traits::is_complex_v);
        REQUIRE_FOR_ALL_TYPES_VECTOR(Traits::is_std_sequence_complex_v);
        REQUIRE_FOR_ALL_TYPES_ARRAY(Traits::is_std_sequence_complex_v);
        REQUIRE_FOR_ALL_TYPES_VECTOR(Traits::is_std_vector_complex_v);
        REQUIRE_FOR_ALL_TYPES_ARRAY(Traits::is_std_array_complex_v);

    } else {
        REQUIRE_FALSE_FOR_ALL_TYPES(Traits::is_std_complex_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_VECTOR(Traits::is_std_sequence_std_complex_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_ARRAY(Traits::is_std_sequence_std_complex_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_VECTOR(Traits::is_std_vector_std_complex_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_ARRAY(Traits::is_std_array_std_complex_v);
        REQUIRE_FALSE_FOR_ALL_TYPES(Traits::is_complex_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_VECTOR(Traits::is_std_sequence_complex_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_ARRAY(Traits::is_std_sequence_complex_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_VECTOR(Traits::is_std_vector_complex_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_ARRAY(Traits::is_std_array_complex_v);
    }

    using TestTypePpointer = std::add_pointer<TestType>;
    REQUIRE_FALSE(Traits::is_complex_v<TestTypePpointer>);
    REQUIRE_FALSE(Traits::is_std_vector_complex_v<std::vector<TestTypePpointer>>);
    REQUIRE_FALSE(Traits::is_std_array_complex_v<std::array<TestTypePpointer, 1>>);
    REQUIRE_FALSE(Traits::is_std_sequence_complex_v<std::vector<TestTypePpointer>>);
    REQUIRE_FALSE(Traits::is_std_complex_v<TestTypePpointer>);
    REQUIRE_FALSE(Traits::is_std_vector_std_complex_v<std::vector<TestTypePpointer>>);
    REQUIRE_FALSE(Traits::is_std_array_std_complex_v<std::array<TestTypePpointer, 1>>);
    REQUIRE_FALSE(Traits::is_std_sequence_std_complex_v<std::vector<TestTypePpointer>>);
}


TEMPLATE_TEST_CASE("Traits:: (sequence of) string(_view)", "[noa][traits]",
                   uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t,
                   float, double, cfloat_t, cdouble_t,
                   std::complex<float>, std::complex<double>,
                   std::string, std::string_view, bool) {

    if constexpr (std::is_same_v<TestType, std::string> ||
                  std::is_same_v<TestType, std::string_view>) {

        REQUIRE_FOR_ALL_TYPES(Traits::is_string_v);
        REQUIRE_FOR_ALL_TYPES_VECTOR(Traits::is_std_sequence_string_v);
        REQUIRE_FOR_ALL_TYPES_ARRAY(Traits::is_std_sequence_string_v);
        REQUIRE_FOR_ALL_TYPES_VECTOR(Traits::is_std_vector_string_v);
        REQUIRE_FOR_ALL_TYPES_ARRAY(Traits::is_std_array_string_v);

    } else {
        REQUIRE_FALSE_FOR_ALL_TYPES(Traits::is_string_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_VECTOR(Traits::is_std_sequence_string_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_ARRAY(Traits::is_std_sequence_string_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_VECTOR(Traits::is_std_vector_string_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_ARRAY(Traits::is_std_array_string_v);
    }

    using TestTypePpointer = std::add_pointer<TestType>;
    REQUIRE_FALSE(Traits::is_string_v<TestTypePpointer>);
    REQUIRE_FALSE(Traits::is_std_vector_string_v<std::vector<TestTypePpointer>>);
    REQUIRE_FALSE(Traits::is_std_array_string_v<std::array<TestTypePpointer, 1>>);
    REQUIRE_FALSE(Traits::is_std_sequence_string_v<std::vector<TestTypePpointer>>);
}


TEMPLATE_TEST_CASE("Traits:: (sequence of) bool", "[noa][traits]",
                   uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t,
                   float, double, cfloat_t, cdouble_t,
                   std::complex<float>, std::complex<double>,
                   std::string, std::string_view, bool) {

    if constexpr (std::is_same_v<TestType, bool>) {
        REQUIRE_FOR_ALL_TYPES(Traits::is_bool_v);
        REQUIRE_FOR_ALL_TYPES_VECTOR(Traits::is_std_sequence_bool_v);
        REQUIRE_FOR_ALL_TYPES_ARRAY(Traits::is_std_sequence_bool_v);
        REQUIRE_FOR_ALL_TYPES_VECTOR(Traits::is_std_vector_bool_v);
        REQUIRE_FOR_ALL_TYPES_ARRAY(Traits::is_std_array_bool_v);

    } else {
        REQUIRE_FALSE_FOR_ALL_TYPES(Traits::is_bool_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_VECTOR(Traits::is_std_sequence_bool_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_ARRAY(Traits::is_std_sequence_bool_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_VECTOR(Traits::is_std_vector_bool_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_ARRAY(Traits::is_std_array_bool_v);
    }

    using TestTypePpointer = std::add_pointer<TestType>;
    REQUIRE_FALSE(Traits::is_bool_v<TestTypePpointer>);
    REQUIRE_FALSE(Traits::is_std_vector_bool_v<std::vector<TestTypePpointer>>);
    REQUIRE_FALSE(Traits::is_std_array_bool_v<std::array<TestTypePpointer, 1>>);
    REQUIRE_FALSE(Traits::is_std_sequence_bool_v<std::vector<TestTypePpointer>>);
}


TEMPLATE_TEST_CASE("Traits:: (sequence of) scalar", "[noa][traits]",
                   uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t,
                   float, double, cfloat_t, cdouble_t,
                   std::complex<float>, std::complex<double>,
                   std::string, std::string_view, bool) {

    if constexpr (std::is_same_v<TestType, int8_t> ||
                  std::is_same_v<TestType, int16_t> ||
                  std::is_same_v<TestType, int32_t> ||
                  std::is_same_v<TestType, int64_t> ||
                  std::is_same_v<TestType, uint8_t> ||
                  std::is_same_v<TestType, uint16_t> ||
                  std::is_same_v<TestType, uint32_t> ||
                  std::is_same_v<TestType, uint64_t> ||
                  std::is_same_v<TestType, float> ||
                  std::is_same_v<TestType, double>) {

        REQUIRE_FOR_ALL_TYPES(Traits::is_scalar_v);
        REQUIRE_FOR_ALL_TYPES_VECTOR(Traits::is_std_sequence_scalar_v);
        REQUIRE_FOR_ALL_TYPES_ARRAY(Traits::is_std_sequence_scalar_v);
        REQUIRE_FOR_ALL_TYPES_VECTOR(Traits::is_std_vector_scalar_v);
        REQUIRE_FOR_ALL_TYPES_ARRAY(Traits::is_std_array_scalar_v);

    } else {
        REQUIRE_FALSE_FOR_ALL_TYPES(Traits::is_scalar_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_VECTOR(Traits::is_std_sequence_scalar_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_ARRAY(Traits::is_std_sequence_scalar_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_VECTOR(Traits::is_std_vector_scalar_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_ARRAY(Traits::is_std_array_scalar_v);
    }

    using TestTypePpointer = std::add_pointer<TestType>;
    REQUIRE_FALSE(Traits::is_scalar_v<TestTypePpointer>);
    REQUIRE_FALSE(Traits::is_std_vector_scalar_v<std::vector<TestTypePpointer>>);
    REQUIRE_FALSE(Traits::is_std_array_scalar_v<std::array<TestTypePpointer, 1>>);
    REQUIRE_FALSE(Traits::is_std_sequence_scalar_v<std::vector<TestTypePpointer>>);
}


TEMPLATE_TEST_CASE("Traits:: (sequence of) data", "[noa][traits]",
                   uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t,
                   float, double, cfloat_t, cdouble_t,
                   std::complex<float>, std::complex<double>,
                   std::string, std::string_view, bool) {

    if constexpr (std::is_same_v<TestType, int8_t> ||
                  std::is_same_v<TestType, int16_t> ||
                  std::is_same_v<TestType, int32_t> ||
                  std::is_same_v<TestType, int64_t> ||
                  std::is_same_v<TestType, uint8_t> ||
                  std::is_same_v<TestType, uint16_t> ||
                  std::is_same_v<TestType, uint32_t> ||
                  std::is_same_v<TestType, uint64_t> ||
                  std::is_same_v<TestType, float> ||
                  std::is_same_v<TestType, double> ||
                  std::is_same_v<TestType, cfloat_t> ||
                  std::is_same_v<TestType, cdouble_t>) {

        REQUIRE_FOR_ALL_TYPES(Traits::is_data_v);
        REQUIRE_FOR_ALL_TYPES_VECTOR(Traits::is_std_sequence_data_v);
        REQUIRE_FOR_ALL_TYPES_ARRAY(Traits::is_std_sequence_data_v);
        REQUIRE_FOR_ALL_TYPES_VECTOR(Traits::is_std_vector_data_v);
        REQUIRE_FOR_ALL_TYPES_ARRAY(Traits::is_std_array_data_v);

    } else {
        REQUIRE_FALSE_FOR_ALL_TYPES(Traits::is_data_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_VECTOR(Traits::is_std_sequence_data_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_ARRAY(Traits::is_std_sequence_data_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_VECTOR(Traits::is_std_vector_data_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_ARRAY(Traits::is_std_array_data_v);
    }

    using TestTypePpointer = std::add_pointer<TestType>;
    REQUIRE_FALSE(Traits::is_data_v<TestTypePpointer>);
    REQUIRE_FALSE(Traits::is_std_vector_data_v<std::vector<TestTypePpointer>>);
    REQUIRE_FALSE(Traits::is_std_array_data_v<std::array<TestTypePpointer, 1>>);
    REQUIRE_FALSE(Traits::is_std_sequence_data_v<std::vector<TestTypePpointer>>);
}


TEMPLATE_TEST_CASE("Traits: sequence", "[noa][traits]",
                   uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t,
                   float, double, cfloat_t, cdouble_t,
                   std::complex<float>, std::complex<double>,
                   std::string, std::string_view, bool) {

    REQUIRE_FOR_ALL_TYPES_VECTOR(Traits::is_std_sequence_v);
    REQUIRE_FOR_ALL_TYPES_ARRAY(Traits::is_std_sequence_v);
    REQUIRE_FOR_ALL_TYPES_VECTOR(Traits::is_std_vector_v);
    REQUIRE_FOR_ALL_TYPES_ARRAY(Traits::is_std_array_v);

    using TestTypePointer = std::add_pointer<TestType>;
    REQUIRE(Traits::is_std_sequence_v<std::vector<TestTypePointer>>);
    REQUIRE(Traits::is_std_sequence_v<std::array<TestTypePointer, 1>>);
    REQUIRE(Traits::is_std_vector_v<std::vector<TestTypePointer>>);
    REQUIRE(Traits::is_std_array_v<std::array<TestTypePointer, 1>>);

    REQUIRE_FALSE_FOR_ALL_TYPES(Traits::is_std_sequence_v);
    REQUIRE_FALSE_FOR_ALL_TYPES_VECTOR(Traits::is_std_array_v);
    REQUIRE_FALSE_FOR_ALL_TYPES_ARRAY(Traits::is_std_vector_v);
}


//@CLION-formatter:off
TEMPLATE_TEST_CASE("Traits: sequence of same type", "[noa][traits]",
                   uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t,
                   float, double, cfloat_t, cdouble_t,
                   std::complex<float>, std::complex<double>,
                   std::string, std::string_view, bool) {

    using TestTypePointer = std::add_pointer<TestType>;

    REQUIRE(Traits::are_std_sequence_of_same_type_v<std::vector<TestType>, std::vector<TestType>>);
    REQUIRE(Traits::are_std_sequence_of_same_type_v<std::vector<TestType>, std::array<TestType, 1>>);
    REQUIRE(Traits::are_std_sequence_of_same_type_v<std::array<TestType, 1>, std::array<TestType, 1>>);
    REQUIRE(Traits::are_std_sequence_of_same_type_v<std::array<TestType, 1>, std::vector<TestType>>);
    REQUIRE(Traits::are_std_sequence_of_same_type_v<std::vector<TestTypePointer>, std::array<TestTypePointer, 1>>);

    REQUIRE_FALSE(Traits::are_std_sequence_of_same_type_v<std::vector<std::add_const_t<TestType>>, std::vector<TestType>>);
    REQUIRE_FALSE(Traits::are_std_sequence_of_same_type_v<std::array<std::add_lvalue_reference_t<TestType>, 1>, std::vector<TestType>>);
    REQUIRE_FALSE(Traits::are_std_sequence_of_same_type_v<std::vector<int>, std::vector<unsigned int>>);
    REQUIRE_FALSE(Traits::are_std_sequence_of_same_type_v<std::vector<float>, std::vector<double>>);
    REQUIRE_FALSE(Traits::are_std_sequence_of_same_type_v<std::vector<float>, std::vector<const float>>);
    REQUIRE_FALSE(Traits::are_std_sequence_of_same_type_v<std::vector<double>, std::vector<int>>);
    REQUIRE_FALSE(Traits::are_std_sequence_of_same_type_v<std::vector<std::complex<float>>, std::vector<std::complex<double>>>);
    REQUIRE_FALSE(Traits::are_std_sequence_of_same_type_v<std::vector<std::string>, std::vector<std::string_view>>);
}


TEMPLATE_TEST_CASE("Traits: sequence of type", "[noa][traits]",
                   uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t,
                   float, double, cfloat_t, cdouble_t,
                   std::complex<float>, std::complex<double>,
                   std::string, std::string_view, bool) {
    using TestTypePointer = std::add_pointer<TestType>;

    REQUIRE(Traits::is_std_sequence_of_type_v<std::vector<TestType>, TestType>);
    REQUIRE(Traits::is_std_sequence_of_type_v<std::array<TestType, 1>, TestType>);
    REQUIRE(Traits::is_std_sequence_of_type_v<std::vector<TestTypePointer>, TestTypePointer>);

    REQUIRE_FALSE(Traits::is_std_sequence_of_type_v<std::vector<std::add_const_t<TestType>>, TestType>);
    REQUIRE_FALSE(Traits::is_std_sequence_of_type_v<std::array<std::add_lvalue_reference_t<TestType>, 1>, TestType>);
    REQUIRE_FALSE(Traits::is_std_sequence_of_type_v<std::vector<int>, unsigned int>);
    REQUIRE_FALSE(Traits::is_std_sequence_of_type_v<std::vector<float>, double>);
    REQUIRE_FALSE(Traits::is_std_sequence_of_type_v<std::vector<float>, const float>);
    REQUIRE_FALSE(Traits::is_std_sequence_of_type_v<std::vector<double>, int>);
    REQUIRE_FALSE(Traits::is_std_sequence_of_type_v<std::vector<std::complex<float>>, std::complex<double>>);
    REQUIRE_FALSE(Traits::is_std_sequence_of_type_v<std::vector<std::string>, std::string_view>);
}
//@CLION-formatter:on


TEMPLATE_TEST_CASE("Traits: is_same", "[noa][traits]",
                   uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t,
                   float, double, cfloat_t, cdouble_t,
                   std::complex<float>, std::complex<double>,
                   std::string, std::string_view, bool) {

    REQUIRE(Traits::is_same_v<TestType, TestType>);
    REQUIRE(Traits::is_same_v<std::add_const_t<TestType>, TestType>);
    REQUIRE(Traits::is_same_v<std::add_volatile_t<TestType>, TestType>);
    REQUIRE(Traits::is_same_v<std::add_cv_t<TestType>, TestType>);
    REQUIRE(Traits::is_same_v<std::add_lvalue_reference_t<TestType>, TestType>);
    REQUIRE(Traits::is_same_v<std::add_rvalue_reference_t<TestType>, TestType>);
    REQUIRE(Traits::is_same_v<std::add_pointer<TestType>, std::add_pointer<TestType>>);

    REQUIRE_FALSE(Traits::is_same_v<int, unsigned int>);
    REQUIRE_FALSE(Traits::is_same_v<float, double>);
    REQUIRE_FALSE(Traits::is_same_v<double, int>);
    REQUIRE_FALSE(Traits::is_same_v<cfloat_t, cdouble_t>);
    REQUIRE_FALSE(Traits::is_same_v<std::string, std::string_view>);
    REQUIRE_FALSE(Traits::is_same_v<std::add_pointer<TestType>, TestType>);

    // Some important assumptions:
    REQUIRE(Traits::is_same_v<uint16_t, unsigned short>);
    REQUIRE(Traits::is_same_v<uint32_t, unsigned int>);
    REQUIRE(Traits::is_same_v<uint64_t, unsigned long>);
    REQUIRE(Traits::is_same_v<int16_t, short>);
    REQUIRE(Traits::is_same_v<int32_t, int>);
    REQUIRE(Traits::is_same_v<int64_t, long>);
}


TEMPLATE_TEST_CASE("Traits: always_false", "[noa][traits]",
                   uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t,
                   float, double, cfloat_t, cdouble_t,
                   std::complex<float>, std::complex<double>,
                   std::string, std::string_view, bool) {
    REQUIRE_FALSE(Traits::always_false_v<TestType>);
    REQUIRE_FALSE(Traits::always_false_v<std::array<TestType, 3>>);
    REQUIRE_FALSE(Traits::always_false_v<std::vector<TestType>>);
}
