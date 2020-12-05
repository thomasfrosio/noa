/**
 * @file Traits.h
 * @brief Some type traits.
 * @author Thomas - ffyr2w
 * @date 23 Jul 2020
 *
 * Available type traits:
 *  -# @c is_unsigned_v             : (cv qualifiers) unsigned short|int|long|long long
 *  -# @c is_int_v                  : (cv qualifiers) (unsigned) short|int|long|long long
 *  -# @c is_float_v                : (cv qualifiers) float|double|long double
 *  -# @c is_complex_v              : (cv qualifiers) std::complex<float|double|long double>
 *  -# @c is_scalar_v               : is_float_v || is_int_v
 *  -# @c is_data_v                 : is_float_v || is_complex_v
 *  -# @c is_arith_v                : is_float_v || is_int_v || is_complex_v
 *
 *  -# @c is_bool_v                 : (cv qualifiers) bool
 *  -# @c is_string_v               : (cv qualifiers) std::string(_view)
 *
 *  -# @c is_vector_v               : std::vector
 *  -# @c is_vector_of_bool_v       : std::vector<is_bool_v, A>
 *  -# @c is_vector_of_string_v     : std::vector<is_string_v, A>
 *  -# @c is_vector_of_unsigned_v   : std::vector<is_unsigned_v, A>
 *  -# @c is_vector_of_int_v        : std::vector<is_int_v, A>
 *  -# @c is_vector_of_float_v      : std::vector<is_float_v, A>
 *  -# @c is_vector_of_complex_v    : std::vector<is_complex_v, A>
 *  -# @c is_vector_of_scalar_v     : std::vector<(is_float_v|is_int_v), A>
 *  -# @c is_vector_of_data_v       : std::vector<(is_float_v|is_complex_v), A>
 *  -# @c is_vector_of_arith_v      : std::vector<(is_float_v|is_complex_v|is_int_v), A>
 *
 *  -# @c is_array_v                : std::array
 *  -# @c is_array_of_bool_v        : std::array<is_bool_v, N>
 *  -# @c is_array_of_string_v      : std::array<is_string_v, N>
 *  -# @c is_array_of_unsigned_v    : std::array<is_unsigned_v, N>
 *  -# @c is_array_of_int_v         : std::array<is_int_v, N>
 *  -# @c is_array_of_float_v       : std::array<is_float_v, N>
 *  -# @c is_array_of_complex_v     : std::array<is_complex_v, N>
 *  -# @c is_array_of_scalar_v      : std::array<(is_float_v|is_int_v), N>
 *  -# @c is_array_of_data_v        : std::array<(is_float_v|is_complex_v), N>
 *  -# @c is_array_of_arith_v       : std::array<(is_float_v|is_complex_v|is_int_v), N>
 *
 *  -# @c is_sequence_v             : std::(vector|array)
 *  -# @c is_sequence_of_bool_v     : std::(vector|array)<is_bool_v, X>
 *  -# @c is_sequence_of_string_v   : std::(vector|array)<is_string_v, X>
 *  -# @c is_sequence_of_unsigned_v : std::(vector|array)<is_unsigned_v, A>
 *  -# @c is_sequence_of_int_v      : std::(vector|array)<is_int_v, X>
 *  -# @c is_sequence_of_float_v    : std::(vector|array)<is_float_v, X>
 *  -# @c is_sequence_of_complex_v  : std::(vector|array)<is_complex_v, X>
 *  -# @c is_sequence_of_scalar_v   : std::(vector|array)<(is_float_v|is_int_v), X>
 *  -# @c is_sequence_of_data_v     : std::(vector|array)<(is_float_v|is_complex_v), X>
 *  -# @c is_sequence_of_arith_v    : std::(vector|array)<(is_float_v|is_complex_v|is_int_v), X>
 *
 *  -# @c is_sequence_of_type_v<T1, V2>         T1 = std::(vector|array)<V1>; check if V1 == V2
 *  -# @c are_sequence_of_same_type_v<T1, T2>   T1|T2 = std::(vector|array)<V1|V2>; check if V1 == V2
 *  -# @c is_same_v<T1, T2>                     T1|T2 = (cv) V1|V2(&); check if V1 == V2
 *  -# @c remove_ref_cv<T>                      std::remove_cv_t<std::remove_reference_t<T>>
 */
#pragma once

#include "noa/Base.h"


/** Gathers a bunch of type traits. */
namespace Noa::Traits {
    template<typename>
    struct p_is_unsigned : public std::false_type {
    };
    template<>
    struct p_is_unsigned<uint8_t> : public std::true_type {
    };
    template<>
    struct p_is_unsigned<unsigned short> : public std::true_type {
    };
    template<>
    struct p_is_unsigned<unsigned int> : public std::true_type {
    };
    template<>
    struct p_is_unsigned<unsigned long> : public std::true_type {
    };
    template<>
    struct p_is_unsigned<unsigned long long> : public std::true_type {
    };
    template<typename T>
    struct NOA_API is_unsigned
            : p_is_unsigned<typename std::remove_cv_t<typename std::remove_reference_t<T>>>::type {
    };
    template<typename T>
    NOA_API inline constexpr bool is_unsigned_v = is_unsigned<T>::value;
}


namespace Noa::Traits {
    template<typename>
    struct p_is_int : public std::false_type {
    };
    template<>
    struct p_is_int<int8_t> : public std::true_type {
    };
    template<>
    struct p_is_int<uint8_t> : public std::true_type {
    };
    template<>
    struct p_is_int<short> : public std::true_type {
    };
    template<>
    struct p_is_int<unsigned short> : public std::true_type {
    };
    template<>
    struct p_is_int<int> : public std::true_type {
    };
    template<>
    struct p_is_int<unsigned int> : public std::true_type {
    };
    template<>
    struct p_is_int<long> : public std::true_type {
    };
    template<>
    struct p_is_int<unsigned long> : public std::true_type {
    };
    template<>
    struct p_is_int<long long> : public std::true_type {
    };
    template<>
    struct p_is_int<unsigned long long> : public std::true_type {
    };
    template<typename T>
    struct NOA_API is_int
            : p_is_int<typename std::remove_cv_t<typename std::remove_reference_t<T>>>::type {
    };
    template<typename T>
    NOA_API inline constexpr bool is_int_v = is_int<T>::value;
}


namespace Noa::Traits {
    template<typename>
    struct p_is_float : std::false_type {
    };
    template<>
    struct p_is_float<float> : std::true_type {
    };
    template<>
    struct p_is_float<double> : std::true_type {
    };
    template<>
    struct p_is_float<long double> : std::true_type {
    };
    template<typename T>
    struct NOA_API is_float
            : p_is_float<typename std::remove_cv_t<typename std::remove_reference_t<T>>>::type {
    };
    template<typename T>
    NOA_API inline constexpr bool is_float_v = is_float<T>::value;
}


namespace Noa::Traits {
    template<typename>
    struct p_is_complex : std::false_type {
    };
    template<>
    struct p_is_complex<std::complex<float>> : std::true_type {
    };
    template<>
    struct p_is_complex<std::complex<double>> : std::true_type {
    };
    template<>
    struct p_is_complex<std::complex<long double>> : std::true_type {
    };
    template<typename T>
    struct NOA_API is_complex
            : p_is_complex<typename std::remove_cv_t<typename std::remove_reference_t<T>>>::type {
    };
    template<typename T>
    NOA_API inline constexpr bool is_complex_v = is_complex<T>::value;
}


namespace Noa::Traits {
    template<typename T>
    struct NOA_API is_scalar {
        static constexpr const bool value = is_float<T>::value || is_int<T>::value;
    };
    template<typename T>
    NOA_API inline constexpr bool is_scalar_v = is_scalar<T>::value;
}


namespace Noa::Traits {
    template<typename T>
    struct NOA_API is_data {
        static constexpr bool value = is_float<T>::value || is_complex<T>::value;
    };
    template<typename T>
    NOA_API inline constexpr bool is_data_v = is_data<T>::value;
}


namespace Noa::Traits {
    template<typename T>
    struct NOA_API is_arith {
        static constexpr bool value = (is_float<T>::value ||
                                       is_int<T>::value ||
                                       is_complex<T>::value);
    };
    template<typename T>
    NOA_API inline constexpr bool is_arith_v = is_arith<T>::value;
}


namespace Noa::Traits {
    template<typename>
    struct p_is_bool : std::false_type {
    };
    template<>
    struct p_is_bool<bool> : std::true_type {
    };
    template<typename T>
    struct NOA_API is_bool
            : p_is_bool<typename std::remove_cv_t<typename std::remove_reference_t<T>>>::type {
    };
    template<typename T>
    NOA_API inline constexpr bool is_bool_v = is_bool<T>::value;
}


namespace Noa::Traits {
    template<typename>
    struct p_is_string : std::false_type {
    };
    template<>
    struct p_is_string<std::string> : std::true_type {
    };
    template<>
    struct p_is_string<std::string_view> : std::true_type {
    };
    template<typename T>
    struct NOA_API is_string
            : p_is_string<typename std::remove_cv_t<typename std::remove_reference_t<T>>>::type {
    };
    template<typename T>
    NOA_API inline constexpr bool is_string_v = is_string<T>::value;
}


// is_vector
namespace Noa::Traits {
    template<typename T>
    struct p_is_vector : std::false_type {
    };
    template<typename T, typename A>
    struct p_is_vector<std::vector<T, A>> : std::true_type {
    };
    template<typename T>
    struct NOA_API is_vector {
        static constexpr bool value = p_is_vector<
                typename std::remove_cv_t<typename std::remove_reference_t<T>>>::value;
    };
    template<typename T>
    NOA_API inline constexpr bool is_vector_v = is_vector<T>::value;
}


// is_vector_of_unsigned
namespace Noa::Traits {
    template<typename T>
    struct p_is_vector_of_unsigned : std::false_type {
    };
    template<typename T, typename A>
    struct p_is_vector_of_unsigned<std::vector<T, A>> {
        static constexpr bool value = is_unsigned_v<T>; // noa::p_is_unsigned<T>::value
    };
    template<typename T>
    struct NOA_API is_vector_of_unsigned {
        static constexpr bool value = p_is_vector_of_unsigned<
                typename std::remove_cv_t<typename std::remove_reference_t<T>>>::value;
    };
    template<typename T>
    NOA_API inline constexpr bool is_vector_of_unsigned_v = is_vector_of_unsigned<T>::value;
}


// is_vector_of_int
namespace Noa::Traits {
    template<typename T>
    struct p_is_vector_of_int : std::false_type {
    };
    template<typename T, typename A>
    struct p_is_vector_of_int<std::vector<T, A>> {
        static constexpr bool value = is_int_v<T>; // noa::p_is_int<T>::value
    };
    template<typename T>
    struct NOA_API is_vector_of_int {
        static constexpr bool value = p_is_vector_of_int<
                typename std::remove_cv_t<typename std::remove_reference_t<T>>>::value;
    };
    template<typename T>
    NOA_API inline constexpr bool is_vector_of_int_v = is_vector_of_int<T>::value;
}


// is_vector_of_float
namespace Noa::Traits {
    template<typename T>
    struct p_is_vector_of_float : std::false_type {
    };
    template<typename T, typename A>
    struct p_is_vector_of_float<std::vector<T, A>> {
        static constexpr bool value = is_float_v<T>; // noa::p_is_float<T>::value
    };
    template<typename T>
    struct NOA_API is_vector_of_float {
        static constexpr bool value = p_is_vector_of_float<
                typename std::remove_cv_t<typename std::remove_reference_t<T>>>::value;
    };
    template<typename T>
    NOA_API inline constexpr bool is_vector_of_float_v = is_vector_of_float<T>::value;
}


// is_vector_of_complex
namespace Noa::Traits {
    template<typename T>
    struct p_is_vector_of_complex : std::false_type {
    };
    template<typename T, typename A>
    struct p_is_vector_of_complex<std::vector<T, A>> {
        static constexpr bool value = is_complex_v<T>; // noa::p_is_complex<T>::value
    };
    template<typename T>
    struct NOA_API is_vector_of_complex {
        static constexpr bool value = p_is_vector_of_complex<
                typename std::remove_cv_t<typename std::remove_reference_t<T>>>::value;
    };
    template<typename T>
    NOA_API inline constexpr bool is_vector_of_complex_v = is_vector_of_complex<T>::value;
}


// is_vector_of_scalar
namespace Noa::Traits {
    template<typename T>
    struct p_is_vector_of_scalar : std::false_type {
    };
    template<typename T, typename A>
    struct p_is_vector_of_scalar<std::vector<T, A>> {
        static constexpr bool value = is_int_v<T> || is_float_v<T>;
    };
    template<typename T>
    struct NOA_API is_vector_of_scalar {
        static constexpr bool value = p_is_vector_of_scalar<
                typename std::remove_cv_t<typename std::remove_reference_t<T>>>::value;
    };
    template<typename T>
    NOA_API inline constexpr bool is_vector_of_scalar_v = is_vector_of_scalar<T>::value;
}


// is_vector_of_data
namespace Noa::Traits {
    template<typename T>
    struct p_is_vector_of_data : std::false_type {
    };
    template<typename T, typename A>
    struct p_is_vector_of_data<std::vector<T, A>> {
        static constexpr bool value = is_complex_v<T> || is_float_v<T>;
    };
    template<typename T>
    struct NOA_API is_vector_of_data {
        static constexpr bool value = p_is_vector_of_data<
                typename std::remove_cv_t<typename std::remove_reference_t<T>>>::value;
    };
    template<typename T>
    NOA_API inline constexpr bool is_vector_of_data_v = is_vector_of_data<T>::value;
}


// is_vector_of_arith
namespace Noa::Traits {
    template<typename T>
    struct p_is_vector_of_arith : std::false_type {
    };
    template<typename T, typename A>
    struct p_is_vector_of_arith<std::vector<T, A>> {
        static constexpr bool value = is_int_v<T> || is_float_v<T> || is_complex_v<T>;
    };
    template<typename T>
    struct NOA_API is_vector_of_arith {
        static constexpr bool value = p_is_vector_of_arith<
                typename std::remove_cv_t<typename std::remove_reference_t<T>>>::value;
    };
    template<typename T>
    NOA_API inline constexpr bool is_vector_of_arith_v = is_vector_of_arith<T>::value;
}


// is_vector_of_bool
namespace Noa::Traits {
    template<typename>
    struct p_is_vector_of_bool : std::false_type {
    };
    template<typename T, typename A>
    struct p_is_vector_of_bool<std::vector<T, A>> {
        static constexpr bool value = is_bool_v<T>;
    };
    template<typename T>
    struct NOA_API is_vector_of_bool {
        static constexpr bool value = p_is_vector_of_bool<
                typename std::remove_cv_t<typename std::remove_reference_t<T>>>::value;
    };
    template<typename T>
    NOA_API inline constexpr bool is_vector_of_bool_v = is_vector_of_bool<T>::value;
}


// is_vector_of_string
namespace Noa::Traits {
    template<typename T>
    struct p_is_vector_of_string : std::false_type {
    };
    template<typename T, typename A>
    struct p_is_vector_of_string<std::vector<T, A>> {
        static constexpr bool value = is_string_v<T>;
    };
    template<typename T>
    struct NOA_API is_vector_of_string {
        static constexpr bool value = p_is_vector_of_string<
                typename std::remove_cv_t<typename std::remove_reference_t<T>>>::value;
    };
    template<typename T>
    NOA_API inline constexpr bool is_vector_of_string_v = is_vector_of_string<T>::value;
}


namespace Noa::Traits {
    template<typename T>
    struct p_is_array : std::false_type {
    };
    template<typename T, std::size_t N>
    struct p_is_array<std::array<T, N>> : std::true_type {
    };
    template<typename T>
    struct NOA_API is_array
            : p_is_array<typename std::remove_cv_t<typename std::remove_reference_t<T>>>::type {
    };
    template<typename T>
    NOA_API inline constexpr bool is_array_v = is_array<T>::value;
}


namespace Noa::Traits {
    template<typename T>
    struct p_is_array_of_unsigned : std::false_type {
    };
    template<typename T, std::size_t N>
    struct p_is_array_of_unsigned<std::array<T, N>> {
        static constexpr bool value = is_unsigned_v<T>; // noa::p_is_unsigned<T>::value
    };
    template<typename T>
    struct NOA_API is_array_of_unsigned {
        static constexpr bool value = p_is_array_of_unsigned<
                typename std::remove_cv_t<typename std::remove_reference_t<T>>>::value;
    };
    template<typename T>
    NOA_API inline constexpr bool is_array_of_unsigned_v = is_array_of_unsigned<T>::value;
}


namespace Noa::Traits {
    template<typename T>
    struct p_is_array_of_int : std::false_type {
    };
    template<typename T, std::size_t N>
    struct p_is_array_of_int<std::array<T, N>> {
        static constexpr bool value = is_int_v<T>; // noa::p_is_int<T>::value
    };
    template<typename T>
    struct NOA_API is_array_of_int {
        static constexpr bool value = p_is_array_of_int<
                typename std::remove_cv_t<typename std::remove_reference_t<T>>>::value;
    };
    template<typename T>
    NOA_API inline constexpr bool is_array_of_int_v = is_array_of_int<T>::value;
}


namespace Noa::Traits {
    template<typename T>
    struct p_is_array_of_float : std::false_type {
    };
    template<typename T, std::size_t N>
    struct p_is_array_of_float<std::array<T, N>> {
        static constexpr bool value = is_float_v<T>; // noa::p_is_float<T>::value
    };
    template<typename T>
    struct NOA_API is_array_of_float {
        static constexpr bool value = p_is_array_of_float<
                typename std::remove_cv_t<typename std::remove_reference_t<T>>>::value;
    };
    template<typename T>
    NOA_API inline constexpr bool is_array_of_float_v = is_array_of_float<T>::value;
}


namespace Noa::Traits {
    template<typename T>
    struct p_is_array_of_complex : std::false_type {
    };
    template<typename T, std::size_t N>
    struct p_is_array_of_complex<std::array<T, N>> {
        static constexpr bool value = is_complex_v<T>; // noa::p_is_complex<T>::value
    };
    template<typename T>
    struct NOA_API is_array_of_complex {
        static constexpr bool value = p_is_array_of_complex<
                typename std::remove_cv_t<typename std::remove_reference_t<T>>>::value;
    };
    template<typename T>
    NOA_API inline constexpr bool is_array_of_complex_v = is_array_of_complex<T>::value;
}


namespace Noa::Traits {
    template<typename T>
    struct p_is_array_of_scalar : std::false_type {
    };
    template<typename T, std::size_t N>
    struct p_is_array_of_scalar<std::array<T, N>> {
        static constexpr bool value = is_int_v<T> || is_float_v<T>;
    };
    template<typename T>
    struct NOA_API is_array_of_scalar {
        static constexpr bool value = p_is_array_of_scalar<
                typename std::remove_cv_t<typename std::remove_reference_t<T>>>::value;
    };
    template<typename T>
    NOA_API inline constexpr bool is_array_of_scalar_v = is_array_of_scalar<T>::value;
}


namespace Noa::Traits {
    template<typename T>
    struct p_is_array_of_data : std::false_type {
    };
    template<typename T, std::size_t N>
    struct p_is_array_of_data<std::array<T, N>> {
        static constexpr bool value = is_complex_v<T> || is_float_v<T>;
    };
    template<typename T>
    struct NOA_API is_array_of_data {
        static constexpr bool value = p_is_array_of_data<
                typename std::remove_cv_t<typename std::remove_reference_t<T>>>::value;
    };
    template<typename T>
    NOA_API inline constexpr bool is_array_of_data_v = is_array_of_data<T>::value;
}


namespace Noa::Traits {
    template<typename T>
    struct p_is_array_of_arith : std::false_type {
    };
    template<typename T, std::size_t N>
    struct p_is_array_of_arith<std::array<T, N>> {
        static constexpr bool value = is_complex_v<T> || is_int_v<T> || is_float_v<T>;
    };
    template<typename T>
    struct NOA_API is_array_of_arith {
        static constexpr bool value = p_is_array_of_arith<
                typename std::remove_cv_t<typename std::remove_reference_t<T>>>::value;
    };
    template<typename T>
    NOA_API inline constexpr bool is_array_of_arith_v = is_array_of_arith<T>::value;
}


namespace Noa::Traits {
    template<typename T>
    struct p_is_array_of_bool : std::false_type {
    };
    template<typename T, std::size_t N>
    struct p_is_array_of_bool<std::array<T, N>> {
        static constexpr bool value = is_bool_v<T>;
    };
    template<typename T>
    struct NOA_API is_array_of_bool {
        static constexpr bool value = p_is_array_of_bool<
                typename std::remove_cv_t<typename std::remove_reference_t<T>>>::value;
    };
    template<typename T>
    NOA_API inline constexpr bool is_array_of_bool_v = is_array_of_bool<T>::value;
}


namespace Noa::Traits {
    template<typename T>
    struct p_is_array_of_string : std::false_type {
    };
    template<typename T, std::size_t N>
    struct p_is_array_of_string<std::array<T, N>> {
        static constexpr bool value = is_string_v<T>;
    };
    template<typename T>
    struct NOA_API is_array_of_string {
        static constexpr bool value = p_is_array_of_string<
                typename std::remove_cv_t<typename std::remove_reference_t<T>>>::value;
    };
    template<typename T>
    NOA_API inline constexpr bool is_array_of_string_v = is_array_of_string<T>::value;
}


namespace Noa::Traits {
    template<typename T>
    struct NOA_API is_sequence {
        static constexpr bool value = (is_array<T>::value || is_vector<T>::value);
    };
    template<typename T>
    NOA_API inline constexpr bool is_sequence_v = is_sequence<T>::value;
}


namespace Noa::Traits {
    template<typename T>
    struct NOA_API is_sequence_of_unsigned {
        static constexpr bool value = (is_array_of_unsigned<T>::value ||
                                       is_vector_of_unsigned<T>::value);
    };
    template<typename T>
    NOA_API inline constexpr bool is_sequence_of_unsigned_v = is_sequence_of_unsigned<T>::value;
}


namespace Noa::Traits {
    template<typename T>
    struct NOA_API is_sequence_of_int {
        static constexpr bool value = (is_array_of_int<T>::value ||
                                       is_vector_of_int<T>::value);
    };
    template<typename T>
    NOA_API inline constexpr bool is_sequence_of_int_v = is_sequence_of_int<T>::value;
}


namespace Noa::Traits {
    template<typename T>
    struct NOA_API is_sequence_of_float {
        static constexpr bool value = (is_array_of_float<T>::value ||
                                       is_vector_of_float<T>::value);
    };
    template<typename T>
    NOA_API inline constexpr bool is_sequence_of_float_v = is_sequence_of_float<T>::value;
}


namespace Noa::Traits {
    template<typename T>
    struct NOA_API is_sequence_of_complex {
        static constexpr bool value = (is_array_of_complex<T>::value ||
                                       is_vector_of_complex<T>::value);
    };
    template<typename T>
    NOA_API inline constexpr bool is_sequence_of_complex_v = is_sequence_of_complex<T>::value;
}


namespace Noa::Traits {
    template<typename T>
    struct NOA_API is_sequence_of_scalar {
        static constexpr bool value = (is_array_of_scalar<T>::value ||
                                       is_vector_of_scalar<T>::value);
    };
    template<typename T>
    NOA_API inline constexpr bool is_sequence_of_scalar_v = is_sequence_of_scalar<T>::value;
}


namespace Noa::Traits {
    template<typename T>
    struct NOA_API is_sequence_of_data {
        static constexpr bool value = (is_array_of_data<T>::value ||
                                       is_vector_of_data<T>::value);
    };
    template<typename T>
    NOA_API inline constexpr bool is_sequence_of_data_v = is_sequence_of_data<T>::value;
}


namespace Noa::Traits {
    template<typename T>
    struct NOA_API is_sequence_of_arith {
        static constexpr bool value = (is_array_of_arith<T>::value ||
                                       is_vector_of_arith<T>::value);
    };
    template<typename T>
    NOA_API inline constexpr bool is_sequence_of_arith_v = is_sequence_of_arith<T>::value;
}


namespace Noa::Traits {
    template<typename T>
    struct NOA_API is_sequence_of_bool {
        static constexpr bool value = (is_array_of_bool<T>::value ||
                                       is_vector_of_bool<T>::value);
    };
    template<typename T>
    NOA_API inline constexpr bool is_sequence_of_bool_v = is_sequence_of_bool<T>::value;
}


namespace Noa::Traits {
    template<typename T>
    struct NOA_API is_sequence_of_string {
        static constexpr bool value = (is_array_of_string<T>::value ||
                                       is_vector_of_string<T>::value);
    };
    template<typename T>
    NOA_API inline constexpr bool is_sequence_of_string_v = is_sequence_of_string<T>::value;
}


namespace Noa::Traits {
    template<typename, typename>
    struct p_is_sequence_of_type : std::false_type {
    };
    template<typename V1, typename A, typename V2>
    struct p_is_sequence_of_type<std::vector<V1, A>, V2> {
        static constexpr bool value = std::is_same_v<V1, V2>;
    };
    template<typename V1, size_t N, typename V2>
    struct p_is_sequence_of_type<std::array<V1, N>, V2> {
        static constexpr bool value = std::is_same_v<V1, V2>;
    };
    template<typename T, typename V>
    struct NOA_API is_sequence_of_type {
        static constexpr bool value = p_is_sequence_of_type<
                typename std::remove_cv_t<typename std::remove_reference_t<T>>, V>::value;
    };
    template<typename T, typename V>
    NOA_API inline constexpr bool is_sequence_of_type_v = is_sequence_of_type<T, V>::value;
}


namespace Noa::Traits {
    template<typename, typename>
    struct p_are_sequence_of_same_type : std::false_type {
    };
    template<typename V1, typename V2, typename X>
    struct p_are_sequence_of_same_type<std::vector<V1, X>, std::vector<V2, X>> {
        static constexpr bool value = std::is_same_v<V1, V2>;
    };
    template<typename V1, typename V2, size_t X>
    struct p_are_sequence_of_same_type<std::array<V1, X>, std::array<V2, X>> {
        static constexpr bool value = std::is_same_v<V1, V2>;
    };
    template<typename V1, typename V2, typename X1, size_t X2>
    struct p_are_sequence_of_same_type<std::vector<V1, X1>, std::array<V2, X2>> {
        static constexpr bool value = std::is_same_v<V1, V2>;
    };
    template<typename V1, typename V2, size_t X1, typename X2>
    struct p_are_sequence_of_same_type<std::array<V1, X1>, std::vector<V2, X2>> {
        static constexpr bool value = std::is_same_v<V1, V2>;
    };
    template<typename T1, typename T2>
    struct NOA_API are_sequence_of_same_type {
        static constexpr bool value = p_are_sequence_of_same_type<
                typename std::remove_cv_t<typename std::remove_reference_t<T1>>,
                typename std::remove_cv_t<typename std::remove_reference_t<T2>>
        >::value;
    };
    template<typename T1, typename T2>
    NOA_API inline constexpr bool are_sequence_of_same_type_v = are_sequence_of_same_type<T1, T2>::value;
}


namespace Noa::Traits {
    template<typename T1, typename T2>
    struct NOA_API is_same {
        static constexpr bool value = std::is_same_v<
                std::remove_cv_t<std::remove_reference_t<T1>>,
                std::remove_cv_t<std::remove_reference_t<T2>>
        >;
    };
    template<typename T1, typename T2>
    NOA_API inline constexpr bool is_same_v = is_same<T1, T2>::value;
}


namespace Noa::Traits {
    template<typename T>
    struct NOA_API always_false {
        static constexpr bool value = false;
    };
    template<typename T>
    NOA_API inline constexpr bool always_false_v = always_false<T>::value;
}


namespace Noa::Traits {
    template<typename T>
    struct NOA_API remove_ref_cv {
        using type = typename std::remove_const_t<typename std::remove_reference_t<T>>;
    };
    template<typename T>
    NOA_API using remove_ref_cv_t = typename remove_ref_cv<T>::type;
}
