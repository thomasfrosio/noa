/**
 * @file Traits.h
 * @brief Some type traits
 * @author Thomas - ffyr2w
 * @date 23 Jul 2020
 *
 * @details:
 *
 * is_int_v                 short/int/long/long long and cv qualifiers
 * is_float_v               float/double/long double and cv qualifiers
 * is_arith_v               is_int_v || is_float_v
 * is_bool_v                bool and cv qualifiers
 *
 * is_sequence_v            std::vector or std::array
 * is_sequence_of_int_v     std::vector<is_int_v, A> || std::array<is_int_v, T>
 * is_sequence_of_float_v   std::vector<is_float_v, A> || std::array<is_float_v, T>
 * is_sequence_of_arith_v   std::vector<is_int|float_v, A> || std::array<is_int|float_v, T>
 * is_sequence_of_bool_v    std::vector<is_bool_v, A> || std::array<is_bool_v, T>
 *
 * is_vector_v              std::vector
 * is_vector_of_int_v       std::vector<is_int_v, A>
 * is_vector_of_float_v     std::vector<is_float_v, A>
 * is_vector_of_arith_v     std::vector<is_int|float_v, A>
 * is_vector_of_bool_v      std::vector<is_bool_v, A>
 *
 * is_array_v               std::array
 * is_array_of_int_v        std::array<is_int_v, T>
 * is_array_of_float_v      std::array<is_float_v, T>
 * is_array_of_arith_v      std::array<is_int|float_v, T>
 * is_array_of_bool_v       std::array<is_bool_v, T>
 */
#pragma once

#include "Core.h"


// is_int
namespace Noa::Traits {
    template<typename>
    struct p_is_int : public std::false_type {
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
    struct is_int : p_is_int<typename std::decay<T>>::type {
    };
    template<typename T>
    inline constexpr bool is_int_v = is_int<T>::value;
}

// is_float
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
    struct is_float : p_is_float<typename std::decay<T>>::type {
    };
    template<typename T>
    inline constexpr bool is_float_v = is_float<T>::value;
}

// is_arith
namespace Noa::Traits {
    template<typename T>
    struct is_arith {
        static constexpr const bool value = is_float<T>::value || is_int<T>::value;
    };
    template<typename T>
    inline constexpr bool is_arith_v = is_arith<T>::value;
}

// is_bool
namespace Noa::Traits {
    template<typename>
    struct p_is_bool : std::false_type {
    };
    template<>
    struct p_is_bool<bool> : std::true_type {
    };
    template<typename T>
    struct is_bool : p_is_bool<typename std::decay<T>>::type {
    };
    template<typename T>
    inline constexpr bool is_bool_v = is_bool<T>::value;
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
    struct is_vector {
        static constexpr const bool value = p_is_vector<typename std::decay_t<T>>::value;
    };
    template<typename T>
    inline constexpr bool is_vector_v = is_vector<T>::value;
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
    struct is_vector_of_int {
        static constexpr bool value = p_is_vector_of_int<typename std::decay_t<T>>::value;
    };
    template<typename T>
    inline constexpr bool is_vector_of_int_v = is_vector_of_int<T>::value;
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
    struct is_vector_of_float {
        static constexpr bool value = p_is_vector_of_float<typename std::decay_t<T>>::value;
    };
    template<typename T>
    inline constexpr bool is_vector_of_float_v = is_vector_of_float<T>::value;
}

// is_vector_of_arith
namespace Noa::Traits {
    template<typename T>
    struct p_is_vector_of_arith : std::false_type {
    };
    template<typename T, typename A>
    struct p_is_vector_of_arith<std::vector<T, A>> {
        static constexpr bool value = is_int_v<T> || is_float_v<T>;
    };
    template<typename T>
    struct is_vector_of_arith {
        static constexpr bool value = is_vector_of_arith<typename std::decay_t<T>>::value;
    };
    template<typename T>
    inline constexpr bool is_vector_of_arith_v = is_vector_of_arith<T>::value;
}

// is_vector_of_bool
namespace Noa::Traits {
    template<typename T>
    struct p_is_vector_of_bool : std::false_type {
    };
    template<typename T, typename A>
    struct p_is_vector_of_bool<std::vector<T, A>> {
        static constexpr bool value = is_bool_v<T>;
    };
    template<typename T>
    struct is_vector_of_bool {
        static constexpr bool value = is_vector_of_bool<typename std::decay_t<T>>::value;
    };
    template<typename T>
    inline constexpr bool is_vector_of_bool_v = is_vector_of_bool<T>::value;
}

// is_array
namespace Noa::Traits {
    template<typename T>
    struct p_is_array : std::false_type {
    };
    template<typename T, std::size_t N>
    struct p_is_array<std::array<T, N>> : std::true_type {
    };
    template<typename T>
    struct is_array : p_is_array<typename std::decay_t<T>>::type {
    };
    template<typename T>
    inline constexpr bool is_array_v = is_array<T>::value;
}

// is_array_of_int_v
namespace Noa::Traits {
    template<typename T>
    struct p_is_array_of_int : std::false_type {
    };
    template<typename T, std::size_t N>
    struct p_is_array_of_int<std::array<T, N>> {
        static constexpr bool value = is_int_v<T>; // noa::p_is_int<T>::value
    };
    template<typename T>
    struct is_array_of_int {
        static constexpr bool value = p_is_array_of_int<typename std::decay_t<T>>::value;
    };
    template<typename T>
    inline constexpr bool is_array_of_int_v = is_array_of_int<T>::value;
}

// is_array_of_float
namespace Noa::Traits {
    template<typename T>
    struct p_is_array_of_float : std::false_type {
    };
    template<typename T, std::size_t N>
    struct p_is_array_of_float<std::array<T, N>> {
        static constexpr bool value = is_float_v<T>; // noa::p_is_float<T>::value
    };
    template<typename T>
    struct is_array_of_float {
        static constexpr bool value = p_is_array_of_float<typename std::decay_t<T>>::value;
    };
    template<typename T>
    inline constexpr bool is_array_of_float_v = is_array_of_float<T>::value;
}

// is_array_of_arith
namespace Noa::Traits {
    template<typename T>
    struct p_is_array_of_arith : std::false_type {
    };
    template<typename T, std::size_t N>
    struct p_is_array_of_arith<std::array<T, N>> {
        static constexpr bool value = is_int_v<T> || is_float_v<T>;
    };
    template<typename T>
    struct is_array_of_arith {
        static constexpr bool value = is_array_of_arith<typename std::decay_t<T>>::value;
    };
    template<typename T>
    inline constexpr bool is_array_of_arith_v = is_array_of_arith<T>::value;
}

// is_array_of_bool
namespace Noa::Traits {
    template<typename T>
    struct p_is_array_of_bool : std::false_type {
    };
    template<typename T, std::size_t N>
    struct p_is_array_of_bool<std::array<T, N>> {
        static constexpr bool value = is_bool_v<T>;
    };
    template<typename T>
    struct is_array_of_bool {
        static constexpr bool value = is_array_of_bool<typename std::decay_t<T>>::value;
    };
    template<typename T>
    inline constexpr bool is_array_of_bool_v = is_array_of_bool<T>::value;
}

// is_sequence
namespace Noa::Traits {
    template<typename T>
    struct is_sequence {
        static constexpr const bool value = (is_array<T>::value || is_vector<T>::value);
    };
    template<typename T>
    inline constexpr bool is_sequence_v = is_sequence<T>::value;
}

// is_sequence_of_int
namespace Noa::Traits {
    template<typename T>
    struct is_sequence_of_int {
        static constexpr const bool value = (is_array_of_int<T>::value ||
                                             is_vector_of_int<T>::value);
    };
    template<typename T>
    inline constexpr bool is_sequence_of_int_v = is_sequence_of_int<T>::value;
}

// is_sequence_of_float
namespace Noa::Traits {
    template<typename T>
    struct is_sequence_of_float {
        static constexpr const bool value = (is_array_of_float<T>::value ||
                                             is_vector_of_float<T>::value);
    };
    template<typename T>
    inline constexpr bool is_sequence_of_float_v = is_sequence_of_float<T>::value;
}

// is_sequence_of_arith
namespace Noa::Traits {
    template<typename T>
    struct is_sequence_of_arith {
        static constexpr const bool value = (is_array_of_arith<T>::value ||
                                             is_vector_of_arith<T>::value);
    };
    template<typename T>
    inline constexpr bool is_sequence_of_arith_v = is_sequence_of_arith<T>::value;
}

// is_sequence_of_bool
namespace Noa::Traits {
    template<typename T>
    struct is_sequence_of_bool {
        static constexpr const bool value = (is_array_of_bool<T>::value ||
                                             is_vector_of_bool<T>::value);
    };
    template<typename T>
    inline constexpr bool is_sequence_of_bool_v = is_sequence_of_bool<T>::value;
}