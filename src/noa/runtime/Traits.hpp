#pragma once

#include "noa/runtime/core/Traits.hpp"

namespace noa::traits {
    NOA_GENERATE_PROCLAIM_FULL(array);
    NOA_GENERATE_PROCLAIM(array_rc);
    NOA_GENERATE_PROCLAIM(array_view);
    template<typename... T> concept array_rc = array<T...> and are_array_rc_v<T...>;
    template<typename... T> concept array_view = array<T...> and are_array_view_v<T...>;
    template<typename... T> concept array_decay = array<std::decay_t<T>...>;
    template<typename... T> concept array_rc_decay = array_rc<std::decay_t<T>...>;
    template<typename... T> concept array_view_decay = array_view<std::decay_t<T>...>;

    template<typename T> constexpr usize array_size_v = std::remove_reference_t<T>::SIZE;
    template<typename T> constexpr usize array_ssize_v = std::remove_reference_t<T>::SSIZE;

    template<typename T, usize... N> concept array_nd = array<T> and ((std::remove_reference_t<T>::SIZE == N) or ...);
    template<typename T, usize... N> concept array_rc_nd = array_rc<T> and ((std::remove_reference_t<T>::SIZE == N) or ...);
    template<typename T, usize... N> concept array_view_nd = array_view<T> and ((std::remove_reference_t<T>::SIZE == N) or ...);
    template<typename T, usize... N> concept array_decay_nd = array_nd<std::decay_t<T>, N...>;
    template<typename T, usize... N> concept array_rc_decay_nd = array_rc_nd<std::decay_t<T>, N...>;
    template<typename T, usize... N> concept array_view_decay_nd = array_view_nd<std::decay_t<T>, N...>;

    template<typename... T> concept array_with_same_nd = array<T...> and ((T::SIZE == first_t<T...>::SIZE) and ...);
    template<typename... T> concept array_decay_with_same_nd = array_with_same_nd<std::decay_t<T>...>;

    template<typename... T> concept readable_array = array<T...> and ((not std::is_void_v<value_type_t<T>>) and ...);
    template<typename... T> concept writable_array = readable_array<T...> and ((not std::is_const_v<value_type_t<T>>) and ...);
    template<typename... T> concept readable_array_decay = array_decay<T...> and ((not std::is_void_v<value_type_t<T>>) and ...);
    template<typename... T> concept writable_array_decay = readable_array_decay<T...> and ((not std::is_const_v<value_type_t<T>>) and ...);

    #define NOA_TRAITS_GENERATE_VARRAY(name)                                                                                    \
    template<typename... T> concept array_of_##name = array<T...> and name<value_type_t<T>...>;                               \
    template<typename... T> concept array_decay_of_##name = array_decay<T...> and name<value_type_t<T>...>;                   \
    template<typename... T> concept readable_array_of_##name = readable_array<T...> and name<value_type_t<T>...>;             \
    template<typename... T> concept writable_array_of_##name = writable_array<T...> and name<value_type_t<T>...>;             \
    template<typename... T> concept readable_array_decay_of_##name = readable_array_decay<T...> and name<value_type_t<T>...>; \
    template<typename... T> concept writable_array_decay_of_##name = writable_array_decay<T...> and name<value_type_t<T>...>
    NOA_TRAITS_GENERATE_VARRAY(integer);
    NOA_TRAITS_GENERATE_VARRAY(sinteger);
    NOA_TRAITS_GENERATE_VARRAY(uinteger);
    NOA_TRAITS_GENERATE_VARRAY(real);
    NOA_TRAITS_GENERATE_VARRAY(complex);
    NOA_TRAITS_GENERATE_VARRAY(real_or_complex);
    NOA_TRAITS_GENERATE_VARRAY(scalar);
    NOA_TRAITS_GENERATE_VARRAY(numeric);
    NOA_TRAITS_GENERATE_VARRAY(boolean);
    NOA_TRAITS_GENERATE_VARRAY(byte);
    #undef NOA_TRAITS_GENERATE_VARRAY

    #define NOA_TRAITS_GENERATE_VARRAY_SAME_AS(suffix, constrain)                                                                                               \
    template<typename T, typename... U> concept array_of_##suffix = array<T> and constrain<value_type_t<T>, U...>;                                            \
    template<typename T, typename... U> concept array_decay_of_##suffix = array_decay<T> and constrain<value_type_t<T>, U...>;                                \
    template<typename T, typename... U> concept readable_array_decay_of_##suffix = readable_array_decay<T> and constrain<value_type_t<T>, U...>;              \
    template<typename T, typename... U> concept writable_array_decay_of_##suffix = writable_array_decay<T> and constrain<value_type_t<T>, U...>;              \
    template<typename T, typename... U> concept array_or_value_of_##suffix = array_of_##suffix<T, U...> or constrain<T, U...>;                                \
    template<typename T, typename... U> concept array_decay_or_value_of_##suffix = array_decay_of_##suffix<T, U...> or constrain<T, U...>;                    \
    template<typename T, typename... U> concept readable_array_decay_or_value_of_##suffix = readable_array_decay_of_##suffix<T, U...> or constrain<T, U...>;  \
    template<typename T, typename... U> concept writable_array_decay_or_value_of_##suffix = writable_array_decay_of_##suffix<T, U...> or constrain<T, U...>

    NOA_TRAITS_GENERATE_VARRAY_SAME_AS(any, any_of);
    NOA_TRAITS_GENERATE_VARRAY_SAME_AS(almost_any, almost_any_of);
    #undef NOA_TRAITS_GENERATE_VARRAY_SAME_AS

    template<typename T, typename... U> concept array_of_almost_same_type = array<T, U...> and almost_same_as<value_type_t<T>, value_type_t<U>...>;
    template<typename T, typename... U> concept array_decay_of_almost_same_type = array_decay<T, U...> and almost_same_as<value_type_t<T>, value_type_t<std::decay_t<U>>...>;
    template<typename T, typename... U> concept readable_array_decay_of_almost_same_type = readable_array_decay<T, U...> and almost_same_as<value_type_t<T>, value_type_t<std::decay_t<U>>...>;
    template<typename T, typename... U> concept writable_array_decay_of_almost_same_type = writable_array_decay<T, U...> and almost_same_as<value_type_t<T>, value_type_t<std::decay_t<U>>...>;

    template<typename T, typename... U> concept array_with_compatible_types = array<T, U...> and (compatible_types<value_type_t<T>, value_type_t<U>> and ...);
    template<typename T, typename... U> concept array_decay_with_compatible_types = array_decay<T, U...> and (compatible_types<value_type_t<T>, value_type_t<U>> and ...);

    template<typename T, typename... U> concept array_with_spectrum_types = array<T, U...> and (spectrum_types<value_type_t<T>, value_type_t<U>> and ...);
    template<typename T, typename... U> concept array_decay_with_spectrum_types = array_decay<T, U...> and (spectrum_types<value_type_t<T>, value_type_t<U>> and ...);

    template<typename T, typename... U> concept array_with_compatible_or_spectrum_types = array<T, U...> and (compatible_or_spectrum_types<value_type_t<T>, value_type_t<U>> and ...);
    template<typename T, typename... U> concept array_decay_with_compatible_or_spectrum_types = array_decay<T, U...> and (compatible_or_spectrum_types<value_type_t<T>, value_type_t<U>> and ...);

    template<typename T, typename U> concept array_compatible_with = array<T> and compatible_types<value_type_t<T>, value_type_t<U>>;
    template<typename T, typename U> concept array_decay_compatible_with = array_decay<T> and compatible_types<value_type_t<T>, value_type_t<U>>;
}
