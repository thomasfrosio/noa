#pragma once

#include "noa/runtime/Traits.hpp"
#include "noa/xform/core/Traits.hpp"

namespace noa::traits {
    NOA_GENERATE_PROCLAIM_FULL(texture);
    namespace details {
        template<texture T> struct array_size<T> { static constexpr usize value = T::SIZE; };
        template<texture T> struct array_ssize<T> { static constexpr isize value = T::SSIZE; };

        template<typename> struct texture_size { static constexpr usize value = 0; };
        template<typename> struct texture_ssize { static constexpr isize value = 0; };
        template<texture T> struct texture_size<T> { static constexpr usize value = T::SIZE; };
        template<texture T> struct texture_ssize<T> { static constexpr isize value = T::SSIZE; };

        template<typename> struct texture_rank { static constexpr usize value = 0; };
        template<texture T> struct texture_rank<T> { static constexpr usize value = T::RANK; };
    }
    template<typename T> constexpr usize texture_size_v = details::texture_size<std::decay_t<T>>::value;
    template<typename T> constexpr usize texture_ssize_v = details::texture_ssize<std::decay_t<T>>::value;
    template<typename T> constexpr usize texture_rank_v = details::texture_rank<std::decay_t<T>>::value;

    template<typename T> using is_texture_decay = is_texture<std::decay_t<T>>;
    NOA_GENERATE_PROCLAIM_UTILS(texture_decay);
    template<typename... T> concept texture_decay = are_texture_decay_v<T...>;

    template<typename T, usize... N> concept texture_nd = texture<T> and ((std::remove_reference_t<T>::SIZE == N) or ...);
    template<typename T, usize... R> concept texture_rd = texture<T> and ((std::remove_reference_t<T>::RANK == R) or ...);
    template<typename T, usize... N> concept texture_decay_nd = texture_decay<T> and ((std::remove_reference_t<T>::SIZE == N) or ...);
    template<typename T, usize... R> concept texture_decay_rd = texture_decay<T> and ((std::remove_reference_t<T>::RANK == R) or ...);

    #define NOA_TRAITS_GENERATE_TEXTURE(name)                                                           \
    template<typename... T> concept texture_of_##name = texture<T...> and name<value_type_t<T>...>;     \
    template<typename... T> concept texture_decay_of_##name = texture_decay<T...> and name<value_type_t<T>...>
    NOA_TRAITS_GENERATE_TEXTURE(real);
    NOA_TRAITS_GENERATE_TEXTURE(complex);
    NOA_TRAITS_GENERATE_TEXTURE(real_or_complex);
    #undef NOA_TRAITS_GENERATE_TEXTURE

    #define NOA_TRAITS_GENERATE_TEXTURE_SAME_AS(suffix, constrain)                                                                  \
    template<typename T, typename... U> concept texture_of_##suffix = texture<T> and constrain<value_type_t<T>, U...>;              \
    template<typename T, typename... U> concept texture_decay_of_##suffix = texture_decay<T> and constrain<value_type_t<T>, U...>;  \
    template<typename T, typename... U> concept texture_or_value_of_##suffix = texture_of_##suffix<T, U...> or constrain<T, U...>;  \
    template<typename T, typename... U> concept texture_decay_or_value_of_##suffix = texture_decay_of_##suffix<T, U...> or constrain<T, U...>
    NOA_TRAITS_GENERATE_TEXTURE_SAME_AS(any, any_of);
    NOA_TRAITS_GENERATE_TEXTURE_SAME_AS(almost_any, almost_any_of);
    #undef NOA_TRAITS_GENERATE_TEXTURE_SAME_AS

    template<typename... T>
    concept array_or_texture = ((array<T> or texture<T>) and ...);

    template<typename... T>
    concept array_or_texture_decay = ((array_decay<T> or texture_decay<T>) and ...);

    template<typename T, usize R>
    concept readable_array_or_texture_rd_decay = (readable_array_decay<T> or texture_decay_rd<T, R>) and nt::array_size_v<T> >= R;

    template<typename T, typename... U>
    concept array_or_texture_with_spectrum_types =
        (array<T, U...> or (texture<T> and array<U...>)) and
        (spectrum_types<value_type_t<T>, value_type_t<U>> and ...);

    template<typename T, typename... U>
    concept array_or_texture_decay_with_spectrum_types =
        (array_decay<T, U...> or (texture_decay<T> and array_decay<U...>)) and
        (spectrum_types<value_type_t<T>, value_type_t<U>> and ...);

    #define NOA_TRAITS_GENERATE_VARRAY_OR_TEXTURE(name)                                                                 \
    template<typename... T> concept array_or_texture_of_##name = array_or_texture<T...> and name<value_type_t<T>...>;   \
    template<typename... T> concept array_or_texture_decay_of_##name = array_or_texture_decay<T...> and name<value_type_t<T>...>
    NOA_TRAITS_GENERATE_VARRAY_OR_TEXTURE(real);
    NOA_TRAITS_GENERATE_VARRAY_OR_TEXTURE(complex);
    NOA_TRAITS_GENERATE_VARRAY_OR_TEXTURE(real_or_complex);
    #undef NOA_TRAITS_GENERATE_VARRAY_OR_TEXTURE
}
