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
    }
    template<typename T> constexpr usize texture_size_v = details::texture_size<std::decay_t<T>>::value;
    template<typename T> constexpr usize texture_ssize_v = details::texture_ssize<std::decay_t<T>>::value;

    template<typename T> using is_texture_decay = is_texture<std::decay_t<T>>;
    NOA_GENERATE_PROCLAIM_UTILS(texture_decay);
    template<typename... T> concept texture_decay = are_texture_decay_v<T...>;

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

    template<typename T, usize N, typename U = std::remove_reference_t<T>, typename V = value_type_t<T>>
    concept transform_affine_nd =
        mat_of_shape<U, N, N + 1> or
        mat_of_shape<U, N + 1, N + 1> or
        (nt::array<U> and (mat_of_shape<V, N, N + 1> or mat_of_shape<V, N + 1, N + 1>));

    template<typename T, usize N, typename U = std::remove_reference_t<T>, typename V = value_type_t<T>>
    concept transform_projection_nd =
        mat_of_shape<U, N - 1, N + 1> or
        mat_of_shape<U, N + 1, N + 1> or
        (nt::array<U> and (mat_of_shape<V, N - 1, N + 1> or mat_of_shape<V, N + 1, N + 1>));
}
