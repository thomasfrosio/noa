#pragma once

#include "noa/runtime/Traits.hpp"

namespace noa::traits {
    NOA_GENERATE_PROCLAIM_FULL(mat);

    template<typename, size_t, size_t> struct proclaim_is_mat_of_shape : std::false_type {};
    template<typename T, size_t R, size_t C> using is_mat_of_shape = proclaim_is_mat_of_shape<std::remove_cv_t<T>, R, C>;
    template<typename T, size_t R, size_t C> constexpr bool is_mat_of_shape_v = is_mat_of_shape<T, R, C>::value;
    template<typename T, size_t R, size_t C> concept mat_of_shape = mat<T> and is_mat_of_shape_v<T, R, C>;
    template<size_t R, size_t C, typename... T> using are_mat_of_shape = conjunction_or_false<is_mat_of_shape<T, R, C>...>;
    template<size_t R, size_t C, typename... T> constexpr bool are_mat_of_shape_v = are_mat_of_shape<R, C, std::decay_t<T>...>::value;

    #define NOA_TRAITS_MAT_(R, C)                                                   \
    template<typename T> using is_mat##R##C = is_mat_of_shape<T, R, C>::value;      \
    template<typename T> constexpr bool is_mat##R##C##_v = is_mat##R##C<T>::value;  \
    template<typename T> concept mat##R##C = mat_of_shape<T, R, C>
    NOA_TRAITS_MAT_(2, 2);
    NOA_TRAITS_MAT_(2, 3);
    NOA_TRAITS_MAT_(3, 3);
    NOA_TRAITS_MAT_(3, 4);
    NOA_TRAITS_MAT_(4, 4);

    NOA_GENERATE_PROCLAIM_FULL(quaternion);
    template<typename... T> concept quaternion_f32 = quaternion<T...> and same_as<f32, value_type_t<T>...>;
    template<typename... T> concept quaternion_f64 = quaternion<T...> and same_as<f64, value_type_t<T>...>;

    NOA_GENERATE_PROCLAIM_FULL(interpolator);
    template<typename T, size_t N> struct proclaim_is_interpolator_nd : std::false_type {};
    template<typename T, size_t... N> using is_interpolator_nd = std::disjunction<proclaim_is_interpolator_nd<std::remove_cv_t<T>, N>...>;
    template<typename T, size_t... N> constexpr bool is_interpolator_nd_v = is_interpolator_nd<T, N...>::value;
    template<typename T, size_t... N> concept interpolator_nd = interpolator<T> and is_interpolator_nd<T, N...>::value;
    template<size_t N, typename... T> using are_interpolator_nd = conjunction_or_false<is_interpolator_nd<T, N>...>;
    template<size_t N, typename... T> constexpr bool are_interpolator_nd_v = are_interpolator_nd<N, T...>::value;

    NOA_GENERATE_PROCLAIM_FULL(interpolator_spectrum);
    template<typename T, size_t N> struct proclaim_is_interpolator_spectrum_nd : std::false_type {};
    template<typename T, size_t... N> using is_interpolator_spectrum_nd = std::disjunction<proclaim_is_interpolator_spectrum_nd<std::remove_cv_t<T>, N>...>;
    template<typename T, size_t... N> constexpr bool is_interpolator_spectrum_nd_v = is_interpolator_spectrum_nd<T, N...>::value;
    template<typename T, size_t... N> concept interpolator_spectrum_nd = interpolator_spectrum<T> and is_interpolator_spectrum_nd<T, N...>::value;
    template<size_t N, typename... T> using are_interpolator_spectrum_nd = conjunction_or_false<is_interpolator_spectrum_nd<T, N>...>;
    template<size_t N, typename... T> constexpr bool are_interpolator_spectrum_nd_v = are_interpolator_spectrum_nd<N, T...>::value;

    template<typename... T> concept interpolator_or_empty = conjunction_or_false<std::disjunction<is_interpolator<T>, std::is_empty<T>>...>::value;
    template<typename... T> concept interpolator_spectrum_or_empty = conjunction_or_false<std::disjunction<is_interpolator_spectrum<T>, std::is_empty<T>>...>::value;
    template<typename T, size_t... N> concept interpolator_nd_or_empty = interpolator_nd<T, N...> or empty<T>;
    template<typename T, size_t... N> concept interpolator_spectrum_nd_or_empty = interpolator_spectrum_nd<T, N...> or empty<T>;
}

namespace noa::traits {
    NOA_GENERATE_PROCLAIM_FULL(texture);

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
    concept varray_or_texture = ((varray<T> or texture<T>) and ...);

    template<typename... T>
    concept varray_or_texture_decay = ((varray_decay<T> or texture_decay<T>) and ...);

    template<typename T, typename... U>
    concept varray_or_texture_with_spectrum_types =
        (varray<T, U...> or (texture<T> and varray<U...>)) and
        (spectrum_types<value_type_t<T>, value_type_t<U>> and ...);

    template<typename T, typename... U>
    concept varray_or_texture_decay_with_spectrum_types =
        (varray_decay<T, U...> or (texture_decay<T> and varray_decay<U...>)) and
        (spectrum_types<value_type_t<T>, value_type_t<U>> and ...);

    #define NOA_TRAITS_GENERATE_VARRAY_OR_TEXTURE(name)                                                         \
    template<typename... T> concept varray_or_texture_of_##name = varray_or_texture<T...> and name<value_type_t<T>...>;    \
    template<typename... T> concept varray_or_texture_decay_of_##name = varray_or_texture_decay<T...> and name<value_type_t<T>...>
    NOA_TRAITS_GENERATE_VARRAY_OR_TEXTURE(real);
    NOA_TRAITS_GENERATE_VARRAY_OR_TEXTURE(complex);
    NOA_TRAITS_GENERATE_VARRAY_OR_TEXTURE(real_or_complex);
    #undef NOA_TRAITS_GENERATE_VARRAY_OR_TEXTURE

    template<typename T, usize N, typename U = std::remove_reference_t<T>, typename V = value_type_t<T>>
    concept transform_affine_nd =
        mat_of_shape<U, N, N + 1> or
        mat_of_shape<U, N + 1, N + 1> or
        (nt::varray<U> and (mat_of_shape<V, N, N + 1> or mat_of_shape<V, N + 1, N + 1>));

    template<typename T, usize N, typename U = std::remove_reference_t<T>, typename V = value_type_t<T>>
    concept transform_projection_nd =
        mat_of_shape<U, N - 1, N + 1> or
        mat_of_shape<U, N + 1, N + 1> or
        (nt::varray<U> and (mat_of_shape<V, N - 1, N + 1> or mat_of_shape<V, N + 1, N + 1>));
}
