#pragma once

#include "noa/base/Traits.hpp"

namespace noa::traits {
    // Accessor
    NOA_GENERATE_PROCLAIM_FULL(accessor);

    NOA_GENERATE_PROCLAIM(accessor_restrict);
    NOA_GENERATE_PROCLAIM(accessor_contiguous);
    NOA_GENERATE_PROCLAIM(accessor_pure);
    NOA_GENERATE_PROCLAIM(accessor_reference);
    NOA_GENERATE_PROCLAIM(accessor_value);
    template<typename... T> concept accessor_restrict = accessor<T...> and are_accessor_restrict_v<T...>;
    template<typename... T> concept accessor_contiguous = accessor<T...> and are_accessor_contiguous_v<T...>;
    template<typename... T> concept accessor_pure = accessor<T...> and are_accessor_pure_v<T...>;
    template<typename... T> concept accessor_reference = accessor<T...> and are_accessor_reference_v<T...>;
    template<typename... T> concept accessor_value = accessor<T...> and are_accessor_value_v<T...>;

    NOA_GENERATE_PROCLAIM_FULL_ND(accessor_nd);
    template<typename T, usize... N> using is_accessor_pure_nd = std::conjunction<is_accessor_pure<T>, is_accessor_nd<T, N...>>;
    template<typename T, usize... N>  constexpr bool is_accessor_pure_nd_v = is_accessor_pure_nd<T, N...>::value;
    template<typename T, size_t... N> concept accessor_pure_nd = accessor_nd<T, N...> and accessor_pure<T>;
    template<size_t N, typename... T> using are_accessor_pure_nd = conjunction_or_false<is_accessor_pure_nd<T, N>...>;
    template<size_t N, typename... T> constexpr bool are_accessor_pure_nd_v = are_accessor_pure_nd<N, T...>::value;

    // Shape and Strides
    #define NOA_TRAITS_SHAPE_(name)                                                                                                         \
    NOA_GENERATE_PROCLAIM_FULL(name);                                                                                                       \
    template<typename, typename> struct proclaim_is_##name##_of_type : std::false_type {};                                                  \
    template<typename T, typename U> using is_##name##_of_type = proclaim_is_##name##_of_type<std::remove_cv_t<T>, U>;                      \
    template<typename T, typename U> constexpr bool is_##name##_of_type_v = is_##name##_of_type<T, U>::value;                               \
    template<typename T, typename U> concept name##_of_type = name<T> and is_##name##_of_type_v<T, U>;                                      \
    template<typename, size_t> struct proclaim_is_##name##_of_size : std::false_type {};                                                    \
    template<typename T, size_t... N> using is_##name##_of_size = std::disjunction<proclaim_is_##name##_of_size<std::remove_cv_t<T>, N>...>;\
    template<typename T, size_t... N> constexpr bool is_##name##_of_size_v = is_##name##_of_size<T, N...>::value;                           \
    template<typename T, size_t... N> concept name##_of_size = name<T> and is_##name##_of_size_v<T, N...>
    NOA_TRAITS_SHAPE_(shape);
    NOA_TRAITS_SHAPE_(strides);

    template<typename T> using is_shape_or_strides = std::disjunction<is_strides<T>, is_shape<T>>;
    NOA_GENERATE_PROCLAIM_UTILS(shape_or_strides);
    template<typename... T> concept shape_or_strides = conjunction_or_false<is_shape_or_strides<T>...>::value;

    template<typename T> using is_vec_shape_or_strides = std::disjunction<is_strides<T>, is_shape<T>, is_vec<T>>;
    NOA_GENERATE_PROCLAIM_UTILS(vec_shape_or_strides);
    template<typename... T> concept vec_shape_or_strides = conjunction_or_false<is_vec_shape_or_strides<T>...>::value;

    template<typename T, size_t... N> concept vec_shape_or_strides_of_size =
        vec_of_size<T, N...> or shape_of_size<T, N...> or strides_of_size<T, N...>;

    // Tuple and Pair
    NOA_GENERATE_PROCLAIM_FULL(tuple_of_accessor);
    NOA_GENERATE_PROCLAIM_FULL(tuple_of_accessor_pure);
    NOA_GENERATE_PROCLAIM_FULL(tuple_of_accessor_reference);
    NOA_GENERATE_PROCLAIM_FULL(tuple_of_accessor_value);
    NOA_GENERATE_PROCLAIM_FULL(tuple_of_accessor_or_empty);

    NOA_GENERATE_PROCLAIM_FULL_ND(tuple_of_accessor_nd);

    template<typename T, size_t... N> concept tuple_of_accessor_nd_or_empty = empty_tuple<T> or tuple_of_accessor_nd<T, N...>;
    template<typename T, size_t... N> concept tuple_of_accessor_pure_nd = tuple_of_accessor_pure<T> and tuple_of_accessor_nd<T, N...>;
    template<typename T, size_t... N> concept tuple_of_accessor_pure_nd_or_empty = empty_tuple<T> or tuple_of_accessor_pure_nd<T, N...>;

    template<typename T> using is_tuple_of_accessor_pure_or_empty = std::disjunction<is_empty_tuple<T>, is_tuple_of_accessor_pure<T>>;
    template<typename T> using is_tuple_of_accessor_value_or_empty = std::disjunction<is_empty_tuple<T>, is_tuple_of_accessor_value<T>>;
    NOA_GENERATE_PROCLAIM_UTILS(tuple_of_accessor_pure_or_empty);
    NOA_GENERATE_PROCLAIM_UTILS(tuple_of_accessor_value_or_empty);
    template<typename... T> concept tuple_of_accessor_pure_or_empty = conjunction_or_false<is_tuple_of_accessor_pure_or_empty<T>...>::value;
    template<typename... T> concept tuple_of_accessor_value_or_empty = conjunction_or_false<is_tuple_of_accessor_value_or_empty<T>...>::value;

    NOA_GENERATE_PROCLAIM_FULL(batch);

    NOA_GENERATE_PROCLAIM_FULL(span);
    NOA_GENERATE_PROCLAIM_FULL(span_contiguous);
    NOA_GENERATE_PROCLAIM_FULL_ND(span_nd);
    NOA_GENERATE_PROCLAIM_FULL_ND(span_contiguous_nd);
}

namespace noa::inline types {
    template<typename, size_t, size_t>
    class Vec;
}

namespace noa::traits {
     template<typename From, typename To>
    concept spectrum_types = nt::complex<From, To> or nt::real<From, To> or (nt::complex<From> and nt::real<To>);

    template<typename From, typename To>
    concept compatible_or_spectrum_types = (nt::complex<From> and nt::real<To>) or static_castable_to<From, To>;

    template<typename T, size_t N>
    concept indexable_nd =
        nt::integer<typename T::index_type> and
        std::convertible_to<decltype(T::SIZE), size_t> and
        std::convertible_to<decltype(std::declval<const T&>().template stride<N - 1>()), size_t>;

    template<typename T, typename... U>
    concept indexable =
        nt::integer<typename T::index_type> and
        std::convertible_to<decltype(T::SIZE), size_t> and
        ((integer<U...> and std::convertible_to<decltype(std::declval<const T&>().template stride<sizeof...(U) - 1>()), size_t>) or
         (vec_integer<U...> and std::convertible_to<decltype(std::declval<const T&>().template stride<size_or_v<U..., 0> - 1>()), size_t>));

    template<size_t N, typename... T>
    concept offset_indexing =
        (sizeof...(T) <= N and integer<T...>) or
        (sizeof...(T) == 1 and vec_integer<T...> and size_or_v<T..., 0> <= N);

    template<size_t N, typename... T>
    concept iwise_indexing =
        (sizeof...(T) == N and integer<T...>) or
        (sizeof...(T) == 1 and vec_integer<T...> and are_vec_of_size_v<N, T...>);

    template<size_t N, typename I, typename... T>
    concept iwise_core_indexing =
        (sizeof...(T) == N and same_as<I, T...>) or
        (sizeof...(T) == 1 and are_vec_of_type_v<I, T...> and are_vec_of_size_v<N, T...>);

    namespace details { // nvcc workaround
        template<typename T, size_t S, typename I, I... J>
        concept readable_nd_c =
            std::convertible_to<decltype(std::declval<const T&>()(J...)), mutable_value_type_t<T>> and
            std::convertible_to<decltype(std::declval<const T&>()(Vec<I, S, 0>{})), mutable_value_type_t<T>>;

        template<typename T, typename I, size_t... N>
        struct readable_nd_t {
            template<size_t S, I... J>
            static consteval bool has_op(std::integer_sequence<I, J...>) { return readable_nd_c<T, S, I, J...>; }
            static constexpr bool value = (has_op<N>(std::make_integer_sequence<I, N>{}) or ...);
        };

        template<typename T, size_t S, typename I, I... J>
        concept writable_nd_c =
            std::same_as<decltype(std::declval<T&>()(J...)), reference_type_t<T>> and
            std::same_as<decltype(std::declval<T&>()(Vec<I, S, 0>{})), reference_type_t<T>>;

        template<typename T, typename I, size_t... N>
        struct writable_nd_t {
            template<size_t S, I... J>
            static consteval bool has_op(std::integer_sequence<I, J...>) { return writable_nd_c<T, S, I, J...>; }
            static constexpr bool value = (has_op<N>(std::make_integer_sequence<I, N>{}) or ...);
        };

        template<typename T, size_t S, i32... J>
        concept atomic_addable_nd_c =
            pointer<decltype(std::declval<const T&>().get())> and
            same_as<decltype(std::declval<const T&>().offset_pointer(std::declval<const T&>().get(), J...)),
                    decltype(std::declval<const T&>().get())> and
            same_as<decltype(std::declval<const T&>().offset_pointer(std::declval<const T&>().get(), Vec<i32, S, 0>{})),
                    decltype(std::declval<const T&>().get())>;

        template<typename T, size_t... N>
        struct atomic_addable_nd_t {
            template<size_t S, i32... J>
            static consteval bool has_op(std::integer_sequence<i32, J...>) { return atomic_addable_nd_c<T, S, J...>; }
            static constexpr bool value = (has_op<N>(std::make_integer_sequence<i32, N>{}) or ...);
        };
    }

    template<typename T, size_t... N>
    concept readable_nd = std::copyable<std::remove_cv_t<T>> and requires {
        typename T::value_type;
        typename T::index_type;
        requires integer<decltype(T::SIZE)>;
    } and details::readable_nd_t<T, typename T::index_type, N...>::value;

    template<typename T, size_t... N>
    concept readable_nd_or_empty = readable_nd<T, N...> or nt::empty<T>;

    template<typename T, size_t... N>
    concept readable_nd_optional = readable_nd<T, N...> and static_castable_to<decltype(std::declval<const T&>()), bool>;

    template<typename T, size_t... N>
    concept writable_nd = readable_nd<T, N...> and
        not std::is_const_v<value_type_t<T>> and
        not std::is_const_v<mutable_value_type_t<T>> and
        details::writable_nd_t<T, typename T::index_type, N...>::value;

    template<typename T, size_t... N>
    concept writable_nd_optional = writable_nd<T, N...> and static_castable_to<decltype(std::declval<const T&>()), bool>;

    template<typename T, size_t... N>
    concept writable_nd_or_empty = writable_nd<T, N...> or nt::empty<T>;

    template<typename T, size_t... N>
    concept atomic_addable_nd =
        std::copyable<std::remove_cv_t<T>> and
        numeric<typename T::value_type> and
        details::atomic_addable_nd_t<T, N...>::value;

    template<typename T, size_t... N>
    concept atomic_addable_nd_optional = atomic_addable_nd<T, N...> and static_castable_to<decltype(std::declval<const T&>()), bool>;

    template<typename T, size_t... N>
    concept atomic_addable_nd_or_empty = atomic_addable_nd<T, N...> or nt::empty<T>;

    template<typename T>
    concept readable_pointer_like =
        std::copyable<std::remove_cv_t<T>> and
        std::convertible_to<decltype(std::declval<const T&>()[size_t{}]), mutable_value_type_t<T>>;

    template<typename T>
    concept writable_pointer_like =
        readable_pointer_like<T> and
        std::same_as<decltype(std::declval<T&>()[size_t{}]), reference_type_t<T>>;

    template<typename T> concept marked_contiguous = same_as<std::remove_cv_t<decltype(T::IS_CONTIGUOUS)>, bool> and T::IS_CONTIGUOUS == true;
    template<typename T> concept marked_restrict =   same_as<std::remove_cv_t<decltype(T::IS_RESTRICT)>,   bool> and T::IS_RESTRICT   == true;
}

namespace noa::traits {
    template<typename T>
    concept allocatable_type =
        not std::is_pointer_v<T> and
        not std::is_reference_v<T> and
        not std::is_const_v<T> and
        std::is_trivially_destructible_v<T>;
}
