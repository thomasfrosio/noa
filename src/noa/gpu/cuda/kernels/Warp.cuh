#pragma once

#include "noa/core/Config.hpp"
#include "noa/gpu/cuda/Types.hpp"

namespace noa::cuda::guts {
    // TODO Add a way for the users to add a specialisation for wrap_reduce with their own types?
    template<typename T>
    constexpr bool is_valid_suffle_v = nt::is_numeric_v<T> or nt::is_any_v<T, half, half2>;

    template<typename T> requires guts::is_valid_suffle_v<T>
    NOA_FD T warp_shuffle(T value, i32 source, i32 width = 32, u32 mask = 0xffffffff) {
        if constexpr (nt::is_almost_same_v<c16, T>) {
            __half2 tmp = __shfl_sync(mask, *reinterpret_cast<__half2*>(&value), source, width);
            return *reinterpret_cast<c16*>(&tmp);
        } else if constexpr (std::is_same_v<f16, T>) {
            return T(__shfl_sync(mask, value.native(), source, width));
        } else if constexpr (nt::is_complex_v<T>) {
            return T(__shfl_sync(mask, value.real, source, width),
                     __shfl_sync(mask, value.imag, source, width));
        } else if (nt::is_int_v<T> && sizeof(T) < 4) {
            return static_cast<T>(__shfl_sync(mask, static_cast<i32>(value), source, width));
        } else {
            return __shfl_sync(mask, value, source, width);
        }
    }

    template<typename T> requires guts::is_valid_suffle_v<T>
    NOA_FD T warp_suffle_down(T value, u32 delta, i32 width = 32, u32 mask = 0xffffffff) {
        if constexpr (nt::is_almost_same_v<c16, T>) {
            __half2 tmp = __shfl_down_sync(mask, *reinterpret_cast<__half2*>(&value), delta, width);
            return *reinterpret_cast<c16*>(&tmp);
        } else if constexpr (std::is_same_v<f16, T>) {
            return T(__shfl_down_sync(mask, value.native(), delta, width));
        } else if constexpr (nt::is_complex_v<T>) {
            return T(__shfl_down_sync(mask, value.real, delta, width),
                     __shfl_down_sync(mask, value.imag, delta, width));
        } else if (nt::is_int_v<T> && sizeof(T) < 4) {
            return static_cast<T>(__shfl_down_sync(mask, static_cast<i32>(value), delta, width));
        } else {
            return __shfl_down_sync(mask, value, delta, width);
        }
    }

    template<typename Tup>
    struct has_warp_reduce : std::bool_constant<is_valid_suffle_v<Tup>> {};

    template<typename... T>
    struct has_warp_reduce<Tuple<T...>> {
        static constexpr bool value = nt::bool_and<is_valid_suffle_v<typename T::value_type>...>::value;
    };

    template<typename T>
    constexpr bool has_warp_reduce_v = has_warp_reduce<T>::value;

    // Reduces one warp to one element.
    // The first thread of the warp returns with the reduced value.
    // The returned value is undefined for the other threads.
    // value:       Per-thread value.
    // reduce_op:   Reduction operator.
    template<typename T, typename ReduceOp> requires has_warp_reduce_v<T>
    NOA_ID T warp_reduce(T value, ReduceOp reduce_op) {
        T reduce;
        for (i32 delta = 1; delta < 32; delta *= 2) {
            reduce = warp_suffle_down(value, delta);
            value = reduce_op(reduce, value);
        }
        return value;
    }

    // Overload for pairs.
    template<typename Interface, typename Op, typename Reduced>
    requires (nt::is_tuple_of_accessor_value_v<Reduced> and has_warp_reduce_v<Reduced>)
    NOA_ID auto warp_reduce(Op op, Reduced reduced) -> Reduced {
        for (i32 delta = 1; delta < 32; delta *= 2) {
            Reduced reduce = reduced.map([delta]<typename T>(T& accessor_value) {
                // TODO If int/float, there's a faster reduction, which I forgot the name...
                if constexpr (guts::is_valid_suffle_v<nt::value_type_t<T>>)
                    return T(warp_suffle_down(accessor_value.deref(), delta));
                else
                    static_assert(nt::always_false_v<T>);
            });
            Interface::join(op, reduce, reduced);
        }
        return reduced;
    }
}
