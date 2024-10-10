#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/types/Vec.hpp"
#include "noa/gpu/cuda/Runtime.hpp"

namespace noa::cuda::guts {
    // TODO Add a way for the users to add a specialisation for wrap_reduce with their own types?
    template<typename T>
    concept shuffable = nt::numeric<T> or nt::any_of<T, half, half2>;

    template<shuffable T>
    NOA_FD T warp_shuffle(T value, i32 source, i32 width = 32, u32 mask = 0xffffffff) {
        if constexpr (nt::almost_same_as<c16, T>) {
            __half2 tmp = __shfl_sync(mask, *reinterpret_cast<__half2*>(&value), source, width);
            return *reinterpret_cast<c16*>(&tmp);
        } else if constexpr (nt::same_as<f16, T>) {
            return T(__shfl_sync(mask, value.native(), source, width));
        } else if constexpr (nt::complex<T>) {
            return T(__shfl_sync(mask, value.real, source, width),
                     __shfl_sync(mask, value.imag, source, width));
        } else if (nt::integer<T> and sizeof(T) < 4) {
            return static_cast<T>(__shfl_sync(mask, static_cast<i32>(value), source, width));
        } else {
            return __shfl_sync(mask, value, source, width);
        }
    }

    template<shuffable T>
    NOA_FD T warp_suffle_down(T value, u32 delta, i32 width = 32, u32 mask = 0xffffffff) {
        if constexpr (nt::almost_same_as<c16, T>) {
            __half2 tmp = __shfl_down_sync(mask, *reinterpret_cast<__half2*>(&value), delta, width);
            return *reinterpret_cast<c16*>(&tmp);
        } else if constexpr (nt::same_as<f16, T>) {
            return T(__shfl_down_sync(mask, value.native(), delta, width));
        } else if constexpr (nt::complex<T>) {
            return T(__shfl_down_sync(mask, value.real, delta, width),
                     __shfl_down_sync(mask, value.imag, delta, width));
        } else if (nt::integer<T> and sizeof(T) < 4) {
            return static_cast<T>(__shfl_down_sync(mask, static_cast<i32>(value), delta, width));
        } else {
            return __shfl_down_sync(mask, value, delta, width);
        }
    }

    template<typename T>
    struct has_warp_reduce : std::bool_constant<shuffable<T>> {};

    template<typename... T>
    struct has_warp_reduce<Tuple<T...>> {
        static constexpr bool value = (shuffable<nt::value_type_t<T>> and ...);
    };

    template<typename T>
    concept wrap_reduceable = has_warp_reduce<T>::value;

    // Reduces one warp to one element.
    // The first thread of the warp returns with the reduced value.
    // The returned value is undefined for the other threads.
    template<typename BinaryOp, shuffable T>
    NOA_ID auto warp_reduce(BinaryOp op, T value) -> T {
        T reduce;
        for (i32 delta = 1; delta < 32; delta *= 2) {
            reduce = warp_suffle_down(value, delta);
            value = op(reduce, value);
        }
        return value;
    }

    // Reduces one warp to one element.
    // The first thread of the warp returns with the reduced value.
    // The returned value is undefined for the other threads.
    template<typename Interface, typename Op, nt::tuple_of_accessor_value Reduced>
    requires wrap_reduceable<Reduced>
    NOA_ID auto warp_reduce(Op op, Reduced reduced) -> Reduced {
        for (i32 delta = 1; delta < 32; delta *= 2) {
            Reduced reduce = reduced.map([delta]<typename T>(T& accessor_value) {
                // TODO If int/float, there's a faster reduction, which I forgot the name...
                return T(warp_suffle_down(accessor_value.ref(), delta));
            });
            Interface::join(op, reduce, reduced);
        }
        return reduced;
    }
}
