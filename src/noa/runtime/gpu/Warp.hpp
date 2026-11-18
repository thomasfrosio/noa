#pragma once
#include "noa/runtime/gpu/IncludeGuard.hpp"

#include "noa/base/Config.hpp"
#include "noa/base/Complex.hpp"
#include "noa/runtime/core/Interfaces.hpp"
#include "noa/runtime/core/Accessor.hpp"
#include "noa/runtime/gpu/Constants.hpp"

namespace noa::traits {
    NOA_GENERATE_PROCLAIM_FULL(warp_reduce_add);
    template<> struct proclaim_is_warp_reduce_add<Plus> : std::true_type {};
    template<> struct proclaim_is_warp_reduce_add<PlusEq> : std::true_type {};
    template<typename T> struct proclaim_is_warp_reduce_add<std::plus<T>> : std::true_type {};

    NOA_GENERATE_PROCLAIM_FULL(warp_reduce_min);
    template<> struct proclaim_is_warp_reduce_min<Min> : std::true_type {};
    template<> struct proclaim_is_warp_reduce_min<MinEq> : std::true_type {};

    NOA_GENERATE_PROCLAIM_FULL(warp_reduce_max);
    template<> struct proclaim_is_warp_reduce_max<Max> : std::true_type {};
    template<> struct proclaim_is_warp_reduce_max<MaxEq> : std::true_type {};

    NOA_GENERATE_PROCLAIM_FULL(warp_reduce_bit_and);
    template<> struct proclaim_is_warp_reduce_bit_and<BitAnd> : std::true_type {};
    template<> struct proclaim_is_warp_reduce_bit_and<BitAndEq> : std::true_type {};
    template<typename T> struct proclaim_is_warp_reduce_bit_and<std::bit_and<T>> : std::true_type {};

    NOA_GENERATE_PROCLAIM_FULL(warp_reduce_bit_or);
    template<> struct proclaim_is_warp_reduce_bit_or<BitOr> : std::true_type {};
    template<> struct proclaim_is_warp_reduce_bit_or<BitOrEq> : std::true_type {};
    template<typename T> struct proclaim_is_warp_reduce_bit_or<std::bit_or<T>> : std::true_type {};

    NOA_GENERATE_PROCLAIM_FULL(warp_reduce_bit_xor);
    template<> struct proclaim_is_warp_reduce_bit_or<BitXOr> : std::true_type {};
    template<> struct proclaim_is_warp_reduce_bit_or<BitXOrEq> : std::true_type {};
    template<typename T> struct proclaim_is_warp_reduce_bit_or<std::bit_xor<T>> : std::true_type {};

    template<typename T>
    concept warp_shuffable = std::is_trivially_copyable_v<T>;
}

#if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
namespace noa::gpu::details {
    template<nt::warp_shuffable T>
    NOA_FD auto warp_shuffle_down(T value, u32 delta) -> T {
        constexpr u32 ALL_MASK = 0xffffffff;
        constexpr u32 WIDTH = Constant::WARP_SIZE;

        if constexpr (nt::any_of<T, int, unsigned int, long, unsigned long, long long, unsigned long long, float, double, __half, __half2>) {
            return __shfl_down_sync(ALL_MASK, value, delta, WIDTH);
        } else if constexpr (nt::same_as<T, f16>) {
            return f16(__shfl_down_sync(ALL_MASK, value.native(), delta, WIDTH));
        } else if constexpr (nt::same_as<T, c16>) {
            auto half2 = reinterpret_cast<__half2*>(&value);
            *half2 = __shfl_down_sync(ALL_MASK, *half2, delta, WIDTH);
            return *reinterpret_cast<c16*>(half2);
        } else if constexpr (nt::any_of<T, c32, c64>) {
            return T(__shfl_down_sync(ALL_MASK, value.real, delta, WIDTH),
                     __shfl_down_sync(ALL_MASK, value.imag, delta, WIDTH));
        } else if constexpr (nt::same_as<T, bool> or (nt::integer<T> and sizeof(T) < 4)) { // TODO redundant branch
            return static_cast<T>(__shfl_down_sync(ALL_MASK, static_cast<i32>(value), delta, WIDTH));
        } else {
            // Use generic implementation (which should compile to pretty much the shuffle down) for unknown types.
            // Same as cub::ShuffleDown, and equivalent to AdaptiveCpp.
            static constexpr bool IS_INT_MULTIPLE = (sizeof(T) % sizeof(i32) == 0) && (alignof(T) % alignof(i32) == 0);
            static constexpr bool IS_SHORT_MULTIPLE = (sizeof(T) % sizeof(i16) == 0) && (alignof(T) % alignof(i16) == 0);
            using shuffle_word_t =
                std::conditional_t<IS_INT_MULTIPLE, u32,
                std::conditional_t<IS_SHORT_MULTIPLE, u16, u8>>;

            constexpr i32 WORDS = (sizeof(T) + sizeof(shuffle_word_t) - 1) / sizeof(shuffle_word_t);

            T output;
            shuffle_word_t* output_alias = reinterpret_cast<shuffle_word_t*>(&output);
            shuffle_word_t* input_alias  = reinterpret_cast<shuffle_word_t*>(&value);

            auto shuffle_word = __shfl_down_sync(ALL_MASK, static_cast<u32>(input_alias[0]), delta, WIDTH);
            output_alias[0] = static_cast<shuffle_word_t>(shuffle_word);

            #pragma unroll
            for (int word = 1; word < WORDS; ++word) {
                shuffle_word = __shfl_down_sync(ALL_MASK, static_cast<u32>(input_alias[word]), delta, WIDTH);
                output_alias[word] = static_cast<shuffle_word_t>(shuffle_word);
            }

            return output;
        }
    }

    template<typename BinaryOp, nt::warp_shuffable T>
    NOA_ID auto warp_reduce(BinaryOp op, T value) -> T {
        // Fast paths using hardware reductions.
        NOA_IF_TARGET_SM80((
            if constexpr (nt::warp_reduce_add<BinaryOp> and nt::any_of<T, i32, u32>)
                return __reduce_add_sync(0xffffffff, value);
            else if constexpr (nt::warp_reduce_min<BinaryOp> and nt::any_of<T, i32, u32>)
                return __reduce_min_sync(0xffffffff, value);
            else if constexpr (nt::warp_reduce_max<BinaryOp> and nt::any_of<T, i32, u32>)
                return __reduce_max_sync(0xffffffff, value);
            else if constexpr (nt::warp_reduce_bit_and<BinaryOp> and nt::any_of<T, i32, u32>)
                return static_cast<T>(__reduce_and_sync(0xffffffff, static_cast<u32>(value)));
            else if constexpr (nt::warp_reduce_bit_or<BinaryOp> and nt::any_of<T, i32, u32>)
                return static_cast<T>(__reduce_or_sync(0xffffffff, static_cast<u32>(value)));
            else if constexpr (nt::warp_reduce_bit_xor<BinaryOp> and nt::any_of<T, i32, u32>)
                return static_cast<T>(__reduce_xor_sync(0xffffffff, static_cast<u32>(value)));
        ), ())

        // Slower path using shuffle down.
        #pragma unroll
        for (u32 delta = 1; delta < Constant::WARP_SIZE; delta *= 2)
            value = op(value, warp_shuffle_down(value, delta));
        return value;
    }

    template<typename Interface, typename Op, nt::tuple_of_accessor_value Reduced>
    NOA_ID auto warp_reduce(Op op, Reduced reduced) -> Reduced {
        NOA_IF_TARGET_SM80((
            // Fast paths using hardware reductions.
            if constexpr (nt::has_join_type_v<Op>) {
                using join_t = nt::join_type_t<Op>;
                using ai32 = AccessorValue<i32>;
                using au32 = AccessorValue<u32>;
                if constexpr (nt::warp_reduce_add<join_t> and nt::tuple_value_type_any_of<Reduced, ai32, au32>) {
                    return reduced.map([&]<typename T>(T& accessor_value) {
                        return T(__reduce_add_sync(0xffffffff, accessor_value.ref()));
                    });
                } else if constexpr (nt::warp_reduce_min<join_t> and nt::tuple_value_type_any_of<Reduced, ai32, au32>) {
                    return reduced.map([&]<typename T>(T& accessor_value) {
                        return T(__reduce_min_sync(0xffffffff, accessor_value.ref()));
                    });
                } else if constexpr (nt::warp_reduce_max<join_t> and nt::tuple_value_type_any_of<Reduced, ai32, au32>) {
                    return reduced.map([&]<typename T>(T& accessor_value) {
                        return T(__reduce_max_sync(0xffffffff, accessor_value.ref()));
                    });
                } else if constexpr (nt::warp_reduce_bit_and<join_t> and nt::tuple_value_type_any_of<Reduced, ai32, au32>) {
                    return reduced.map([&]<typename T>(T& accessor_value) {
                        using int_t = nt::mutable_value_type_t<T>;
                        return T(static_cast<int_t>(__reduce_and_sync(0xffffffff, static_cast<u32>(accessor_value.ref()))));
                    });
                } else if constexpr (nt::warp_reduce_bit_or<join_t> and nt::tuple_value_type_any_of<Reduced, ai32, au32>) {
                    return reduced.map([&]<typename T>(T& accessor_value) {
                        using int_t = nt::mutable_value_type_t<T>;
                        return T(static_cast<int_t>(__reduce_or_sync(0xffffffff, static_cast<u32>(accessor_value.ref()))));
                    });
                } else if constexpr (nt::warp_reduce_bit_xor<join_t> and nt::tuple_value_type_any_of<Reduced, ai32, au32>) {
                    return reduced.map([&]<typename T>(T& accessor_value) {
                        using int_t = nt::mutable_value_type_t<T>;
                        return T(static_cast<int_t>(__reduce_xor_sync(0xffffffff, static_cast<u32>(accessor_value.ref()))));
                    });
                }
            }
        ), ())

        // Slower path using shuffle down.
        #pragma unroll
        for (u32 delta = 1; delta < Constant::WARP_SIZE; delta *= 2) {
            Reduced reduce = reduced.map([delta]<typename T>(T& accessor_value) {
                return T(warp_shuffle_down(accessor_value.ref(), delta));
            });
            Interface::join(op, reduce, reduced);
        }
        return reduced;
    }
}
#elif defined(NOA_ENABLE_SYCL)
namespace noa::gpu::details {
    template<nt::warp_shuffable T, i32 N>
    NOA_FD auto warp_shuffle_down(const sycl::nd_item<N>& item, T value, u32 delta) -> T {
        if constexpr (nt::complex<T>) {
            return T{warp_shuffle_down(item, value.real, delta),
                     warp_shuffle_down(item, value.imag, delta)};
        } else if constexpr (nt::same_as<T, f16>) {
            return f16(sycl::shift_group_left(item.get_sub_group(), value.native(), delta));
        } else if constexpr (nt::same_as<T, bool> or (nt::integer<T> and sizeof(T) < 4)) { // TODO redundant branch
            return static_cast<T>(sycl::shift_group_left(item.get_sub_group(), static_cast<i32>(value), delta));
        } else {
            return sycl::shift_group_left(item.get_sub_group(), value, delta); // any trivially copyable type
        }
    }

    template<typename BinaryOp, nt::warp_shuffable T, i32 N>
    NOA_FD auto warp_reduce(const sycl::nd_item<N>& item, BinaryOp op, T value) -> T {
        if constexpr (nt::any_of<T, i32, u32, i64, u64, f32, f64>) {
            if constexpr (nt::warp_reduce_add<BinaryOp>) {
                value = sycl::reduce_over_group(item.get_sub_group(), value, sycl::plus<T>());
                return true;
            } else if constexpr (nt::warp_reduce_min<BinaryOp>) {
                value = sycl::reduce_over_group(item.get_sub_group(), value, sycl::minimum<T>());
                return true;
            } else if constexpr (nt::warp_reduce_max<BinaryOp>) {
                value = sycl::reduce_over_group(item.get_sub_group(), value, sycl::maximum<T>());
                return true;
            }
        }
        if constexpr (nt::any_of<T, i32, u32, i64, u64>) {
            if constexpr (nt::warp_reduce_bit_and<BinaryOp>) {
                value = sycl::reduce_over_group(item.get_sub_group(), value, sycl::bit_and<T>());
                return true;
            } else if constexpr (nt::warp_reduce_bit_or<BinaryOp>) {
                value = sycl::reduce_over_group(item.get_sub_group(), value, sycl::bit_or<T>());
                return true;
            } else if constexpr (nt::warp_reduce_bit_xor<BinaryOp>) {
                value = sycl::reduce_over_group(item.get_sub_group(), value, sycl::bit_xor<T>());
                return true;
            }
        }

        #pragma unroll
        for (u32 delta = 1; delta < Constant::WARP_SIZE; delta *= 2)
            value = op(value, warp_shuffle_down(item, value, delta));
        return value;
    }

    template<typename Interface, typename Op, nt::tuple_of_accessor_value Reduced, i32 N>
    NOA_FD auto warp_reduce(const sycl::nd_item<N>& item, Op op, Reduced reduced) -> Reduced {
        if constexpr (nt::has_join_type_v<Op>) {
            using join_t = nt::join_type_t<Op>;
            using ai32 = AccessorValue<i32>;
            using au32 = AccessorValue<u32>;
            using ai64 = AccessorValue<i64>;
            using au64 = AccessorValue<u64>;
            using af32 = AccessorValue<f32>;
            using af64 = AccessorValue<f64>;
            if constexpr (nt::warp_reduce_add<join_t> and nt::tuple_value_type_any_of<Reduced, ai32, au32, ai64, au64, af32, af64>) {
                const auto sg = item.get_sub_group();
                return reduced.map([&]<typename T>(T& v) {
                    return T(sycl::reduce_over_group(sg, v.ref(), sycl::plus{}));
                });
            } else if constexpr (nt::warp_reduce_min<join_t> and nt::tuple_value_type_any_of<Reduced, ai32, au32, ai64, au64, af32, af64>) {
                const auto sg = item.get_sub_group();
                return reduced.map([&]<typename T>(T& v) {
                    return T(sycl::reduce_over_group(sg, v.ref(), sycl::minimum{}));
                });
            } else if constexpr (nt::warp_reduce_max<join_t> and nt::tuple_value_type_any_of<Reduced, ai32, au32, ai64, au64, af32, af64>) {
                const auto sg = item.get_sub_group();
                return reduced.map([&]<typename T>(T& v) {
                    return T(sycl::reduce_over_group(sg, v.ref(), sycl::maximum{}));
                });
            } else if constexpr (nt::warp_reduce_bit_and<join_t> and nt::tuple_value_type_any_of<Reduced, ai32, au32, ai64, au64>) {
                const auto sg = item.get_sub_group();
                return reduced.map([&]<typename T>(T& v) {
                    return T(sycl::reduce_over_group(sg, v.ref(), sycl::bit_and{}));
                });
            } else if constexpr (nt::warp_reduce_bit_or<join_t> and nt::tuple_value_type_any_of<Reduced, ai32, au32, ai64, au64>) {
                const auto sg = item.get_sub_group();
                return reduced.map([&]<typename T>(T& v) {
                    return T(sycl::reduce_over_group(sg, v.ref(), sycl::bit_or{}));
                });
            } else if constexpr (nt::warp_reduce_bit_xor<join_t> and nt::tuple_value_type_any_of<Reduced, ai32, au32, ai64, au64>) {
                const auto sg = item.get_sub_group();
                return reduced.map([&]<typename T>(T& v) {
                    return T(sycl::reduce_over_group(sg, v.ref(), sycl::bit_xor{}));
                });
            }
        }

        #pragma unroll
        for (u32 delta = 1; delta < Constant::WARP_SIZE; delta *= 2) {
            Reduced reduce = reduced.map([&, delta]<typename T>(T& v) {
                return T(warp_shuffle_down(item, v.ref(), delta));
            });
            Interface::join(op, reduce, reduced);
        }
        return reduced;
    }
}
#endif
