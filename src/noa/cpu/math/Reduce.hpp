#pragma once

#include <algorithm>

#include "noa/core/Types.hpp"

namespace noa::cpu::math::details {
    template<typename T>
    constexpr bool is_valid_min_max_median_v =
            noa::traits::is_any_v<T, i16, i32, i64, u16, u32, u64, f16, f32, f64>;

    template<typename T, typename PreProcessOp = noa::copy_t>
    constexpr bool is_valid_sum_mean_v =
            noa::traits::is_any_v<T, i32, i64, u32, u64, f32, f64, c32, c64> &&
            noa::traits::is_any_v<PreProcessOp,
                                  noa::copy_t, noa::nonzero_t, noa::square_t,
                                  noa::abs_t, noa::abs_squared_t>;

    // invoke_result could be used, but I think noa::nonzero_t would return bool and not T.
    template<typename T, typename PreProcessOp>
    using sum_mean_return_t = std::conditional_t<
            noa::traits::is_any_v<PreProcessOp, noa::copy_t, noa::square_t>,
            T, noa::traits::value_type_t<T>>;

    template<typename T, typename U>
    constexpr bool is_valid_var_std_v =
            noa::traits::is_any_v<T, f32, f64, c32, c64> &&
            std::is_same_v<U, noa::traits::value_type_t<T>>;
}

namespace noa::cpu::math {
    template<typename Value, typename = std::enable_if_t<details::is_valid_min_max_median_v<Value>>>
    [[nodiscard]] Value min(const Value* input,
                            const Strides4<i64>& strides,
                            const Shape4<i64>& shape,
                            i64 threads);

    template<typename Value, typename = std::enable_if_t<details::is_valid_min_max_median_v<Value>>>
    [[nodiscard]] Value max(const Value* input,
                            const Strides4<i64>& strides,
                            const Shape4<i64>& shape,
                            i64 threads);

    template<typename Value, typename PreProcessOp,
             typename Reduced = details::sum_mean_return_t<Value, PreProcessOp>,
             typename = std::enable_if_t<details::is_valid_sum_mean_v<Value, PreProcessOp>>>
    [[nodiscard]] Reduced sum(const Value* input,
                              const Strides4<i64>& strides,
                              const Shape4<i64>& shape,
                              PreProcessOp pre_process_op,
                              i64 threads);

    template<typename Value, typename PreProcessOp,
             typename Reduced = details::sum_mean_return_t<Value, PreProcessOp>,
             typename = std::enable_if_t<details::is_valid_sum_mean_v<Value, PreProcessOp>>>
    [[nodiscard]] Reduced mean(const Value* input,
                               const Strides4<i64>& strides,
                               const Shape4<i64>& shape,
                               PreProcessOp pre_process_op,
                               i64 threads);

    template<typename Input, typename Output = noa::traits::value_type_t<Input>,
             typename = std::enable_if_t<details::is_valid_var_std_v<Input, Output>>>
    [[nodiscard]] Output var(const Input* input,
                             const Strides4<i64>& strides,
                             const Shape4<i64>& shape,
                             i64 ddof, i64 threads);

    template<typename Input, typename Output = noa::traits::value_type_t<Input>,
             typename = std::enable_if_t<details::is_valid_var_std_v<Input, Output>>>
    [[nodiscard]] auto mean_var(const Input* input,
                                const Strides4<i64>& strides,
                                const Shape4<i64>& shape,
                                i64 ddof, i64 threads) -> std::pair<Input, Output>;

    template<typename Input, typename Output = noa::traits::value_type_t<Input>,
             typename = std::enable_if_t<details::is_valid_var_std_v<Input, Output>>>
    [[nodiscard]] Output std(const Input* input,
                             const Strides4<i64>& strides,
                             const Shape4<i64>& shape,
                             i64 ddof, i64 threads);

    template<typename Value, typename = std::enable_if_t<details::is_valid_min_max_median_v<Value>>>
    [[nodiscard]] Value median(Value* input,
                               Strides4<i64> strides,
                               Shape4<i64> shape,
                               bool overwrite);

    template<typename Value, typename = std::enable_if_t<noa::traits::is_any_v<Value, f32, f64>>>
    [[nodiscard]] Value rmsd(const Value* lhs, const Strides4<i64>& lhs_strides,
                             const Value* rhs, const Strides4<i64>& rhs_strides,
                             const Shape4<i64>& shape, i64 threads);
}

namespace noa::cpu::math {
    template<typename Value, typename = std::enable_if_t<details::is_valid_min_max_median_v<Value>>>
    void min(const Value* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
             Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
             i64 threads);

    template<typename Value, typename = std::enable_if_t<details::is_valid_min_max_median_v<Value>>>
    void max(const Value* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
             Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
             i64 threads);

    template<typename Value, typename = std::enable_if_t<details::is_valid_sum_mean_v<Value>>>
    void sum(const Value* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
             Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
             i64 threads);

    template<typename Value, typename = std::enable_if_t<details::is_valid_sum_mean_v<Value>>>
    void mean(const Value* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
              Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
              i64 threads);

    template<typename Input, typename Output, typename = std::enable_if_t<details::is_valid_var_std_v<Input, Output>>>
    void var(const Input* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
             Output* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
             i64 ddof, i64 threads);

    template<typename Input, typename Output, typename = std::enable_if_t<details::is_valid_var_std_v<Input, Output>>>
    void std(const Input* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
             Output* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
             i64 ddof, i64 threads);
}
