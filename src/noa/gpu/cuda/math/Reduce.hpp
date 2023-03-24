#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Stream.hpp"

namespace noa::cuda::math::details {
    template<typename T>
    constexpr bool is_valid_min_max_median_v =
            noa::traits::is_any_v<T, i16, i32, i64, u16, u32, u64, f16, f32, f64>;

    template<typename T>
    constexpr bool is_valid_sum_mean_v =
            traits::is_any_v<T, i32, i64, u32, u64, f32, f64, c32, c64>;

    template<typename T, typename U>
    constexpr bool is_valid_var_std_v =
            traits::is_any_v<T, f32, f64, c32, c64> &&
            std::is_same_v<U, traits::value_type_t<T>>;
}

namespace noa::cuda::math {
    template<typename Value, typename = std::enable_if_t<details::is_valid_min_max_median_v<Value>>>
    [[nodiscard]] Value min(const Value* input,
                            const Strides4<i64>& strides,
                            const Shape4<i64>& shape,
                            Stream& stream);

    template<typename Value, typename = std::enable_if_t<details::is_valid_min_max_median_v<Value>>>
    [[nodiscard]] Value max(const Value* input,
                            const Strides4<i64>& strides,
                            const Shape4<i64>& shape,
                            Stream& stream);

    template<typename Value, typename = std::enable_if_t<details::is_valid_sum_mean_v<Value>>>
    [[nodiscard]] Value sum(const Value* input,
                            const Strides4<i64>& strides,
                            const Shape4<i64>& shape,
                            Stream& stream);

    template<typename Value, typename = std::enable_if_t<details::is_valid_sum_mean_v<Value>>>
    [[nodiscard]] Value mean(const Value* input,
                             const Strides4<i64>& strides,
                             const Shape4<i64>& shape,
                             Stream& stream);

    template<typename Input, typename Output = noa::traits::value_type_t<Input>,
             typename = std::enable_if_t<details::is_valid_var_std_v<Input, Output>>>
    [[nodiscard]] Output var(const Input* input,
                             const Strides4<i64>& strides,
                             const Shape4<i64>& shape,
                             i64 ddof, Stream& stream);

    template<typename Input, typename Output = noa::traits::value_type_t<Input>,
             typename = std::enable_if_t<details::is_valid_var_std_v<Input, Output>>>
    [[nodiscard]] auto mean_var(const Input* input,
                                const Strides4<i64>& strides,
                                const Shape4<i64>& shape,
                                i64 ddof, Stream& stream) -> std::pair<Input, Output>;

    template<typename Input, typename Output = noa::traits::value_type_t<Input>,
             typename = std::enable_if_t<details::is_valid_var_std_v<Input, Output>>>
    [[nodiscard]] Output std(const Input* input,
                             const Strides4<i64>& strides,
                             const Shape4<i64>& shape,
                             i64 ddof, Stream& stream);

    template<typename Value, typename = std::enable_if_t<details::is_valid_min_max_median_v<Value>>>
    [[nodiscard]] Value median(Value* input,
                               Strides4<i64> strides,
                               Shape4<i64> shape,
                               bool overwrite,
                               Stream& stream);
}

namespace noa::cuda::math {
    template<typename Value, typename = std::enable_if_t<details::is_valid_min_max_median_v<Value>>>
    void min(const Value* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
             Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
             Stream& stream);

    template<typename Value, typename = std::enable_if_t<details::is_valid_min_max_median_v<Value>>>
    void max(const Value* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
             Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
             Stream& stream);

    template<typename Value, typename = std::enable_if_t<details::is_valid_sum_mean_v<Value>>>
    void sum(const Value* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
             Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
             Stream& stream);

    template<typename Value, typename = std::enable_if_t<details::is_valid_sum_mean_v<Value>>>
    void mean(const Value* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
              Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
              Stream& stream);

    template<typename Input, typename Output, typename = std::enable_if_t<details::is_valid_var_std_v<Input, Output>>>
    void var(const Input* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
             Output* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
             i64 ddof, Stream& stream);

    template<typename Input, typename Output, typename = std::enable_if_t<details::is_valid_var_std_v<Input, Output>>>
    void std(const Input* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
             Output* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
             i64 ddof, Stream& stream);
}
