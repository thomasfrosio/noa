#pragma once

#include "noa/runtime/View.hpp"
#include "noa/io/Encoding.hpp"

namespace noa::io {
    /// Cast, with type erased output.
    /// \param[in] input    Input array
    /// \param[out] output  Output array.
    /// \param output_dtype Output data type. Should be a "static type" (U4 and CI16 are currently not supported).
    /// \param clamp        Whether to use clamp_cast to clamp input values to the output value range.
    template<nt::readable_varray_decay_of_numeric Input, nt::writable_varray_decay_of_byte Output>
    void cast(
        Input&& input,
        Output&& output,
        const DataType& output_dtype,
        bool clamp = false
    ) {
        using value_t = nt::mutable_value_type_t<Input>;
        switch (output_dtype) {
            case DataType::I8:
                if constexpr (nt::scalar<value_t>)
                    return cast(std::forward<Input>(input), std::forward<Output>(output).template reinterpret_as<i8>(), clamp);
                break;
            case DataType::U8:
                if constexpr (nt::scalar<value_t>)
                    return cast(std::forward<Input>(input), std::forward<Output>(output).template reinterpret_as<u8>(), clamp);
                break;
            case DataType::I16:
                if constexpr (nt::scalar<value_t>)
                    return cast(std::forward<Input>(input), std::forward<Output>(output).template reinterpret_as<i16>(), clamp);
                break;
            case DataType::U16:
                if constexpr (nt::scalar<value_t>)
                    return cast(std::forward<Input>(input), std::forward<Output>(output).template reinterpret_as<u16>(), clamp);
                break;
            case DataType::I32:
                if constexpr (nt::scalar<value_t>)
                    return cast(std::forward<Input>(input), std::forward<Output>(output).template reinterpret_as<i32>(), clamp);
                break;
            case DataType::U32:
                if constexpr (nt::scalar<value_t>)
                    return cast(std::forward<Input>(input), std::forward<Output>(output).template reinterpret_as<u32>(), clamp);
                break;
            case DataType::I64:
                if constexpr (nt::scalar<value_t>)
                    return cast(std::forward<Input>(input), std::forward<Output>(output).template reinterpret_as<i64>(), clamp);
                break;
            case DataType::U64:
                if constexpr (nt::scalar<value_t>)
                    return cast(std::forward<Input>(input), std::forward<Output>(output).template reinterpret_as<u64>(), clamp);
                break;
            case DataType::F16:
                if constexpr (nt::scalar<value_t>)
                    return cast(std::forward<Input>(input), std::forward<Output>(output).template reinterpret_as<f16>(), clamp);
                break;
            case DataType::F32:
                if constexpr (nt::scalar<value_t>)
                    return cast(std::forward<Input>(input), std::forward<Output>(output).template reinterpret_as<f32>(), clamp);
                break;
            case DataType::F64:
                if constexpr (nt::scalar<value_t>)
                    return cast(std::forward<Input>(input), std::forward<Output>(output).template reinterpret_as<f64>(), clamp);
                break;
            case DataType::C16:
                if constexpr (nt::complex<value_t>)
                    return cast(std::forward<Input>(input), std::forward<Output>(output).template reinterpret_as<c16>(), clamp);
                break;
            case DataType::C32:
                if constexpr (nt::complex<value_t>)
                    return cast(std::forward<Input>(input), std::forward<Output>(output).template reinterpret_as<c32>(), clamp);
                break;
            case DataType::C64:
                if constexpr (nt::complex<value_t>)
                    return cast(std::forward<Input>(input), std::forward<Output>(output).template reinterpret_as<c64>(), clamp);
                break;
            case DataType::CI16:
            case DataType::U4:
                panic("TODO unimplemented cast, use encode instead");
            case DataType::UNKNOWN:
                panic("Unknown data type");
        }
        panic("{} cannot be cast to {}", nd::stringify<value_t>(), output_dtype);
    }

    /// Cast, with type erased input.
    /// \param[in] input    Input array
    /// \param input_dtype  Input data type. Should be a "static type" (U4 and CI16 are currently not supported).
    /// \param[out] output  Output array.
    /// \param clamp        Whether to use clamp_cast to clamp input values to the output value range.
    template<nt::readable_varray_decay_of_byte Input, nt::writable_varray_decay_of_numeric Output>
    void cast(
        Input&& input,
        const DataType& input_dtype,
        Output&& output,
        bool clamp = false
    ) {
        using value_t = nt::mutable_value_type_t<Output>;
        switch (input_dtype) {
            case DataType::I8:
                if constexpr (nt::scalar<value_t>)
                    return cast(std::forward<Input>(input).template reinterpret_as<i8>(), std::forward<Output>(output), clamp);
                break;
            case DataType::U8:
                if constexpr (nt::scalar<value_t>)
                    return cast(std::forward<Input>(input).template reinterpret_as<u8>(), std::forward<Output>(output), clamp);
                break;
            case DataType::I16:
                if constexpr (nt::scalar<value_t>)
                    return cast(std::forward<Input>(input).template reinterpret_as<i16>(), std::forward<Output>(output), clamp);
                break;
            case DataType::U16:
                if constexpr (nt::scalar<value_t>)
                    return cast(std::forward<Input>(input).template reinterpret_as<u16>(), std::forward<Output>(output), clamp);
                break;
            case DataType::I32:
                if constexpr (nt::scalar<value_t>)
                    return cast(std::forward<Input>(input).template reinterpret_as<i32>(), std::forward<Output>(output), clamp);
                break;
            case DataType::U32:
                if constexpr (nt::scalar<value_t>)
                    return cast(std::forward<Input>(input).template reinterpret_as<u32>(), std::forward<Output>(output), clamp);
                break;
            case DataType::I64:
                if constexpr (nt::scalar<value_t>)
                    return cast(std::forward<Input>(input).template reinterpret_as<i64>(), std::forward<Output>(output), clamp);
                break;
            case DataType::U64:
                if constexpr (nt::scalar<value_t>)
                    return cast(std::forward<Input>(input).template reinterpret_as<u64>(), std::forward<Output>(output), clamp);
                break;
            case DataType::F16:
                if constexpr (nt::scalar<value_t>)
                    return cast(std::forward<Input>(input).template reinterpret_as<f16>(), std::forward<Output>(output), clamp);
                break;
            case DataType::F32:
                if constexpr (nt::scalar<value_t>)
                    return cast(std::forward<Input>(input).template reinterpret_as<f32>(), std::forward<Output>(output), clamp);
                break;
            case DataType::F64:
                if constexpr (nt::scalar<value_t>)
                    return cast(std::forward<Input>(input).template reinterpret_as<f64>(), std::forward<Output>(output), clamp);
                break;
            case DataType::C16:
                if constexpr (nt::real_or_complex<value_t>)
                    return cast(std::forward<Input>(input).template reinterpret_as<c16>(), std::forward<Output>(output), clamp);
                break;
            case DataType::C32:
                if constexpr (nt::real_or_complex<value_t>)
                    return cast(std::forward<Input>(input).template reinterpret_as<c32>(), std::forward<Output>(output), clamp);
                break;
            case DataType::C64:
                if constexpr (nt::real_or_complex<value_t>)
                    return cast(std::forward<Input>(input).template reinterpret_as<c64>(), std::forward<Output>(output), clamp);
                break;
            case DataType::CI16:
            case DataType::U4:
                panic("TODO unimplemented cast, use encode instead");
            case DataType::UNKNOWN:
                panic("Unknown data type");
        }
        panic("{} cannot be cast to {}", input_dtype, nd::stringify<value_t>());
    }

    /// Cast, with type erased input and output.
    /// \param[in] input    Input array
    /// \param input_dtype  Input data type. Should be a "static type" (U4 and CI16 are currently not supported).
    /// \param[out] output  Output array.
    /// \param output_dtype Output data type. Should be a "static type" (U4 and CI16 are currently not supported).
    /// \param clamp        Whether to use clamp_cast to clamp input values to the output value range.
    template<nt::readable_varray_decay_of_byte Input, nt::writable_varray_decay_of_byte Output>
    void cast(
        Input&& input,
        const DataType& input_dtype,
        Output&& output,
        const DataType& output_dtype,
        bool clamp = false
    ) {
        switch (input_dtype) {
            case DataType::I8:  return cast(std::forward<Input>(input).template reinterpret_as<const i8>(),  std::forward<Output>(output), output_dtype, clamp);
            case DataType::I16: return cast(std::forward<Input>(input).template reinterpret_as<const i16>(), std::forward<Output>(output), output_dtype, clamp);
            case DataType::I32: return cast(std::forward<Input>(input).template reinterpret_as<const i32>(), std::forward<Output>(output), output_dtype, clamp);
            case DataType::I64: return cast(std::forward<Input>(input).template reinterpret_as<const i64>(), std::forward<Output>(output), output_dtype, clamp);
            case DataType::U8:  return cast(std::forward<Input>(input).template reinterpret_as<const u8>(),  std::forward<Output>(output), output_dtype, clamp);
            case DataType::U16: return cast(std::forward<Input>(input).template reinterpret_as<const u16>(), std::forward<Output>(output), output_dtype, clamp);
            case DataType::U32: return cast(std::forward<Input>(input).template reinterpret_as<const u32>(), std::forward<Output>(output), output_dtype, clamp);
            case DataType::U64: return cast(std::forward<Input>(input).template reinterpret_as<const u64>(), std::forward<Output>(output), output_dtype, clamp);
            case DataType::F16: return cast(std::forward<Input>(input).template reinterpret_as<const f16>(), std::forward<Output>(output), output_dtype, clamp);
            case DataType::F32: return cast(std::forward<Input>(input).template reinterpret_as<const f32>(), std::forward<Output>(output), output_dtype, clamp);
            case DataType::F64: return cast(std::forward<Input>(input).template reinterpret_as<const f64>(), std::forward<Output>(output), output_dtype, clamp);
            case DataType::C16: return cast(std::forward<Input>(input).template reinterpret_as<const c16>(), std::forward<Output>(output), output_dtype, clamp);
            case DataType::C32: return cast(std::forward<Input>(input).template reinterpret_as<const c32>(), std::forward<Output>(output), output_dtype, clamp);
            case DataType::C64: return cast(std::forward<Input>(input).template reinterpret_as<const c64>(), std::forward<Output>(output), output_dtype, clamp);
            case DataType::U4:
            case DataType::CI16:
                panic("TODO unimplemented cast, use encode instead");
            case DataType::UNKNOWN:
                panic("Unknown data type");
        }
        panic("TODO missing dtype");
    }
}
