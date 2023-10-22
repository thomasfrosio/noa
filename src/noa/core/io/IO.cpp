#include "noa/core/Exception.hpp"
#include "noa/core/indexing/Offset.hpp"
#include "noa/core/indexing/Layout.hpp"
#include "noa/core/io/IO.hpp"

namespace {
    using namespace ::noa;

    // Converts input into the desired serial_t type and serializes the result in output.
    // If the input values don't fit in the serial_t range or if we simply don't know, clamp should be true.
    template<typename Output, typename Input>
    inline void serialize_(
            const Input* input, std::byte* output, int64_t elements,
            bool clamp, bool swap_endian
    ) {
        constexpr int64_t OUTPUT_SIZE = sizeof(Output);
        if constexpr (std::is_same_v<Input, Output>) {
            // TODO On platform where sizeof(long) == 8, there's the possibility that T=long long and Output=long.
            //      We could check for this since a simple memcpy should be fine between these two types.
            std::memcpy(output, reinterpret_cast<const std::byte*>(input), static_cast<size_t>(elements * OUTPUT_SIZE));
        } else {
            Output tmp;
            if (clamp) {
                for (int64_t idx = 0; idx < elements; ++idx) {
                    tmp = clamp_cast<Output>(input[idx]);
                    std::memcpy(output + idx * OUTPUT_SIZE, &tmp, OUTPUT_SIZE);
                }
            } else {
                for (int64_t idx = 0; idx < elements; ++idx) {
                    tmp = static_cast<Output>(input[idx]);
                    std::memcpy(output + idx * OUTPUT_SIZE, &tmp, OUTPUT_SIZE);
                }
            }
        }
        // TODO Merge this on the conversion loop?
        if (swap_endian)
            io::swap_endian(output, elements, OUTPUT_SIZE);
    }

    // Same as above, but support a stridded array.
    template<typename Output, typename Input>
    inline void serialize_(
            const Input* input, const Strides4<int64_t>& strides, const Shape4<int64_t>& shape,
            std::byte* output, bool clamp, bool swap_endian
    ) {
        if (are_contiguous(strides, shape))
            return serialize_<Output>(input, output, shape.elements(), clamp, swap_endian);

        constexpr int64_t OUTPUT_SIZE = sizeof(Output);
        Output tmp;
        int64_t idx{0};
        // TODO Move the if inside the loop since branch prediction should take care of it.
        //      Although I'm not sure the compiler will see through the memcpy with the branch.
        //      Compiler explorer help!
        if (clamp) {
            for (int64_t i = 0; i < shape[0]; ++i) {
                for (int64_t j = 0; j < shape[1]; ++j) {
                    for (int64_t k = 0; k < shape[2]; ++k) {
                        for (int64_t l = 0; l < shape[3]; ++l, ++idx) {
                            tmp = clamp_cast<Output>(input[offset_at(i, j, k, l, strides)]);
                            std::memcpy(output + idx * OUTPUT_SIZE, &tmp, sizeof(Output));
                        }
                    }
                }
            }
        } else {
            for (int64_t i = 0; i < shape[0]; ++i) {
                for (int64_t j = 0; j < shape[1]; ++j) {
                    for (int64_t k = 0; k < shape[2]; ++k) {
                        for (int64_t l = 0; l < shape[3]; ++l, ++idx) {
                            tmp = static_cast<Output>(input[offset_at(i, j, k, l, strides)]);
                            std::memcpy(output + idx * OUTPUT_SIZE, &tmp, sizeof(Output));
                        }
                    }
                }
            }
        }
        if (swap_endian)
            io::swap_endian(output, shape.elements(), sizeof(Output));
    }

    template<typename Input, typename Output>
    inline void deserialize_(
            const std::byte* input, Output* output, int64_t elements,
            bool clamp, bool swap_endian
    ) {
        if constexpr (std::is_same_v<Output, Input>) {
            auto* output_ptr = reinterpret_cast<std::byte*>(output);
            std::memcpy(output_ptr, input, static_cast<size_t>(elements) * sizeof(Output));
            if (swap_endian)
                io::swap_endian(output_ptr, elements, sizeof(Input));
        } else {
            // Branch prediction should work nicely.
            // std::memcpy is removed.
            // std::reverse is translated in bswap
            // https://godbolt.org/z/Eavdcv8PM
            Input tmp;
            constexpr int64_t INPUT_SIZE = sizeof(Input);
            for (int64_t idx = 0; idx < elements; ++idx) {
                std::memcpy(&tmp, input + idx * INPUT_SIZE, sizeof(Input));
                if (swap_endian)
                    io::guts::reverse<sizeof(Input)>(reinterpret_cast<std::byte*>(&tmp));
                output[idx] = clamp ? clamp_cast<Output>(tmp) : static_cast<Output>(tmp);
            }
        }
    }

    template<typename Input, typename Output>
    inline void deserialize_(
            const std::byte* input, Output* output,
            const Strides4<int64_t>& strides, const Shape4<int64_t>& shape,
            bool clamp, bool swap_endian
    ) {
        if (are_contiguous(strides, shape))
            return deserialize_<Input>(input, output, shape.elements(), clamp, swap_endian);

        constexpr int64_t INPUT_SIZE = sizeof(Input);
        Input tmp;
        int64_t idx{0};
        for (int64_t i = 0; i < shape[0]; ++i) {
            for (int64_t j = 0; j < shape[1]; ++j) {
                for (int64_t k = 0; k < shape[2]; ++k) {
                    for (int64_t l = 0; l < shape[3]; ++l, ++idx) {
                        std::memcpy(&tmp, input + idx * INPUT_SIZE, sizeof(Input));
                        if (swap_endian)
                            io::guts::reverse<sizeof(Input)>(reinterpret_cast<std::byte*>(&tmp));
                        output[offset_at(i, j, k, l, strides)] =
                                clamp ? clamp_cast<Output>(tmp) : static_cast<Output>(tmp);
                    }
                }
            }
        }
    }

    template<typename T>
    void serialize_row_4bits_(const T* input, std::byte* output, int64_t elements_row, bool is_odd, bool clamp) {
        // The order of the first and second elements in the output are the 4 LSB and 4 MSB of the CPU, respectively.
        // If the row has an odd number of elements, the last byte of the row has the 4 MSB unset.
        static_assert(nt::is_scalar_v<T>);
        uint32_t tmp{}, l_val{}, h_val{};

        if (clamp) {
            for (int64_t i = 0; i < elements_row / 2; ++i) {
                l_val = clamp_cast<uint32_t>(input[2 * i]); // If IEEE float, default round to nearest
                h_val = clamp_cast<uint32_t>(input[2 * i + 1]);
                l_val = noa::clamp(l_val, 0U, 15U); // 2^4-1
                h_val = noa::clamp(h_val, 0U, 15U);
                tmp = l_val + (h_val << 4);
                std::memcpy(output + i, &tmp, 1);
            }
            if (is_odd) {
                l_val = clamp_cast<uint32_t>(input[elements_row - 1]);
                l_val = noa::clamp(l_val, 0U, 15U);
                std::memcpy(output + elements_row / 2, &l_val, 1);
            }
        } else {
            // std::round could be used instead, but we assume values are positive so +0.5f is enough
            for (int64_t i = 0; i < elements_row / 2; ++i) {
                l_val = static_cast<uint32_t>(noa::round(input[2 * i]));
                h_val = static_cast<uint32_t>(noa::round(input[2 * i + 1] ));
                tmp = l_val + (h_val << 4);
                std::memcpy(output + i, &tmp, 1);
            }
            if (is_odd) {
                l_val = static_cast<uint32_t>(noa::round(input[elements_row - 1]));
                std::memcpy(output + elements_row / 2, &l_val, 1);
            }
        }
    }

    template<typename T>
    inline void deserialize_row_4bits_(const std::byte* input, T* output, int64_t elements_row, bool is_odd) {
        // This is assuming that the first and second elements are at the LSB and MSB of the CPU, respectively.
        // If the row has an odd number of elements, the last bytes has the 4 MSB unused.
        constexpr unsigned char MASK_4LSB{0b00001111};
        for (int64_t i = 0; i < elements_row / 2; ++i) {
            const auto tmp = static_cast<unsigned char>(input[i]);
            output[i * 2] = static_cast<T>(tmp & MASK_4LSB);
            output[i * 2 + 1] = static_cast<T>((tmp >> 4) & MASK_4LSB);
        }
        if (is_odd) {
            const auto tmp = static_cast<unsigned char>(input[elements_row / 2]);
            output[elements_row - 1] = static_cast<T>(tmp & MASK_4LSB);
        }
    }
}

namespace noa::io {
    template<typename Input, typename>
    void serialize(
            const Input* input, const Strides4<int64_t>& strides, const Shape4<int64_t>& shape,
            std::byte* output, DataType data_type,
            bool clamp, bool swap_endian
    ) {
        const auto elements = shape.elements();
        switch (data_type) {
            case DataType::U4:
                if constexpr (nt::is_scalar_v<Input>) {
                    NOA_ASSERT(noa::all(shape > 0));
                    NOA_ASSERT(noa::indexing::are_contiguous(strides, shape));
                    if (!(shape[3] % 2)) { // if even, data can be serialized contiguously
                        serialize_row_4bits_(input, output, elements, false, clamp);
                    } else { // otherwise, there's a "padding" of 4bits at the end of each row
                        const auto rows = shape[0] * shape[1] * shape[2];
                        const auto bytes_per_row = (shape[3] + 1) / 2;
                        for (int64_t row = 0; row < rows; ++row)
                            serialize_row_4bits_(input + shape[3] * row,
                                                 output + bytes_per_row * row,
                                                 shape[3], true, clamp);
                    }
                    return;
                }
                break;
            case DataType::I8:
                if constexpr (nt::is_scalar_v<Input>)
                    return serialize_<int8_t>(input, strides, shape, output, clamp, swap_endian);
                break;
            case DataType::U8:
                if constexpr (nt::is_scalar_v<Input>)
                    return serialize_<u8>(input, strides, shape, output, clamp, swap_endian);
                break;
            case DataType::I16:
                if constexpr (nt::is_scalar_v<Input>)
                    return serialize_<i16>(input, strides, shape, output, clamp, swap_endian);
                break;
            case DataType::U16:
                if constexpr (nt::is_scalar_v<Input>)
                    return serialize_<u16>(input, strides, shape, output, clamp, swap_endian);
                break;
            case DataType::I32:
                if constexpr (nt::is_scalar_v<Input>)
                    return serialize_<i32>(input, strides, shape, output, clamp, swap_endian);
                break;
            case DataType::U32:
                if constexpr (nt::is_scalar_v<Input>)
                    return serialize_<uint32_t>(input, strides, shape, output, clamp, swap_endian);
                break;
            case DataType::I64:
                if constexpr (nt::is_scalar_v<Input>)
                    return serialize_<int64_t>(input, strides, shape, output, clamp, swap_endian);
                break;
            case DataType::U64:
                if constexpr (nt::is_scalar_v<Input>)
                    return serialize_<u64>(input, strides, shape, output, clamp, swap_endian);
                break;
            case DataType::F16:
                if constexpr (nt::is_scalar_v<Input>)
                    return serialize_<f16>(input, strides, shape, output, clamp, swap_endian);
                break;
            case DataType::F32:
                if constexpr (nt::is_scalar_v<Input>)
                    return serialize_<f32>(input, strides, shape, output, clamp, swap_endian);
                break;
            case DataType::F64:
                if constexpr (nt::is_scalar_v<Input>)
                    return serialize_<f64>(input, strides, shape, output, clamp, swap_endian);
                break;
            case DataType::CI16:
                if constexpr (nt::is_complex_v<Input>) {
                    using real_t = typename Input::value_type;
                    const auto real = ReinterpretLayout(shape, strides, input).template as<const real_t>();
                    return serialize(real.ptr, real.strides, real.shape,
                                     output, DataType::I16,
                                     clamp, swap_endian);
                }
                break;
            case DataType::C16:
                if constexpr (nt::is_complex_v<Input>) {
                    using real_t = typename Input::value_type;
                    const auto real = ReinterpretLayout(shape, strides, input).template as<const real_t>();
                    return serialize(real.ptr, real.strides, real.shape,
                                     output, DataType::F16,
                                     clamp, swap_endian);
                }
                break;
            case DataType::C32:
                if constexpr (nt::is_complex_v<Input>) {
                    using real_t = typename Input::value_type;
                    const auto real = ReinterpretLayout(shape, strides, input).template as<const real_t>();
                    return serialize(real.ptr, real.strides, real.shape,
                                     output, DataType::F32,
                                     clamp, swap_endian);
                }
                break;
            case DataType::C64:
                if constexpr (nt::is_complex_v<Input>) {
                    using real_t = typename Input::value_type;
                    const auto real = ReinterpretLayout(shape, strides, input).template as<const real_t>();
                    return serialize(real.ptr, real.strides, real.shape,
                                     output, DataType::F64,
                                     clamp, swap_endian);
                }
                break;
            case DataType::UNKNOWN:
                break;
        }
        NOA_THROW("{} cannot be serialized into the data type {}", to_human_readable<Input>(), data_type);
    }

    void serialize(
            const void* input, DataType input_data_type,
            const Strides4<int64_t>& strides, const Shape4<int64_t>& shape,
            std::byte* output, DataType output_data_type,
            bool clamp, bool swap_endian
    ) {
        switch (input_data_type) {
            case DataType::I8:
                return serialize(reinterpret_cast<const int8_t*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::U8:
                return serialize(reinterpret_cast<const u8*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::I16:
                return serialize(reinterpret_cast<const i16*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::U16:
                return serialize(reinterpret_cast<const u16*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::I32:
                return serialize(reinterpret_cast<const i32*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::U32:
                return serialize(reinterpret_cast<const uint32_t*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::I64:
                return serialize(reinterpret_cast<const int64_t*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::U64:
                return serialize(reinterpret_cast<const u64*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::F16:
                return serialize(reinterpret_cast<const f16*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::F32:
                return serialize(reinterpret_cast<const f32*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::F64:
                return serialize(reinterpret_cast<const f64*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::C16:
                return serialize(reinterpret_cast<const c16*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::C32:
                return serialize(reinterpret_cast<const c32*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::C64:
                return serialize(reinterpret_cast<const c64*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            default:
                NOA_THROW("Data type {} cannot be converted into a real type", input_data_type);
        }
    }

    template<typename T>
    void serialize(
            const T* input, const Strides4<int64_t>& strides, const Shape4<int64_t>& shape,
            std::ostream& output, DataType data_type,
            bool clamp, bool swap_endian
    ) {
        // Ignore all previous errors on that stream. If these errors cannot be recovered from,
        // the failbit will be reset by write() anyway and an exception will be thrown.
        output.clear();

        const bool are_contiguous = noa::are_contiguous(strides, shape);
        const int64_t elements = shape.elements();
        constexpr int64_t SIZEOF_T = sizeof(T);

        if (are_contiguous && !swap_endian && data_type == dtype<T>()) {
            output.write(reinterpret_cast<const char*>(input), SIZEOF_T * elements);
            if (output.fail()) {
                output.clear();
                NOA_THROW("Stream error. Failed while writing {} bytes", SIZEOF_T * elements);
            }
            return;

        } else if (data_type == DataType::U4) {
            const int64_t bytes = serialized_size(DataType::U4, elements, shape[3]);
            const auto buffer = std::make_unique<std::byte[]>(static_cast<size_t>(bytes));
            serialize(input, strides, shape, buffer.get(), DataType::U4, clamp);
            output.write(reinterpret_cast<const char*>(buffer.get()), bytes);
            if (output.fail()) {
                output.clear();
                NOA_THROW("Stream error. Failed while writing {} bytes", bytes);
            }

        } else if (are_contiguous) {
            constexpr int64_t bytes_per_batch = 1 << 26; // 67MB
            const int64_t bytes_per_element = serialized_size(data_type, 1);
            int64_t bytes_remain = bytes_per_element * elements;
            int64_t bytes_buffer = bytes_remain > bytes_per_batch ? bytes_per_batch : bytes_remain;
            const auto buffer = std::make_unique<std::byte[]>(static_cast<size_t>(bytes_buffer));
            const auto* buffer_ptr = reinterpret_cast<const char*>(buffer.get());

            // Read until there's nothing left.
            for (; bytes_remain > 0; bytes_remain -= bytes_buffer) {
                bytes_buffer = std::min(bytes_remain, bytes_buffer);
                const int64_t elements_buffer = bytes_buffer / bytes_per_element;
                const Shape4<int64_t> buffer_shape{1, 1, 1, elements_buffer};

                // Serialize according to data type and write.
                serialize(input, buffer_shape.strides(), buffer_shape,
                          buffer.get(), data_type,
                          clamp, swap_endian);
                output.write(buffer_ptr, bytes_buffer);
                if (output.fail()) {
                    output.clear();
                    NOA_THROW("Stream error. Failed while writing {} bytes", bytes_buffer);
                }

                input += elements_buffer;
            }

        } else {
            const int64_t elements_per_slice = shape[2] * shape[3];
            const int64_t bytes_per_slice = serialized_size(data_type, elements_per_slice);
            const Shape4<int64_t> slice_shape{1, 1, shape[2], shape[3]};
            const Strides4<int64_t> slice_strides{0, 0, strides[2], strides[3]};
            const auto buffer = std::make_unique<std::byte[]>(static_cast<size_t>(bytes_per_slice));
            const auto* buffer_ptr = reinterpret_cast<const char*>(buffer.get());

            for (int64_t i = 0; i < shape[0]; ++i) {
                for (int64_t j = 0; j < shape[1]; ++j) {
                    const T* input_ptr = input + offset_at(i, j, strides);
                    serialize(input_ptr, slice_strides, slice_shape,
                              buffer.get(), data_type,
                              clamp, swap_endian);
                    output.write(buffer_ptr, bytes_per_slice);
                    if (output.fail()) {
                        output.clear();
                        NOA_THROW("Stream error. Failed while writing {} bytes", bytes_per_slice);
                    }
                }
            }
        }
    }

    void serialize(const void* input, DataType input_data_type,
                   const Strides4<int64_t>& strides, const Shape4<int64_t>& shape,
                   std::ostream& output, DataType output_data_type,
                   bool clamp, bool swap_endian) {
        switch (input_data_type) {
            case DataType::I8:
                return serialize(reinterpret_cast<const int8_t*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::U8:
                return serialize(reinterpret_cast<const u8*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::I16:
                return serialize(reinterpret_cast<const i16*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::U16:
                return serialize(reinterpret_cast<const u16*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::I32:
                return serialize(reinterpret_cast<const i32*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::U32:
                return serialize(reinterpret_cast<const uint32_t*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::I64:
                return serialize(reinterpret_cast<const int64_t*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::U64:
                return serialize(reinterpret_cast<const u64*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::F16:
                return serialize(reinterpret_cast<const f16*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::F32:
                return serialize(reinterpret_cast<const f32*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::F64:
                return serialize(reinterpret_cast<const f64*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::C16:
                return serialize(reinterpret_cast<const c16*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::C32:
                return serialize(reinterpret_cast<const c32*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::C64:
                return serialize(reinterpret_cast<const c64*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            default:
                NOA_THROW("data type {} cannot be converted into a supported real type", input_data_type);
        }
    }
}

namespace noa::io {
    template<typename T>
    void deserialize(
            const std::byte* input, DataType data_type,
            T* output, const Strides4<int64_t>& strides, const Shape4<int64_t>& shape,
            bool clamp, bool swap_endian
    ) {
        switch (data_type) {
            case DataType::U4:
                if constexpr (nt::is_scalar_v<T>) {
                    NOA_ASSERT(noa::all(shape > 0));
                    NOA_ASSERT(noa::indexing::are_contiguous(strides, shape));
                    const auto elements = shape.elements();
                    if (!(shape[3] % 2)) { // if even, data can be deserialized contiguously
                        deserialize_row_4bits_(input, output, elements, false);
                    } else { // otherwise, there's a "padding" of 4bits at the end of each row
                        const auto rows = shape[0] * shape[1] * shape[2];
                        const auto bytes_per_row = (shape[3] + 1) / 2;
                        for (int64_t row = 0; row < rows; ++row)
                            deserialize_row_4bits_(input + bytes_per_row * row,
                                                   output + shape[3] * row,
                                                   shape[3], true);
                    }
                    return;
                }
                break;
            case DataType::I8:
                if constexpr (nt::is_scalar_v<T>)
                    return deserialize_<int8_t>(input, output, strides, shape, clamp, swap_endian);
                break;
            case DataType::U8:
                if constexpr (nt::is_scalar_v<T>)
                    return deserialize_<u8>(input, output, strides, shape, clamp, swap_endian);
                break;
            case DataType::I16:
                if constexpr (nt::is_scalar_v<T>)
                    return deserialize_<i16>(input, output, strides, shape, clamp, swap_endian);
                break;
            case DataType::U16:
                if constexpr (nt::is_scalar_v<T>)
                    return deserialize_<u16>(input, output, strides, shape, clamp, swap_endian);
                break;
            case DataType::I32:
                if constexpr (nt::is_scalar_v<T>)
                    return deserialize_<i32>(input, output, strides, shape, clamp, swap_endian);
                break;
            case DataType::U32:
                if constexpr (nt::is_scalar_v<T>)
                    return deserialize_<uint32_t>(input, output, strides, shape, clamp, swap_endian);
                break;
            case DataType::I64:
                if constexpr (nt::is_scalar_v<T>)
                    return deserialize_<int64_t>(input, output, strides, shape, clamp, swap_endian);
                break;
            case DataType::U64:
                if constexpr (nt::is_scalar_v<T>)
                    return deserialize_<u64>(input, output, strides, shape, clamp, swap_endian);
                break;
            case DataType::F16:
                if constexpr (nt::is_scalar_v<T>)
                    return deserialize_<f16>(input, output, strides, shape, clamp, swap_endian);
                break;
            case DataType::F32:
                if constexpr (nt::is_scalar_v<T>)
                    return deserialize_<f32>(input, output, strides, shape, clamp, swap_endian);
                break;
            case DataType::F64:
                if constexpr (nt::is_scalar_v<T>)
                    return deserialize_<f64>(input, output, strides, shape, clamp, swap_endian);
                break;
            case DataType::CI16:
                if constexpr (nt::is_complex_v<T>) {
                    using real_t = typename T::value_type;
                    const auto real = ReinterpretLayout(shape, strides, output).template as<real_t>();
                    return deserialize(input, DataType::I16,
                                       real.ptr, real.strides,
                                       real.shape, clamp, swap_endian);
                }
                break;
            case DataType::C16:
                if constexpr (nt::is_complex_v<T>) {
                    using real_t = typename T::value_type;
                    const auto real = ReinterpretLayout(shape, strides, output).template as<real_t>();
                    return deserialize(input, DataType::F16,
                                       real.ptr, real.strides,
                                       real.shape, clamp, swap_endian);
                }
                break;
            case DataType::C32:
                if constexpr (nt::is_complex_v<T>) {
                    using real_t = typename T::value_type;
                    const auto real = ReinterpretLayout(shape, strides, output).template as<real_t>();
                    return deserialize(input, DataType::F32,
                                       real.ptr, real.strides,
                                       real.shape, clamp, swap_endian);
                }
                break;
            case DataType::C64:
                if constexpr (nt::is_complex_v<T>) {
                    using real_t = typename T::value_type;
                    const auto real = ReinterpretLayout(shape, strides, output).template as<real_t>();
                    return deserialize(input, DataType::F64,
                                       real.ptr, real.strides,
                                       real.shape, clamp, swap_endian);
                }
                break;
            case DataType::UNKNOWN:
                break;
        }
        NOA_THROW("data type {} cannot be deserialized into {}", data_type, to_human_readable<T>());
    }

    void deserialize(
            const std::byte* input, DataType input_data_type,
            void* output, DataType output_data_type,
            const Strides4<int64_t>& strides, const Shape4<int64_t>& shape,
            bool clamp, bool swap_endian
    ) {
        switch (output_data_type) {
            case DataType::I8:
                return deserialize(input, input_data_type,
                                   static_cast<int8_t*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::U8:
                return deserialize(input, input_data_type,
                                   static_cast<u8*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::I16:
                return deserialize(input, input_data_type,
                                   static_cast<i16*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::U16:
                return deserialize(input, input_data_type,
                                   static_cast<u16*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::I32:
                return deserialize(input, input_data_type,
                                   static_cast<i32*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::U32:
                return deserialize(input, input_data_type,
                                   static_cast<uint32_t*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::I64:
                return deserialize(input, input_data_type,
                                   static_cast<int64_t*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::U64:
                return deserialize(input, input_data_type,
                                   static_cast<u64*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::F16:
                return deserialize(input, input_data_type,
                                   static_cast<f16*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::F32:
                return deserialize(input, input_data_type,
                                   static_cast<f32*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::F64:
                return deserialize(input, input_data_type,
                                   static_cast<f64*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::C16:
                return deserialize(input, input_data_type,
                                   static_cast<c16*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::C32:
                return deserialize(input, input_data_type,
                                   static_cast<c32*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::C64:
                return deserialize(input, input_data_type,
                                   static_cast<c64*>(output), strides,
                                   shape, clamp, swap_endian);
            default:
                NOA_THROW("data type {} cannot be converted into a supported real type", output_data_type);
        }
    }

    template<typename T>
    void deserialize(
            std::istream& input, DataType data_type,
            T* output, const Strides4<int64_t>& strides, const Shape4<int64_t>& shape,
            bool clamp, bool swap_endian
    ) {
        input.clear();
        const bool are_contiguous = noa::are_contiguous(strides, shape);
        const int64_t elements = shape.elements();
        constexpr int64_t SIZEOF_T = sizeof(T);

        if (are_contiguous && data_type == dtype<T>()) {
            input.read(reinterpret_cast<char*>(output), SIZEOF_T * elements);
            if (input.fail()) {
                input.clear();
                NOA_THROW("Stream error. Failed while reading {} bytes", SIZEOF_T * elements);
            } else if (swap_endian) {
                if constexpr (nt::is_complex_v<T>)
                    noa::io::swap_endian(reinterpret_cast<std::byte*>(output), elements * 2, SIZEOF_T / 2);
                else
                    noa::io::swap_endian(reinterpret_cast<std::byte*>(output), elements, SIZEOF_T);
            }
            return;

        } else if (data_type == DataType::U4) {
            const auto bytes = serialized_size(DataType::U4, elements, shape[3]);
            const auto buffer = std::make_unique<std::byte[]>(static_cast<size_t>(bytes));
            input.read(reinterpret_cast<char*>(buffer.get()), bytes);
            if (input.fail()) {
                input.clear();
                NOA_THROW("Stream error. Failed while reading {} bytes", bytes);
            }
            deserialize(buffer.get(), DataType::U4,
                        output, strides,
                        shape, clamp);

        } else if (are_contiguous) {
            constexpr int64_t bytes_per_batch = 1 << 26; // 67MB
            const int64_t bytes_per_element = serialized_size(data_type, 1);
            int64_t bytes_remain = bytes_per_element * elements;
            int64_t bytes_buffer = bytes_remain > bytes_per_batch ? bytes_per_batch : bytes_remain;
            const auto buffer = std::make_unique<std::byte[]>(static_cast<size_t>(bytes_buffer));
            auto* buffer_ptr = reinterpret_cast<char*>(buffer.get());

            // Read until there's nothing left.
            for (; bytes_remain > 0; bytes_remain -= bytes_buffer) {
                bytes_buffer = std::min(bytes_remain, bytes_buffer);
                const int64_t elements_buffer = bytes_buffer / bytes_per_element;
                const Shape4<int64_t> buffer_shape{1, 1, 1, elements_buffer};

                // Read, swap and deserialize according to data type.
                input.read(buffer_ptr, bytes_buffer);
                if (input.fail()) {
                    input.clear();
                    NOA_THROW("Stream error. Failed while reading {} bytes", bytes_buffer);
                }
                deserialize(buffer.get(), data_type,
                            output, buffer_shape.strides(),
                            buffer_shape, clamp, swap_endian);

                output += elements_buffer;
            }
        } else {
            const int64_t elements_per_slice = shape[2] * shape[3];
            const int64_t bytes_per_slice = serialized_size(data_type, elements_per_slice);
            const Shape4<int64_t> slice_shape{1, 1, shape[2], shape[3]};
            const Strides4<int64_t> slice_strides{0, 0, strides[2], strides[3]};
            const auto buffer = std::make_unique<std::byte[]>(static_cast<size_t>(bytes_per_slice));
            auto* buffer_ptr = reinterpret_cast<char*>(buffer.get());

            for (int64_t i = 0; i < shape[0]; ++i) {
                for (int64_t j = 0; j < shape[1]; ++j) {
                    input.read(buffer_ptr, bytes_per_slice);
                    if (input.fail()) {
                        input.clear();
                        NOA_THROW("Stream error. Failed while reading {} bytes", bytes_per_slice);
                    }
                    T* output_ptr = output + offset_at(i, j, strides);
                    deserialize(buffer.get(), data_type,
                                output_ptr, slice_strides,
                                slice_shape, clamp, swap_endian);
                }
            }
        }
    }

    void deserialize(
            std::istream& input, DataType input_data_type,
            void* output, DataType output_data_type,
            const Strides4<int64_t>& strides, const Shape4<int64_t>& shape,
            bool clamp, bool swap_endian
    ) {
        switch (output_data_type) {
            case DataType::I8:
                return deserialize(input, input_data_type,
                                   static_cast<int8_t*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::U8:
                return deserialize(input, input_data_type,
                                   static_cast<u8*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::I16:
                return deserialize(input, input_data_type,
                                   static_cast<i16*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::U16:
                return deserialize(input, input_data_type,
                                   static_cast<u16*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::I32:
                return deserialize(input, input_data_type,
                                   static_cast<i32*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::U32:
                return deserialize(input, input_data_type,
                                   static_cast<uint32_t*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::I64:
                return deserialize(input, input_data_type,
                                   static_cast<int64_t*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::U64:
                return deserialize(input, input_data_type,
                                   static_cast<u64*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::F16:
                return deserialize(input, input_data_type,
                                   static_cast<f16*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::F32:
                return deserialize(input, input_data_type,
                                   static_cast<f32*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::F64:
                return deserialize(input, input_data_type,
                                   static_cast<f64*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::C16:
                return deserialize(input, input_data_type,
                                   static_cast<c16*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::C32:
                return deserialize(input, input_data_type,
                                   static_cast<c32*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::C64:
                return deserialize(input, input_data_type,
                                   static_cast<c64*>(output), strides,
                                   shape, clamp, swap_endian);
            default:
                NOA_THROW("data type {} cannot be converted into a supported real type", output_data_type);
        }
    }
}
