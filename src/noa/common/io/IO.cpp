#include "noa/common/io/IO.h"
#include "noa/common/Exception.h"

namespace {
    using namespace ::noa;

    // Converts input into the desired serial_t type and serializes the result in output.
    // If the input values don't fit in the serial_t range or if we simply don't know, clamp should be true.
    template<typename serial_t, typename T>
    inline void serialize_(const T* input, byte_t* output, size_t elements,
                           bool clamp, bool swap_endian) {
        if constexpr (std::is_same_v<T, serial_t>) {
            // TODO On platform where sizeof(long) == 8, there's the possibility that T=long long and serial_t=long.
            //      We could check for this since a simple memcpy should be fine between these two types.
            std::memcpy(output, reinterpret_cast<const byte_t*>(input), elements * sizeof(T));
        } else {
            serial_t tmp;
            if (clamp) {
                for (size_t idx = 0; idx < elements; ++idx) {
                    tmp = clamp_cast<serial_t>(input[idx]);
                    std::memcpy(output + idx * sizeof(serial_t), &tmp, sizeof(serial_t));
                }
            } else {
                for (size_t idx = 0; idx < elements; ++idx) {
                    tmp = static_cast<serial_t>(input[idx]);
                    std::memcpy(output + idx * sizeof(serial_t), &tmp, sizeof(serial_t));
                }
            }
        }
        // TODO Merge this on the conversion loop?
        if (swap_endian)
            io::swapEndian(output, elements, sizeof(serial_t));
    }

    // Same as above, but support any stridded array.
    template<typename serial_t, typename T>
    inline void serialize_(const T* input, size4_t strides, size4_t shape, byte_t* output,
                           bool clamp, bool swap_endian) {
        if (indexing::areContiguous(strides, shape))
            return serialize_<serial_t>(input, output, shape.elements(), clamp, swap_endian);

        serial_t tmp;
        size_t idx{0};
        // TODO Move the if inside the loop since branch prediction should take care of it.
        //      Although I'm not sure the compiler will see through the memcpy with the branch...
        if (clamp) {
            for (size_t i = 0; i < shape[0]; ++i) {
                for (size_t j = 0; j < shape[1]; ++j) {
                    for (size_t k = 0; k < shape[2]; ++k) {
                        for (size_t l = 0; l < shape[3]; ++l, ++idx) {
                            tmp = clamp_cast<serial_t>(input[indexing::at(i, j, k, l, strides)]);
                            std::memcpy(output + idx * sizeof(serial_t), &tmp, sizeof(serial_t));
                        }
                    }
                }
            }
        } else {
            for (size_t i = 0; i < shape[0]; ++i) {
                for (size_t j = 0; j < shape[1]; ++j) {
                    for (size_t k = 0; k < shape[2]; ++k) {
                        for (size_t l = 0; l < shape[3]; ++l, ++idx) {
                            tmp = static_cast<serial_t>(input[indexing::at(i, j, k, l, strides)]);
                            std::memcpy(output + idx * sizeof(serial_t), &tmp, sizeof(serial_t));
                        }
                    }
                }
            }
        }
        if (swap_endian)
            io::swapEndian(output, shape.elements(), sizeof(serial_t));
    }

    template<typename serial_t, typename T>
    inline void deserialize_(const byte_t* input, T* output, size_t elements,
                            bool clamp, bool swap_endian) {
        if constexpr (std::is_same_v<T, serial_t>) {
            auto* output_ptr = reinterpret_cast<byte_t*>(output);
            std::memcpy(output_ptr, input, elements * sizeof(T));
            if (swap_endian)
                io::swapEndian(output_ptr, elements, sizeof(serial_t));
        } else {
            // Branch prediction should work nicely.
            // std::memcpy is removed.
            // std::reverse is translated in bswap
            // https://godbolt.org/z/Eavdcv8PM
            serial_t tmp;
            for (size_t idx = 0; idx < elements; ++idx) {
                std::memcpy(&tmp, input + idx * sizeof(serial_t), sizeof(serial_t));
                if (swap_endian)
                    io::details::reverse<sizeof(serial_t)>(reinterpret_cast<byte_t*>(&tmp));
                output[idx] = clamp ? clamp_cast<T>(tmp) : static_cast<T>(tmp);
            }
        }
    }

    template<typename serial_t, typename T>
    inline void deserialize_(const byte_t* input, T* output, size4_t strides, size4_t shape,
                            bool clamp, bool swap_endian) {
        if (indexing::areContiguous(strides, shape))
            return deserialize_<serial_t>(input, output, shape.elements(), clamp, swap_endian);

        serial_t tmp;
        size_t idx{0};
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    for (size_t l = 0; l < shape[3]; ++l, ++idx) {
                        std::memcpy(&tmp, input + idx * sizeof(serial_t), sizeof(serial_t));
                        if (swap_endian)
                            io::details::reverse<sizeof(serial_t)>(reinterpret_cast<byte_t*>(&tmp));
                        output[indexing::at(i, j, k, l, strides)] = clamp ? clamp_cast<T>(tmp) : static_cast<T>(tmp);
                    }
                }
            }
        }
    }

    template<typename T>
    void serializeRow4bits_(const T* input, byte_t* output, size_t elements_row, bool is_odd, bool clamp) {
        // The order of the first and second elements in the output are the 4 LSB and 4 MSB of the CPU, respectively.
        // If the row has an odd number of elements, the last byte of the row has the 4 MSB unset.
        static_assert(traits::is_scalar_v<T>);
        uint32_t tmp, l_val, h_val;

        if (clamp) {
            for (size_t i = 0; i < elements_row / 2; ++i) {
                l_val = clamp_cast<uint32_t>(input[2 * i]); // If IEEE float, default round to nearest
                h_val = clamp_cast<uint32_t>(input[2 * i + 1]);
                l_val = math::clamp(l_val, 0U, 15U); // 2^4-1
                h_val = math::clamp(h_val, 0U, 15U);
                tmp = l_val + (h_val << 4);
                std::memcpy(output + i, &tmp, 1);
            }
            if (is_odd) {
                l_val = clamp_cast<uint32_t>(input[elements_row - 1]);
                l_val = math::clamp(l_val, 0U, 15U);
                std::memcpy(output + elements_row / 2, &l_val, 1);
            }
        } else {
            // std::round could be used instead, but we assume values are positive so +0.5f is enough
            for (size_t i = 0; i < elements_row / 2; ++i) {
                l_val = static_cast<uint32_t>(math::round(input[2 * i]));
                h_val = static_cast<uint32_t>(math::round(input[2 * i + 1] ));
                tmp = l_val + (h_val << 4);
                std::memcpy(output + i, &tmp, 1);
            }
            if (is_odd) {
                l_val = static_cast<uint32_t>(math::round(input[elements_row - 1]));
                std::memcpy(output + elements_row / 2, &l_val, 1);
            }
        }
    }

    template<typename T>
    inline void deserializeRow4bits_(const byte_t* input, T* output, size_t elements_row, bool is_odd) {
        // This is assuming that the first and second elements are at the LSB and MSB of the CPU, respectively.
        // If the row has an odd number of elements, the last bytes has the 4 MSB unused.
        constexpr unsigned char MASK_4LSB{0b00001111};
        for (size_t i = 0; i < elements_row / 2; ++i) {
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
    template<typename T>
    void serialize(const T* input, size4_t strides, size4_t shape,
                   byte_t* output, DataType data_type,
                   bool clamp, bool swap_endian) {
        const size_t elements = shape.elements();
        switch (data_type) {
            case DataType::UINT4:
                if constexpr (traits::is_scalar_v<T>) {
                    NOA_ASSERT(all(shape > 0));
                    NOA_ASSERT(indexing::areContiguous(strides, shape));
                    if (!(shape[3] % 2)) { // if even, data can be serialized contiguously
                        serializeRow4bits_(input, output, elements, false, clamp);
                    } else { // otherwise, there's a "padding" of 4bits at the end of each row
                        size_t rows = shape[0] * shape[1] * shape[2];
                        size_t bytes_per_row = (shape[3] + 1) / 2;
                        for (size_t row = 0; row < rows; ++row)
                            serializeRow4bits_(input + shape[3] * row,
                                               output + bytes_per_row * row,
                                               shape[3], true, clamp);
                    }
                    break;
                }
            case DataType::INT8:
                if constexpr (traits::is_scalar_v<T>)
                    return serialize_<int8_t>(input, strides, shape, output, clamp, swap_endian);
            case DataType::UINT8:
                if constexpr (traits::is_scalar_v<T>)
                    return serialize_<uint8_t>(input, strides, shape, output, clamp, swap_endian);
            case DataType::INT16:
                if constexpr (traits::is_scalar_v<T>)
                    return serialize_<int16_t>(input, strides, shape, output, clamp, swap_endian);
            case DataType::UINT16:
                if constexpr (traits::is_scalar_v<T>)
                    return serialize_<uint16_t>(input, strides, shape, output, clamp, swap_endian);
            case DataType::INT32:
                if constexpr (traits::is_scalar_v<T>)
                    return serialize_<int32_t>(input, strides, shape, output, clamp, swap_endian);
            case DataType::UINT32:
                if constexpr (traits::is_scalar_v<T>)
                    return serialize_<uint32_t>(input, strides, shape, output, clamp, swap_endian);
            case DataType::INT64:
                if constexpr (traits::is_scalar_v<T>)
                    return serialize_<int64_t>(input, strides, shape, output, clamp, swap_endian);
            case DataType::UINT64:
                if constexpr (traits::is_scalar_v<T>)
                    return serialize_<uint64_t>(input, strides, shape, output, clamp, swap_endian);
            case DataType::FLOAT16:
                if constexpr (traits::is_scalar_v<T>)
                    return serialize_<half_t>(input, strides, shape, output, clamp, swap_endian);
            case DataType::FLOAT32:
                if constexpr (traits::is_scalar_v<T>)
                    return serialize_<float>(input, strides, shape, output, clamp, swap_endian);
            case DataType::FLOAT64:
                if constexpr (traits::is_scalar_v<T>)
                    return serialize_<double>(input, strides, shape, output, clamp, swap_endian);
            case DataType::CINT16:
                if constexpr (traits::is_complex_v<T>) {
                    using real_t = typename T::value_type;
                    const auto real = indexing::Reinterpret(shape, strides, input).template as<const real_t>();
                    return serialize(real.ptr, real.strides, real.shape,
                                     output, DataType::INT16,
                                     clamp, swap_endian);
                }
            case DataType::CFLOAT16:
                if constexpr (traits::is_complex_v<T>) {
                    using real_t = typename T::value_type;
                    const auto real = indexing::Reinterpret(shape, strides, input).template as<const real_t>();
                    return serialize(real.ptr, real.strides, real.shape,
                                     output, DataType::FLOAT16,
                                     clamp, swap_endian);
                }
            case DataType::CFLOAT32:
                if constexpr (traits::is_complex_v<T>) {
                    using real_t = typename T::value_type;
                    const auto real = indexing::Reinterpret(shape, strides, input).template as<const real_t>();
                    return serialize(real.ptr, real.strides, real.shape,
                                     output, DataType::FLOAT32,
                                     clamp, swap_endian);
                }
            case DataType::CFLOAT64:
                if constexpr (traits::is_complex_v<T>) {
                    using real_t = typename T::value_type;
                    const auto real = indexing::Reinterpret(shape, strides, input).template as<const real_t>();
                    return serialize(real.ptr, real.strides, real.shape,
                                     output, DataType::FLOAT64,
                                     clamp, swap_endian);
                }
            default:
                NOA_THROW("{} cannot be serialized into the data type {}", string::human<T>(), data_type);
        }
    }

    void serialize(const void* input, DataType input_data_type, size4_t strides, size4_t shape,
                   byte_t* output, DataType output_data_type,
                   bool clamp, bool swap_endian) {
        switch (input_data_type) {
            case DataType::INT8:
                return serialize(reinterpret_cast<const int8_t*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::UINT8:
                return serialize(reinterpret_cast<const uint8_t*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::INT16:
                return serialize(reinterpret_cast<const int16_t*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::UINT16:
                return serialize(reinterpret_cast<const uint16_t*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::INT32:
                return serialize(reinterpret_cast<const int32_t*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::UINT32:
                return serialize(reinterpret_cast<const uint32_t*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::INT64:
                return serialize(reinterpret_cast<const int64_t*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::UINT64:
                return serialize(reinterpret_cast<const uint64_t*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::FLOAT16:
                return serialize(reinterpret_cast<const half_t*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::FLOAT32:
                return serialize(reinterpret_cast<const float*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::FLOAT64:
                return serialize(reinterpret_cast<const double*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::CFLOAT16:
                return serialize(reinterpret_cast<const chalf_t*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::CFLOAT32:
                return serialize(reinterpret_cast<const cfloat_t*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::CFLOAT64:
                return serialize(reinterpret_cast<const cdouble_t*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            default:
                NOA_THROW("Data type {} cannot be converted into a real type", input_data_type);
        }
    }

    template<typename T>
    void serialize(const T* input, size4_t strides, size4_t shape,
                   std::ostream& output, DataType data_type,
                   bool clamp, bool swap_endian) {
        // Ignore all previous errors on that stream. If these errors cannot be recovered from,
        // the failbit will be reset by write() anyway and an exception will be thrown.
        output.clear();

        const bool are_contiguous = indexing::areContiguous(strides, shape);
        const size_t elements = shape.elements();

        if (are_contiguous && !swap_endian && data_type == dtype<T>()) {
            output.write(reinterpret_cast<const char*>(input), static_cast<std::streamsize>(sizeof(T) * elements));
            if (output.fail()) {
                output.clear();
                NOA_THROW("Stream error. Failed while writing {} bytes", sizeof(T) * elements);
            }
            return;

        } else if (data_type == DataType::UINT4) {
            size_t bytes = serializedSize(DataType::UINT4, elements, shape[3]);
            std::unique_ptr<byte_t[]> buffer = std::make_unique<byte_t[]>(bytes);
            serialize(input, strides, shape, buffer.get(), DataType::UINT4, clamp);
            output.write(reinterpret_cast<const char*>(buffer.get()), static_cast<std::streamsize>(bytes));
            if (output.fail()) {
                output.clear();
                NOA_THROW("Stream error. Failed while writing {} bytes", bytes);
            }

        } else if (are_contiguous) {
            constexpr size_t bytes_per_batch = 1 << 26; // 67MB
            const size_t bytes_per_element = serializedSize(data_type, 1);
            size_t bytes_remain = bytes_per_element * elements;
            size_t bytes_buffer = bytes_remain > bytes_per_batch ? bytes_per_batch : bytes_remain;
            std::unique_ptr<byte_t[]> buffer = std::make_unique<byte_t[]>(bytes_buffer);
            const auto* buffer_ptr = reinterpret_cast<const char*>(buffer.get());

            // Read until there's nothing left.
            for (; bytes_remain > 0; bytes_remain -= bytes_buffer) {
                bytes_buffer = std::min(bytes_remain, bytes_buffer);
                const size_t elements_buffer = bytes_buffer / bytes_per_element;
                const size4_t buffer_shape{1, 1, 1, elements_buffer};

                // Serialize according to data type and write.
                serialize(input, buffer_shape.strides(), buffer_shape,
                          buffer.get(), data_type,
                          clamp, swap_endian);
                output.write(buffer_ptr, static_cast<std::streamsize>(bytes_buffer));
                if (output.fail()) {
                    output.clear();
                    NOA_THROW("Stream error. Failed while writing {} bytes", bytes_buffer);
                }

                input += elements_buffer;
            }

        } else {
            const size_t elements_per_slice = shape[2] * shape[3];
            const size_t bytes_per_slice = serializedSize(data_type, elements_per_slice);
            const size4_t slice_shape{1, 1, shape[2], shape[3]};
            const size4_t slice_strides{0, 0, strides[2], strides[3]};
            std::unique_ptr<byte_t[]> buffer = std::make_unique<byte_t[]>(bytes_per_slice);
            const auto* buffer_ptr = reinterpret_cast<const char*>(buffer.get());

            for (size_t i = 0; i < shape[0]; ++i) {
                for (size_t j = 0; j < shape[1]; ++j) {
                    const T* input_ptr = input + indexing::at(i, j, strides);
                    serialize(input_ptr, slice_strides, slice_shape,
                              buffer.get(), data_type,
                              clamp, swap_endian);
                    output.write(buffer_ptr, static_cast<std::streamsize>(bytes_per_slice));
                    if (output.fail()) {
                        output.clear();
                        NOA_THROW("Stream error. Failed while writing {} bytes", bytes_per_slice);
                    }
                }
            }
        }
    }

    void serialize(const void* input, DataType input_data_type, size4_t strides, size4_t shape,
                   std::ostream& output, DataType output_data_type,
                   bool clamp, bool swap_endian) {
        switch (input_data_type) {
            case DataType::INT8:
                return serialize(reinterpret_cast<const int8_t*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::UINT8:
                return serialize(reinterpret_cast<const uint8_t*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::INT16:
                return serialize(reinterpret_cast<const int16_t*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::UINT16:
                return serialize(reinterpret_cast<const uint16_t*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::INT32:
                return serialize(reinterpret_cast<const int32_t*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::UINT32:
                return serialize(reinterpret_cast<const uint32_t*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::INT64:
                return serialize(reinterpret_cast<const int64_t*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::UINT64:
                return serialize(reinterpret_cast<const uint64_t*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::FLOAT16:
                return serialize(reinterpret_cast<const half_t*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::FLOAT32:
                return serialize(reinterpret_cast<const float*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::FLOAT64:
                return serialize(reinterpret_cast<const double*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::CFLOAT16:
                return serialize(reinterpret_cast<const chalf_t*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::CFLOAT32:
                return serialize(reinterpret_cast<const cfloat_t*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            case DataType::CFLOAT64:
                return serialize(reinterpret_cast<const cdouble_t*>(input), strides, shape,
                                 output, output_data_type,
                                 clamp, swap_endian);
            default:
                NOA_THROW("data type {} cannot be converted into a supported real type", input_data_type);
        }
    }
}

namespace noa::io {
    template<typename T>
    void deserialize(const byte_t* input, DataType data_type,
                     T* output, size4_t strides,
                     size4_t shape, bool clamp, bool swap_endian) {
        switch (data_type) {
            case DataType::UINT4:
                if constexpr (traits::is_scalar_v<T>) {
                    NOA_ASSERT(all(shape > 0));
                    NOA_ASSERT(indexing::areContiguous(strides, shape));
                    const size_t elements = shape.elements();
                    if (!(shape[3] % 2)) { // if even, data can be deserialized contiguously
                        deserializeRow4bits_(input, output, elements, false);
                    } else { // otherwise, there's a "padding" of 4bits at the end of each row
                        size_t rows = shape[0] * shape[1] * shape[2];
                        size_t bytes_per_row = (shape[3] + 1) / 2;
                        for (size_t row = 0; row < rows; ++row)
                            deserializeRow4bits_(input + bytes_per_row * row,
                                                 output + shape[3] * row,
                                                 shape[3], true);
                    }
                    break;
                }
            case DataType::INT8:
                if constexpr (traits::is_scalar_v<T>)
                    return deserialize_<int8_t>(input, output, strides, shape, clamp, swap_endian);
            case DataType::UINT8:
                if constexpr (traits::is_scalar_v<T>)
                    return deserialize_<uint8_t>(input, output, strides, shape, clamp, swap_endian);
            case DataType::INT16:
                if constexpr (traits::is_scalar_v<T>)
                    return deserialize_<int16_t>(input, output, strides, shape, clamp, swap_endian);
            case DataType::UINT16:
                if constexpr (traits::is_scalar_v<T>)
                    return deserialize_<uint16_t>(input, output, strides, shape, clamp, swap_endian);
            case DataType::INT32:
                if constexpr (traits::is_scalar_v<T>)
                    return deserialize_<int32_t>(input, output, strides, shape, clamp, swap_endian);
            case DataType::UINT32:
                if constexpr (traits::is_scalar_v<T>)
                    return deserialize_<uint32_t>(input, output, strides, shape, clamp, swap_endian);
            case DataType::INT64:
                if constexpr (traits::is_scalar_v<T>)
                    return deserialize_<int64_t>(input, output, strides, shape, clamp, swap_endian);
            case DataType::UINT64:
                if constexpr (traits::is_scalar_v<T>)
                    return deserialize_<uint64_t>(input, output, strides, shape, clamp, swap_endian);
            case DataType::FLOAT16:
                if constexpr (traits::is_scalar_v<T>)
                    return deserialize_<half_t>(input, output, strides, shape, clamp, swap_endian);
            case DataType::FLOAT32:
                if constexpr (traits::is_scalar_v<T>)
                    return deserialize_<float>(input, output, strides, shape, clamp, swap_endian);
            case DataType::FLOAT64:
                if constexpr (traits::is_scalar_v<T>)
                    return deserialize_<double>(input, output, strides, shape, clamp, swap_endian);
            case DataType::CINT16:
                if constexpr (traits::is_complex_v<T>) {
                    using real_t = typename T::value_type;
                    const auto real = indexing::Reinterpret(shape, strides, output).template as<real_t>();
                    return deserialize(input, DataType::INT16,
                                       real.ptr, real.strides,
                                       real.shape, clamp, swap_endian);
                }
            case DataType::CFLOAT16:
                if constexpr (traits::is_complex_v<T>) {
                    using real_t = typename T::value_type;
                    const auto real = indexing::Reinterpret(shape, strides, output).template as<real_t>();
                    return deserialize(input, DataType::FLOAT16,
                                       real.ptr, real.strides,
                                       real.shape, clamp, swap_endian);
                }
            case DataType::CFLOAT32:
                if constexpr (traits::is_complex_v<T>) {
                    using real_t = typename T::value_type;
                    const auto real = indexing::Reinterpret(shape, strides, output).template as<real_t>();
                    return deserialize(input, DataType::FLOAT32,
                                       real.ptr, real.strides,
                                       real.shape, clamp, swap_endian);
                }
            case DataType::CFLOAT64:
                if constexpr (traits::is_complex_v<T>) {
                    using real_t = typename T::value_type;
                    const auto real = indexing::Reinterpret(shape, strides, output).template as<real_t>();
                    return deserialize(input, DataType::FLOAT64,
                                       real.ptr, real.strides,
                                       real.shape, clamp, swap_endian);
                }
            default:
                NOA_THROW("data type {} cannot be deserialized into {}", data_type, string::human<T>());
        }
    }

    void deserialize(const byte_t* input, DataType input_data_type,
                     void* output, DataType output_data_type, size4_t strides,
                     size4_t shape, bool clamp, bool swap_endian) {
        switch (output_data_type) {
            case DataType::INT8:
                return deserialize(input, input_data_type,
                                   static_cast<int8_t*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::UINT8:
                return deserialize(input, input_data_type,
                                   static_cast<uint8_t*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::INT16:
                return deserialize(input, input_data_type,
                                   static_cast<int16_t*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::UINT16:
                return deserialize(input, input_data_type,
                                   static_cast<uint16_t*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::INT32:
                return deserialize(input, input_data_type,
                                   static_cast<int32_t*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::UINT32:
                return deserialize(input, input_data_type,
                                   static_cast<uint32_t*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::INT64:
                return deserialize(input, input_data_type,
                                   static_cast<int64_t*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::UINT64:
                return deserialize(input, input_data_type,
                                   static_cast<uint64_t*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::FLOAT16:
                return deserialize(input, input_data_type,
                                   static_cast<half_t*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::FLOAT32:
                return deserialize(input, input_data_type,
                                   static_cast<float*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::FLOAT64:
                return deserialize(input, input_data_type,
                                   static_cast<double*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::CFLOAT16:
                return deserialize(input, input_data_type,
                                   static_cast<chalf_t*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::CFLOAT32:
                return deserialize(input, input_data_type,
                                   static_cast<cfloat_t*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::CFLOAT64:
                return deserialize(input, input_data_type,
                                   static_cast<cdouble_t*>(output), strides,
                                   shape, clamp, swap_endian);
            default:
                NOA_THROW("data type {} cannot be converted into a supported real type", output_data_type);
        }
    }

    template<typename T>
    void deserialize(std::istream& input, DataType data_type,
                     T* output, size4_t strides,
                     size4_t shape, bool clamp, bool swap_endian) {
        input.clear();
        const bool are_contiguous = indexing::areContiguous(strides, shape);
        const size_t elements = shape.elements();

        if (are_contiguous && data_type == dtype<T>()) {
            input.read(reinterpret_cast<char*>(output), static_cast<std::streamsize>(sizeof(T) * elements));
            if (input.fail()) {
                input.clear();
                NOA_THROW("Stream error. Failed while reading {} bytes", sizeof(T) * elements);
            } else if (swap_endian) {
                if constexpr (traits::is_complex_v<T>)
                    swapEndian(reinterpret_cast<byte_t*>(output), elements * 2, sizeof(T) / 2);
                else
                    swapEndian(reinterpret_cast<byte_t*>(output), elements, sizeof(T));
            }
            return;

        } else if (data_type == DataType::UINT4) {
            const size_t bytes = serializedSize(DataType::UINT4, elements, shape[3]);
            std::unique_ptr<byte_t[]> buffer = std::make_unique<byte_t[]>(bytes);
            input.read(reinterpret_cast<char*>(buffer.get()), static_cast<std::streamsize>(bytes));
            if (input.fail()) {
                input.clear();
                NOA_THROW("Stream error. Failed while reading {} bytes", bytes);
            }
            deserialize(buffer.get(), DataType::UINT4,
                        output, strides,
                        shape, clamp);

        } else if (are_contiguous) {
            constexpr size_t bytes_per_batch = 1 << 26; // 67MB
            const size_t bytes_per_element = serializedSize(data_type, 1);
            size_t bytes_remain = bytes_per_element * elements;
            size_t bytes_buffer = bytes_remain > bytes_per_batch ? bytes_per_batch : bytes_remain;
            std::unique_ptr<byte_t[]> buffer = std::make_unique<byte_t[]>(bytes_buffer);
            auto* buffer_ptr = reinterpret_cast<char*>(buffer.get());

            // Read until there's nothing left.
            for (; bytes_remain > 0; bytes_remain -= bytes_buffer) {
                bytes_buffer = std::min(bytes_remain, bytes_buffer);
                const size_t elements_buffer = bytes_buffer / bytes_per_element;
                const size4_t buffer_shape{1, 1, 1, elements_buffer};

                // Read, swap and deserialize according to data type.
                input.read(buffer_ptr, static_cast<std::streamsize>(bytes_buffer));
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
            const size_t elements_per_slice = shape[2] * shape[3];
            const size_t bytes_per_slice = serializedSize(data_type, elements_per_slice);
            const size4_t slice_shape{1, 1, shape[2], shape[3]};
            const size4_t slice_strides{0, 0, strides[2], strides[3]};
            std::unique_ptr<byte_t[]> buffer = std::make_unique<byte_t[]>(bytes_per_slice);
            auto* buffer_ptr = reinterpret_cast<char*>(buffer.get());

            for (size_t i = 0; i < shape[0]; ++i) {
                for (size_t j = 0; j < shape[1]; ++j) {
                    input.read(buffer_ptr, static_cast<std::streamsize>(bytes_per_slice));
                    if (input.fail()) {
                        input.clear();
                        NOA_THROW("Stream error. Failed while reading {} bytes", bytes_per_slice);
                    }
                    T* output_ptr = output + indexing::at(i, j, strides);
                    deserialize(buffer.get(), data_type,
                                output_ptr, slice_strides,
                                slice_shape, clamp, swap_endian);
                }
            }
        }
    }

    void deserialize(std::istream& input, DataType input_data_type,
                     void* output, DataType output_data_type, size4_t strides,
                     size4_t shape, bool clamp, bool swap_endian) {
        switch (output_data_type) {
            case DataType::INT8:
                return deserialize(input, input_data_type,
                                   static_cast<int8_t*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::UINT8:
                return deserialize(input, input_data_type,
                                   static_cast<uint8_t*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::INT16:
                return deserialize(input, input_data_type,
                                   static_cast<int16_t*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::UINT16:
                return deserialize(input, input_data_type,
                                   static_cast<uint16_t*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::INT32:
                return deserialize(input, input_data_type,
                                   static_cast<int32_t*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::UINT32:
                return deserialize(input, input_data_type,
                                   static_cast<uint32_t*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::INT64:
                return deserialize(input, input_data_type,
                                   static_cast<int64_t*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::UINT64:
                return deserialize(input, input_data_type,
                                   static_cast<uint64_t*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::FLOAT16:
                return deserialize(input, input_data_type,
                                   static_cast<half_t*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::FLOAT32:
                return deserialize(input, input_data_type,
                                   static_cast<float*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::FLOAT64:
                return deserialize(input, input_data_type,
                                   static_cast<double*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::CFLOAT16:
                return deserialize(input, input_data_type,
                                   static_cast<chalf_t*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::CFLOAT32:
                return deserialize(input, input_data_type,
                                   static_cast<cfloat_t*>(output), strides,
                                   shape, clamp, swap_endian);
            case DataType::CFLOAT64:
                return deserialize(input, input_data_type,
                                   static_cast<cdouble_t*>(output), strides,
                                   shape, clamp, swap_endian);
            default:
                NOA_THROW("data type {} cannot be converted into a supported real type", output_data_type);
        }
    }
}
