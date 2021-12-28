#include "noa/common/io/IO.h"
#include "noa/common/Exception.h"

namespace noa::io::details {
    // Converts input into the desired DTYPE and serializes the result in output.
    // If the input values don't fit in the DTYPE range or if we simply don't know, clamp should be true.
    template<typename DTYPE, typename T>
    inline void serialize(const T* input, char* output, size_t elements, bool clamp) {
        if constexpr (std::is_same_v<T, DTYPE>) {
            // On platform where sizeof(long) == 8, there's the possibility that T=long long and DTYPE=long.
            // We could check for this since a simple memcpy should be fine between these two types.
            std::memcpy(output, reinterpret_cast<const char*>(input), elements * sizeof(T)); // cast isn't necessary
        } else {
            DTYPE tmp;
            if (clamp) {
                for (size_t idx= 0; idx < elements; ++idx) {
                    tmp = clamp_cast<DTYPE>(input[idx]);
                    std::memcpy(output + idx * sizeof(DTYPE), &tmp, sizeof(DTYPE));
                }
            } else {
                for (size_t idx= 0; idx < elements; ++idx) {
                    tmp = static_cast<DTYPE>(input[idx]);
                    std::memcpy(output + idx * sizeof(DTYPE), &tmp, sizeof(DTYPE));
                }
            }
        }
    }

    template<typename DTYPE, typename T>
    inline void deserialize(const char* input, T* output, size_t elements, bool clamp) {
        if constexpr (std::is_same_v<T, DTYPE>) {
            std::memcpy(reinterpret_cast<char*>(output), input, elements * sizeof(T));
        } else {
            DTYPE tmp;
            if (clamp) {
                for (size_t idx= 0; idx < elements; ++idx) {
                    std::memcpy(&tmp, input + idx * sizeof(DTYPE), sizeof(DTYPE));
                    output[idx] = clamp_cast<T>(tmp);
                }
            } else {
                for (size_t idx= 0; idx < elements; ++idx) {
                    std::memcpy(&tmp, input + idx * sizeof(DTYPE), sizeof(DTYPE));
                    output[idx] = static_cast<T>(tmp);
                }
            }
        }
    }

    template<typename T>
    void serializeRow4bits(const T* input, char* output, size_t elements_row, bool is_odd, bool clamp) {
        // The order of the first and second elements in the output are the 4 LSB and 4 MSB of the CPU, respectively.
        // If the row has an odd number of elements, the last byte of the row has the 4 MSB unset.
        static_assert(noa::traits::is_scalar_v<T>);
        uint32_t tmp, l_val, h_val;

        if (clamp) {
            for (size_t i = 0; i < elements_row / 2; ++i) {
                l_val = clamp_cast<uint32_t>(input[2 * i]); // If IEEE float, round to nearest
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
            constexpr T half = noa::traits::is_float_v<T> ? static_cast<T>(0.5) : 0;
            for (size_t i = 0; i < elements_row / 2; ++i) {
                l_val = static_cast<uint32_t>(input[2 * i] + half);
                h_val = static_cast<uint32_t>(input[2 * i + 1] + half);
                tmp = l_val + (h_val << 4);
                std::memcpy(output + i, &tmp, 1);
            }
            if (is_odd) {
                l_val = static_cast<uint32_t>(input[elements_row - 1] + half);
                std::memcpy(output + elements_row / 2, &l_val, 1);
            }
        }
    }

    template<typename T>
    inline void deserializeRow4bits(const char* input, T* output, size_t elements_row, bool is_odd) {
        // This is assuming that the first and second elements are at the LSB and MSB of the CPU, respectively.
        // If the row has an odd number of elements, the last bytes has the 4 MSB unused.
        for (size_t i = 0; i < elements_row / 2; ++i) {
            char tmp = input[i];
            output[i * 2] = static_cast<T>(tmp & 0b00001111);
            output[i * 2 + 1] = static_cast<T>((tmp >> 4) & 0b00001111);
        }
        if (is_odd)
            output[elements_row - 1] = static_cast<T>(input[elements_row / 2] & 0b00001111);
    }
}

namespace noa::io {
    template<typename T>
    void serialize(const T* input, char* output, DataType data_type,
                   size_t elements, bool clamp, bool swap_endian, size_t elements_per_row) {
        switch (data_type) {
            case DataType::UINT4:
                if constexpr (noa::traits::is_scalar_v<T>) {
                    if (elements_per_row == 0 || !(elements_per_row % 2)) { // i.e. contiguous serialized data
                        details::serializeRow4bits(input, output, elements, false, clamp);
                    } else {
                        NOA_ASSERT(!(elements % elements_per_row)); // otherwise, last partial row is ignored
                        size_t rows = elements / elements_per_row;
                        size_t bytes_per_row = (elements_per_row + 1) / 2;
                        for (size_t row = 0; row < rows; ++row)
                            details::serializeRow4bits(input + elements_per_row * row,
                                                       output + bytes_per_row * row,
                                                       elements_per_row, true, clamp);
                    }
                    break;
                }
            case DataType::INT8:
                if constexpr (noa::traits::is_scalar_v<T>) {
                    details::serialize<int8_t>(input, output, elements, clamp);
                    break;
                }
            case DataType::UINT8:
                if constexpr (noa::traits::is_scalar_v<T>) {
                    details::serialize<uint8_t>(input, output, elements, clamp);
                    break;
                }
            case DataType::INT16:
                if constexpr (noa::traits::is_scalar_v<T>) {
                    details::serialize<int16_t>(input, output, elements, clamp);
                    break;
                }
            case DataType::UINT16:
                if constexpr (noa::traits::is_scalar_v<T>) {
                    details::serialize<uint16_t>(input, output, elements, clamp);
                    break;
                }
            case DataType::INT32:
                if constexpr (noa::traits::is_scalar_v<T>) {
                    details::serialize<int32_t>(input, output, elements, clamp);
                    break;
                }
            case DataType::UINT32:
                if constexpr (noa::traits::is_scalar_v<T>) {
                    details::serialize<uint32_t>(input, output, elements, clamp);
                    break;
                }
            case DataType::INT64:
                if constexpr (noa::traits::is_scalar_v<T>) {
                    details::serialize<int64_t>(input, output, elements, clamp);
                    break;
                }
            case DataType::UINT64:
                if constexpr (noa::traits::is_scalar_v<T>) {
                    details::serialize<uint64_t>(input, output, elements, clamp);
                    break;
                }
            case DataType::FLOAT32:
                if constexpr (noa::traits::is_scalar_v<T>) {
                    details::serialize<float>(input, output, elements, clamp);
                    break;
                }
            case DataType::FLOAT64:
                if constexpr (noa::traits::is_scalar_v<T>) {
                    details::serialize<double>(input, output, elements, clamp);
                    break;
                }
            case DataType::CINT16:
                if constexpr (noa::traits::is_complex_v<T>) {
                    using real_t = typename T::value_type;
                    return serialize(reinterpret_cast<const real_t*>(input),
                                     output, DataType::INT16, elements * 2, clamp, swap_endian);
                }
            case DataType::CFLOAT32:
                if constexpr (noa::traits::is_complex_v<T>) {
                    using real_t = typename T::value_type;
                    return serialize(reinterpret_cast<const real_t*>(input),
                                     output, DataType::FLOAT32, elements * 2, clamp, swap_endian);
                }
            case DataType::CFLOAT64:
                if constexpr (noa::traits::is_complex_v<T>) {
                    using real_t = typename T::value_type;
                    return serialize(reinterpret_cast<const real_t*>(input),
                                     output, DataType::FLOAT64, elements * 2, clamp, swap_endian);
                }
            default:
                NOA_THROW("{} cannot be serialized into the data type {}", string::typeName<T>(), data_type);
        }
        if (swap_endian)
            swapEndian(output, elements, getSerializedSize(data_type, 1));
    }

    void serialize(const void* input, DataType input_data_type, char* output, DataType output_data_type,
                   size_t elements, bool clamp, bool swap_endian, size_t elements_per_row) {
        switch (input_data_type) {
            case DataType::INT8:
                return serialize(reinterpret_cast<const int8_t*>(input), output, output_data_type,
                                 elements, clamp, swap_endian, elements_per_row);
            case DataType::UINT8:
                return serialize(reinterpret_cast<const uint8_t*>(input), output, output_data_type,
                                 elements, clamp, swap_endian, elements_per_row);
            case DataType::INT16:
                return serialize(reinterpret_cast<const int16_t*>(input), output, output_data_type,
                                 elements, clamp, swap_endian, elements_per_row);
            case DataType::UINT16:
                return serialize(reinterpret_cast<const uint16_t*>(input), output, output_data_type,
                                 elements, clamp, swap_endian, elements_per_row);
            case DataType::INT32:
                return serialize(reinterpret_cast<const int32_t*>(input), output, output_data_type,
                                 elements, clamp, swap_endian, elements_per_row);
            case DataType::UINT32:
                return serialize(reinterpret_cast<const uint32_t*>(input), output, output_data_type,
                                 elements, clamp, swap_endian, elements_per_row);
            case DataType::INT64:
                // Reinterpreting to int64_t on platforms where int64_t==long *might* break the
                // strict aliasing rule since T could have been long long originally. While it
                // shouldn't cause any issue, we might recommend using fixed-size integers in this case.
                return serialize(reinterpret_cast<const int64_t*>(input), output, output_data_type,
                                 elements, clamp, swap_endian, elements_per_row);
            case DataType::UINT64:
                return serialize(reinterpret_cast<const uint64_t*>(input), output, output_data_type,
                                 elements, clamp, swap_endian, elements_per_row);
            case DataType::FLOAT32:
                return serialize(reinterpret_cast<const float*>(input), output, output_data_type,
                                 elements, clamp, swap_endian, elements_per_row);
            case DataType::FLOAT64:
                return serialize(reinterpret_cast<const double*>(input), output, output_data_type,
                                 elements, clamp, swap_endian, elements_per_row);
            case DataType::CFLOAT32:
                return serialize(reinterpret_cast<const cfloat_t*>(input), output, output_data_type,
                                 elements, clamp, swap_endian, elements_per_row);
            case DataType::CFLOAT64:
                return serialize(reinterpret_cast<const cdouble_t*>(input), output, output_data_type,
                                 elements, clamp, swap_endian, elements_per_row);
            default:
                NOA_THROW("data type {} cannot be converted into a supported real type", input_data_type);
        }
    }

    template<typename T>
    void serialize(const T* input, std::ostream& output, DataType data_type, size_t elements,
                   bool clamp, bool swap_endian, size_t elements_per_row) {
        // Ignore all previous errors on that stream. If these errors cannot be recovered from,
        // the failbit will be reset by write() anyway and an exception will be thrown.
        output.clear();

        // Shortcut: doesn't need any conversion
        if (!swap_endian && data_type == getDataType<T>()) {
            output.write(reinterpret_cast<const char*>(input), static_cast<std::streamsize>(sizeof(T) * elements));
            if (output.fail()) {
                output.clear();
                NOA_THROW("Stream error. Failed while writing {} bytes", sizeof(T) * elements);
            }
            return;

        } else if (data_type == DataType::UINT4) {
            // UINT4 is not necessarily contiguous, so ignore the batch system and read all at once.
            size_t bytes = getSerializedSize(DataType::UINT4, elements, elements_per_row);
            std::unique_ptr<char[]> buffer = std::make_unique<char[]>(bytes);
            serialize(input, buffer.get(), DataType::UINT4, elements, clamp, false, elements_per_row);
            output.write(buffer.get(), static_cast<std::streamsize>(bytes));
            if (output.fail()) {
                output.clear();
                NOA_THROW("Stream error. Failed while writing {} bytes", bytes);
            }

        } else {
            constexpr size_t bytes_per_batch = 1 << 26;
            size_t bytes_per_element = getSerializedSize(data_type, 1);
            size_t bytes_remain = bytes_per_element * elements;
            size_t bytes_buffer = bytes_remain > bytes_per_batch ? bytes_per_batch : bytes_remain;
            std::unique_ptr<char[]> buffer = std::make_unique<char[]>(bytes_buffer);

            // Read until there's nothing left.
            for (; bytes_remain > 0; bytes_remain -= bytes_buffer) {
                bytes_buffer = std::min(bytes_remain, bytes_buffer);
                size_t elements_buffer = bytes_buffer / bytes_per_element;

                // Serialize according to data type and write.
                serialize(input, buffer.get(), data_type, elements_buffer, clamp, swap_endian);
                output.write(buffer.get(), static_cast<std::streamsize>(bytes_buffer));
                if (output.fail()) {
                    output.clear();
                    NOA_THROW("Stream error. Failed while writing {} bytes", bytes_buffer);
                }

                input += elements_buffer;
            }
        }
    }

    void serialize(const void* input, DataType input_data_type, std::ostream& output, DataType output_data_type,
                   size_t elements, bool clamp, bool swap_endian, size_t elements_per_row) {
        switch (input_data_type) {
            case DataType::INT8:
                return serialize(reinterpret_cast<const int8_t*>(input), output, output_data_type,
                                 elements, clamp, swap_endian, elements_per_row);
            case DataType::UINT8:
                return serialize(reinterpret_cast<const uint8_t*>(input), output, output_data_type,
                                 elements, clamp, swap_endian, elements_per_row);
            case DataType::INT16:
                return serialize(reinterpret_cast<const int16_t*>(input), output, output_data_type,
                                 elements, clamp, swap_endian, elements_per_row);
            case DataType::UINT16:
                return serialize(reinterpret_cast<const uint16_t*>(input), output, output_data_type,
                                 elements, clamp, swap_endian, elements_per_row);
            case DataType::INT32:
                return serialize(reinterpret_cast<const int32_t*>(input), output, output_data_type,
                                 elements, clamp, swap_endian, elements_per_row);
            case DataType::UINT32:
                return serialize(reinterpret_cast<const uint32_t*>(input), output, output_data_type,
                                 elements, clamp, swap_endian, elements_per_row);
            case DataType::INT64:
                return serialize(reinterpret_cast<const int64_t*>(input), output, output_data_type,
                                 elements, clamp, swap_endian, elements_per_row);
            case DataType::UINT64:
                return serialize(reinterpret_cast<const uint64_t*>(input), output, output_data_type,
                                 elements, clamp, swap_endian, elements_per_row);
            case DataType::FLOAT32:
                return serialize(reinterpret_cast<const float*>(input), output, output_data_type,
                                 elements, clamp, swap_endian, elements_per_row);
            case DataType::FLOAT64:
                return serialize(reinterpret_cast<const double*>(input), output, output_data_type,
                                 elements, clamp, swap_endian, elements_per_row);
            case DataType::CFLOAT32:
                return serialize(reinterpret_cast<const cfloat_t*>(input), output, output_data_type,
                                 elements, clamp, swap_endian, elements_per_row);
            case DataType::CFLOAT64:
                return serialize(reinterpret_cast<const cdouble_t*>(input), output, output_data_type,
                                 elements, clamp, swap_endian, elements_per_row);
            default:
                NOA_THROW("data type {} cannot be converted into a supported real type", input_data_type);
        }
    }

    template<typename T>
    void deserialize(const char* input, DataType data_type, T* output,
                     size_t elements, bool clamp, size_t elements_per_row) {
        switch (data_type) {
            case DataType::UINT4:
                if (elements_per_row == 0 || !(elements_per_row % 2)) {
                    details::deserializeRow4bits(input, output, elements, false);
                } else {
                    NOA_ASSERT(!(elements % elements_per_row)); // otherwise, last partial row is ignored
                    size_t rows = elements / elements_per_row;
                    size_t bytes_per_row = (elements_per_row + 1) / 2;
                    for (size_t row = 0; row < rows; ++row)
                        details::deserializeRow4bits(input + bytes_per_row * row,
                                                     output + elements_per_row * row,
                                                     elements_per_row, true);
                }
                break;
            case DataType::INT8:
                return details::deserialize<int8_t>(input, output, elements, clamp);
            case DataType::UINT8:
                return details::deserialize<uint8_t>(input, output, elements, clamp);
            case DataType::INT16:
                return details::deserialize<int16_t>(input, output, elements, clamp);
            case DataType::UINT16:
                return details::deserialize<uint16_t>(input, output, elements, clamp);
            case DataType::INT32:
                return details::deserialize<int32_t>(input, output, elements, clamp);
            case DataType::UINT32:
                return details::deserialize<uint32_t>(input, output, elements, clamp);
            case DataType::INT64:
                return details::deserialize<int64_t>(input, output, elements, clamp);
            case DataType::UINT64:
                return details::deserialize<uint64_t>(input, output, elements, clamp);
            case DataType::FLOAT32:
                return details::deserialize<float>(input, output, elements, clamp);
            case DataType::FLOAT64:
                return details::deserialize<double>(input, output, elements, clamp);
            case DataType::CINT16:
                if constexpr (noa::traits::is_complex_v<T>) {
                    using real_t = typename T::value_type;
                    return deserialize(input, DataType::INT16,
                                       reinterpret_cast<real_t*>(output), elements * 2, clamp);
                }
            case DataType::CFLOAT32:
                if constexpr (noa::traits::is_complex_v<T>) {
                    using real_t = typename T::value_type;
                    return deserialize(input, DataType::FLOAT32,
                                       reinterpret_cast<real_t*>(output), elements * 2, clamp);
                }
            case DataType::CFLOAT64:
                if constexpr (noa::traits::is_complex_v<T>) {
                    using real_t = typename T::value_type;
                    return deserialize(input, DataType::FLOAT64,
                                       reinterpret_cast<real_t*>(output), elements * 2, clamp);
                }
            default:
                NOA_THROW("data type {} cannot be deserialized into {}", data_type, string::typeName<T>());
        }
    }

    void deserialize(const char* input, DataType input_data_type, void* output, DataType output_data_type,
                     size_t elements, bool clamp, size_t elements_per_row) {
        switch (output_data_type) {
            case DataType::INT8:
                return deserialize(input, input_data_type, static_cast<int8_t*>(output),
                                   elements, clamp, elements_per_row);
            case DataType::UINT8:
                return deserialize(input, input_data_type, static_cast<uint8_t*>(output),
                                   elements, clamp, elements_per_row);
            case DataType::INT16:
                return deserialize(input, input_data_type, static_cast<int16_t*>(output),
                                   elements, clamp, elements_per_row);
            case DataType::UINT16:
                return deserialize(input, input_data_type, static_cast<uint16_t*>(output),
                                   elements, clamp, elements_per_row);
            case DataType::INT32:
                return deserialize(input, input_data_type, static_cast<int32_t*>(output),
                                   elements, clamp, elements_per_row);
            case DataType::UINT32:
                return deserialize(input, input_data_type, static_cast<uint32_t*>(output),
                                   elements, clamp, elements_per_row);
            case DataType::INT64:
                return deserialize(input, input_data_type, static_cast<int64_t*>(output),
                                   elements, clamp, elements_per_row);
            case DataType::UINT64:
                return deserialize(input, input_data_type, static_cast<uint64_t*>(output),
                                   elements, clamp, elements_per_row);
            case DataType::FLOAT32:
                return deserialize(input, input_data_type, static_cast<float*>(output),
                                   elements, clamp, elements_per_row);
            case DataType::FLOAT64:
                return deserialize(input, input_data_type, static_cast<double*>(output),
                                   elements, clamp, elements_per_row);
            case DataType::CFLOAT32:
                return deserialize(input, input_data_type, static_cast<cfloat_t*>(output),
                                   elements, clamp, elements_per_row);
            case DataType::CFLOAT64:
                return deserialize(input, input_data_type, static_cast<cdouble_t*>(output),
                                   elements, clamp, elements_per_row);
            default:
                NOA_THROW("data type {} cannot be converted into a supported real type", output_data_type);
        }
    }

    template<typename T>
    void deserialize(std::istream& input, DataType data_type, T* output, size_t elements,
                     bool clamp, bool swap_endian, size_t elements_per_row) {
        input.clear();

        // Shortcut: doesn't need any conversion
        if (data_type == getDataType<T>()) {
            input.read(reinterpret_cast<char*>(output), static_cast<std::streamsize>(sizeof(T) * elements));
            if (input.fail()) {
                input.clear();
                NOA_THROW("Stream error. Failed while reading {} bytes", sizeof(T) * elements);
            } else if (swap_endian) {
                if constexpr (noa::traits::is_complex_v<T>)
                    swapEndian(reinterpret_cast<char*>(output), elements * 2, sizeof(T) / 2);
                else
                    swapEndian(reinterpret_cast<char*>(output), elements, sizeof(T));
            }
            return;

        } else if (data_type == DataType::UINT4) {
            // UINT4 is not necessarily contiguous, so ignore the batch system and read all at once.
            size_t bytes = getSerializedSize(DataType::UINT4, elements, elements_per_row);
            std::unique_ptr<char[]> buffer = std::make_unique<char[]>(bytes);
            input.read(buffer.get(), static_cast<std::streamsize>(bytes));
            if (input.fail()) {
                input.clear();
                NOA_THROW("Stream error. Failed while reading {} bytes", bytes);
            }
            deserialize(buffer.get(), DataType::UINT4, output, elements, clamp, elements_per_row);

        } else {
            constexpr size_t bytes_per_batch = 1 << 26; // 2^26
            size_t bytes_per_element = getSerializedSize(data_type, 1);
            size_t bytes_remain = bytes_per_element * elements;
            size_t bytes_buffer = bytes_remain > bytes_per_batch ? bytes_per_batch : bytes_remain;
            std::unique_ptr<char[]> buffer = std::make_unique<char[]>(bytes_buffer);

            // Read until there's nothing left.
            for (; bytes_remain > 0; bytes_remain -= bytes_buffer) {
                bytes_buffer = std::min(bytes_remain, bytes_buffer);
                size_t elements_buffer = bytes_buffer / bytes_per_element;

                // Read, swap and deserialize according to data type.
                input.read(buffer.get(), static_cast<std::streamsize>(bytes_buffer));
                if (input.fail()) {
                    input.clear();
                    NOA_THROW("Stream error. Failed while reading {} bytes", bytes_buffer);
                } else if (swap_endian) {
                    if constexpr (noa::traits::is_complex_v<T>)
                        swapEndian(buffer.get(), elements_buffer * 2, bytes_per_element / 2);
                    else
                        swapEndian(buffer.get(), elements_buffer, bytes_per_element);
                }
                deserialize(buffer.get(), data_type, output, elements_buffer, clamp);

                output += elements_buffer;
            }
        }
    }

    void deserialize(std::istream& input, DataType input_data_type, void* output, DataType output_data_type,
                     size_t elements, bool clamp, bool swap_endian, size_t elements_per_row) {
        switch (output_data_type) {
            case DataType::INT8:
                return deserialize(input, input_data_type, static_cast<int8_t*>(output),
                                   elements, clamp, swap_endian, elements_per_row);
            case DataType::UINT8:
                return deserialize(input, input_data_type, static_cast<uint8_t*>(output),
                                   elements, clamp, swap_endian, elements_per_row);
            case DataType::INT16:
                return deserialize(input, input_data_type, static_cast<int16_t*>(output),
                                   elements, clamp, swap_endian, elements_per_row);
            case DataType::UINT16:
                return deserialize(input, input_data_type, static_cast<uint16_t*>(output),
                                   elements, clamp, swap_endian, elements_per_row);
            case DataType::INT32:
                return deserialize(input, input_data_type, static_cast<int32_t*>(output),
                                   elements, clamp, swap_endian, elements_per_row);
            case DataType::UINT32:
                return deserialize(input, input_data_type, static_cast<uint32_t*>(output),
                                   elements, clamp, swap_endian, elements_per_row);
            case DataType::INT64:
                return deserialize(input, input_data_type, static_cast<int64_t*>(output),
                                   elements, clamp, swap_endian, elements_per_row);
            case DataType::UINT64:
                return deserialize(input, input_data_type, static_cast<uint64_t*>(output),
                                   elements, clamp, swap_endian, elements_per_row);
            case DataType::FLOAT32:
                return deserialize(input, input_data_type, static_cast<float*>(output),
                                   elements, clamp, swap_endian, elements_per_row);
            case DataType::FLOAT64:
                return deserialize(input, input_data_type, static_cast<double*>(output),
                                   elements, clamp, swap_endian, elements_per_row);
            case DataType::CFLOAT32:
                return deserialize(input, input_data_type, static_cast<cfloat_t*>(output),
                                   elements, clamp, swap_endian, elements_per_row);
            case DataType::CFLOAT64:
                return deserialize(input, input_data_type, static_cast<cdouble_t*>(output),
                                   elements, clamp, swap_endian, elements_per_row);
            default:
                NOA_THROW("data type {} cannot be converted into a supported real type", output_data_type);
        }
    }
}
