#include "noa/core/Error.hpp"
#include "noa/core/indexing/Layout.hpp"
#include "noa/core/io/IO.hpp"

namespace {
    using namespace ::noa;

    template<typename Output, typename Input>
    void serialize_1d_(
        SpanContiguous<const Input, 1> input,
        SpanContiguous<Byte, 1> output,
        bool clamp, bool swap_endian
    ) {
        const i64 n_elements = input.ssize();
        const size_t n_bytes = static_cast<size_t>(n_elements) * sizeof(Output);
        check(n_bytes >= output.size(),
              "The size of the output buffer is not big enough to contain the input buffer");

        constexpr i64 OUTPUT_SIZE = sizeof(Output);
        if constexpr (nt::same_as<Input, Output>) {
            // TODO On platform where sizeof(long) == 8, there's the possibility that T=long long and Output=long.
            //      We could check for this since a simple memcpy should be fine between these two types.
            std::memcpy(output.get(), input.get(), static_cast<size_t>(n_bytes));
        } else {
            Output tmp;
            if (clamp) {
                for (i64 idx{}; idx < n_elements; ++idx) {
                    tmp = clamp_cast<Output>(input[idx]);
                    std::memcpy(output.get() + idx * OUTPUT_SIZE, &tmp, OUTPUT_SIZE);
                }
            } else {
                for (i64 idx{}; idx < n_elements; ++idx) {
                    tmp = static_cast<Output>(input[idx]);
                    std::memcpy(output.get() + idx * OUTPUT_SIZE, &tmp, OUTPUT_SIZE);
                }
            }
        }
        // TODO Merge this on the conversion loop?
        if (swap_endian)
            noa::io::swap_endian(output.get(), n_elements, OUTPUT_SIZE);
    }

    template<typename Output, typename Input>
    void serialize_4d_(
        const Span<const Input, 4>& input,
        const SpanContiguous<Byte, 1>& output,
        bool clamp,
        bool swap_endian
    ) {
        if (input.are_contiguous())
            return serialize_1d_<Output>(input.as_contiguous_1d(), output, clamp, swap_endian);

        const i64 n_elements = input.ssize();
        const size_t n_bytes = static_cast<size_t>(n_elements) * sizeof(Output);
        check(n_bytes >= output.size(),
              "The size of the output buffer is not big enough to contain the input buffer");

        constexpr i64 OUTPUT_SIZE = sizeof(Output);
        Output tmp;
        i64 idx{};
        // TODO Move the if inside the loop since branch prediction should take care of it.
        //      Although I'm not sure the compiler will see through the memcpy with the branch.
        //      Compiler explorer help!
        if (clamp) {
            for (i64 i{}; i < input.shape()[0]; ++i) {
                for (i64 j{}; j < input.shape()[1]; ++j) {
                    for (i64 k{}; k < input.shape()[2]; ++k) {
                        for (i64 l{}; l < input.shape()[3]; ++l, ++idx) {
                            tmp = clamp_cast<Output>(input(i, j, k, l));
                            std::memcpy(output.get() + idx * OUTPUT_SIZE, &tmp, sizeof(Output));
                        }
                    }
                }
            }
        } else {
            for (i64 i{}; i < input.shape()[0]; ++i) {
                for (i64 j{}; j < input.shape()[1]; ++j) {
                    for (i64 k{}; k < input.shape()[2]; ++k) {
                        for (i64 l{}; l < input.shape()[3]; ++l, ++idx) {
                            tmp = static_cast<Output>(input(i, j, k, l));
                            std::memcpy(output.get() + idx * OUTPUT_SIZE, &tmp, sizeof(Output));
                        }
                    }
                }
            }
        }
        if (swap_endian)
            noa::io::swap_endian(output.get(), n_elements, sizeof(Output));
    }

    template<typename Input, typename Output>
    void deserialize_1d_(
        SpanContiguous<const Byte, 1> input,
        SpanContiguous<Output, 1> output,
        bool clamp, bool swap_endian
    ) {
        const auto n_elements = output.ssize();
        if constexpr (nt::same_as<Output, Input>) {
            auto* output_ptr = reinterpret_cast<std::byte*>(output.get());
            std::memcpy(output_ptr, input.get(), static_cast<size_t>(n_elements) * sizeof(Output));
            if (swap_endian)
                noa::io::swap_endian(output_ptr, n_elements, sizeof(Input));
        } else {
            // Branch prediction should work nicely.
            // std::memcpy is removed.
            // std::reverse is translated in bswap
            // https://godbolt.org/z/Eavdcv8PM
            Input tmp;
            constexpr i64 INPUT_SIZE = sizeof(Input);
            for (i64 idx{}; idx < n_elements; ++idx) {
                std::memcpy(&tmp, input.get() + idx * INPUT_SIZE, sizeof(Input));
                if (swap_endian)
                    noa::io::guts::reverse<sizeof(Input)>(reinterpret_cast<std::byte*>(&tmp));
                output[idx] = clamp ? clamp_cast<Output>(tmp) : static_cast<Output>(tmp);
            }
        }
    }

    template<typename Input, typename Output>
    void deserialize_4d_(
        const SpanContiguous<const Byte, 1>& input,
        const Span<Output, 4>& output,
        bool clamp,
        bool swap_endian
    ) {
        if (output.are_contiguous())
            return deserialize_1d_<Input>(input, output.as_contiguous_1d(), clamp, swap_endian);

        constexpr i64 INPUT_SIZE = sizeof(Input);
        Input tmp;
        i64 idx{};
        for (i64 i{}; i < output.shape()[0]; ++i) {
            for (i64 j{}; j < output.shape()[1]; ++j) {
                for (i64 k{}; k < output.shape()[2]; ++k) {
                    for (i64 l{}; l < output.shape()[3]; ++l, ++idx) {
                        std::memcpy(&tmp, input.get() + idx * INPUT_SIZE, sizeof(Input));
                        if (swap_endian)
                            noa::io::guts::reverse<sizeof(Input)>(reinterpret_cast<std::byte*>(&tmp));
                        output(i, j, k, l) = clamp ? clamp_cast<Output>(tmp) : static_cast<Output>(tmp);
                    }
                }
            }
        }
    }

    template<typename T>
    void serialize_row_4bits_(const T* input, std::byte* output, i64 elements_row, bool is_odd, bool clamp) {
        // The order of the first and second elements in the output are the 4 LSB and 4 MSB of the CPU, respectively.
        // If the row has an odd number of elements, the last byte of the row has the 4 MSB unset.
        static_assert(nt::scalar<T>);
        u32 tmp{}, l_val{}, h_val{};

        if (clamp) {
            for (i64 i{}; i < elements_row / 2; ++i) {
                l_val = clamp_cast<u32>(input[2 * i]); // If IEEE float, default round to nearest
                h_val = clamp_cast<u32>(input[2 * i + 1]);
                l_val = noa::clamp(l_val, 0U, 15U); // 2^4-1
                h_val = noa::clamp(h_val, 0U, 15U);
                tmp = l_val + (h_val << 4);
                std::memcpy(output + i, &tmp, 1);
            }
            if (is_odd) {
                l_val = clamp_cast<u32>(input[elements_row - 1]);
                l_val = noa::clamp(l_val, 0U, 15U);
                std::memcpy(output + elements_row / 2, &l_val, 1);
            }
        } else {
            // std::round could be used instead, but we assume values are positive so +0.5f is enough
            for (i64 i{}; i < elements_row / 2; ++i) {
                l_val = static_cast<u32>(noa::round(input[2 * i]));
                h_val = static_cast<u32>(noa::round(input[2 * i + 1] ));
                tmp = l_val + (h_val << 4);
                std::memcpy(output + i, &tmp, 1);
            }
            if (is_odd) {
                l_val = static_cast<u32>(noa::round(input[elements_row - 1]));
                std::memcpy(output + elements_row / 2, &l_val, 1);
            }
        }
    }

    template<typename T>
    void deserialize_row_4bits_(const std::byte* input, T* output, i64 elements_row, bool is_odd) {
        // This is assuming that the first and second elements are at the LSB and MSB of the CPU, respectively.
        // If the row has an odd number of elements, the last bytes has the 4 MSB unused.
        constexpr unsigned char MASK_4LSB{0b00001111};
        for (i64 i{}; i < elements_row / 2; ++i) {
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
    template<nt::numeric T>
    void serialize(
        const Span<const T, 4>& input,
        const SpanContiguous<Byte, 1>& output,
        Encoding encoding
    ) {
        switch (encoding.format) {
            case Encoding::U4:
                if constexpr (nt::scalar<T>) {
                    check(not input.is_empty() and input.are_contiguous());
                    if (is_even(input.shape()[3])) { // if even, data can be serialized contiguously
                        serialize_row_4bits_(input.get(), output.get(), input.ssize(), false, encoding.clamp);
                    } else { // otherwise, there's a "padding" of 4bits at the end of each row
                        const auto n_rows = input.shape().pop_back().n_elements();
                        const auto bytes_per_row = (input.shape()[3] + 1) / 2;
                        for (i64 row{}; row < n_rows; ++row)
                            serialize_row_4bits_(input.get() + input.shape()[3] * row,
                                                 output.get() + bytes_per_row * row,
                                                 input.shape()[3], true, encoding.clamp);
                    }
                    return;
                }
                break;
            case Encoding::I8:
                if constexpr (nt::scalar<T>)
                    return serialize_4d_<i8>(input, output, encoding.clamp, encoding.endian_swap);
                break;
            case Encoding::U8:
                if constexpr (nt::scalar<T>)
                    return serialize_4d_<u8>(input, output, encoding.clamp, encoding.endian_swap);
                break;
            case Encoding::I16:
                if constexpr (nt::scalar<T>)
                    return serialize_4d_<i16>(input, output, encoding.clamp, encoding.endian_swap);
                break;
            case Encoding::U16:
                if constexpr (nt::scalar<T>)
                    return serialize_4d_<u16>(input, output, encoding.clamp, encoding.endian_swap);
                break;
            case Encoding::I32:
                if constexpr (nt::scalar<T>)
                    return serialize_4d_<i32>(input, output, encoding.clamp, encoding.endian_swap);
                break;
            case Encoding::U32:
                if constexpr (nt::scalar<T>)
                    return serialize_4d_<u32>(input, output, encoding.clamp, encoding.endian_swap);
                break;
            case Encoding::I64:
                if constexpr (nt::scalar<T>)
                    return serialize_4d_<i64>(input, output, encoding.clamp, encoding.endian_swap);
                break;
            case Encoding::U64:
                if constexpr (nt::scalar<T>)
                    return serialize_4d_<u64>(input, output, encoding.clamp, encoding.endian_swap);
                break;
            case Encoding::F16:
                if constexpr (nt::scalar<T>)
                    return serialize_4d_<f16>(input, output, encoding.clamp, encoding.endian_swap);
                break;
            case Encoding::F32:
                if constexpr (nt::scalar<T>)
                    return serialize_4d_<f32>(input, output, encoding.clamp, encoding.endian_swap);
                break;
            case Encoding::F64:
                if constexpr (nt::scalar<T>)
                    return serialize_4d_<f64>(input, output, encoding.clamp, encoding.endian_swap);
                break;
            case Encoding::CI16:
                if constexpr (nt::complex<T>) {
                    using real_t = const T::value_type;
                    auto new_encoding = Encoding{Encoding::I16, encoding.clamp, encoding.endian_swap};
                    return serialize(input.template as<real_t>(), output, new_encoding);
                }
                break;
            case Encoding::C16:
                if constexpr (nt::complex<T>) {
                    using real_t = const T::value_type;
                    auto new_encoding = Encoding{Encoding::F16, encoding.clamp, encoding.endian_swap};
                    return serialize(input.template as<real_t>(), output, new_encoding);
                }
                break;
            case Encoding::C32:
                if constexpr (nt::complex<T>) {
                    using real_t = const T::value_type;
                    auto new_encoding = Encoding{Encoding::F32, encoding.clamp, encoding.endian_swap};
                    return serialize(input.template as<real_t>(), output, new_encoding);
                }
                break;
            case Encoding::C64:
                if constexpr (nt::complex<T>) {
                    using real_t = const T::value_type;
                    auto new_encoding = Encoding{Encoding::F64, encoding.clamp, encoding.endian_swap};
                    return serialize(input.template as<real_t>(), output, new_encoding);
                }
                break;
            case Encoding::Format::UNKNOWN:
                break;
        }
        panic("{} cannot be serialized into {}", ns::stringify<T>(), encoding.format);
    }

    template<nt::numeric T>
    void serialize(
        const Span<const T, 4>& input,
        std::ostream& output,
        Encoding encoding
    ) {
        // Ignore all previous errors on that stream. If these errors cannot be recovered from,
        // the failbit will be reset by write() anyway and an exception will be thrown.
        output.clear();

        const bool are_contiguous = input.are_contiguous();
        const i64 n_elements = input.ssize();
        constexpr auto n_bytes_per_elements = static_cast<i64>(sizeof(T));

        if (are_contiguous and not encoding.endian_swap and encoding.format == Encoding::to_format<T>()) {
            output.write(reinterpret_cast<const char*>(input.get()), n_bytes_per_elements * n_elements);
            if (output.fail()) {
                output.clear();
                panic("Stream error. Failed while writing {} bytes", n_bytes_per_elements * n_elements);
            }
        } else if (encoding.format == Encoding::U4) {
            const i64 n_bytes = encoding.encoded_size(n_elements, input.shape()[3]);
            const auto buffer = std::make_unique<Byte[]>(static_cast<size_t>(n_bytes));
            serialize(input, SpanContiguous<Byte>(buffer.get(), n_bytes), encoding);
            output.write(reinterpret_cast<const char*>(buffer.get()), n_bytes);
            if (output.fail()) {
                output.clear();
                panic("Stream error. Failed while writing {} bytes", n_bytes);
            }
        } else if (are_contiguous) {
            constexpr i64 bytes_per_batch = 1 << 26; // 67MB
            const i64 bytes_per_element = encoding.encoded_size(1);
            i64 bytes_remain = bytes_per_element * n_elements;
            i64 bytes_buffer = std::min(bytes_remain, bytes_per_batch);
            const auto buffer = std::make_unique<Byte[]>(static_cast<size_t>(bytes_buffer));
            const auto* buffer_ptr = reinterpret_cast<const char*>(buffer.get());

            // Read until there's nothing left.
            for (auto* input_ptr = input.get(); bytes_remain > 0; bytes_remain -= bytes_buffer) {
                bytes_buffer = std::min(bytes_remain, bytes_buffer);
                const i64 n_elements_buffer = bytes_buffer / bytes_per_element;

                // Serialize according to data type and write.
                serialize(Span<const T, 4>(input_ptr, n_elements_buffer),
                          SpanContiguous<Byte, 1>(buffer.get(), bytes_buffer), encoding);
                output.write(buffer_ptr, bytes_buffer);
                if (output.fail()) {
                    output.clear();
                    panic("Stream error. Failed while writing {} bytes", bytes_buffer);
                }

                input_ptr += n_elements_buffer;
            }
        } else {
            const i64 n_elements_per_slice = input.shape().filter(2, 3).n_elements();
            const i64 n_bytes_per_slice = encoding.encoded_size(n_elements_per_slice);
            const auto buffer = std::make_unique<Byte[]>(static_cast<size_t>(n_bytes_per_slice));
            const auto buffer_span = SpanContiguous<Byte, 1>(buffer.get(), n_bytes_per_slice);
            const auto* buffer_ptr = reinterpret_cast<const char*>(buffer.get());

            for (i64 i{}; i < input.shape()[0]; ++i) {
                for (i64 j{}; j < input.shape()[1]; ++j) {
                    serialize(input.subregion(i, j), buffer_span, encoding);
                    output.write(buffer_ptr, n_bytes_per_slice);
                    if (output.fail()) {
                        output.clear();
                        panic("Stream error. Failed while writing {} bytes", n_bytes_per_slice);
                    }
                }
            }
        }
    }

    template<nt::numeric T>
    void deserialize(
        const SpanContiguous<const Byte, 1>& input,
        Encoding encoding,
        const Span<T, 4>& output
    ) {
        switch (encoding.format) {
            case Encoding::U4:
                if constexpr (nt::scalar<T>) {
                    check(not input.is_empty() and output.are_contiguous());
                    if (is_even(output.shape()[3])) { // if even, data can be deserialized contiguously
                        deserialize_row_4bits_(input.get(), output.get(), output.n_elements(), false);
                    } else { // otherwise, there's a "padding" of 4bits at the end of each row
                        const auto rows = output.shape().pop_back().n_elements();
                        const auto bytes_per_row = (output.shape()[3] + 1) / 2;
                        for (i64 row{}; row < rows; ++row)
                            deserialize_row_4bits_(input.get() + bytes_per_row * row,
                                                   output.get() + output.shape()[3] * row,
                                                   output.shape()[3], true);
                    }
                    return;
                }
                break;
            case Encoding::I8:
                if constexpr (nt::scalar<T>)
                    return deserialize_4d_<i8>(input, output, encoding.clamp, encoding.endian_swap);
                break;
            case Encoding::U8:
                if constexpr (nt::scalar<T>)
                    return deserialize_4d_<u8>(input, output, encoding.clamp, encoding.endian_swap);
                break;
            case Encoding::I16:
                if constexpr (nt::scalar<T>)
                    return deserialize_4d_<i16>(input, output, encoding.clamp, encoding.endian_swap);
                break;
            case Encoding::U16:
                if constexpr (nt::scalar<T>)
                    return deserialize_4d_<u16>(input, output, encoding.clamp, encoding.endian_swap);
                break;
            case Encoding::I32:
                if constexpr (nt::scalar<T>)
                    return deserialize_4d_<i32>(input, output, encoding.clamp, encoding.endian_swap);
                break;
            case Encoding::U32:
                if constexpr (nt::scalar<T>)
                    return deserialize_4d_<u32>(input, output, encoding.clamp, encoding.endian_swap);
                break;
            case Encoding::I64:
                if constexpr (nt::scalar<T>)
                    return deserialize_4d_<i64>(input, output, encoding.clamp, encoding.endian_swap);
                break;
            case Encoding::U64:
                if constexpr (nt::scalar<T>)
                    return deserialize_4d_<u64>(input, output, encoding.clamp, encoding.endian_swap);
                break;
            case Encoding::F16:
                if constexpr (nt::scalar<T>)
                    return deserialize_4d_<f16>(input, output, encoding.clamp, encoding.endian_swap);
                break;
            case Encoding::F32:
                if constexpr (nt::scalar<T>)
                    return deserialize_4d_<f32>(input, output, encoding.clamp, encoding.endian_swap);
                break;
            case Encoding::F64:
                if constexpr (nt::scalar<T>)
                    return deserialize_4d_<f64>(input, output, encoding.clamp, encoding.endian_swap);
                break;
            case Encoding::CI16:
                if constexpr (nt::complex<T>) {
                    auto new_encoding = Encoding{Encoding::I16, encoding.clamp, encoding.endian_swap};
                    return deserialize(input, new_encoding, output.template as<typename T::value_type>());
                }
                break;
            case Encoding::C16:
                if constexpr (nt::complex<T>) {
                    auto new_encoding = Encoding{Encoding::F16, encoding.clamp, encoding.endian_swap};
                    return deserialize(input, new_encoding, output.template as<typename T::value_type>());
                }
                break;
            case Encoding::C32:
                if constexpr (nt::complex<T>) {
                    auto new_encoding = Encoding{Encoding::F32, encoding.clamp, encoding.endian_swap};
                    return deserialize(input, new_encoding, output.template as<typename T::value_type>());
                }
                break;
            case Encoding::C64:
                if constexpr (nt::complex<T>) {
                    auto new_encoding = Encoding{Encoding::F64, encoding.clamp, encoding.endian_swap};
                    return deserialize(input, new_encoding, output.template as<typename T::value_type>());
                }
                break;
            case Encoding::UNKNOWN:
                break;
        }
        panic("{} cannot be deserialized into {}", encoding.format, ns::stringify<T>());
    }

    template<nt::numeric T>
    void deserialize(
        std::istream& input,
        Encoding encoding,
        const Span<T, 4>& output
    ) {
        input.clear();
        const bool are_contiguous = output.are_contiguous();
        const i64 n_elements = output.n_elements();
        constexpr i64 SIZEOF_T = sizeof(T);

        if (are_contiguous and encoding.format == Encoding::to_format<T>()) {
            input.read(reinterpret_cast<char*>(output.get()), SIZEOF_T * n_elements);
            if (input.fail()) {
                input.clear();
                panic("Stream error. Failed while reading {} bytes", SIZEOF_T * n_elements);
            } else if (encoding.endian_swap) {
                if constexpr (nt::complex<T>)
                    swap_endian(reinterpret_cast<Byte*>(output.get()), n_elements * 2, SIZEOF_T / 2);
                else
                    swap_endian(reinterpret_cast<Byte*>(output.get()), n_elements, SIZEOF_T);
            }
        } else if (encoding.format == Encoding::U4) {
            const auto n_bytes = encoding.encoded_size(n_elements, output.shape()[3]);
            const auto buffer = std::make_unique<Byte[]>(static_cast<size_t>(n_bytes));
            input.read(reinterpret_cast<char*>(buffer.get()), n_bytes);
            if (input.fail()) {
                input.clear();
                panic("Stream error. Failed while reading {} bytes", n_bytes);
            }
            deserialize(SpanContiguous<const Byte, 1>(buffer.get(), n_bytes), encoding, output);

        } else if (are_contiguous) {
            constexpr i64 bytes_per_batch = 1 << 26; // 67MB
            const i64 bytes_per_element = encoding.encoded_size(1);
            i64 bytes_remain = bytes_per_element * n_elements;
            i64 bytes_buffer = std::min(bytes_remain, bytes_per_batch);
            const auto buffer = std::make_unique<Byte[]>(static_cast<size_t>(bytes_buffer));
            auto* buffer_ptr = reinterpret_cast<char*>(buffer.get());
            const auto buffer_span = SpanContiguous<Byte, 1>(buffer.get(), bytes_buffer);

            // Read until there's nothing left.
            auto* output_ptr = output.get();
            for (; bytes_remain > 0; bytes_remain -= bytes_buffer) {
                bytes_buffer = std::min(bytes_remain, bytes_buffer);
                const i64 n_elements_buffer = bytes_buffer / bytes_per_element;

                // Read, swap and deserialize according to data type.
                input.read(buffer_ptr, bytes_buffer);
                if (input.fail()) {
                    input.clear();
                    panic("Stream error. Failed while reading {} bytes", bytes_buffer);
                }
                deserialize(buffer_span, encoding, Span<T, 4>(output_ptr, n_elements_buffer));

                output_ptr += n_elements_buffer;
            }
        } else {
            const i64 n_elements_per_slice = output.shape().filter(2, 3).n_elements();
            const i64 n_bytes_per_slice = encoding.encoded_size(n_elements_per_slice);
            const auto buffer = std::make_unique<std::byte[]>(static_cast<size_t>(n_bytes_per_slice));
            const auto buffer_span = SpanContiguous<Byte, 1>(buffer.get(), n_bytes_per_slice);

            for (i64 i{}; i < output.shape()[0]; ++i) {
                for (i64 j{}; j < output.shape()[1]; ++j) {
                    input.read(reinterpret_cast<char*>(buffer.get()), n_bytes_per_slice);
                    if (input.fail()) {
                        input.clear();
                        panic("Stream error. Failed while reading {} bytes", n_bytes_per_slice);
                    }
                    deserialize(buffer_span, encoding, output.subregion(i, j));
                }
            }
        }
    }

    #define NOA_IO_SERIALIZE_(T)                                                                    \
    template void serialize<T>(const Span<const T, 4>&, const SpanContiguous<Byte, 1>&, Encoding);  \
    template void serialize<T>(const Span<const T, 4>&, std::ostream&, Encoding);                   \
    template void deserialize<T>(const SpanContiguous<const Byte, 1>&, Encoding, const Span<T, 4>&);\
    template void deserialize<T>(std::istream&, Encoding, const Span<T, 4>&)

    NOA_IO_SERIALIZE_(i8);
    NOA_IO_SERIALIZE_(u8);
    NOA_IO_SERIALIZE_(i16);
    NOA_IO_SERIALIZE_(u16);
    NOA_IO_SERIALIZE_(i32);
    NOA_IO_SERIALIZE_(u32);
    NOA_IO_SERIALIZE_(i64);
    NOA_IO_SERIALIZE_(u64);
    NOA_IO_SERIALIZE_(f16);
    NOA_IO_SERIALIZE_(f32);
    NOA_IO_SERIALIZE_(f64);
    NOA_IO_SERIALIZE_(c16);
    NOA_IO_SERIALIZE_(c32);
    NOA_IO_SERIALIZE_(c64);
}

namespace noa::io {
    auto operator<<(std::ostream& os, Open mode) -> std::ostream& {
        std::array flags{mode.read, mode.write, mode.truncate, mode.binary, mode.append, mode.at_the_end};
        std::array names{"read", "write", "truncate", "binary", "append", "at_the_end"};

        bool add{};
        os << "Open{";
        for (auto i: irange(flags.size())) {
            if (flags[i]) {
                if (add)
                    os << '|';
                os << names[i];
                add = true;
            }
        }
        os << '}';
        return os;
    }

    auto operator<<(std::ostream& os, Encoding::Format data_type) -> std::ostream& {
        switch (data_type) {
            case Encoding::UNKNOWN: return os << "Encoding::UNKNOWN";
            case Encoding::U4: return os << "Encoding::U4";
            case Encoding::I8: return os << "Encoding::I8";
            case Encoding::U8: return os << "Encoding::U8";
            case Encoding::I16: return os << "Encoding::I16";
            case Encoding::U16: return os << "Encoding::U16";
            case Encoding::I32: return os << "Encoding::I32";
            case Encoding::U32: return os << "Encoding::U32";
            case Encoding::I64: return os << "Encoding::I64";
            case Encoding::U64: return os << "Encoding::U64";
            case Encoding::F16: return os << "Encoding::F16";
            case Encoding::F32: return os << "Encoding::F32";
            case Encoding::F64: return os << "Encoding::F64";
            case Encoding::CI16: return os << "Encoding::CI16";
            case Encoding::C16: return os << "Encoding::C16";
            case Encoding::C32: return os << "Encoding::C32";
            case Encoding::C64: return os << "Encoding::C64";
        }
        return os; // unreachable
    }
}
