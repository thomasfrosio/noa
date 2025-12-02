#include <omp.h>
#include <cstring>

#include "noa/core/indexing/Layout.hpp"
#include "noa/core/io/Encoding.hpp"

namespace {
    using namespace ::noa;

    // Multithreading really helps for large arrays, and thanks to memory mapping,
    // we can easily encode data in files directly in-place and in parallel.
    auto actual_n_threads_(i64 n_elements, i32 n_threads) -> i32 {
        // The encoding is very cheap to compute, using multiple threads should only be useful for huge arrays.
        constexpr i64 N_ELEMENTS_PER_THREAD = 8'388'608;
        i64 actual_n_threads = n_elements <= N_ELEMENTS_PER_THREAD ? 1 : n_threads;
        if (actual_n_threads > 1)
            actual_n_threads = min<i64>(n_threads, n_elements / N_ELEMENTS_PER_THREAD);
        return static_cast<i32>(actual_n_threads);
    }

    auto remaining_bytes_from_file(std::FILE* file) {
        // Get the current position.
        i64 pos = std::ftell(file);
        check_runtime(pos != -1, std::strerror(errno));

        // Get the position at the end of the file.
        check_runtime(std::fseek(file, 0, SEEK_END) == 0, std::strerror(errno));
        i64 size = std::ftell(file);
        check_runtime(size != -1, std::strerror(errno));

        // Reset the current position.
        check_runtime(std::fseek(file, pos, SEEK_SET) == 0, std::strerror(errno));

        return size - pos;
    }

    template<typename Input, typename Output>
    struct Encoder {
        bool clamp{};
        bool swap_endian{};

        constexpr auto operator()(const Input& value) -> Output {
            auto encoded = clamp ? clamp_cast<Output>(value) : static_cast<Output>(value);
            if (swap_endian)
                encoded = noa::io::swap_endian(encoded);
            return encoded;
        }
    };

    template<typename Output, typename Input>
    void encode_1d_(
        const Input* NOA_RESTRICT_ATTRIBUTE input,
        std::byte* NOA_RESTRICT_ATTRIBUTE output,
        i64 n_elements, bool clamp, bool swap_endian, i32 n_threads
    ) {
        auto ptr = reinterpret_cast<Output*>(output);
        auto encoder = Encoder<Input, Output>{clamp, swap_endian};

        // TODO This still can be auto vectorized...
        #pragma omp parallel for num_threads(n_threads)
        for (i64 idx = 0; idx < n_elements; ++idx)
            ptr[idx] = encoder(input[idx]);
    }

    template<typename Output, typename Input>
    void encode_4d_(
        const Span<const Input, 4>& input,
        const SpanContiguous<std::byte, 1>& output,
        bool clamp, bool swap_endian, i32 n_threads
    ) {
        if (input.are_contiguous())
            return encode_1d_<Output>(input.data(), output.data(), input.n_elements(), clamp, swap_endian, n_threads);

        auto* ptr = reinterpret_cast<Output*>(output.get());
        auto encoder = Encoder<Input, Output>{clamp, swap_endian};

        if (n_threads > 1) {
            // Collapse manually since we need to keep track of a linear index anyway...
            #pragma omp parallel for num_threads(n_threads)
            for (i64 i = 0; i < input.ssize(); ++i)
                ptr[i] = encoder(input(ni::offset2index(i, input.shape())));
        } else {
            for (i64 i = 0; i < input.shape()[0]; ++i)
                for (i64 j = 0; j < input.shape()[1]; ++j)
                    for (i64 k = 0; k < input.shape()[2]; ++k)
                        for (i64 l = 0; l < input.shape()[3]; ++l)
                            *(ptr++) = encoder(input(i, j, k, l));
        }
    }

    template<typename Input, typename Output>
    struct Decoder {
        bool clamp{};
        bool swap_endian{};

        constexpr auto operator()(Input value) -> Output {
            if (swap_endian)
                value = noa::io::swap_endian(value);
            return clamp ? clamp_cast<Output>(value) : static_cast<Output>(value);
        }
    };

    template<typename Input, typename Output>
    void decode_1d_(
        const std::byte* NOA_RESTRICT_ATTRIBUTE input,
        Output* NOA_RESTRICT_ATTRIBUTE output,
        i64 n_elements, bool clamp, bool swap_endian, i32 n_threads
    ) {
        auto ptr = reinterpret_cast<const Input*>(input);
        auto decoder = Decoder<Input, Output>{clamp, swap_endian};

        // TODO This still can be auto vectorized...
        #pragma omp parallel for num_threads(n_threads)
        for (i64 idx = 0; idx < n_elements; ++idx)
            output[idx] = decoder(ptr[idx]);
    }

    template<typename Input, typename Output>
    void decode_4d_(
        SpanContiguous<const std::byte, 1> input,
        const Span<Output, 4>& output,
        bool clamp, bool swap_endian, i32 n_threads
    ) {
        if (output.are_contiguous())
            return decode_1d_<Input>(input.data(), output.data(), output.n_elements(), clamp, swap_endian, n_threads);

        auto* ptr = reinterpret_cast<const Input*>(input.get());
        auto decoder = Decoder<Input, Output>{clamp, swap_endian};

        if (n_threads > 1) {
            // Collapse manually since we need to keep track of a linear index anyway...
            #pragma omp parallel for num_threads(n_threads)
            for (i64 i = 0; i < output.ssize(); ++i)
                output(ni::offset2index(i, output.shape())) = decoder(ptr[i]);
        } else {
            for (i64 i = 0; i < output.shape()[0]; ++i)
                for (i64 j = 0; j < output.shape()[1]; ++j)
                    for (i64 k = 0; k < output.shape()[2]; ++k)
                        for (i64 l = 0; l < output.shape()[3]; ++l)
                            output(i, j, k, l) = decoder(*(ptr++));
        }
    }

    template<nt::scalar T>
    void encode_4bits_(
        const T* NOA_RESTRICT_ATTRIBUTE input,
        std::byte* NOA_RESTRICT_ATTRIBUTE output,
        i64 n_elements, i32 n_threads
    ) {
        // The order of the first and second elements in the output are the 4 LSB and 4 MSB, respectively.
        // Note: We don't support odd rows, but if the row had an odd number of elements, the last byte of
        // the row has the 4 MSB unset.
        #pragma omp parallel for num_threads(n_threads)
        for (i64 i = 0; i < n_elements / 2; ++i) {
            u32 l_val = clamp_cast<u32>(noa::round(input[2 * i]));
            u32 h_val = clamp_cast<u32>(noa::round(input[2 * i + 1]));
            l_val = noa::clamp(l_val, 0u, 15u);
            h_val = noa::clamp(h_val, 0u, 15u);
            u32 tmp = l_val + (h_val << 4);
            std::memcpy(output + i, &tmp, 1);
        }
    }

    template<typename T>
    void decode_4bits_(
        const std::byte* NOA_RESTRICT_ATTRIBUTE input,
        T* NOA_RESTRICT_ATTRIBUTE output,
        i64 n_elements, i32 n_threads
    ) {
        constexpr unsigned char MASK_4LSB{0b00001111};

        #pragma omp parallel for num_threads(n_threads)
        for (i64 i = 0; i < n_elements / 2; ++i) {
            const auto tmp = static_cast<unsigned char>(input[i]);
            output[i * 2] = static_cast<T>(tmp & MASK_4LSB);
            output[i * 2 + 1] = static_cast<T>((tmp >> 4) & MASK_4LSB);
        }
    }

    struct u4_encoding {};

    template<typename Output, typename Input>
    void encode_file_(
        const Span<const Input, 4>& input,
        std::FILE* output,
        bool clamp, bool swap_endian,
        i32 n_threads
    ) {
        constexpr i64 N_BYTES_PER_BLOCK = 1 << 16; // 64KB
        constexpr i64 N_BYTES_PER_ELEMENT = sizeof(Output);
        static_assert(is_multiple_of(N_BYTES_PER_BLOCK, N_BYTES_PER_ELEMENT));

        const i64 n_elements = input.ssize();
        const i64 n_bytes = std::same_as<Output, u4_encoding> ? n_elements / 2 : n_elements * N_BYTES_PER_ELEMENT;
        const i64 n_blocks = noa::divide_up(n_bytes, N_BYTES_PER_BLOCK);

        const auto buffer_ptr = std::make_unique<char[]>(static_cast<size_t>(n_threads * N_BYTES_PER_BLOCK));
        const auto buffer_span = Span(buffer_ptr.get(), Shape<i64, 2>{n_threads, N_BYTES_PER_BLOCK});

        const i64 start_offset = std::ftell(output);
        check(start_offset != -1, "Could not get the current position of the stream, {}", std::strerror(errno));

        const bool input_is_contiguous = input.are_contiguous();

        #pragma omp parallel for num_threads(n_threads)
        for (i64 n_block = 0; n_block < n_blocks; ++n_block) {
            i32 thread_id = omp_get_thread_num();
            char* per_thread_buffer = buffer_span[thread_id].data();

            const i64 offset = n_block * N_BYTES_PER_BLOCK;
            const i64 n_bytes_to_write = std::min(N_BYTES_PER_BLOCK, n_bytes - offset);
            NOA_ASSERT(n_bytes_to_write > 0);

            const i64 n_elements_to_write = n_bytes_to_write / N_BYTES_PER_ELEMENT;
            const i64 n_elements_offset = offset / N_BYTES_PER_ELEMENT;

            const Input* input_ptr = input.get() + n_elements_offset;
            for (i64 i = 0; i < n_elements_to_write; ++i) {
                if constexpr (std::same_as<Output, u4_encoding>) {
                    u32 l_val = clamp_cast<u32>(noa::round(input_ptr[2 * i]));
                    u32 h_val = clamp_cast<u32>(noa::round(input_ptr[2 * i + 1]));
                    l_val = noa::clamp(l_val, 0u, 15u);
                    h_val = noa::clamp(h_val, 0u, 15u);
                    u32 tmp = l_val + (h_val << 4);
                    std::memcpy(per_thread_buffer + i, &tmp, 1);
                } else {
                    auto* output_ptr = reinterpret_cast<Output*>(per_thread_buffer);
                    auto encoder = Encoder<Input, Output>{clamp, swap_endian};
                    if (input_is_contiguous) {
                        output_ptr[i] = encoder(input_ptr[i]);
                    } else {
                        auto indices = noa::indexing::offset2index(n_elements_offset + i, input.shape());
                        output_ptr[i] = encoder(input(indices));
                    }
                }
            }

            #pragma omp critical
            {
                check(std::fseek(output, start_offset + offset, SEEK_SET) == 0,
                      "Failed to seek at position {}. {}",
                      start_offset + offset, std::strerror(errno));
                check(static_cast<size_t>(n_elements_to_write) == std::fwrite(
                          per_thread_buffer,
                          static_cast<size_t>(N_BYTES_PER_ELEMENT),
                          static_cast<size_t>(n_elements_to_write),
                          output),
                      "Failed to read from the file");
            }
        }
    }

    template<typename Input, typename Output>
    void decode_file_(
        std::FILE* input,
        const Span<Output, 4>& output,
        bool clamp, bool swap_endian,
        i32 n_threads
    ) {
        constexpr i64 N_BYTES_PER_BLOCK = 1 << 16; // 64KB
        constexpr i64 N_BYTES_PER_ELEMENT = sizeof(Input);
        static_assert(is_multiple_of(N_BYTES_PER_BLOCK, N_BYTES_PER_ELEMENT));

        const i64 n_elements = output.ssize();
        const i64 n_bytes = std::same_as<Input, u4_encoding> ? n_elements / 2 : n_elements * N_BYTES_PER_ELEMENT;
        const i64 n_blocks = noa::divide_up(n_bytes, N_BYTES_PER_BLOCK);

        const auto buffer_ptr = std::make_unique<char[]>(static_cast<size_t>(n_threads * N_BYTES_PER_BLOCK));
        const auto buffer_span = Span(buffer_ptr.get(), Shape<i64, 2>{n_threads, N_BYTES_PER_BLOCK});

        const i64 start_offset = std::ftell(input);
        check(start_offset != -1, "Could not get the current position of the stream, {}", std::strerror(errno));

        const bool output_is_contiguous = output.are_contiguous();

        #pragma omp parallel for num_threads(n_threads)
        for (i64 n_block = 0; n_block < n_blocks; ++n_block) {
            i32 thread_id = omp_get_thread_num();
            char* per_thread_buffer = buffer_span[thread_id].data();

            const i64 offset = n_block * N_BYTES_PER_BLOCK;
            const i64 n_bytes_to_read = std::min(N_BYTES_PER_BLOCK, n_bytes - offset);
            NOA_ASSERT(n_bytes_to_read > 0);

            const i64 n_elements_to_read = n_bytes_to_read / N_BYTES_PER_ELEMENT;
            const i64 n_elements_offset = offset / N_BYTES_PER_ELEMENT;

            #pragma omp critical
            {
                check(std::fseek(input, start_offset + offset, SEEK_SET) == 0,
                      "Failed to seek at position {}. {}",
                      start_offset + offset, std::strerror(errno));
                check(static_cast<size_t>(n_elements_to_read) == std::fread(
                          per_thread_buffer,
                          static_cast<size_t>(N_BYTES_PER_ELEMENT),
                          static_cast<size_t>(n_elements_to_read),
                          input),
                      "Failed to read from the file");
            }

            Output* output_ptr = output.get() + n_elements_offset;
            for (i64 i = 0; i < n_elements_to_read; ++i) {
                if constexpr (std::same_as<Input, u4_encoding>) {
                    constexpr char MASK_4LSB{0b00001111};
                    output_ptr[i * 2] = static_cast<Output>(per_thread_buffer[i] & MASK_4LSB);
                    output_ptr[i * 2 + 1] = static_cast<Output>((per_thread_buffer[i] >> 4) & MASK_4LSB);
                } else {
                    auto* input_ptr = reinterpret_cast<Input*>(per_thread_buffer);
                    auto decoded = Decoder<Input, Output>{clamp, swap_endian}(input_ptr[i]);
                    if (output_is_contiguous) {
                        output_ptr[i] = decoded;
                    } else {
                        auto indices = noa::indexing::offset2index(n_elements_offset + i, output.shape());
                        output(indices) = decoded;
                    }
                }
            }
        }
    }
}

namespace noa::io {
    template<nt::numeric T>
    void encode(
        const Span<const T, 4>& input,
        const SpanContiguous<std::byte, 1>& output,
        const DataType& output_dtype,
        const EncodeOptions& options
    ) {
        const i64 n_elements = input.ssize();
        const i64 n_encoded_bytes = output_dtype.n_bytes(n_elements);
        const auto n_threads = actual_n_threads_(n_elements, options.n_threads);

        check(n_encoded_bytes <= output.ssize(), "The encoded array is not big enough to contain the input array");
        check(not ni::are_overlapped(input, output), "The input and output arrays should not overlap");

        switch (output_dtype) {
            case DataType::U4:
                if constexpr (nt::scalar<T>) {
                    check(input.are_contiguous() and is_even(input.shape()[3]),
                          "u4 encoding requires the input array to be contiguous and have even rows");
                    return encode_4bits_(input.data(), output.data(), input.n_elements(), n_threads);
                }
                break;
            case DataType::I8:
                if constexpr (nt::scalar<T>)
                    return encode_4d_<i8>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::U8:
                if constexpr (nt::scalar<T>)
                    return encode_4d_<u8>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::I16:
                if constexpr (nt::scalar<T>)
                    return encode_4d_<i16>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::U16:
                if constexpr (nt::scalar<T>)
                    return encode_4d_<u16>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::I32:
                if constexpr (nt::scalar<T>)
                    return encode_4d_<i32>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::U32:
                if constexpr (nt::scalar<T>)
                    return encode_4d_<u32>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::I64:
                if constexpr (nt::scalar<T>)
                    return encode_4d_<i64>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::U64:
                if constexpr (nt::scalar<T>)
                    return encode_4d_<u64>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::F16:
                if constexpr (nt::scalar<T>)
                    return encode_4d_<f16>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::F32:
                if constexpr (nt::scalar<T>)
                    return encode_4d_<f32>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::F64:
                if constexpr (nt::scalar<T>)
                    return encode_4d_<f64>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::CI16:
                if constexpr (nt::complex<T>)
                    return encode(input.template as<nt::const_value_type_t<T>>(), output, DataType::I16, options);
                break;
            case DataType::C16:
                if constexpr (nt::complex<T>)
                    return encode(input.template as<nt::const_value_type_t<T>>(), output, DataType::F16, options);
                break;
            case DataType::C32:
                if constexpr (nt::complex<T>)
                    return encode(input.template as<nt::const_value_type_t<T>>(), output, DataType::F32, options);
                break;
            case DataType::C64:
                if constexpr (nt::complex<T>)
                    return encode(input.template as<nt::const_value_type_t<T>>(), output, DataType::F64, options);
                break;
            case DataType::UNKNOWN:
                break;
        }
        panic("{} cannot be encoded into {}", noa::string::stringify<T>(), output_dtype);
    }

    void encode(
        const SpanContiguous<const std::byte, 1>& input,
        const DataType& input_dtype,
        const SpanContiguous<std::byte, 1>& output,
        const DataType& output_dtype,
        const EncodeOptions& options
    ) {
        switch (input_dtype) {
            case DataType::I8:
                return encode(input.as_strided<const i8, 4>(), output, output_dtype, options);
            case DataType::I16:
                return encode(input.as_strided<const i16, 4>(), output, output_dtype, options);
            case DataType::I32:
                return encode(input.as_strided<const i32, 4>(), output, output_dtype, options);
            case DataType::I64:
                return encode(input.as_strided<const i64, 4>(), output, output_dtype, options);
            case DataType::U8:
                return encode(input.as_strided<const u8, 4>(), output, output_dtype, options);
            case DataType::U16:
                return encode(input.as_strided<const u16, 4>(), output, output_dtype, options);
            case DataType::U32:
                return encode(input.as_strided<const u32, 4>(), output, output_dtype, options);
            case DataType::U64:
                return encode(input.as_strided<const u64, 4>(), output, output_dtype, options);
            case DataType::F16:
                return encode(input.as_strided<const f16, 4>(), output, output_dtype, options);
            case DataType::F32:
                return encode(input.as_strided<const f32, 4>(), output, output_dtype, options);
            case DataType::F64:
                return encode(input.as_strided<const f64, 4>(), output, output_dtype, options);
            case DataType::C16:
                return encode(input.as_strided<const c16, 4>(), output, output_dtype, options);
            case DataType::C32:
                return encode(input.as_strided<const c32, 4>(), output, output_dtype, options);
            case DataType::C64:
                return encode(input.as_strided<const c64, 4>(), output, output_dtype, options);
            case DataType::U4:
            case DataType::CI16:
                panic("TODO: u4 and ci16 cannot be reinterpreted to valid types, they would require special cases");
            case DataType::UNKNOWN:
                break;
        }
    }

    template<nt::numeric T>
    void encode(
        const Span<const T, 4>& input,
        std::FILE* output,
        const DataType& output_dtype,
        const EncodeOptions& options
    ) {
        const i64 n_elements = input.ssize();
        const i64 n_encoded_bytes = output_dtype.n_bytes(n_elements);
        const i64 remaining_bytes = remaining_bytes_from_file(output);
        const auto n_threads = actual_n_threads_(n_elements, options.n_threads);

        check(n_encoded_bytes <= remaining_bytes,
              "The file (remaining_bytes={}) is not big enough to contain the encoded array",
              remaining_bytes);

        switch (output_dtype) {
            case DataType::U4:
                if constexpr (nt::scalar<T>) {
                    check(input.are_contiguous() and is_even(n_elements),
                          "u4 encoding requires the input array to be contiguous and have an even number of elements");
                    return encode_file_<u4_encoding>(input, output, options.clamp, options.endian_swap, n_threads);
                }
                break;
            case DataType::I8:
                if constexpr (nt::scalar<T>)
                    return encode_file_<i8>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::U8:
                if constexpr (nt::scalar<T>)
                    return encode_file_<u8>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::I16:
                if constexpr (nt::scalar<T>)
                    return encode_file_<i16>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::U16:
                if constexpr (nt::scalar<T>)
                    return encode_file_<u16>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::I32:
                if constexpr (nt::scalar<T>)
                    return encode_file_<i32>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::U32:
                if constexpr (nt::scalar<T>)
                    return encode_file_<u32>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::I64:
                if constexpr (nt::scalar<T>)
                    return encode_file_<i64>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::U64:
                if constexpr (nt::scalar<T>)
                    return encode_file_<u64>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::F16:
                if constexpr (nt::scalar<T>)
                    return encode_file_<f16>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::F32:
                if constexpr (nt::scalar<T>)
                    return encode_file_<f32>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::F64:
                if constexpr (nt::scalar<T>)
                    return encode_file_<f64>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::CI16:
                if constexpr (nt::complex<T>)
                    return encode(input.template as<nt::const_value_type_t<T>>(), output, DataType::I16, options);
                break;
            case DataType::C16:
                if constexpr (nt::complex<T>)
                    return encode(input.template as<nt::const_value_type_t<T>>(), output, DataType::F16, options);
                break;
            case DataType::C32:
                if constexpr (nt::complex<T>)
                    return encode(input.template as<nt::const_value_type_t<T>>(), output, DataType::F32, options);
                break;
            case DataType::C64:
                if constexpr (nt::complex<T>)
                    return encode(input.template as<nt::const_value_type_t<T>>(), output, DataType::F64, options);
                break;
            case DataType::UNKNOWN:
                break;
        }
        panic("{} cannot be encoded into {}", noa::string::stringify<T>(), output_dtype);
    }

    void encode(
        const SpanContiguous<const std::byte, 1>& input,
        const DataType& input_dtype,
        std::FILE* output,
        const DataType& output_dtype,
        const EncodeOptions& options
    ) {
        switch (input_dtype) {
            case DataType::I8:
                return encode(input.as_strided<const i8, 4>(), output, output_dtype, options);
            case DataType::I16:
                return encode(input.as_strided<const i16, 4>(), output, output_dtype, options);
            case DataType::I32:
                return encode(input.as_strided<const i32, 4>(), output, output_dtype, options);
            case DataType::I64:
                return encode(input.as_strided<const i64, 4>(), output, output_dtype, options);
            case DataType::U8:
                return encode(input.as_strided<const u8, 4>(), output, output_dtype, options);
            case DataType::U16:
                return encode(input.as_strided<const u16, 4>(), output, output_dtype, options);
            case DataType::U32:
                return encode(input.as_strided<const u32, 4>(), output, output_dtype, options);
            case DataType::U64:
                return encode(input.as_strided<const u64, 4>(), output, output_dtype, options);
            case DataType::F16:
                return encode(input.as_strided<const f16, 4>(), output, output_dtype, options);
            case DataType::F32:
                return encode(input.as_strided<const f32, 4>(), output, output_dtype, options);
            case DataType::F64:
                return encode(input.as_strided<const f64, 4>(), output, output_dtype, options);
            case DataType::C16:
                return encode(input.as_strided<const c16, 4>(), output, output_dtype, options);
            case DataType::C32:
                return encode(input.as_strided<const c32, 4>(), output, output_dtype, options);
            case DataType::C64:
                return encode(input.as_strided<const c64, 4>(), output, output_dtype, options);
            case DataType::U4:
            case DataType::CI16:
                panic("TODO: u4 and ci16 cannot be reinterpreted to valid types, they would require special cases");
            case DataType::UNKNOWN:
                break;
        }
    }

    template<nt::numeric T>
    void decode(
        const SpanContiguous<const std::byte, 1>& input,
        const DataType& input_dtype,
        const Span<T, 4>& output,
        const DecodeOptions& options
    ) {
        const i64 n_elements = output.ssize();
        const i64 n_encoded_bytes = input_dtype.n_bytes(n_elements);
        const auto n_threads = actual_n_threads_(n_elements, options.n_threads);

        check(n_encoded_bytes <= input.ssize(), "The encoded array is not big enough to contain the output array");
        check(not ni::are_overlapped(input, output), "The input and output arrays should not overlap");

        switch (input_dtype) {
            case DataType::U4:
                if constexpr (nt::scalar<T>) {
                    check(output.are_contiguous() and is_even(n_elements),
                          "u4 encoding requires the output array to be contiguous and have an even number of elements");
                    return decode_4bits_(input.data(), output.data(), output.n_elements(), n_threads);
                }
                break;
            case DataType::I8:
                if constexpr (nt::scalar<T>)
                    return decode_4d_<i8>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::U8:
                if constexpr (nt::scalar<T>)
                    return decode_4d_<u8>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::I16:
                if constexpr (nt::scalar<T>)
                    return decode_4d_<i16>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::U16:
                if constexpr (nt::scalar<T>)
                    return decode_4d_<u16>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::I32:
                if constexpr (nt::scalar<T>)
                    return decode_4d_<i32>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::U32:
                if constexpr (nt::scalar<T>)
                    return decode_4d_<u32>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::I64:
                if constexpr (nt::scalar<T>)
                    return decode_4d_<i64>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::U64:
                if constexpr (nt::scalar<T>)
                    return decode_4d_<u64>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::F16:
                if constexpr (nt::scalar<T>)
                    return decode_4d_<f16>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::F32:
                if constexpr (nt::scalar<T>)
                    return decode_4d_<f32>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::F64:
                if constexpr (nt::scalar<T>)
                    return decode_4d_<f64>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::CI16:
                if constexpr (nt::complex<T>)
                    return decode(input, DataType::I16, output.template as<typename T::value_type>(), options);
                break;
            case DataType::C16:
                if constexpr (nt::complex<T>)
                    return decode(input, DataType::F16, output.template as<typename T::value_type>(), options);
                break;
            case DataType::C32:
                if constexpr (nt::complex<T>)
                    return decode(input, DataType::F32, output.template as<typename T::value_type>(), options);
                break;
            case DataType::C64:
                if constexpr (nt::complex<T>)
                    return decode(input, DataType::F64, output.template as<typename T::value_type>(), options);
                break;
            case DataType::UNKNOWN:
                break;
        }
        panic("{} cannot be decoded into {}", input_dtype, noa::string::stringify<T>());
    }

    void decode(
        const SpanContiguous<const std::byte, 1>& input,
        const DataType& input_dtype,
        const SpanContiguous<std::byte, 1>& output,
        const DataType& output_dtype,
        const DecodeOptions& options
    ) {
        switch (output_dtype) {
            case DataType::I8:
                return decode(input, input_dtype, output.as_strided<i8, 4>(), options);
            case DataType::I16:
                return decode(input, input_dtype, output.as_strided<i16, 4>(), options);
            case DataType::I32:
                return decode(input, input_dtype, output.as_strided<i32, 4>(), options);
            case DataType::I64:
                return decode(input, input_dtype, output.as_strided<i64, 4>(), options);
            case DataType::U8:
                return decode(input, input_dtype, output.as_strided<u8, 4>(), options);
            case DataType::U16:
                return decode(input, input_dtype, output.as_strided<u16, 4>(), options);
            case DataType::U32:
                return decode(input, input_dtype, output.as_strided<u32, 4>(), options);
            case DataType::U64:
                return decode(input, input_dtype, output.as_strided<u64, 4>(), options);
            case DataType::F16:
                return decode(input, input_dtype, output.as_strided<f16, 4>(), options);
            case DataType::F32:
                return decode(input, input_dtype, output.as_strided<f32, 4>(), options);
            case DataType::F64:
                return decode(input, input_dtype, output.as_strided<f64, 4>(), options);
            case DataType::C16:
                return decode(input, input_dtype, output.as_strided<c16, 4>(), options);
            case DataType::C32:
                return decode(input, input_dtype, output.as_strided<c32, 4>(), options);
            case DataType::C64:
                return decode(input, input_dtype, output.as_strided<c64, 4>(), options);
            case DataType::U4:
            case DataType::CI16:
                panic("TODO: u4 and ci16 cannot be reinterpreted to valid types, they would require special cases");
            case DataType::UNKNOWN:
                break;
        }
    }

    template<nt::numeric T>
    void decode(
        std::FILE* input,
        const DataType& input_dtype,
        const Span<T, 4>& output,
        const DecodeOptions& options
    ) {
        const i64 n_elements = output.n_elements();
        const i64 n_encoded_bytes = input_dtype.n_bytes(n_elements);
        const i64 remaining_bytes = remaining_bytes_from_file(input);
        const auto n_threads = actual_n_threads_(n_elements, options.n_threads);

        check(n_encoded_bytes <= remaining_bytes,
              "The file (remaining_bytes={}) is not big enough to contain the decoded array",
              remaining_bytes);

        switch (input_dtype) {
            case DataType::U4:
                if constexpr (nt::scalar<T>) {
                    check(output.are_contiguous() and is_even(n_elements),
                          "u4 encoding requires the output array to be contiguous and have an even number of elements");
                    return decode_file_<u4_encoding>(input, output, options.clamp, options.endian_swap, n_threads);
                }
                break;
            case DataType::I8:
                if constexpr (nt::scalar<T>)
                    return decode_file_<i8>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::U8:
                if constexpr (nt::scalar<T>)
                    return decode_file_<u8>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::I16:
                if constexpr (nt::scalar<T>)
                    return decode_file_<i16>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::U16:
                if constexpr (nt::scalar<T>)
                    return decode_file_<u16>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::I32:
                if constexpr (nt::scalar<T>)
                    return decode_file_<i32>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::U32:
                if constexpr (nt::scalar<T>)
                    return decode_file_<u32>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::I64:
                if constexpr (nt::scalar<T>)
                    return decode_file_<i64>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::U64:
                if constexpr (nt::scalar<T>)
                    return decode_file_<u64>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::F16:
                if constexpr (nt::scalar<T>)
                    return decode_file_<f16>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::F32:
                if constexpr (nt::scalar<T>)
                    return decode_file_<f32>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::F64:
                if constexpr (nt::scalar<T>)
                    return decode_file_<f64>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::CI16:
                if constexpr (nt::complex<T>) {
                    return decode(input, DataType::I16, output.template as<typename T::value_type>(), options);
                }
                break;
            case DataType::C16:
                if constexpr (nt::complex<T>) {
                    return decode(input, DataType::F16, output.template as<typename T::value_type>(), options);
                }
                break;
            case DataType::C32:
                if constexpr (nt::complex<T>) {
                    return decode(input, DataType::F32, output.template as<typename T::value_type>(), options);
                }
                break;
            case DataType::C64:
                if constexpr (nt::complex<T>) {
                    return decode(input, DataType::F64, output.template as<typename T::value_type>(), options);
                }
                break;
            case DataType::UNKNOWN:
                break;
        }
        panic("{} cannot be decoded into {}", input_dtype, noa::string::stringify<T>());
    }

    void decode(
        std::FILE* input,
        const DataType& input_dtype,
        const SpanContiguous<std::byte, 1>& output,
        const DataType& output_dtype,
        const DecodeOptions& options
    ) {
        switch (output_dtype) {
            case DataType::I8:
                return decode(input, input_dtype, output.as_strided<i8, 4>(), options);
            case DataType::I16:
                return decode(input, input_dtype, output.as_strided<i16, 4>(), options);
            case DataType::I32:
                return decode(input, input_dtype, output.as_strided<i32, 4>(), options);
            case DataType::I64:
                return decode(input, input_dtype, output.as_strided<i64, 4>(), options);
            case DataType::U8:
                return decode(input, input_dtype, output.as_strided<u8, 4>(), options);
            case DataType::U16:
                return decode(input, input_dtype, output.as_strided<u16, 4>(), options);
            case DataType::U32:
                return decode(input, input_dtype, output.as_strided<u32, 4>(), options);
            case DataType::U64:
                return decode(input, input_dtype, output.as_strided<u64, 4>(), options);
            case DataType::F16:
                return decode(input, input_dtype, output.as_strided<f16, 4>(), options);
            case DataType::F32:
                return decode(input, input_dtype, output.as_strided<f32, 4>(), options);
            case DataType::F64:
                return decode(input, input_dtype, output.as_strided<f64, 4>(), options);
            case DataType::C16:
                return decode(input, input_dtype, output.as_strided<c16, 4>(), options);
            case DataType::C32:
                return decode(input, input_dtype, output.as_strided<c32, 4>(), options);
            case DataType::C64:
                return decode(input, input_dtype, output.as_strided<c64, 4>(), options);
            case DataType::U4:
            case DataType::CI16:
                panic("TODO: u4 and ci16 cannot be reinterpreted to valid types, they would require special cases");
            case DataType::UNKNOWN:
                break;
        }
    }

    #define NOA_IO_ENCODE_(T) \
    template void encode<T>(const Span<const T, 4>&, const SpanContiguous<std::byte, 1>&, const DataType&, const EncodeOptions&);  \
    template void encode<T>(const Span<const T, 4>&, std::FILE*, const DataType&, const EncodeOptions&);                           \
    template void decode<T>(const SpanContiguous<const std::byte, 1>&, const DataType&, const Span<T, 4>&, const DecodeOptions&);  \
    template void decode<T>(std::FILE*, const DataType&, const Span<T, 4>&, const DecodeOptions&)

    NOA_IO_ENCODE_(i8);
    NOA_IO_ENCODE_(u8);
    NOA_IO_ENCODE_(i16);
    NOA_IO_ENCODE_(u16);
    NOA_IO_ENCODE_(i32);
    NOA_IO_ENCODE_(u32);
    NOA_IO_ENCODE_(i64);
    NOA_IO_ENCODE_(u64);
    NOA_IO_ENCODE_(f16);
    NOA_IO_ENCODE_(f32);
    NOA_IO_ENCODE_(f64);
    NOA_IO_ENCODE_(c16);
    NOA_IO_ENCODE_(c32);
    NOA_IO_ENCODE_(c64);
}

namespace noa::io {
    auto operator<<(std::ostream& os, DataType::Enum dtype) -> std::ostream& {
        switch (dtype) {
            case DataType::UNKNOWN: return os << "<unknown>";
            case DataType::U4: return os << "u4";
            case DataType::I8: return os << "i8";
            case DataType::U8: return os << "u8";
            case DataType::I16: return os << "i16";
            case DataType::U16: return os << "u16";
            case DataType::I32: return os << "i32";
            case DataType::U32: return os << "u32";
            case DataType::I64: return os << "i64";
            case DataType::U64: return os << "u64";
            case DataType::F16: return os << "f16";
            case DataType::F32: return os << "f32";
            case DataType::F64: return os << "f64";
            case DataType::CI16: return os << "ci16";
            case DataType::C16: return os << "c16";
            case DataType::C32: return os << "c32";
            case DataType::C64: return os << "c64";
        }
        return os; // unreachable
    }

    auto operator<<(std::ostream& os, DataType encoding) -> std::ostream& {
        return os << encoding.value;
    }
}
