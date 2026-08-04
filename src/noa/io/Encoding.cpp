#include "noa/io/Encoding.hpp"
#include "noa/runtime/cpu/Allocators.hpp"

namespace {
    using namespace ::noa;

    // Multithreading really helps for large arrays, and thanks to memory mapping,
    // we can easily encode data in files directly in-place and in parallel.
    auto actual_n_threads_(isize n_elements, i32 n_threads) -> i32 {
        // The encoding is very cheap to compute, using multiple threads should only be useful for huge arrays.
        constexpr isize N_ELEMENTS_PER_THREAD = 8'388'608;
        isize actual_n_threads = n_elements <= N_ELEMENTS_PER_THREAD ? 1 : n_threads;
        if (actual_n_threads > 1)
            actual_n_threads = min<isize>(n_threads, n_elements / N_ELEMENTS_PER_THREAD);
        return static_cast<i32>(actual_n_threads);
    }

    auto remaining_bytes_from_file(std::FILE* file) {
        // Get the current position.
        isize pos = std::ftell(file);
        check_runtime(pos != -1, std::strerror(errno));

        // Get the position at the end of the file.
        check_runtime(std::fseek(file, 0, SEEK_END) == 0, std::strerror(errno));
        isize size = std::ftell(file);
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

    template<typename Output, typename Input, usize N, StridesTraits S>
    void actual_encode_(
        const Span<const Input, N, isize, S>& input,
        const SpanContiguous<std::byte, 1>& output,
        bool clamp, bool swap_endian, i32 n_threads
    ) {
        auto* ptr = reinterpret_cast<Output*>(output.get());
        auto encoder = Encoder<Input, Output>{clamp, swap_endian};

        if constexpr (N == 2) {
            if (n_threads == 1) {
                for (isize i = 0; i < input.shape()[0]; ++i)
                    for (isize j = 0; j < input.shape()[1]; ++j)
                        ptr[i * input.shape()[1] + j] = encoder(input(i, j));
            } else {
                #pragma omp parallel for num_threads(n_threads) collapse(2)
                for (isize i = 0; i < input.shape()[0]; ++i)
                    for (isize j = 0; j < input.shape()[1]; ++j)
                        ptr[i * input.shape()[1] + j] = encoder(input(i, j));
            }
        } else {
            if (n_threads > 1) {
                auto strides = input.shape().strides();
                #pragma omp parallel for num_threads(n_threads) collapse(6)
                for (isize i = 0; i < input.shape()[0]; ++i)
                    for (isize j = 0; j < input.shape()[1]; ++j)
                        for (isize k = 0; k < input.shape()[2]; ++k)
                            for (isize l = 0; l < input.shape()[3]; ++l)
                                for (isize m = 0; m < input.shape()[4]; ++m)
                                    for (isize n = 0; n < input.shape()[5]; ++n)
                                        ptr[noa::offset_at(strides, i, j, k, l, m, n)] =
                                            encoder(input(i, j, k, l, m, n));
            } else {
                for (isize i = 0; i < input.shape()[0]; ++i)
                    for (isize j = 0; j < input.shape()[1]; ++j)
                        for (isize k = 0; k < input.shape()[2]; ++k)
                            for (isize l = 0; l < input.shape()[3]; ++l)
                                for (isize m = 0; m < input.shape()[4]; ++m)
                                    for (isize n = 0; n < input.shape()[5]; ++n)
                                        *(ptr++) = encoder(input(i, j, k, l, m, n));
            }
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

    template<typename Input, typename Output, usize N, StridesTraits S>
    void actual_decode_(
        SpanContiguous<const std::byte, 1> input,
        const Span<Output, N, isize, S>& output,
        bool clamp, bool swap_endian, i32 n_threads
    ) {
        auto* ptr = reinterpret_cast<const Input*>(input.get());
        auto decoder = Decoder<Input, Output>{clamp, swap_endian};

        if constexpr (N == 2) {
            if (n_threads == 1) {
                for (isize i = 0; i < output.shape()[0]; ++i)
                    for (isize j = 0; j < output.shape()[1]; ++j)
                        output(i, j) = decoder(ptr[i * output.shape()[1] + j]);
            } else {
                #pragma omp parallel for num_threads(n_threads) collapse(2)
                for (isize i = 0; i < output.shape()[0]; ++i)
                    for (isize j = 0; j < output.shape()[1]; ++j)
                        output(i, j) = decoder(ptr[i * output.shape()[1] + j]);
            }
        } else {
            if (n_threads > 1) {
                auto strides = output.shape().strides();
                #pragma omp parallel for num_threads(n_threads) collapse(6)
                for (isize i = 0; i < output.shape()[0]; ++i)
                    for (isize j = 0; j < output.shape()[1]; ++j)
                        for (isize k = 0; k < output.shape()[2]; ++k)
                            for (isize l = 0; l < output.shape()[3]; ++l)
                                for (isize m = 0; m < output.shape()[4]; ++m)
                                    for (isize n = 0; n < output.shape()[5]; ++n)
                                        output(i, j, k, l, m, n) =
                                            decoder(ptr[noa::offset_at(strides, i, j, k, l, m, n)]);
            } else {
                for (isize i = 0; i < output.shape()[0]; ++i)
                    for (isize j = 0; j < output.shape()[1]; ++j)
                        for (isize k = 0; k < output.shape()[2]; ++k)
                            for (isize l = 0; l < output.shape()[3]; ++l)
                                for (isize m = 0; m < output.shape()[4]; ++m)
                                    for (isize n = 0; n < output.shape()[5]; ++n)
                                        output(i, j, k, l, m, n) = decoder(*(ptr++));
            }
        }
    }

    template<nt::scalar T>
    void encode_4bits_(
        const T* NOA_RESTRICT_ATTRIBUTE input,
        std::byte* NOA_RESTRICT_ATTRIBUTE output,
        isize n_elements, i32 n_threads
    ) {
        // The order of the first and second elements in the output are the 4 LSB and 4 MSB, respectively.
        // Note: We don't support odd rows, but if the row had an odd number of elements, the last byte of
        // the row has the 4 MSB unset.
        #pragma omp parallel for num_threads(n_threads)
        for (isize i = 0; i < n_elements / 2; ++i) {
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
        isize n_elements, i32 n_threads
    ) {
        constexpr unsigned char MASK_4LSB{0b00001111};

        #pragma omp parallel for num_threads(n_threads)
        for (isize i = 0; i < n_elements / 2; ++i) {
            const auto tmp = static_cast<unsigned char>(input[i]);
            output[i * 2] = static_cast<T>(tmp & MASK_4LSB);
            output[i * 2 + 1] = static_cast<T>((tmp >> 4) & MASK_4LSB);
        }
    }

    struct u4_encoding {};

    #if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wsign-conversion"
    #pragma GCC diagnostic ignored "-Wsign-compare"
    #elif defined(NOA_COMPILER_MSVC)
    #pragma warning(push, 0)
    #endif

    template<typename Output, typename Input, usize N, StridesTraits S>
    void encode_file_(
        const Span<const Input, N, isize, S>& input,
        std::FILE* output,
        bool clamp, bool swap_endian,
        i32 n_threads
    ) {
        // Chunk size.
        constexpr isize N_BYTES_PER_CHUNK = 32 * 1024 * 1024;
        constexpr isize N_BYTES_PER_ELEMENT = sizeof(Output);

        const isize n_elements = input.ssize();
        const isize n_bytes = std::same_as<Output, u4_encoding> ? n_elements / 2 : n_elements * N_BYTES_PER_ELEMENT;

        // Allocate two double-buffers to encode and write in parallel.
        constexpr usize ALIGNMENT = 64;
        std::unique_ptr buffer_0 = noa::cpu::AllocatorHeap::allocate<char, ALIGNMENT>(N_BYTES_PER_CHUNK);
        std::unique_ptr buffer_1 = noa::cpu::AllocatorHeap::allocate<char, ALIGNMENT>(N_BYTES_PER_CHUNK);

        // Encode and write.
        const i32 io_threads = 1;
        const i32 encode_threads = std::clamp(n_threads - 1, 1, 6);

        #pragma omp parallel num_threads(io_threads + encode_threads)
        #pragma omp single
        {
            isize current_byte_offset = 0;
            char* active_buffer = buffer_0.get();
            char* alternative_buffer = buffer_1.get();

            // Calculate chunk sizes for the absolute first encoding pass.
            isize n_bytes_to_encode = std::min(N_BYTES_PER_CHUNK, n_bytes - current_byte_offset);
            isize n_elements_to_encode = n_bytes_to_encode / N_BYTES_PER_ELEMENT;

            while (n_bytes_to_encode > 0) {
                // Prepare boundaries for the encoding task
                char* encoding_buffer = active_buffer;
                isize encoding_offset = current_byte_offset;
                isize encoding_elements = n_elements_to_encode;

                #pragma omp task firstprivate(encoding_buffer, encoding_offset, encoding_elements) depend(out: encoding_buffer[0])
                {
                    const isize n_elements_offset = encoding_offset / N_BYTES_PER_ELEMENT;
                    if constexpr (N == 1) {
                        #pragma omp taskloop num_tasks(encode_threads) shared(input)
                        for (isize i = 0; i < encoding_elements; ++i) {
                            const Input* input_ptr = input.get() + n_elements_offset;
                            if constexpr (std::same_as<Output, u4_encoding>) {
                                u32 l_val = clamp_cast<u32>(noa::round(input_ptr[i * 2]));
                                u32 h_val = clamp_cast<u32>(noa::round(input_ptr[i * 2 + 1]));
                                l_val = noa::clamp(l_val, 0u, 15u);
                                h_val = noa::clamp(h_val, 0u, 15u);
                                encoding_buffer[i] = static_cast<char>(l_val + (h_val << 4));
                            } else {
                                auto* output_ptr = reinterpret_cast<Output*>(encoding_buffer);
                                output_ptr[i] = Encoder<Input, Output>{clamp, swap_endian}(input_ptr[i]);
                            }
                        }
                    } else {
                        if constexpr (not std::same_as<Output, u4_encoding>) {
                            #pragma omp taskloop num_tasks(encode_threads) shared(input)
                            for (isize i = 0; i < encoding_elements; ++i) {
                                auto* output_ptr = reinterpret_cast<Output*>(encoding_buffer);
                                auto indices = noa::offset2index(n_elements_offset + i, input.shape());
                                output_ptr[i] = Encoder<Input, Output>{clamp, swap_endian}(input(indices));
                            }
                        } else {
                            unreachable();
                        }
                    }
                }

                // While the encoding is running on the main buffer, write the previous chunk located in the alternative buffer.
                // For the first iteration, there's nothing to write, this is skipped.
                isize n_bytes_to_write = encoding_offset;
                if (n_bytes_to_write > 0) {
                    isize prev_chunk_bytes = std::min(N_BYTES_PER_CHUNK, n_bytes - (encoding_offset - N_BYTES_PER_CHUNK));
                    usize bytes_written = std::fwrite(alternative_buffer, 1, static_cast<usize>(prev_chunk_bytes), output);
                    check(bytes_written == static_cast<usize>(prev_chunk_bytes), "Write failed");
                }

                // Wait for the encoding task and prepare next iteration.
                #pragma omp taskwait

                current_byte_offset += n_bytes_to_encode;
                isize next_bytes_to_encode = std::min(N_BYTES_PER_CHUNK, n_bytes - current_byte_offset);
                if (next_bytes_to_encode == 0) {
                    // Encoding is all done, this is the last iteration, so write the encoded chunk.
                    usize bytes_written = std::fwrite(encoding_buffer, 1, static_cast<usize>(n_bytes_to_encode), output);
                    check(bytes_written == static_cast<usize>(n_bytes_to_encode), "Write failed");
                }

                std::swap(active_buffer, alternative_buffer);
                n_bytes_to_encode = next_bytes_to_encode;
                n_elements_to_encode = n_bytes_to_encode / N_BYTES_PER_ELEMENT;
            }
        }
    }

    template<typename Input, typename Output, usize N, StridesTraits S>
    void decode_file_(
        std::FILE* input,
        const Span<Output, N, isize, S>& output,
        bool clamp, bool swap_endian,
        i32 n_threads
    ) {
        // Chunk size.
        constexpr isize N_BYTES_PER_CHUNK = 32 * 1024 * 1024;
        constexpr isize N_BYTES_PER_ELEMENT = sizeof(Input);

        const isize n_elements = output.ssize();
        const isize n_bytes = std::same_as<Input, u4_encoding> ? n_elements / 2 : n_elements * N_BYTES_PER_ELEMENT;

        // Allocate two double-buffers to read and decode in parallel.
        constexpr usize ALIGNMENT = 64;
        std::unique_ptr buffer_0 = noa::cpu::AllocatorHeap::allocate<char, ALIGNMENT>(N_BYTES_PER_CHUNK);
        std::unique_ptr buffer_1 = noa::cpu::AllocatorHeap::allocate<char, ALIGNMENT>(N_BYTES_PER_CHUNK);

        // Read and decode.
        const i32 io_threads = 1;
        const i32 decode_threads = std::clamp(n_threads - 1, 1, 6);

        #pragma omp parallel num_threads(io_threads + decode_threads)
        #pragma omp single
        {
            isize current_byte_offset = 0;
            char* active_buffer = buffer_0.get();
            char* alternative_buffer = buffer_1.get();

            // Read the first chunk into the active buffer before entering the loop.
            isize n_bytes_to_read = std::min(N_BYTES_PER_CHUNK, n_bytes - current_byte_offset);
            isize n_elements_to_read = n_bytes_to_read / N_BYTES_PER_ELEMENT;
            usize bytes_read = std::fread(active_buffer, 1, static_cast<usize>(n_bytes_to_read), input);
            check(bytes_read == static_cast<usize>(n_bytes_to_read), "Read failed");

            while (n_bytes_to_read > 0) {
                // Prepare and launch the decoding task.
                char* decoding_buffer = active_buffer;
                isize decoding_offset = current_byte_offset;
                isize decoding_elements = n_elements_to_read;

                #pragma omp task firstprivate(decoding_buffer, decoding_offset, decoding_elements) depend(out: decoding_buffer[0])
                {
                    const isize n_elements_offset = decoding_offset / N_BYTES_PER_ELEMENT;
                    if constexpr (N == 1) {
                        #pragma omp taskloop num_tasks(decode_threads) shared(output)
                        for (isize i = 0; i < decoding_elements; ++i) {
                            Output* output_ptr = output.get() + n_elements_offset;
                            if constexpr (std::same_as<Input, u4_encoding>) {
                                constexpr char MASK_4LSB{0b00001111};
                                const char byte = decoding_buffer[i];
                                output_ptr[i * 2] = static_cast<Output>(byte & MASK_4LSB);
                                output_ptr[i * 2 + 1] = static_cast<Output>((byte >> 4) & MASK_4LSB);
                            } else {
                                auto* input_ptr = reinterpret_cast<Input*>(decoding_buffer);
                                output_ptr[i] = Decoder<Input, Output>{clamp, swap_endian}(input_ptr[i]);
                            }
                        }
                    } else {
                        if constexpr (not std::same_as<Input, u4_encoding>) {
                            #pragma omp taskloop num_tasks(decode_threads) shared(output)
                            for (isize i = 0; i < decoding_elements; ++i) {
                                auto* input_ptr = reinterpret_cast<Input*>(decoding_buffer);
                                auto indices = noa::offset2index(n_elements_offset + i, output.shape());
                                output(indices) = Decoder<Input, Output>{clamp, swap_endian}(input_ptr[i]);
                            }
                        } else {
                            unreachable();
                        }
                    }
                }

                // While the decoding is running on the main buffer, read the next chunk into the alternative buffer.
                current_byte_offset += n_bytes_to_read;
                isize next_bytes_to_read = std::min(N_BYTES_PER_CHUNK, n_bytes - current_byte_offset);
                if (next_bytes_to_read > 0) {
                    usize next_read = std::fread(alternative_buffer, 1, static_cast<usize>(next_bytes_to_read), input);
                    check(next_read == static_cast<usize>(next_bytes_to_read), "Read failed");
                }

                // Wait for the decoding task and prepare next iteration.
                #pragma omp taskwait
                std::swap(active_buffer, alternative_buffer);
                n_bytes_to_read = next_bytes_to_read;
                n_elements_to_read = n_bytes_to_read / N_BYTES_PER_ELEMENT;
            }
        }
    }

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
#pragma GCC diagnostic pop
#elif defined(NOA_COMPILER_MSVC)
#pragma warning(pop)
#endif
}

namespace noa::io::details {
    template<typename T, usize N, StridesTraits S> // instantiated: 2-contiguous or 6-strided
    void encode_(
        const Span<const T, N, isize, S>& input,
        const SpanContiguous<std::byte, 1>& output,
        const DataType& output_dtype,
        const EncodeOptions& options
    ) {
        const isize n_elements = input.ssize();
        const isize n_encoded_bytes = output_dtype.n_bytes(n_elements);
        const auto n_threads = actual_n_threads_(n_elements, options.n_threads);

        check(n_encoded_bytes <= output.ssize(), "The encoded array is not big enough to contain the input array");
        check(not are_overlapped(input, output), "The input and output arrays should not overlap");

        switch (output_dtype) {
            case DataType::U4:
                if constexpr (nt::scalar<T>) {
                    check(input.is_contiguous() and is_even(n_elements),
                          "u4 encoding requires the input array to be contiguous and have an even number of elements");
                    return encode_4bits_(input.data(), output.data(), input.n_elements(), n_threads);
                }
                break;
            case DataType::I8:
                if constexpr (nt::scalar<T>)
                    return actual_encode_<i8>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::U8:
                if constexpr (nt::scalar<T>)
                    return actual_encode_<u8>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::I16:
                if constexpr (nt::scalar<T>)
                    return actual_encode_<i16>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::U16:
                if constexpr (nt::scalar<T>)
                    return actual_encode_<u16>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::I32:
                if constexpr (nt::scalar<T>)
                    return actual_encode_<i32>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::U32:
                if constexpr (nt::scalar<T>)
                    return actual_encode_<u32>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::I64:
                if constexpr (nt::scalar<T>)
                    return actual_encode_<i64>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::U64:
                if constexpr (nt::scalar<T>)
                    return actual_encode_<u64>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::F16:
                if constexpr (nt::scalar<T>)
                    return actual_encode_<f16>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::F32:
                if constexpr (nt::scalar<T>)
                    return actual_encode_<f32>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::F64:
                if constexpr (nt::scalar<T>)
                    return actual_encode_<f64>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::CI16:
                if constexpr (nt::complex<T>)
                    return details::encode_(input.template as<nt::const_value_type_t<T>>(), output, DataType::I16, options);
                break;
            case DataType::C16:
                if constexpr (nt::complex<T>)
                    return details::encode_(input.template as<nt::const_value_type_t<T>>(), output, DataType::F16, options);
                break;
            case DataType::C32:
                if constexpr (nt::complex<T>)
                    return details::encode_(input.template as<nt::const_value_type_t<T>>(), output, DataType::F32, options);
                break;
            case DataType::C64:
                if constexpr (nt::complex<T>)
                    return details::encode_(input.template as<nt::const_value_type_t<T>>(), output, DataType::F64, options);
                break;
            case DataType::UNKNOWN:
                break;
        }
        panic("{} cannot be encoded into {}", nd::stringify<T>(), output_dtype);
    }

    template<typename T, usize N, StridesTraits S> // instantiated: 1-contiguous or 6-strided
    void encode_(
        const Span<const T, N, isize, S>& input,
        std::FILE* output,
        const DataType& output_dtype,
        const EncodeOptions& options
    ) {
        const isize n_elements = input.ssize();
        const isize n_encoded_bytes = output_dtype.n_bytes(n_elements);
        const isize remaining_bytes = remaining_bytes_from_file(output);
        const auto n_threads = actual_n_threads_(n_elements, options.n_threads);

        check(n_encoded_bytes <= remaining_bytes,
              "The file (remaining_bytes={}) is not big enough to contain the encoded array",
              remaining_bytes);

        switch (output_dtype) {
            case DataType::U4:
                if constexpr (nt::scalar<T>) {
                    check(input.is_contiguous() and is_even(n_elements),
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
                    return encode_(input.template as<nt::const_value_type_t<T>>(), output, DataType::I16, options);
                break;
            case DataType::C16:
                if constexpr (nt::complex<T>)
                    return encode_(input.template as<nt::const_value_type_t<T>>(), output, DataType::F16, options);
                break;
            case DataType::C32:
                if constexpr (nt::complex<T>)
                    return encode_(input.template as<nt::const_value_type_t<T>>(), output, DataType::F32, options);
                break;
            case DataType::C64:
                if constexpr (nt::complex<T>)
                    return encode_(input.template as<nt::const_value_type_t<T>>(), output, DataType::F64, options);
                break;
            case DataType::UNKNOWN:
                break;
        }
        panic("{} cannot be encoded into {}", nd::stringify<T>(), output_dtype);
    }

    template<typename T, usize N, StridesTraits S> // instantiated: 2-contiguous or 6-strided
    void decode_(
        const SpanContiguous<const std::byte, 1>& input,
        const DataType& input_dtype,
        const Span<T, N, isize, S>& output,
        const DecodeOptions& options
    ) {
        const isize n_elements = output.ssize();
        const isize n_encoded_bytes = input_dtype.n_bytes(n_elements);
        const auto n_threads = actual_n_threads_(n_elements, options.n_threads);

        check(n_encoded_bytes <= input.ssize(), "The encoded array is not big enough to contain the output array");
        check(not are_overlapped(input, output), "The input and output arrays should not overlap");

        switch (input_dtype) {
            case DataType::U4:
                if constexpr (nt::scalar<T>) {
                    check(output.is_contiguous() and is_even(n_elements),
                          "u4 encoding requires the output array to be contiguous and have an even number of elements");
                    return decode_4bits_(input.data(), output.data(), output.n_elements(), n_threads);
                }
                break;
            case DataType::I8:
                if constexpr (nt::scalar<T>)
                    return actual_decode_<i8>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::U8:
                if constexpr (nt::scalar<T>)
                    return actual_decode_<u8>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::I16:
                if constexpr (nt::scalar<T>)
                    return actual_decode_<i16>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::U16:
                if constexpr (nt::scalar<T>)
                    return actual_decode_<u16>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::I32:
                if constexpr (nt::scalar<T>)
                    return actual_decode_<i32>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::U32:
                if constexpr (nt::scalar<T>)
                    return actual_decode_<u32>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::I64:
                if constexpr (nt::scalar<T>)
                    return actual_decode_<i64>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::U64:
                if constexpr (nt::scalar<T>)
                    return actual_decode_<u64>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::F16:
                if constexpr (nt::scalar<T>)
                    return actual_decode_<f16>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::F32:
                if constexpr (nt::scalar<T>)
                    return actual_decode_<f32>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::F64:
                if constexpr (nt::scalar<T>)
                    return actual_decode_<f64>(input, output, options.clamp, options.endian_swap, n_threads);
                break;
            case DataType::CI16:
                if constexpr (nt::complex<T>)
                    return details::decode_(input, DataType::I16, output.template as<typename T::value_type>(), options);
                break;
            case DataType::C16:
                if constexpr (nt::complex<T>)
                    return details::decode_(input, DataType::F16, output.template as<typename T::value_type>(), options);
                break;
            case DataType::C32:
                if constexpr (nt::complex<T>)
                    return details::decode_(input, DataType::F32, output.template as<typename T::value_type>(), options);
                break;
            case DataType::C64:
                if constexpr (nt::complex<T>)
                    return details::decode_(input, DataType::F64, output.template as<typename T::value_type>(), options);
                break;
            case DataType::UNKNOWN:
                break;
        }
        panic("{} cannot be decoded into {}", input_dtype, nd::stringify<T>());
    }

    template<typename T, usize N, StridesTraits S> // instantiated: 1-contiguous or 6-strided
    void decode_(
        std::FILE* input,
        const DataType& input_dtype,
        const Span<T, N, isize, S>& output,
        const DecodeOptions& options
    ) {
        const isize n_elements = output.n_elements();
        const isize n_encoded_bytes = input_dtype.n_bytes(n_elements);
        const isize remaining_bytes = remaining_bytes_from_file(input);
        const auto n_threads = actual_n_threads_(n_elements, options.n_threads);

        check(n_encoded_bytes <= remaining_bytes,
              "The file (remaining_bytes={}) is not big enough to contain the decoded array",
              remaining_bytes);

        switch (input_dtype) {
            case DataType::U4:
                if constexpr (nt::scalar<T>) {
                    check(output.is_contiguous() and is_even(n_elements),
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
                    return details::decode_(input, DataType::I16, output.template as<typename T::value_type>(), options);
                }
                break;
            case DataType::C16:
                if constexpr (nt::complex<T>) {
                    return details::decode_(input, DataType::F16, output.template as<typename T::value_type>(), options);
                }
                break;
            case DataType::C32:
                if constexpr (nt::complex<T>) {
                    return details::decode_(input, DataType::F32, output.template as<typename T::value_type>(), options);
                }
                break;
            case DataType::C64:
                if constexpr (nt::complex<T>) {
                    return details::decode_(input, DataType::F64, output.template as<typename T::value_type>(), options);
                }
                break;
            case DataType::UNKNOWN:
                break;
        }
        panic("{} cannot be decoded into {}", input_dtype, nd::stringify<T>());
    }
}

namespace noa::io {
    void encode(
        const SpanContiguous<const std::byte, 1>& input,
        const DataType& input_dtype,
        const SpanContiguous<std::byte, 1>& output,
        const DataType& output_dtype,
        const EncodeOptions& options
    ) {
        switch (input_dtype) {
            case DataType::I8:
                return details::encode_(input.as_contiguous<const i8, 2>(), output, output_dtype, options);
            case DataType::I16:
                return details::encode_(input.as_contiguous<const i16, 2>(), output, output_dtype, options);
            case DataType::I32:
                return details::encode_(input.as_contiguous<const i32, 2>(), output, output_dtype, options);
            case DataType::I64:
                return details::encode_(input.as_contiguous<const i64, 2>(), output, output_dtype, options);
            case DataType::U8:
                return details::encode_(input.as_contiguous<const u8, 2>(), output, output_dtype, options);
            case DataType::U16:
                return details::encode_(input.as_contiguous<const u16, 2>(), output, output_dtype, options);
            case DataType::U32:
                return details::encode_(input.as_contiguous<const u32, 2>(), output, output_dtype, options);
            case DataType::U64:
                return details::encode_(input.as_contiguous<const u64, 2>(), output, output_dtype, options);
            case DataType::F16:
                return details::encode_(input.as_contiguous<const f16, 2>(), output, output_dtype, options);
            case DataType::F32:
                return details::encode_(input.as_contiguous<const f32, 2>(), output, output_dtype, options);
            case DataType::F64:
                return details::encode_(input.as_contiguous<const f64, 2>(), output, output_dtype, options);
            case DataType::C16:
                return details::encode_(input.as_contiguous<const c16, 2>(), output, output_dtype, options);
            case DataType::C32:
                return details::encode_(input.as_contiguous<const c32, 2>(), output, output_dtype, options);
            case DataType::C64:
                return details::encode_(input.as_contiguous<const c64, 2>(), output, output_dtype, options);
            case DataType::U4:
            case DataType::CI16:
                panic("TODO: u4 and ci16 cannot be reinterpreted to valid types, they would require special cases");
            case DataType::UNKNOWN:
                panic("Unknown data type");
        }
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
                return details::encode_(input.as_contiguous<const i8, 2>(), output, output_dtype, options);
            case DataType::I16:
                return details::encode_(input.as_contiguous<const i16, 2>(), output, output_dtype, options);
            case DataType::I32:
                return details::encode_(input.as_contiguous<const i32, 2>(), output, output_dtype, options);
            case DataType::I64:
                return details::encode_(input.as_contiguous<const i64, 2>(), output, output_dtype, options);
            case DataType::U8:
                return details::encode_(input.as_contiguous<const u8, 2>(), output, output_dtype, options);
            case DataType::U16:
                return details::encode_(input.as_contiguous<const u16, 2>(), output, output_dtype, options);
            case DataType::U32:
                return details::encode_(input.as_contiguous<const u32, 2>(), output, output_dtype, options);
            case DataType::U64:
                return details::encode_(input.as_contiguous<const u64, 2>(), output, output_dtype, options);
            case DataType::F16:
                return details::encode_(input.as_contiguous<const f16, 2>(), output, output_dtype, options);
            case DataType::F32:
                return details::encode_(input.as_contiguous<const f32, 2>(), output, output_dtype, options);
            case DataType::F64:
                return details::encode_(input.as_contiguous<const f64, 2>(), output, output_dtype, options);
            case DataType::C16:
                return details::encode_(input.as_contiguous<const c16, 2>(), output, output_dtype, options);
            case DataType::C32:
                return details::encode_(input.as_contiguous<const c32, 2>(), output, output_dtype, options);
            case DataType::C64:
                return details::encode_(input.as_contiguous<const c64, 2>(), output, output_dtype, options);
            case DataType::U4:
            case DataType::CI16:
                panic("TODO: u4 and ci16 cannot be reinterpreted to valid types, they would require special cases");
            case DataType::UNKNOWN:
                panic("Unknown data type");
        }
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
                return details::decode_(input, input_dtype, output.as_contiguous<i8, 2>(), options);
            case DataType::I16:
                return details::decode_(input, input_dtype, output.as_contiguous<i16, 2>(), options);
            case DataType::I32:
                return details::decode_(input, input_dtype, output.as_contiguous<i32, 2>(), options);
            case DataType::I64:
                return details::decode_(input, input_dtype, output.as_contiguous<i64, 2>(), options);
            case DataType::U8:
                return details::decode_(input, input_dtype, output.as_contiguous<u8, 2>(), options);
            case DataType::U16:
                return details::decode_(input, input_dtype, output.as_contiguous<u16, 2>(), options);
            case DataType::U32:
                return details::decode_(input, input_dtype, output.as_contiguous<u32, 2>(), options);
            case DataType::U64:
                return details::decode_(input, input_dtype, output.as_contiguous<u64, 2>(), options);
            case DataType::F16:
                return details::decode_(input, input_dtype, output.as_contiguous<f16, 2>(), options);
            case DataType::F32:
                return details::decode_(input, input_dtype, output.as_contiguous<f32, 2>(), options);
            case DataType::F64:
                return details::decode_(input, input_dtype, output.as_contiguous<f64, 2>(), options);
            case DataType::C16:
                return details::decode_(input, input_dtype, output.as_contiguous<c16, 2>(), options);
            case DataType::C32:
                return details::decode_(input, input_dtype, output.as_contiguous<c32, 2>(), options);
            case DataType::C64:
                return details::decode_(input, input_dtype, output.as_contiguous<c64, 2>(), options);
            case DataType::U4:
            case DataType::CI16:
                panic("TODO: u4 and ci16 cannot be reinterpreted to valid types, they would require special cases");
            case DataType::UNKNOWN:
                panic("Unknown data type");
        }
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
                return details::decode_(input, input_dtype, output.as_contiguous<i8, 2>(), options);
            case DataType::I16:
                return details::decode_(input, input_dtype, output.as_contiguous<i16, 2>(), options);
            case DataType::I32:
                return details::decode_(input, input_dtype, output.as_contiguous<i32, 2>(), options);
            case DataType::I64:
                return details::decode_(input, input_dtype, output.as_contiguous<i64, 2>(), options);
            case DataType::U8:
                return details::decode_(input, input_dtype, output.as_contiguous<u8, 2>(), options);
            case DataType::U16:
                return details::decode_(input, input_dtype, output.as_contiguous<u16, 2>(), options);
            case DataType::U32:
                return details::decode_(input, input_dtype, output.as_contiguous<u32, 2>(), options);
            case DataType::U64:
                return details::decode_(input, input_dtype, output.as_contiguous<u64, 2>(), options);
            case DataType::F16:
                return details::decode_(input, input_dtype, output.as_contiguous<f16, 2>(), options);
            case DataType::F32:
                return details::decode_(input, input_dtype, output.as_contiguous<f32, 2>(), options);
            case DataType::F64:
                return details::decode_(input, input_dtype, output.as_contiguous<f64, 2>(), options);
            case DataType::C16:
                return details::decode_(input, input_dtype, output.as_contiguous<c16, 2>(), options);
            case DataType::C32:
                return details::decode_(input, input_dtype, output.as_contiguous<c32, 2>(), options);
            case DataType::C64:
                return details::decode_(input, input_dtype, output.as_contiguous<c64, 2>(), options);
            case DataType::U4:
            case DataType::CI16:
                panic("TODO: u4 and ci16 cannot be reinterpreted to valid types, they would require special cases");
            case DataType::UNKNOWN:
                panic("Unknown data type");
        }
    }

    #define NOA_IO_ENCODE_SPAN_(T, N, S) \
        template void details::encode_<T, N, S>(const Span<const T, N, isize, S>&, const SpanContiguous<std::byte, 1>&, const DataType&, const EncodeOptions&); \
        template void details::decode_<T, N, S>(const SpanContiguous<const std::byte, 1>&, const DataType&, const Span<T, N, isize, S>&, const DecodeOptions&)

    #define NOA_IO_ENCODE_FILE_(T, N, S) \
        template void details::encode_<T, N, S>(const Span<const T, N, isize, S>&, std::FILE*, const DataType&, const EncodeOptions&); \
        template void details::decode_<T, N, S>(std::FILE*, const DataType&, const Span<T, N, isize, S>&, const DecodeOptions&)

    #define NOA_IO_ENCODE_ND_(T) \
        NOA_IO_ENCODE_SPAN_(T, 2, StridesTraits::CONTIGUOUS);   \
        NOA_IO_ENCODE_SPAN_(T, 6, StridesTraits::STRIDED);      \
        NOA_IO_ENCODE_FILE_(T, 1, StridesTraits::CONTIGUOUS);   \
        NOA_IO_ENCODE_FILE_(T, 6, StridesTraits::STRIDED)

    NOA_IO_ENCODE_ND_(i8);
    NOA_IO_ENCODE_ND_(u8);
    NOA_IO_ENCODE_ND_(i16);
    NOA_IO_ENCODE_ND_(u16);
    NOA_IO_ENCODE_ND_(i32);
    NOA_IO_ENCODE_ND_(u32);
    NOA_IO_ENCODE_ND_(i64);
    NOA_IO_ENCODE_ND_(u64);
    NOA_IO_ENCODE_ND_(f16);
    NOA_IO_ENCODE_ND_(f32);
    NOA_IO_ENCODE_ND_(f64);
    NOA_IO_ENCODE_ND_(c16);
    NOA_IO_ENCODE_ND_(c32);
    NOA_IO_ENCODE_ND_(c64);
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
