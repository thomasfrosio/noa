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

    template<typename Output, typename Input>
    void encode_1d_(
        SpanContiguous<const Input, 1> input,
        SpanContiguous<std::byte, 1> output,
        bool clamp, bool swap_endian, i32 n_threads
    ) {
        const i64 n_elements = input.ssize();
        auto* ptr = reinterpret_cast<Output*>(output.get());

        if (swap_endian) {
            #pragma omp parallel for num_threads(n_threads) default(none) shared(input, clamp, n_elements, ptr)
            for (i64 idx = 0; idx < n_elements; ++idx) {
                auto value = clamp ? clamp_cast<Output>(input[idx]) : static_cast<Output>(input[idx]);
                ptr[idx] = noa::io::swap_endian(value);
            }
        } else {
            #pragma omp parallel for num_threads(n_threads) default(none) shared(input, clamp, n_elements, ptr)
            for (i64 idx = 0; idx < n_elements; ++idx)
                ptr[idx] = clamp ? clamp_cast<Output>(input[idx]) : static_cast<Output>(input[idx]);
        }
    }

    template<typename Output, typename Input>
    void encode_4d_(
        const Span<const Input, 4>& input,
        const SpanContiguous<std::byte, 1>& output,
        bool clamp, bool swap_endian, i32 n_threads
    ) {
        if (input.are_contiguous())
            return encode_1d_<Output>(input.as_contiguous_1d(), output, clamp, swap_endian, n_threads);

        auto* ptr = reinterpret_cast<Output*>(output.get());

        if (n_threads > 1) {
            // Collapse manually since we need to keep track of a linear index anyway...
            #pragma omp parallel for num_threads(n_threads) default(none) shared(input, clamp, swap_endian, ptr)
            for (i64 i = 0; i < input.ssize(); ++i) {
                auto& value = input(ni::offset2index(i, input.shape()));
                auto encoded = clamp ? clamp_cast<Output>(value) : static_cast<Output>(value);
                ptr[i] = swap_endian ? noa::io::swap_endian(encoded) : encoded;
            }
        } else {
            for (i64 i = 0; i < input.shape()[0]; ++i) {
                for (i64 j = 0; j < input.shape()[1]; ++j) {
                    for (i64 k = 0; k < input.shape()[2]; ++k) {
                        for (i64 l = 0; l < input.shape()[3]; ++l) {
                            auto& value = input(i, j, k, l);
                            auto encoded = clamp ? clamp_cast<Output>(value) : static_cast<Output>(value);
                            *(ptr++) = swap_endian ? noa::io::swap_endian(encoded) : encoded;
                        }
                    }
                }
            }
        }
    }

    template<typename Input, typename Output>
    void decode_1d_(
        SpanContiguous<const std::byte, 1> input,
        SpanContiguous<Output, 1> output,
        bool clamp, bool swap_endian, i32 n_threads
    ) {
        auto* ptr = reinterpret_cast<const Input*>(input.get());
        if (swap_endian) {
            #pragma omp parallel for num_threads(n_threads) default(none) shared(output, clamp, ptr)
            for (i64 idx = 0; idx < output.ssize(); ++idx) {
                auto value = noa::io::swap_endian(ptr[idx]);
                output[idx] = clamp ? clamp_cast<Output>(value) : static_cast<Output>(value);
            }
        } else {
            #pragma omp parallel for num_threads(n_threads) default(none) shared(output, clamp, ptr)
            for (i64 idx = 0; idx < output.ssize(); ++idx)
                output[idx] = clamp ? clamp_cast<Output>(ptr[idx]) : static_cast<Output>(ptr[idx]);
        }
    }

    template<typename Input, typename Output>
    void decode_4d_(
        SpanContiguous<const std::byte, 1> input,
        const Span<Output, 4>& output,
        bool clamp, bool swap_endian, i32 n_threads
    ) {
        if (output.are_contiguous())
            return decode_1d_<Input>(input, output.as_contiguous_1d(), clamp, swap_endian, n_threads);

        auto* ptr = reinterpret_cast<const Input*>(input.get());

        if (n_threads > 1) {
            // Collapse manually since we need to keep track of a linear index anyway...
            const i64 ssize = input.ssize();
            #pragma omp parallel for num_threads(n_threads) default(none) shared(output, clamp, swap_endian, ptr, ssize)
            for (i64 i = 0; i < ssize; ++i) {
                auto value = ptr[i];
                if (swap_endian)
                    value = noa::io::swap_endian(value);
                output(ni::offset2index(i, output.shape())) =
                    clamp ? clamp_cast<Output>(value) : static_cast<Output>(value);
            }
        } else {
            for (i64 i = 0; i < output.shape()[0]; ++i) {
                for (i64 j = 0; j < output.shape()[1]; ++j) {
                    for (i64 k = 0; k < output.shape()[2]; ++k) {
                        for (i64 l = 0; l < output.shape()[3]; ++l) {
                            auto value = *(ptr++);
                            if (swap_endian)
                                value = noa::io::swap_endian(value);
                            output(i, j, k, l) = clamp ? clamp_cast<Output>(value) : static_cast<Output>(value);
                        }
                    }
                }
            }
        }
    }

    template<nt::scalar T>
    void encode_4bits_(
        SpanContiguous<const T, 1> input,
        SpanContiguous<std::byte, 1> output,
        bool clamp, i32 n_threads
    ) {
        // The order of the first and second elements in the output are the 4 LSB and 4 MSB, respectively.
        // Note: We don't support odd rows, but if the row had an odd number of elements, the last byte of
        // the row has the 4 MSB unset.
        if (clamp) {
            #pragma omp parallel for num_threads(n_threads)
            for (i64 i = 0; i < input.ssize() / 2; ++i) {
                u32 l_val = clamp_cast<u32>(input[2 * i]); // If IEEE float, default round to nearest
                u32 h_val = clamp_cast<u32>(input[2 * i + 1]);
                l_val = noa::clamp(l_val, 0U, 15U); // 2^4-1
                h_val = noa::clamp(h_val, 0U, 15U);
                u32 tmp = l_val + (h_val << 4);
                std::memcpy(output.get() + i, &tmp, 1);
            }
        } else {
            #pragma omp parallel for num_threads(n_threads)
            for (i64 i = 0; i < input.ssize() / 2; ++i) {
                // std::round could be used instead, but we assume values are positive so +0.5f is enough
                u32 l_val = static_cast<u32>(noa::round(input[2 * i]));
                u32 h_val = static_cast<u32>(noa::round(input[2 * i + 1] ));
                u32 tmp = l_val + (h_val << 4);
                std::memcpy(output.get() + i, &tmp, 1);
            }
        }
    }

    template<typename T>
    void decode_4bits_(
        SpanContiguous<const std::byte, 1> input,
        SpanContiguous<T, 1> output,
        i32 n_threads
    ) {
        constexpr unsigned char MASK_4LSB{0b00001111};

        #pragma omp parallel for num_threads(n_threads)
        for (i64 i = 0; i < output.ssize() / 2; ++i) {
            const auto tmp = static_cast<unsigned char>(input[i]);
            output[i * 2] = static_cast<T>(tmp & MASK_4LSB);
            output[i * 2 + 1] = static_cast<T>((tmp >> 4) & MASK_4LSB);
        }
    }
}

namespace noa::io {
    template<nt::numeric T>
    void encode(
        const Span<const T, 4>& input,
        SpanContiguous<std::byte, 1> output,
        Encoding encoding,
        i32 n_threads
    ) {
        const i64 n_elements = input.ssize();
        const i64 n_encoded_bytes = encoding.encoded_size(n_elements);
        check(n_encoded_bytes <= output.ssize(), "The encoded array is not big enough to contain the input array");
        n_threads = actual_n_threads_(n_elements, n_threads);

        switch (encoding.dtype) {
            case Encoding::U4:
                if constexpr (nt::scalar<T>) {
                    check(input.are_contiguous() and is_even(input.shape()[3]),
                          "u4 encoding requires the input array to be contiguous and have even rows");
                    return encode_4bits_(input.as_contiguous_1d(), output, encoding.clamp, n_threads);
                }
                break;
            case Encoding::I8:
                if constexpr (nt::scalar<T>)
                    return encode_4d_<i8>(input, output, encoding.clamp, encoding.endian_swap, n_threads);
                break;
            case Encoding::U8:
                if constexpr (nt::scalar<T>)
                    return encode_4d_<u8>(input, output, encoding.clamp, encoding.endian_swap, n_threads);
                break;
            case Encoding::I16:
                if constexpr (nt::scalar<T>)
                    return encode_4d_<i16>(input, output, encoding.clamp, encoding.endian_swap, n_threads);
                break;
            case Encoding::U16:
                if constexpr (nt::scalar<T>)
                    return encode_4d_<u16>(input, output, encoding.clamp, encoding.endian_swap, n_threads);
                break;
            case Encoding::I32:
                if constexpr (nt::scalar<T>)
                    return encode_4d_<i32>(input, output, encoding.clamp, encoding.endian_swap, n_threads);
                break;
            case Encoding::U32:
                if constexpr (nt::scalar<T>)
                    return encode_4d_<u32>(input, output, encoding.clamp, encoding.endian_swap, n_threads);
                break;
            case Encoding::I64:
                if constexpr (nt::scalar<T>)
                    return encode_4d_<i64>(input, output, encoding.clamp, encoding.endian_swap, n_threads);
                break;
            case Encoding::U64:
                if constexpr (nt::scalar<T>)
                    return encode_4d_<u64>(input, output, encoding.clamp, encoding.endian_swap, n_threads);
                break;
            case Encoding::F16:
                if constexpr (nt::scalar<T>)
                    return encode_4d_<f16>(input, output, encoding.clamp, encoding.endian_swap, n_threads);
                break;
            case Encoding::F32:
                if constexpr (nt::scalar<T>)
                    return encode_4d_<f32>(input, output, encoding.clamp, encoding.endian_swap, n_threads);
                break;
            case Encoding::F64:
                if constexpr (nt::scalar<T>)
                    return encode_4d_<f64>(input, output, encoding.clamp, encoding.endian_swap, n_threads);
                break;
            case Encoding::CI16:
                if constexpr (nt::complex<T>) {
                    using real_t = const T::value_type;
                    auto new_encoding = Encoding{Encoding::I16, encoding.clamp, encoding.endian_swap};
                    return encode(input.template as<real_t>(), output, new_encoding, n_threads);
                }
                break;
            case Encoding::C16:
                if constexpr (nt::complex<T>) {
                    using real_t = const T::value_type;
                    auto new_encoding = Encoding{Encoding::F16, encoding.clamp, encoding.endian_swap};
                    return encode(input.template as<real_t>(), output, new_encoding, n_threads);
                }
                break;
            case Encoding::C32:
                if constexpr (nt::complex<T>) {
                    using real_t = const T::value_type;
                    auto new_encoding = Encoding{Encoding::F32, encoding.clamp, encoding.endian_swap};
                    return encode(input.template as<real_t>(), output, new_encoding, n_threads);
                }
                break;
            case Encoding::C64:
                if constexpr (nt::complex<T>) {
                    using real_t = const T::value_type;
                    auto new_encoding = Encoding{Encoding::F64, encoding.clamp, encoding.endian_swap};
                    return encode(input.template as<real_t>(), output, new_encoding, n_threads);
                }
                break;
            case Encoding::UNKNOWN:
                break;
        }
        panic("{} cannot be encoded into {}", ns::stringify<T>(), encoding.dtype);
    }

    template<nt::numeric T>
    void decode(
        SpanContiguous<const std::byte, 1> input,
        Encoding encoding,
        const Span<T, 4>& output,
        i32 n_threads
    ) {
        const i64 n_elements = output.ssize();
        const i64 n_encoded_bytes = encoding.encoded_size(n_elements);
        check(n_encoded_bytes <= input.ssize(), "The encoded array is not big enough to contain the input array");
        n_threads = actual_n_threads_(n_elements, n_threads);

        switch (encoding.dtype) {
            case Encoding::U4:
                if constexpr (nt::scalar<T>) {
                    check(output.are_contiguous() and is_even(output.shape()[3]),
                          "u4 encoding requires the input array to be contiguous and have even rows");
                    return decode_4bits_(input, output.as_contiguous_1d(), n_threads);
                }
                break;
            case Encoding::I8:
                if constexpr (nt::scalar<T>)
                    return decode_4d_<i8>(input, output, encoding.clamp, encoding.endian_swap, n_threads);
                break;
            case Encoding::U8:
                if constexpr (nt::scalar<T>)
                    return decode_4d_<u8>(input, output, encoding.clamp, encoding.endian_swap, n_threads);
                break;
            case Encoding::I16:
                if constexpr (nt::scalar<T>)
                    return decode_4d_<i16>(input, output, encoding.clamp, encoding.endian_swap, n_threads);
                break;
            case Encoding::U16:
                if constexpr (nt::scalar<T>)
                    return decode_4d_<u16>(input, output, encoding.clamp, encoding.endian_swap, n_threads);
                break;
            case Encoding::I32:
                if constexpr (nt::scalar<T>)
                    return decode_4d_<i32>(input, output, encoding.clamp, encoding.endian_swap, n_threads);
                break;
            case Encoding::U32:
                if constexpr (nt::scalar<T>)
                    return decode_4d_<u32>(input, output, encoding.clamp, encoding.endian_swap, n_threads);
                break;
            case Encoding::I64:
                if constexpr (nt::scalar<T>)
                    return decode_4d_<i64>(input, output, encoding.clamp, encoding.endian_swap, n_threads);
                break;
            case Encoding::U64:
                if constexpr (nt::scalar<T>)
                    return decode_4d_<u64>(input, output, encoding.clamp, encoding.endian_swap, n_threads);
                break;
            case Encoding::F16:
                if constexpr (nt::scalar<T>)
                    return decode_4d_<f16>(input, output, encoding.clamp, encoding.endian_swap, n_threads);
                break;
            case Encoding::F32:
                if constexpr (nt::scalar<T>)
                    return decode_4d_<f32>(input, output, encoding.clamp, encoding.endian_swap, n_threads);
                break;
            case Encoding::F64:
                if constexpr (nt::scalar<T>)
                    return decode_4d_<f64>(input, output, encoding.clamp, encoding.endian_swap, n_threads);
                break;
            case Encoding::CI16:
                if constexpr (nt::complex<T>) {
                    auto new_encoding = Encoding{Encoding::I16, encoding.clamp, encoding.endian_swap};
                    return decode(input, new_encoding, output.template as<typename T::value_type>(), n_threads);
                }
                break;
            case Encoding::C16:
                if constexpr (nt::complex<T>) {
                    auto new_encoding = Encoding{Encoding::F16, encoding.clamp, encoding.endian_swap};
                    return decode(input, new_encoding, output.template as<typename T::value_type>(), n_threads);
                }
                break;
            case Encoding::C32:
                if constexpr (nt::complex<T>) {
                    auto new_encoding = Encoding{Encoding::F32, encoding.clamp, encoding.endian_swap};
                    return decode(input, new_encoding, output.template as<typename T::value_type>(), n_threads);
                }
                break;
            case Encoding::C64:
                if constexpr (nt::complex<T>) {
                    auto new_encoding = Encoding{Encoding::F64, encoding.clamp, encoding.endian_swap};
                    return decode(input, new_encoding, output.template as<typename T::value_type>(), n_threads);
                }
                break;
            case Encoding::UNKNOWN:
                break;
        }
        panic("{} cannot be decoded into {}", encoding.dtype, ns::stringify<T>());
    }

    #define NOA_IO_ENCODE_(T)                                                                       \
    template void encode<T>(const Span<const T, 4>&, SpanContiguous<std::byte, 1>, Encoding, i32);  \
    template void decode<T>(SpanContiguous<const std::byte, 1>, Encoding, const Span<T, 4>&, i32)

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
    auto operator<<(std::ostream& os, Encoding::Type dtype) -> std::ostream& {
        switch (dtype) {
            case Encoding::UNKNOWN: return os << "<unknown>";
            case Encoding::U4: return os << "u4";
            case Encoding::I8: return os << "i8";
            case Encoding::U8: return os << "u8";
            case Encoding::I16: return os << "i16";
            case Encoding::U16: return os << "u16";
            case Encoding::I32: return os << "i32";
            case Encoding::U32: return os << "u32";
            case Encoding::I64: return os << "i64";
            case Encoding::U64: return os << "u64";
            case Encoding::F16: return os << "f16";
            case Encoding::F32: return os << "f32";
            case Encoding::F64: return os << "f64";
            case Encoding::CI16: return os << "ci16";
            case Encoding::C16: return os << "c16";
            case Encoding::C32: return os << "c32";
            case Encoding::C64: return os << "c64";
        }
        return os; // unreachable
    }

    auto operator<<(std::ostream& os, Encoding encoding) -> std::ostream& {
        return os
               << "Encoding{.dtype=" << encoding.dtype
               <<         " .clamp=" << encoding.clamp
               <<         " .endian_swap=" << encoding.endian_swap
               << "}";
    }
}
