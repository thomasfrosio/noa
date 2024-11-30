#pragma once

#include "noa/core/Traits.hpp"
#include "noa/core/io/Encoding.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/types/Span.hpp"

namespace noa::traits {
    template<typename T, typename U = f32>
    concept image_encorder = requires(
        T t,
        SpanContiguous<std::byte> file,
        const Span<U, 4>& span,
        const Shape<i64, 4>& shape,
        const Vec<f64, 3>& spacing,
        noa::io::Encoding::Type dtype,
        Vec<i64, 2>& offset,
        bool clamp,
        i32 n_threads,
        std::string_view extension
    ) {
        { t.read_header(file.as_const()) } -> std::same_as<Tuple<Shape<i64, 4>, Vec<f64, 3>, noa::io::Encoding::Type>>;
        { t.write_header(file, shape, spacing, dtype) } -> std::same_as<void>;
        { t.close() } -> std::same_as<void>;
        { t.decode(file, span, offset, clamp, n_threads) } -> std::same_as<void>;
        { t.encode(file, span.as_const(), offset, clamp, n_threads) } -> std::same_as<void>;
        { T::is_supported_extension(extension) } noexcept -> std::same_as<bool>;
        { T::required_file_size(shape, dtype) } noexcept -> std::same_as<i64>;
        { T::closest_supported_dtype(dtype) } noexcept -> std::same_as<noa::io::Encoding::Type>;
    };
}

// TODO Add TIFF, JPEG, PNG and EER.

namespace noa::io {
    /// MRC file encoder and decoder.
    /// \details Limitations/notes:
    ///     - Modifying an existing file is not supported. It is either reading an existing file or writing a new one.
    ///     - The spacing can be set to 0.
    ///     - Reading files with a different endianness is supported and should be transparent to the user.
    ///     - The header, and thus the shape, is set when opening the file.
    ///     - The extended header, the origin (xorg, yorg, zorg), nversion, min/max/mean/std and other parts of the
    ///       header are ignored. When writing a new file, these are set to 0 or to the expected default value.
    ///     - The map ordering should be mapc=1, mapr=2 and maps=3. Anything else is not supported, and an exception
    ///       is thrown when opening a file with a different ordering.
    ///
    /// \see https://bio3d.colorado.edu/imod/doc/mrc_format.txt or
    ///      https://www.ccpem.ac.uk/mrc_format/mrc2014.php
    struct EncoderMrc {
        auto read_header(
            SpanContiguous<const std::byte> file
        ) -> Tuple<Shape<i64, 4>, Vec<f64, 3>, Encoding::Type>;

        void write_header(
            SpanContiguous<std::byte> file,
            const Shape<i64, 4>& shape,
            const Vec<f64, 3>& spacing,
            Encoding::Type dtype
        );

        void close() const {
            // We write the header directly when opening the file, so we have nothing to do here.
        }

        template<typename T>
        void decode(
            SpanContiguous<const std::byte> file,
            const Span<T, 4>& output,
            const Vec<i64, 2>& bd_offset,
            bool clamp,
            i32 n_threads
        ) {
            const auto encoding = Encoding{
                .dtype = m_dtype,
                .clamp = clamp,
                .endian_swap = m_is_endian_swapped
            };
            const i64 byte_offset =
                HEADER_SIZE + m_extended_bytes_nb +
                encoding.encoded_size(ni::offset_at(m_shape.strides(), bd_offset));

            noa::io::decode(file.subregion(ni::Slice{byte_offset, file.ssize()}), encoding, output, n_threads);
        }

        template<typename T>
        void encode(
            SpanContiguous<std::byte> file,
            const Span<const T, 4>& input,
            const Vec<i64, 2>& bd_offset,
            bool clamp,
            i32 n_threads
        ) {
            const auto encoding = Encoding{
                .dtype = m_dtype,
                .clamp = clamp,
                .endian_swap = m_is_endian_swapped
            };
            const i64 byte_offset =
                HEADER_SIZE + m_extended_bytes_nb +
                encoding.encoded_size(ni::offset_at(m_shape.strides(), bd_offset));

            noa::io::encode(input, file.subregion(ni::Slice{byte_offset, file.ssize()}), encoding, n_threads);
        }

        static auto is_supported_extension(std::string_view extension) noexcept -> bool {
            using namespace std::string_view_literals;
            return extension == ".mrc"sv or extension == ".mrcs"sv;
        }

        static auto required_file_size(const Shape<i64, 4>& shape, Encoding::Type dtype) noexcept -> i64 {
            return HEADER_SIZE + Encoding::encoded_size(dtype, shape.n_elements());
        }

        static auto closest_supported_dtype(Encoding::Type dtype) noexcept -> Encoding::Type {
            switch (dtype) {
                case Encoding::I8:
                case Encoding::U8:
                case Encoding::I16:
                case Encoding::U16:
                case Encoding::F16:
                    return dtype;
                case Encoding::I32:
                case Encoding::U32:
                case Encoding::I64:
                case Encoding::U64:
                case Encoding::F32:
                case Encoding::F64:
                    return Encoding::F32;
                case Encoding::C16:
                case Encoding::C32:
                case Encoding::C64:
                    return Encoding::C32;
                default:
                    return Encoding::UNKNOWN;
            }
        }

    private:
        static constexpr i64 HEADER_SIZE = 1024;
        Shape<i64, 4> m_shape{}; // BDHW order
        Vec<f32, 3> m_spacing{}; // DHW order
        Encoding::Type m_dtype{};
        i32 m_extended_bytes_nb{};
        bool m_is_endian_swapped{false};
    };
}
