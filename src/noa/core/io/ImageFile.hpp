#pragma once

#include "noa/core/io/BinaryFile.hpp"
#include "noa/core/io/Encoding.hpp"
#include "noa/core/io/IO.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/types/Span.hpp"

// TODO Add JPEG, PNG and EER.
namespace noa::traits {
    /// Interface of a BasicImageFile encoder.
    template<typename T, typename U = f32>
    concept image_encorder = requires(
        T t,
        std::FILE* file,
        const Span<U, 4>& span,
        const Shape<i64, 4>& shape,
        const Vec<f64, 3>& spacing,
        noa::io::Encoding::Type dtype,
        noa::io::Compression compression,
        Vec<i64, 2>& offset,
        bool clamp,
        i32 n_threads,
        std::string_view extension
    ) {
        /// Name of the encoder, preferably in lower case.
        /// This is called by BasicImageFile::encoder_name().
        { T::name() } noexcept -> std::same_as<std::string_view>;

        /// Whether the encoder supports the file extension.
        /// This is used to decide which encoder to use when creating a new file.
        { T::is_supported_extension(extension) } noexcept -> std::same_as<bool>;

        /// Whether the encoder supports the file content.
        /// This is used to decide which encoder to use when reading from an existing file.
        /// If encoders need to query the stream, as they often do, they must return the stream to its original position.
        { T::is_supported_stream(file) } noexcept -> std::same_as<bool>;

        /// Queries the file size of the new file. The returned size maps to BinaryFile::Parameters::new_size.
        /// When creating a new file, BasicImageFile asks the encoder the desired size, in bytes, that the file stream
        /// should be resized to. When write_header is called, the provided file stream will have that exact size.
        /// Some encoders may return -1, indicating that they will resize the stream on-the-fly when opening the file
        /// (i.e. during write_header) or when writing to the file.
        { T::required_file_size(shape, dtype) } noexcept -> std::same_as<i64>;

        /// Returns the closest supported data-type.
        /// This may allow users to better prepare for encoding files before even creating a new file.
        { T::closest_supported_dtype(dtype) } noexcept -> std::same_as<noa::io::Encoding::Type>;

        /// Reads and extras the image file header from the opened stream.
        { t.read_header(file) } -> std::same_as<Tuple<Shape<i64, 4>, Vec<f64, 3>, noa::io::Encoding::Type, noa::io::Compression>>;

        /// Sets the metadata of the new file.
        /// The file doesn't have to be created at this point, this is just to initialize the encoder with user data.
        /// Encoders are allowed to change the data-type and compression if the provided ones are not supported,
        /// so the actual data-type and compression scheme is returned.
        { t.write_header(file, shape, spacing, dtype, compression) } -> std::same_as<Tuple<noa::io::Encoding::Type, noa::io::Compression>>;

        /// Closes the encoder. After that point, encoders may be reset by calling {read|write}_header again.
        /// Some encoders may do nothing here, some may need to clear some private data, some may need to write
        /// some metadata in the file (in which case they would need to store the file stream from {read|write}_header).
        /// In any case, this function is not supposed to close the file stream.
        { t.close() } -> std::same_as<void>;

        /// Decodes or encodes some data from the opened file.
        { t.decode(file, span, offset, clamp, n_threads) } -> std::same_as<void>;
        { t.encode(file, span.as_const(), offset, clamp, n_threads) } -> std::same_as<void>;
    };

    template<typename T>
    concept image_encorder_supported_value_type =
        nt::any_of<T, i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, c16, c32, c64>;
}

namespace noa::io {
    /// Image file.
    /// \warning This type is not thread-safe. It should be owned and used by a single thread.
    template<nt::image_encorder... Encoders>
    class BasicImageFile {
    public:
        using encoders_type = Tuple<Encoders...>;
        static constexpr size_t N_ENCODERS = sizeof...(Encoders);

        struct Header {
            /// BDHW shape of the (new) file.
            Shape<i64, 4> shape{};

            /// DHW spacing (in Angstrom/pix) of the (new) file.
            Vec<f64, 3> spacing{};

            /// Desired data-type of the (new) file.
            /// Note that encoders are allowed to select a different data-type
            /// (usually the closest related) if this one is not supported.
            Encoding::Type dtype{};

            /// Compression scheme of the (new) file.
            /// Note that encoders are allowed to select a different compression scheme
            /// (a similar scheme or Compression::NONE) if this one is not supported.
            Compression compression{};
        };

    public: // static functions
        [[nodiscard]] static constexpr auto is_supported_extension(std::string_view extension) noexcept -> bool {
            return [&extension]<size_t... Is>(std::index_sequence<Is...>) {
                return ((is_supported_extension_<Is>(extension)) or ...);
            }(std::make_index_sequence<N_ENCODERS>{});
        }

        [[nodiscard]] static auto closest_supported_dtype(
            std::string_view extension,
            Encoding::Type dtype
        ) noexcept -> Encoding::Type {
            [&extension, &dtype]<size_t... Is>(std::index_sequence<Is...>) {
                return (set_closest_dtype_<Is>(extension, dtype) or ...);
            }(std::make_index_sequence<N_ENCODERS>{});
            return dtype;
        }

        template<nt::numeric T>
        [[nodiscard]] static auto closest_supported_dtype(std::string_view extension) noexcept -> Encoding::Type {
            return closest_supported_dtype(extension, Encoding::to_dtype<T>());
        }

    public: // RAII
        BasicImageFile() = default;
        BasicImageFile(const Path& path, Open mode, Header new_header = {}) { open(path, mode, new_header); }

        BasicImageFile(const BasicImageFile&) noexcept = delete;
        BasicImageFile& operator=(const BasicImageFile&) noexcept = delete;
        BasicImageFile(BasicImageFile&&) noexcept = default;
        BasicImageFile& operator=(BasicImageFile&&) noexcept = default;

        ~BasicImageFile() noexcept(false) {
            try {
                close();
            } catch (...) {
                if (not std::uncaught_exceptions()) {
                    std::rethrow_exception(std::current_exception());
                }
            }
        }

    public: // member functions
        /// Opens and memory-maps the file.
        /// \param path File path.
        /// \param mode Open mode. Modifying an existing file is currently not supported.
        ///             Only the following modes are supported:
        ///             1) read:                  r     Readable.           The file should exist.          No backup.
        ///             2) read-write-truncate:   w+    Readable-Writable.  Create or overwrite the file.   Backup move.
        ///             3) write(-truncate):      w     Writable.           Create or overwrite the file.   Backup move.
        /// \param new_header   Header of the opened file. This is ignored in read-only (r) mode.
        ///
        /// \note Currently, we cannot both read and write from the same TIFF file, so while the mode w+ described
        ///       above will correctly create a new file, reading from this newly created file will raise an error.
        void open(const Path& path, Open mode, Header new_header = {}) {
            close();
            check(mode.is_valid() and not mode.append and
                  ((mode.read and not mode.write and not mode.truncate) or
                   (mode.read and mode.write and mode.truncate) or
                   (not mode.read and mode.write)),
                  "Invalid or unsupported open mode: {}", mode);

            if (mode.truncate or (mode.write and not mode.read)) {
                // If we create a new file, use the extension to decide which encoder to select.
                auto extension = path.extension().string();
                const bool has_been_initialized = [&extension, this]<size_t... Is>(std::index_sequence<Is...>) {
                    return (this->initialize_encoder_with_extension_<Is>(extension) or ...);
                }(std::make_index_sequence<N_ENCODERS>{});
                check(has_been_initialized, "Extension \"{}\" is not supported by any encoder", extension);

                // Ask the encoder if we should resize the file. If this returns -1, the file will not be resized,
                // and we expect the encoder to resize it later during the write_(slice|all) operations.
                const i64 new_size = std::visit([&](auto&& f) {
                    return f.required_file_size(new_header.shape, new_header.dtype);
                }, m_encoders);

                // Encoders may need to read the file stream, even in writing mode, so always open in w+ mode.
                m_file.open(path, Open::from_stdio("w+"), {.new_size = new_size});

                // Set the header.
                check(not new_header.shape.is_empty(),
                      "The data shape should be non-zero positive, but got new_header.shape={}", new_header.shape);
                check(all(new_header.spacing >= 0),
                      "The data spacing should be positive, but got new_header.spacing={}", new_header.spacing);
                m_header = new_header;
                std::visit([this](auto& f) {
                    // Encoders are allowed to change the data-type and compression scheme,
                    // so we need to update these in case the user queries the header.
                    auto&& [actual_dtype, actual_compression] = f.write_header(
                        m_file.stream(), m_header.shape, m_header.spacing, m_header.dtype, m_header.compression);
                    m_header.dtype = actual_dtype;
                    m_header.compression = actual_compression;
                }, m_encoders);
            } else {
                // In read-only mode, ask if any encoder recognizes the file.
                m_file.open(path, mode);
                const bool has_been_initialized = [this]<size_t... Is>(std::index_sequence<Is...>) {
                    return (initialize_encoder_with_stream_<Is>(m_file.stream()) or ...);
                }(std::make_index_sequence<N_ENCODERS>{});
                check(has_been_initialized, "{} is not supported by any encoder", path);

                std::visit([this](auto& f) {
                    auto&& [shape, spacing, dtype, compression] = f.read_header(m_file.stream());
                    m_header.shape = shape;
                    m_header.spacing = spacing;
                    m_header.dtype = dtype;
                    m_header.compression = compression;
                }, m_encoders);
            }
        }

        void close() {
            if (is_open()) {
                std::visit([](auto& f) { f.close(); }, m_encoders);
                m_file.close();
            }
        }

        [[nodiscard]] auto is_open() const noexcept -> bool { return m_file.is_open(); }
        [[nodiscard]] auto path() const noexcept -> const Path& { return m_file.path(); }

        [[nodiscard]] auto header() const noexcept -> const Header& { return m_header; }
        [[nodiscard]] auto shape() const noexcept -> const Shape<i64, 4>& { return m_header.shape; }
        [[nodiscard]] auto spacing() const noexcept -> const Vec<f64, 3>& { return m_header.spacing; }
        [[nodiscard]] auto dtype() const noexcept -> const Encoding::Type& { return m_header.dtype; }
        [[nodiscard]] auto compression() const noexcept -> const Compression& { return m_header.compression; }
        [[nodiscard]] auto is_compressed() const noexcept -> bool { return compression() != Compression::NONE; }

        template<typename T>
        [[nodiscard]] auto closest_supported_dtype() const noexcept -> Encoding::Type {
            return std::visit([this](const auto& f) { return f.template closest_supported_dtype<T>(); }, m_encoders);
        }

        /// Returns the encoder name.
        [[nodiscard]] auto encoder_name() const noexcept -> std::string_view {
            return std::visit([this](const auto& f) { return f.name(); }, m_encoders);
        }

        struct Parameters {
            /// Batch and depth offset of the slice(s) within the file.
            /// When reading/writing the whole file, this should be left to zero.
            Vec<i64, 2> bd_offset{};

            /// Whether the values should be clamped to the destination type.
            bool clamp{};

            /// Number of threads to read/write and decode/encode the data.
            /// Multithreading seems mostly relevant when the encoded data is compressed.
            i32 n_threads{1};
        };

        /// Reads one or multiple consecutive 2d slices from a file describing a stack of 2d images or 3d volumes.
        template<nt::image_encorder_supported_value_type T, StridesTraits S>
        void read_slice(const Span<T, 4, i64, S>& output, Parameters parameters) {
            check(is_open(), "The file should be open");

            check(noa::all(parameters.bd_offset >= 0),
                  "Batch-depth offset should be positive, but got bd_offset={}",
                  parameters.bd_offset);

            check(shape()[0] >= parameters.bd_offset[0] + output.shape()[0] and
                  shape()[1] >= parameters.bd_offset[1] + output.shape()[1],
                  "File: {}. Batch-depth shapes do not match, got bd_offset={}, output:shape={} and file:shape={}",
                  path(), parameters.bd_offset, output.shape().filter(0, 1), shape().filter(0, 1));

            check(shape()[2] == output.shape()[2] and shape()[3] == output.shape()[3],
                  "File: {}. Height-width shapes do not match, got output:shape={} and file:shape={}",
                  path(), output.shape().filter(2, 3), shape().filter(2, 3));

            // Read and decode.
            std::visit([&, this](auto& f) {
                f.decode(m_file.stream(), output.as_strided(),
                         parameters.bd_offset, parameters.clamp, parameters.n_threads);
            }, m_encoders);
        }

        /// Reads the whole data from the file.
        template<nt::image_encorder_supported_value_type T, StridesTraits S>
        void read_all(const Span<T, 4, i64, S>& output, Parameters parameters = {}) {
            check(is_open(), "The file should be open");
            check(noa::all(parameters.bd_offset == 0),
                  "Offsets should be 0, but got bd_offset={}",
                  parameters.bd_offset);
            check(all(output.shape() == shape()),
                  "File: {}. Shapes do not match, got output:shape={} and file:shape={}",
                  path(), output.shape(), shape());

            // Read and decode.
            std::visit([&, this](auto& f) {
                f.decode(m_file.stream(), output.as_strided(),
                         parameters.bd_offset, parameters.clamp, parameters.n_threads);
            }, m_encoders);
        }

        /// Writes one or multiple consecutive 2d slices into a file describing a stack of 2d images or 3d volumes.
        /// \note Multithreading is disabled for TIFF files. Parameters::n_threads is ignored.
        /// \note TIFF files can only write slices sequentially, so an error will be thrown if inputs are written
        ///       in a different order.
        template<nt::image_encorder_supported_value_type T, StridesTraits S>
        void write_slice(const Span<const T, 4, i64, S>& input, Parameters parameters) {
            check(is_open(), "The file should be open");

            check(noa::all(parameters.bd_offset >= 0),
                  "Batch-depth offset should be positive, but got bd_offset={}",
                  parameters.bd_offset);

            check(shape()[0] >= parameters.bd_offset[0] + input.shape()[0] and
                  shape()[1] >= parameters.bd_offset[1] + input.shape()[1],
                  "File: {}. Batch-depth shapes do not match, got bd_offset={}, input:shape={} and file:shape={}",
                  path(), parameters.bd_offset, input.shape().filter(0, 1), shape().filter(0, 1));

            check(shape()[2] == input.shape()[2] and shape()[3] == input.shape()[3],
                  "File: {}. Height-width shapes do not match, got input:shape={} and file:shape={}",
                  path(), input.shape().filter(2, 3), shape().filter(2, 3));

            std::visit([&, this](auto& f) {
                f.encode(m_file.stream(), input.as_strided(),
                         parameters.bd_offset, parameters.clamp, parameters.n_threads);
            }, m_encoders);
        }

        /// Writes the data into the file.
        /// \note Multithreading is disabled for TIFF files. Parameters::n_threads is ignored.
        template<nt::image_encorder_supported_value_type T, StridesTraits S>
        void write_all(const Span<const T, 4, i64, S>& input, Parameters parameters = {}) {
            check(is_open(), "The file should be open");
            check(noa::all(parameters.bd_offset == 0),
                  "Offsets should be 0, but got bd_offset={}",
                  parameters.bd_offset);
            check(all(input.shape() == shape()),
                  "File: {}. Shapes do not match, got input:shape={} and file:shape={}",
                  path(), input.shape(), shape());

            std::visit([&, this](auto& f) {
                f.encode(m_file.stream(), input.as_strided(),
                         parameters.bd_offset, parameters.clamp, parameters.n_threads);
            }, m_encoders);
        }

    private:
        template<size_t I>
        [[nodiscard]] static auto is_supported_extension_(std::string_view extension) noexcept -> bool {
            using encoder_t = std::tuple_element_t<I, encoders_type>;
            return encoder_t::is_supported_extension(extension);
        }

        template<size_t I>
        [[nodiscard]] auto initialize_encoder_with_extension_(std::string_view extension) -> bool {
            if (is_supported_extension_<I>(extension)) {
                m_encoders.template emplace<I>();
                return true;
            }
            return false;
        }

        template<size_t I>
        [[nodiscard]] static auto is_supported_stream_(std::FILE* stream) noexcept -> bool {
            using encoder_t = std::tuple_element_t<I, encoders_type>;
            return encoder_t::is_supported_stream(stream);
        }

        template<size_t I>
        [[nodiscard]] auto initialize_encoder_with_stream_(std::FILE* stream) -> bool {
            if (is_supported_stream_<I>(stream)) {
                m_encoders.template emplace<I>();
                return true;
            }
            return false;
        }

        template<size_t I>
        [[nodiscard]] static auto set_closest_dtype_(std::string_view extension, Encoding::Type& dtype) -> bool {
            using encoder_t = std::tuple_element_t<I, encoders_type>;
            if (encoder_t::is_supported_extension(extension)) {
                dtype = encoder_t::closest_supported_dtype(dtype);
                return true;
            }
            return false;
        }

    private:
        BinaryFile m_file{};
        std::variant<Encoders...> m_encoders{};
        Header m_header{};
    };
}

namespace noa::io {
    /// MRC file encoder and decoder.
    /// \details Limitations/notes:
    ///     - Modifying an existing file is not supported. It is either reading an existing file or writing a new one.
    ///     - The spacing can be set to 0, indicating the spacing is unset.
    ///     - The header, and thus the shape, is set when opening the file.
    ///     - The extended header, the origin (xorg, yorg, zorg), nversion, min/max/mean/std and other parts of the
    ///       header are ignored. When writing a new file, these are set to 0 or to the expected default value.
    ///     - The map ordering should be mapc=1, mapr=2 and maps=3. Anything else is not supported, and an exception
    ///       is thrown when opening a file with a different ordering.
    /// \see https://bio3d.colorado.edu/imod/doc/mrc_format.txt or
    ///      https://www.ccpem.ac.uk/mrc_format/mrc2014.php
    struct ImageFileEncoderMrc {
    public:
        static auto name() noexcept -> std::string_view { return "mrc"; }

        static auto is_supported_extension(std::string_view extension) noexcept -> bool {
            using namespace std::string_view_literals;
            return extension == ".mrc"sv or extension == ".mrcs"sv or extension == ".st"sv;
        }

        static auto is_supported_stream(std::FILE* file) noexcept -> bool {
            auto current_pos = std::ftell(file); // get current position
            check(current_pos != -1);

            // Check file size is larger than the MRC header.
            check(std::fseek(file, 0, SEEK_END) == 0);
            auto size = std::ftell(file);
            check(size != -1);
            if (size <= 1024)
                return false;

            // Read the stamp.
            char stamp[4];
            check(std::fseek(file, 212, SEEK_SET) == 0);
            check(std::fread(stamp, 1, 4, file) == 4);

            check(std::fseek(file, current_pos, SEEK_SET) == 0); // go back to the original position

            if (not (stamp[0] == 68 and stamp[1] == 65) and
                not (stamp[0] == 68 and stamp[1] == 68) and
                not (stamp[0] == 17 and stamp[1] == 17))
                return false;
            return stamp[2] == 0 and stamp[3] == 0;
        }

        static auto required_file_size(const Shape<i64, 4>& shape, Encoding::Type dtype) noexcept -> i64 {
            // The MRC encoder doesn't resize the stream, so let the BinaryFile do the resizing when opening the stream.
            return HEADER_SIZE + Encoding::encoded_size(dtype, shape.n_elements());
        }

        static auto closest_supported_dtype(Encoding::Type dtype) noexcept -> Encoding::Type {
            switch (dtype) {
                case Encoding::I8:
                case Encoding::U8:
                case Encoding::I16:
                case Encoding::U16:
                case Encoding::F16:
                case Encoding::U4:
                case Encoding::CI16:
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

    public:
        auto read_header(
            std::FILE* file
        ) -> Tuple<Shape<i64, 4>, Vec<f64, 3>, Encoding::Type, Compression>;

        auto write_header(
            std::FILE* file,
            const Shape<i64, 4>& shape,
            const Vec<f64, 3>& spacing,
            Encoding::Type dtype,
            Compression compression
        ) -> Tuple<Encoding::Type, Compression>;

        void close() const {
            // We write the header directly when opening the file, so we have nothing to do here.
        }

        template<typename T>
        void decode(
            std::FILE* file,
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

            check(std::fseek(file, byte_offset, SEEK_SET) == 0,
                  "Failed to seek at bd_offset={} (bytes={}). {}",
                  bd_offset, byte_offset, std::strerror(errno));
            noa::io::decode(file, encoding, output, n_threads);
        }

        template<typename T>
        void encode(
            std::FILE* file,
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

            check(std::fseek(file, byte_offset, SEEK_SET) == 0,
                  "Failed to seek at bd_offset={} (bytes={}). {}",
                  bd_offset, byte_offset, std::strerror(errno));
            noa::io::encode(input, file, encoding, n_threads);
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

#ifdef NOA_ENABLE_TIFF
namespace noa::io {
    /// Simple tiff file encoder and decoder.
    /// \details Limitations/notes:
    ///     - Only bi-level or grayscale images are supported.
    ///     - Only strips (no tiles) and contiguous samples.
    ///     - Only 2d images are supported.
    ///     - Files are written (see the encode function) uncompressed and using a single thread.
    ///       On the other hand, reading compressed files is allowed, and TIFF directories can be
    ///       distributed amongst multiple threads.
    ///     - Slices should be written sequentially. While our API allows writing slices in any order,
    ///       libtiff doesn't support that, so we check for this and throw an error if slices are written
    ///       non-sequentially.
    /// \see https://download.osgeo.org/libtiff/doc/TIFF6.pdf
    ///      https://libtiff.gitlab.io/libtiff/index.html
    /// \usage
    ///     - In read mode, call read_header(), then decode(), then close().
    ///     See ImageFile for more details.
    struct ImageFileEncoderTiff {
    public:
        static auto name() noexcept -> std::string_view { return "tiff"; }

        static auto is_supported_extension(std::string_view extension) noexcept -> bool {
            using namespace std::string_view_literals;
            return extension == ".tif"sv or extension == ".tiff"sv;
        }

        static auto is_supported_stream(std::FILE* file) noexcept -> bool {
            u16 stamp[2];
            auto current_pos = std::ftell(file); // get current position
            check(current_pos != -1);
            check(std::fseek(file, 0, SEEK_SET) == 0); // go to the beginning
            check(std::fread(stamp, 1, 4, file) == 4); // read stamp
            check(std::fseek(file, current_pos, SEEK_SET) == 0); // go back to the original position
            return (stamp[0] == 0x4949 or stamp[0] == 0x4d4d) and
                   (stamp[1] == 0x002a or stamp[1] == 0x2a00);
        }

        static auto required_file_size(const Shape<i64, 4>&, Encoding::Type) noexcept -> i64 {
            return -1; // the TIFF encoder will handle the stream resizing during writing operations
        }

        static auto closest_supported_dtype(Encoding::Type dtype) noexcept -> Encoding::Type {
            return dtype; // TIFF encoder supports all encoding types
        }

    public:
        auto read_header(
            std::FILE* file
        ) -> Tuple<Shape<i64, 4>, Vec<f64, 3>, Encoding::Type, Compression>;

        auto write_header(
            std::FILE* file,
            const Shape<i64, 4>& shape,
            const Vec<f64, 3>& spacing,
            Encoding::Type dtype,
            Compression compression
        ) -> Tuple<Encoding::Type, Compression>;

        void close() const;

        template<typename T>
        void decode(
            std::FILE* file,
            const Span<T, 4>& output,
            const Vec<i64, 2>& bd_offset,
            bool clamp,
            i32 n_threads
        );

        template<typename T>
        void encode(
            std::FILE* file,
            const Span<const T, 4>& input,
            const Vec<i64, 2>& bd_offset,
            bool clamp,
            i32 n_threads
        );

    public: // move-only - not thread-safe!
        ImageFileEncoderTiff() = default;
        ImageFileEncoderTiff(const ImageFileEncoderTiff& rhs) = delete;
        ImageFileEncoderTiff& operator=(const ImageFileEncoderTiff& rhs) = delete;
        ImageFileEncoderTiff(ImageFileEncoderTiff&& rhs) noexcept {
            m_shape = rhs.m_shape;
            m_spacing = rhs.m_spacing;
            m_dtype = rhs.m_dtype;
            m_handles = std::move(rhs.m_handles);
            current_directory = rhs.current_directory;
            m_is_write = rhs.m_is_write;
        }
        ImageFileEncoderTiff& operator=(ImageFileEncoderTiff&& rhs) noexcept {
            if (this != &rhs) {
                m_shape = rhs.m_shape;
                m_spacing = rhs.m_spacing;
                m_dtype = rhs.m_dtype;
                m_handles = std::move(rhs.m_handles);
                current_directory = rhs.current_directory;
                m_is_write = rhs.m_is_write;
            }
            return *this;
        }

    public:
        // For multithreading support, each thread is assigned its own TIFF handle.
        // Note: we cannot easily use a vector because since we pass a pointer to the tiff library,
        // we need to make sure these handles are not relocated.
        struct Handle {
            std::mutex* mutex{};
            std::FILE* file{};
            long offset{};
        };
        using handle_type = Pair<Handle, void*>;

    private:
        Shape<i64, 3> m_shape{}; // BHW order
        Vec<f32, 2> m_spacing{}; // HW order
        Encoding::Type m_dtype{};
        Compression m_compression{};

        // Create multiple TIFF handles from the same file stream to enable parallel decoding.
        std::mutex m_mutex;
        std::unique_ptr<handle_type[]> m_handles{};
        i32 current_directory{};
        bool m_is_write{};
    };
}
#endif

namespace noa::io {
    using ImageFile = BasicImageFile<
        ImageFileEncoderMrc
        #ifdef NOA_ENABLE_TIFF
        , ImageFileEncoderTiff
        #endif
    >;
}
