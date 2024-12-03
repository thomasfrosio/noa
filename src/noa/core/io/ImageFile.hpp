#pragma once

#include "noa/core/io/BinaryFile.hpp"
#include "noa/core/io/Encoders.hpp"
#include "noa/core/io/IO.hpp"
#include "noa/core/types/Shape.hpp"

namespace noa::io {
    template<nt::image_encorder... Encoders>
    class BasicImageFile {
    public:
        using encoders_type = Tuple<Encoders...>;
        static constexpr size_t N_ENCODERS = sizeof...(Encoders);

        struct Header {
            /// BDHW shape of the new file.
            Shape<i64, 4> shape{};

            /// DHW spacing (in Angstrom/pix) of the new file.
            Vec<f64, 3> spacing{};

            /// Desired data-type of the new file.
            /// Note that encoders are allowed to select a different data-type
            /// (usually the closest related) if the provided one is not supported.
            Encoding::Type dtype{};
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
        /// Opens and memory maps the file.
        /// \param path         File path.
        /// \param mode         Open mode. Only read, read|write|truncate, and write(|truncate) are supported.
        /// \param new_header   Header of the opened file. This is ignored in read-only mode.
        void open(const Path& path, Open mode, Header new_header = {}) {
            close();
            check(mode.is_valid() and not mode.append and
                  ((mode.read and not mode.write and not mode.truncate) or
                   (mode.read and mode.write and mode.truncate) or
                   (not mode.read and mode.write)),
                  "Invalid or unsupported open mode: {}", mode);

            // Select the first encoder supporting the extension.
            auto extension = path.extension().string();
            auto has_been_initialized = [&extension, this]<size_t... Is>(std::index_sequence<Is...>) {
                return (initialize_encoder_<Is>(extension) or ...);
            }(std::make_index_sequence<N_ENCODERS>{});
            check(has_been_initialized, "The file extension \"{}\" is not supported", extension);

            // Get the file size.
            i64 new_size{-1};
            if (mode.write) {
                new_size = std::visit([&](auto&& f) {
                    return f.required_file_size(new_header.shape, new_header.dtype);
                }, m_encoders);
            }

            // Open and mmap the file.
            m_file.open(path, mode, {.new_size = new_size});

            // Save the header.
            if (mode.read) {
                std::visit([this](auto& f) {
                     auto&& [shape, spacing, dtype] = f.read_header(m_file.stream());
                     m_header.shape = shape;
                     m_header.spacing = spacing;
                     m_header.dtype = dtype;
                 }, m_encoders);
            } else {
                check(not new_header.shape.is_empty(),
                      "The data shape should be non-zero positive, but got new_header.shape={}", new_header.shape);
                check(all(new_header.spacing >= 0),
                      "The data spacing should be positive, but got new_header.spacing={}", new_header.spacing);
                check(new_header.dtype != Encoding::Type::UNKNOWN, "The data type is not set");
                m_header = new_header;

                std::visit([this](auto& f) {
                    f.write_header(m_file.stream(), m_header.shape, m_header.spacing, m_header.dtype);
                }, m_encoders);
            }
        }

        void close() {
            if (is_open()) {
                std::visit([this](auto& f) { f.close(); }, m_encoders);
                m_file.close();
            }
        }

        [[nodiscard]] auto is_open() const noexcept -> bool { return m_file.is_open(); }
        [[nodiscard]] auto path() const noexcept -> const Path& { return m_file.path(); }

        [[nodiscard]] auto header() const noexcept -> const Header& { return m_header; }
        [[nodiscard]] auto shape() const noexcept -> const Shape<i64, 4>& { return m_header.shape; }
        [[nodiscard]] auto spacing() const noexcept -> const Vec<f64, 3>& { return m_header.spacing; }
        [[nodiscard]] auto dtype() const noexcept -> const Encoding::Type& { return m_header.dtype; }

        template<typename T>
        [[nodiscard]] auto closest_supported_dtype() const noexcept -> Encoding::Type {
            return std::visit([this](const auto& f) { return f.template closest_supported_dtype<T>(); }, m_encoders);
        }

        struct Parameters {
            /// Batch and depth offset of the slice(s) within the file.
            /// When reading/writing the whole file, this should be left to zero.
            Vec<i64, 2> bd_offset{};

            /// Whether the values should be clamped to the destination type.
            bool clamp{};

            /// Number of threads to read/write and decode/encode the data.
            i32 n_threads{1};
        };

        /// Reads one or multiple consecutive 2d slices from a file describing a stack of 2d images or 3d volumes.
        template<typename T, StridesTraits S>
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
        template<typename T, StridesTraits S>
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
        template<typename T, StridesTraits S>
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
        template<typename T, StridesTraits S>
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

        /// Advise the kernel regarding accesses.
        void optimize_for_sequential_access(i64 offset = 0, i64 size = -1) const {
            m_file.optimize_for_sequential_access(offset, size);
        }
        void optimize_for_random_access(i64 offset = 0, i64 size = -1) const {
            m_file.optimize_for_random_access(offset, size);
        }
        void optimize_for_no_access(i64 offset = 0, i64 size = -1) const {
            m_file.optimize_for_no_access(offset, size);
        }
        void optimize_for_normal_access(i64 offset = 0, i64 size = -1) const {
            m_file.optimize_for_normal_access(offset, size);
        }

    private:
        template<size_t I>
        [[nodiscard]] static auto is_supported_extension_(std::string_view extension) noexcept -> bool {
            using encoder_t = std::tuple_element_t<I, encoders_type>;
            return encoder_t::is_supported_extension(extension);
        }

        template<size_t I>
        [[nodiscard]] auto initialize_encoder_(std::string_view extension) -> bool {
            if (is_supported_extension_<I>(extension)) {
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

    using ImageFile = BasicImageFile<EncoderMrc>;
}
