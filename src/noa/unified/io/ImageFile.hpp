#pragma once

#include <memory>
#include <utility>
#include <variant>
#include "noa/core/io/IO.hpp"
#include "noa/core/io/MRCFile.hpp"
#include "noa/core/io/TIFFFile.hpp"
#include "noa/core/io/Stats.hpp"
#include "noa/unified/Array.hpp"

// TODO(TF) Add JPEG, PNG and EER file format. The doc might need to be updated for the read/write functions.
// TODO(TF) TIFF format is not tested! Add tests!

namespace noa::io {
    /// Manipulate an image file.
    /// \details
    /// - \b Format: ImageFile is an interface to manipulate image file formats. Only MRC and TIFF files are
    ///   currently supported. TIFF files only support (stack of) 2d image(s).\n
    /// - \b Shape: The library, including this class, uses the batch-depth-height-width (BDHW) order.
    ///   As such, a stack of 2d images is \c {batch,1,height,width} and a 3d volume is \c {1,depth,height,width}.\n
    /// - \b Ordering: Data is always writen into the file in the BDHW rightmost order (i.e. row-major), which
    ///   may involve a reordering of the input array. For reading operations, the order in the file is preserved
    ///   and encoded in the strides of the output array.\n
    class ImageFile {
    public:
#ifdef NOA_ENABLE_TIFF
        using variant_type = std::variant<MrcFile, TiffFile>;
#else
        using variant_type = std::variant<MrcFile>;
#endif
        enum class Format { MRC, TIFF };
        using enum Format;

    public:
        /// Creates an empty instance. Use open(), otherwise all other function calls will be ignored.
        ImageFile() = default;

        /// Opens the image file.
        /// \param filename Path of the image file to open. The file format is deduced from the extension.
        /// \param mode     Open mode. See open() for more details.
        ImageFile(const Path& filename, Open mode)
                : m_file_format(extension2format_(filename.extension())) {
            reset_header_(m_file_format);
            open_(filename, mode);
        }

        // No copies.
        ImageFile(const ImageFile&) = delete;
        ImageFile& operator=(const ImageFile&) = delete;

        // Move.
        ImageFile(ImageFile&&) = default;
        ImageFile& operator=(ImageFile&&) = default;

        ~ImageFile() noexcept(false) {
            try {
                close_();
            } catch (...) {
                if (std::uncaught_exceptions() == 0)
                    std::rethrow_exception(std::current_exception());
            }
        }

    public:
        /// (Re)Opens the file.
        /// \param filename     Path of the file to open.
        /// \param mode         Open mode. Should be one of the following combination:
        ///                     (internally, binary is always on, append and at_the_end are always off)
        ///                     1) read:                    File should exists.
        ///                     2) read-write:              File should exists. Backup copy.
        ///                     3) write or write-truncate: Overwrite the file. Backup move.
        ///                     4) read-write-truncate:     Overwrite the file. Backup move.
        ///
        /// \throws Exception   In any of the following cases:
        ///         - If the file does not exist and \p mode is read or read-write.
        ///         - If the permissions do not match the \p mode.
        ///         - If the file format is not recognized, nor supported.
        ///         - If failed to close the current file before (if any).
        ///         - If an underlying OS error was raised.
        ///         - If the file metadata could not be read.
        ///
        /// \note TIFF files don't support modes 2 and 4.
        void open(const Path& filename, Open mode) {
            close();

            // Try to not reset the underlying file.
            const auto old_format = m_file_format;
            m_file_format = extension2format_(filename.extension());
            if (is_empty() or m_file_format != old_format)
                reset_header_(m_file_format);

            open_(filename, mode);
        }

        /// Closes the file. In writing mode and for some file format,
        /// this can trigger a write operation to save the buffered data.
        void close() {
            close_();
        }

        [[nodiscard]] bool is_open() const noexcept { return m_is_open; }
        explicit operator bool() const noexcept { return m_is_open; }
        bool is_empty() const { return m_file.valueless_by_exception(); }

        /// Returns the file format.
        [[nodiscard]] Format format() const noexcept { return m_file_format; }
        [[nodiscard]] bool is_mrc() const noexcept { return m_file_format == Format::MRC; }
        [[nodiscard]] bool is_tiff() const noexcept { return m_file_format == Format::TIFF; }

        /// Gets the path.
        [[nodiscard]] auto filename() const -> const Path& {
            check(is_open(), "The file should be opened");
            return std::visit([](auto&& file) -> const Path& { return file.filename(); }, m_file);
        }

        /// Returns a (brief) description of the file data.
        [[nodiscard]] auto info(bool brief) const -> std::string {
            check(is_open(), "The file should be opened");
            return std::visit([brief](auto&& file) { return file.info_string(brief); }, m_file);
        }

        /// Gets the shape of the data.
        [[nodiscard]] auto shape() const -> Shape4<i64> {
            if (is_empty())
                return {};
            return std::visit([](auto&& file) { return file.shape(); }, m_file);
        }

        /// Sets the shape of the data. In pure read mode, this is usually not allowed
        /// and is likely to throw an exception because the file cannot be modified.
        void set_shape(const Shape4<i64>& shape) {
            check(is_open(), "The file should be opened");
            return std::visit([&](auto&& file) { return file.set_shape(shape); }, m_file);
        }

        /// Gets the pixel size of the data (the batch dimension does not have a pixel size).
        [[nodiscard]] auto pixel_size() const -> Vec3<f64> {
            if (is_empty())
                return {};
            return std::visit([](auto&& file) { return file.pixel_size(); }, m_file);
        }

        /// Sets the pixel size of the data. In pure read mode, this is usually not allowed
        /// and is likely to throw an exception. Passing 0 for one or more dimensions is allowed.
        void set_pixel_size(const Vec3<f64>& pixel_size) {
            check(is_open(), "The file should be opened");
            return std::visit([&](auto&& file) { return file.set_pixel_size(pixel_size); }, m_file);
        }

        /// Gets the encoding format of the serialized data.
        [[nodiscard]] auto encoding_format() const -> Encoding::Format {
            if (is_empty())
                return Encoding::UNKNOWN;
            return std::visit([](auto&& file) { return file.encoding_format(); }, m_file);
        }

        /// Sets the encoding format of the serialized data. This will affect all future writing operation.
        /// In read mode, this is usually not allowed and is likely to throw an exception.
        void set_encoding_format(Encoding::Format encoding_format) {
            check(is_open(), "The file should be opened");
            return std::visit([&](auto&& file) { return file.set_encoding_format(encoding_format); }, m_file);
        }

        /// Gets the statistics of the data.
        /// Some fields might be unset (one should use the has_*() methods of Stats before getting the values).
        [[nodiscard]] auto stats() const -> Stats<f64> {
            if (is_empty())
                return {};
            return std::visit([](auto&& file) { return file.stats(); }, m_file);
        }

        /// Sets the statistics of the data. Depending on the open mode and
        /// file format, this might have no effect on the file.
        void set_stats(const Stats<f64>& stats) {
            check(is_open(), "The file should be opened");
            return std::visit([&](auto&& file) { return file.set_stats(stats); }, m_file);
        }

    public: // Read
        /// Loads the entire file in \p output.
        /// \param[out] output  VArray, of any numeric type, where the deserialized values are saved.
        ///                     Should match the shape of the file.
        /// \param clamp        Whether the deserialized values should be clamped to fit the output type \p T.
        ///                     If false, out of range values are undefined.
        template<nt::writable_varray_decay_of_numeric Output>
        void read(Output&& output, bool clamp = true) {
            check(is_open(), "The file should be opened");
            using value_t = nt::value_type_t<Output>;

            if (output.is_dereferenceable()) {
                std::visit([&](auto&& file) { file.read_all(output.eval().span(), clamp); }, m_file);
            } else {
                auto tmp = Array<value_t>(output.shape());
                std::visit([&](auto&& file) { file.read_all(tmp.span(), clamp); }, m_file);
                std::move(tmp).to(std::forward<Output>(output));
            }
        }

        /// Returns the entire content of the file.
        /// \param option   Output options for the array where the deserialized values are saved.
        /// \param clamp    Whether the deserialized values should be clamped to fit the output type \p T.
        ///                 If false, out of range values are undefined.
        template<typename T>
        [[nodiscard]] auto read(ArrayOption option = {}, bool clamp = true) -> Array<T> {
            check(is_open(), "The file should be opened");

            if (option.is_dereferenceable()) {
                auto output = Array<T>(shape(), option);
                std::visit([&](auto&& file) { file.read_all(output.eval().span(), clamp); }, m_file);
                return output;
            } else {
                auto tmp = Array<T>(shape());
                std::visit([&](auto&& file) { file.read_all(tmp.span(), clamp); }, m_file);
                return std::move(tmp).to(option);
            }
        }

        /// Loads some 2D slices from the file.
        /// The file should describe a (stack of) 2D images(s), or a single volume.
        /// \tparam T           Any data type (integer, floating-point, complex).
        /// \param[out] output  VArray, of any numeric type, where the deserialized slice(s) are saved.
        ///                     Should describe a (stack of) 2D images(s), i.e. the depth should be 1.
        /// \param start        Index of the slice, in the file, where the deserialization starts.
        /// \param clamp        Whether the deserialized values should be clamped to fit the output type \p T.
        ///                     If false, out of range values are undefined.
        template<nt::writable_varray_decay_of_numeric Output>
        void read_slice(Output&& output, i64 start, bool clamp = true) {
            check(is_open(), "The file should be opened");
            using value_t = nt::value_type_t<Output>;

            if (output.is_dereferenceable()) {
                std::visit([&](auto&& file) { file.read_slice(output.eval().span(), start, clamp); }, m_file);
            } else {
                auto tmp = Array<value_t>(output.shape());
                std::visit([&](auto&& file) { file.read_slice(tmp.span(), start, clamp); }, m_file);
                return std::move(tmp).to(std::forward<Output>(output));
            }
        }

    public: // Write
        /// Saves \p input into the file.
        /// \details Writes to the file. If the shape of the file is set, \p input should have the same shape.
        ///          Otherwise, the shape of the file is set to the shape of \p input. If the data type of the
        ///          file is set, it should be compatible with the value type of \p input. Otherwise, the value
        ///          type of \p input (or the closest supported type) is set as the data type of the file.
        /// \param[in] input    VArray, of any numeric type, to serialize.
        /// \param clamp        Whether the input values should be clamped to fit the file data type.
        ///                     If false, out of range values are undefined.
        template<nt::readable_varray_of_numeric Input>
        void write(const Input& input, bool clamp = true) {
            check(is_open(), "The file should be opened");
            using value_t = nt::const_value_type_t<Input>;

            if (not input.is_dereferenceable()) {
                std::visit([&](auto&& file) { file.write_all(input.to_cpu().template span<value_t>(), clamp); }, m_file);
            } else {
                std::visit([&](auto&& file) { file.write_all(input.eval().template span<value_t>(), clamp); }, m_file);
            }
        }

        /// Saves some 2D slices into the file.
        /// \details Writes to the file, which should describe a (stack of) 2D slice(s), or a single volume.
        ///          If the shape of the file is set, \p input should have the same shape.
        ///          Otherwise, the shape of the file is set to the shape of \p input. If the data type of the
        ///          file is set, it should be compatible with the value type of \p input. Otherwise, the value
        ///          type of \p input (or the closest supported type) is set as the data type of the file.
        /// \param[in] input    VArray, of any numeric type, to serialize.
        ///                     Should describe a (stack of) 2D images(s), i.e. the depth should be 1.
        /// \param start        Index of the slice, in the file, where the serialization starts.
        /// \param clamp        Whether the input values should be clamped to fit the file data type.
        ///                     If false, out of range values are undefined.
        template<nt::readable_varray_of_numeric Input>
        void write_slice(const Input& input, i64 start, bool clamp = true) {
            check(is_open(), "The file should be opened");
            using value_t = nt::const_value_type_t<Input>;

            if (not input.is_dereferenceable()) {
                std::visit([&](auto&& file) { file.write_slice(input.to_cpu().template span<value_t>(), start, clamp); }, m_file);
            } else {
                std::visit([&](auto&& file) { file.write_slice(input.eval().template span<value_t>(), start, clamp); }, m_file);
            }
        }

    private:
        void reset_header_(Format new_format) {
            switch (new_format) {
                case Format::MRC:
                    m_file.emplace<MrcFile>();
                    break;
                case Format::TIFF:
                    #if NOA_ENABLE_TIFF
                    m_file.emplace<TiffFile>();
                    break;
                    #else
                    panic("TIFF files are not supported in this build. See CMake option NOA_ENABLE_TIFF");
                    #endif
            }
        }

        static Format extension2format_(const Path& extension) {
            if (MrcFile::is_supported_extension(extension.string()))
                return Format::MRC;
            #if NOA_ENABLE_TIFF
            else if (TiffFile::is_supported_extension(extension.string()))
                return Format::TIFF;
            #endif
            else
                panic("Image file extension \"{}\" not recognized", extension);
        }

        void open_(const Path& filename, Open mode) {
            check(not is_empty());
            std::visit([&](auto&& file) { file.open(filename, mode); }, m_file);
            m_is_open = true;
        }

        void close_() {
            if (is_empty())
                return;
            std::visit([](auto&& file) { file.close(); }, m_file);
            m_is_open = false;
        }

    private:
        variant_type m_file{};
        Format m_file_format{};
        bool m_is_open{};
    };
}

namespace noa::io {
    struct ReadOption {
        /// Whether to enforce the output array to be a stack of 2d images, instead of a single 3d volume.
        bool enforce_2d_stack{};

        /// Whether the deserialized values should be clamped to the output type range.
        bool clamp{true};
    };

    struct WriteOption {
        /// Encoding format used for the serialization.
        /// If Encoding::UNKNOWN, let the file decide the best format given the input value type,
        /// so that no truncation or loss of precision happens.
        Encoding::Format encoding_format{Encoding::UNKNOWN};

        /// Whether the input values should be clamped to the serialized values range.
        bool clamp{true};
    };

    /// Loads the file into a new array with a given type T.
    /// \return BDHW C-contiguous output array containing the whole data array of the file, and its pixel size.
    template<nt::numeric T>
    [[nodiscard]] auto read(
            const Path& filename,
            ReadOption read_option = {},
            ArrayOption array_option = {}
    ) -> Pair<Array<T>, Vec3<f64>> {
        auto file = ImageFile(filename, Open{.read=true});
        Array data = file.read<T>(array_option, read_option.clamp);
        Vec3<f64> pixel_size = file.pixel_size();
        const auto& shape = data.shape();
        if (read_option.enforce_2d_stack and (not shape.is_batched() and shape.height() > 1))
            data = std::move(data).reshape(shape.filter(1, 0, 2, 3)); // TODO Use to_batched()?
        return {data, pixel_size};
    }

    /// Loads the file data into a new array.
    /// Same as the overload above, but without loading the pixel size.
    template<typename T>
    [[nodiscard]] auto read_data(
            const Path& filename,
            ReadOption read_option = {},
            ArrayOption array_option = {}
    ) -> Array<T> {
        Array data = ImageFile(filename, Open{.read=true}).read<T>(array_option, read_option.clamp);
        const auto& shape = data.shape();
        if (read_option.enforce_2d_stack and (not shape.is_batched() and shape.height() > 1))
            data = std::move(data).reshape(shape.filter(1, 0, 2, 3)); // TODO Use to_batched()?
        return data;
    }

    /// Saves the input array into a new file.
    /// \param[in] input        Array to serialize.
    /// \param pixel_size       (D)HW pixel size of \p input.
    /// \param[in] filename     Path of the new file.
    /// \param write_option     Options.
    template<nt::varray_of_numeric Input, nt::vec_real_size<2, 3> PixelSize>
    void write(
            const Input& input,
            const PixelSize& pixel_size,
            const Path& filename,
            WriteOption write_option = {}
    ) {
        auto file = ImageFile(filename, Open{.write=true});
        if constexpr (nt::vec_of_size<PixelSize, 2>)
            file.set_pixel_size(pixel_size.push_front(1).template as<f64>());
        else
            file.set_pixel_size(pixel_size.template as<f64>());
        if (write_option.encoding_format != Encoding::UNKNOWN)
            file.set_encoding_format(write_option.encoding_format);
        file.write(input, write_option.clamp);
    }

    /// Saves the input array into a new file.
    /// Same as the above overload, but without setting a pixel size.
    template<nt::varray_of_numeric Input>
    void write(const Input& input, const Path& filename, WriteOption write_option = {}) {
        auto file = ImageFile(filename, Open{.write=true});
        if (write_option.encoding_format != Encoding::UNKNOWN)
            file.set_encoding_format(write_option.encoding_format);
        file.write(input);
    }
}

// Expose read/write to noa.
namespace noa {
    using noa::io::read;
    using noa::io::read_data;
    using noa::io::write;
}
