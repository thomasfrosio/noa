#pragma once

#include "noa/core/Config.hpp"

#ifdef NOA_IS_OFFLINE
#include <memory>
#include <utility>
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
        /// Creates an empty instance. Use open(), otherwise all other function calls will be ignored.
        ImageFile() = default;

        /// Opens the image file.
        /// \param filename Path of the image file to open. The file format is deduced from the extension.
        /// \param mode     Open mode. See open() for more details.
        ImageFile(const Path& filename, OpenMode mode)
                : m_header_format(extension2format_(filename.extension())) {
            allocate_header_(filename, m_header_format);
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
                if (m_header)
                    m_header->close();
            } catch (...) {
                if (std::uncaught_exceptions() == 0)
                    std::rethrow_exception(std::current_exception());
            }
        }

    public:
        /// (Re)Opens the file.
        /// \param filename     Path of the file to open.
        /// \param mode         Open mode bitflag. Should be one of the following combination:
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
        ///         - If failed to close the file before starting (if any).
        ///         - If an underlying OS error was raised.
        ///         - If the file header could not be read.
        ///
        /// \note TIFF files don't support modes 2 and 4.
        void open(const Path& filename, OpenMode mode) {
            close();
            const auto old_format = m_header_format;
            m_header_format = extension2format_(filename.extension());
            if (!m_header || m_header_format != old_format) {
                allocate_header_(filename, m_header_format);
            } else {
                m_header->reset();
            }
            open_(filename, mode);
        }

        /// Closes the file. In writing mode and for some file format,
        /// this can trigger a write operation to save the buffered data.
        void close() {
            close_();
        }

        /// Whether the file is open.
        [[nodiscard]] bool is_open() const noexcept { return m_is_open; }
        explicit operator bool() const noexcept { return m_is_open; }

        /// Returns the file format.
        [[nodiscard]] Format format() const noexcept { return m_header_format; }
        [[nodiscard]] bool is_mrc() const noexcept { return m_header_format == Format::MRC; }
        [[nodiscard]] bool is_tiff() const noexcept { return m_header_format == Format::TIFF; }
        [[nodiscard]] bool is_eer() const noexcept { return m_header_format == Format::EER; }
        [[nodiscard]] bool is_jpeg() const noexcept { return m_header_format == Format::JPEG; }
        [[nodiscard]] bool is_png() const noexcept { return m_header_format == Format::PNG; }

        /// Gets the path.
        [[nodiscard]] const Path& filename() const noexcept {
            check(is_open(), "The file should be opened");
            NOA_ASSERT(m_header);
            return m_header->filename();
        }

        /// Returns a (brief) description of the file data.
        [[nodiscard]] std::string info(bool brief) const noexcept {
            check(is_open(), "The file should be opened");
            NOA_ASSERT(m_header);
            return m_header->info_string(brief);
        }

        /// Gets the shape of the data.
        [[nodiscard]] Shape4<i64> shape() const noexcept { return m_header ? m_header->shape() : Shape4<i64>{}; }

        /// Sets the shape of the data. In pure read mode, this is usually not allowed
        /// and is likely to throw an exception because the file cannot be modified.
        void set_shape(const Shape4<i64>& shape) {
            check(is_open(), "The file should be opened");
            NOA_ASSERT(m_header);
            m_header->set_shape(shape);
        }

        /// Gets the pixel size of the data (the batch dimension does not have a pixel size).
        [[nodiscard]] Vec3<f32> pixel_size() const noexcept {
            return m_header ? m_header->pixel_size() : Vec3<f32>{};
        }

        /// Sets the pixel size of the data. In pure read mode, this is usually not allowed
        /// and is likely to throw an exception. Passing 0 for one or more dimensions is allowed.
        void set_pixel_size(const Vec3<f32>& pixel_size) {
            check(is_open(), "The file should be opened");
            NOA_ASSERT(m_header);
            m_header->set_pixel_size(pixel_size);
        }

        /// Gets the type of the serialized data.
        [[nodiscard]] DataType dtype() const noexcept {
            return m_header ? m_header->dtype() : DataType::UNKNOWN;
        }

        /// Sets the type of the serialized data. This will affect all future writing operation.
        /// In read mode, this is usually not allowed and is likely to throw an exception.
        void set_dtype(DataType data_type) {
            check(is_open(), "The file should be opened");
            NOA_ASSERT(m_header);
            m_header->set_dtype(data_type);
        }

        /// Gets the statistics of the data.
        /// Some fields might be unset (one should use the has_*() methods of Stats before getting the values).
        [[nodiscard]] Stats<f32> stats() const noexcept {
            return m_header ? m_header->stats() : Stats<f32>{};
        }

        /// Sets the statistics of the data. Depending on the open mode and
        /// file format, this might have no effect on the file.
        void stats(const Stats<f32>& stats) {
            check(is_open(), "The file should be opened");
            NOA_ASSERT(m_header);
            m_header->set_stats(stats);
        }

    public: // Read
        /// Loads the entire file in \p output.
        /// \param[out] output  VArray, of any numeric type, where the deserialized values are saved.
        ///                     Should match the shape of the file.
        /// \param clamp        Whether the deserialized values should be clamped to fit the output type \p T.
        ///                     If false, out of range values are undefined.
        template<typename Output, typename = std::enable_if_t<nt::is_varray_of_numeric_v<Output>>>
        void read(const Output& output, bool clamp = true) {
            check(is_open(), "The file should be opened");
            NOA_ASSERT(m_header);
            using value_t = nt::value_type_t<Output>;

            if (output.is_dereferenceable()) {
                m_header->read_all(output.eval().get(), output.strides(), output.shape(), io::dtype<value_t>(), clamp);
            } else {
                Array<value_t> tmp(output.shape());
                m_header->read_all(tmp.get(), tmp.strides(), tmp.shape(), io::dtype<value_t>(), clamp);
                tmp.to(output);
            }
        }

        /// Returns the entire content of the file.
        /// \param option   Output options for the array where the deserialized values are saved.
        /// \param clamp    Whether the deserialized values should be clamped to fit the output type \p T.
        ///                 If false, out of range values are undefined.
        template<typename T>
        Array<T> read(ArrayOption option = {}, bool clamp = true) {
            check(is_open(), "The file should be opened");
            NOA_ASSERT(m_header);

            if (option.is_dereferenceable()) {
                Array<T> out(shape(), option);
                m_header->read_all(out.eval().get(), out.strides(), out.shape(), io::dtype<T>(), clamp);
                return out;
            } else {
                Array<T> tmp(shape());
                m_header->read_all(tmp.get(), tmp.strides(), tmp.shape(), io::dtype<T>(), clamp);
                return tmp.to(option);
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
        template<typename Output, typename = std::enable_if_t<nt::is_varray_of_numeric_v<Output>>>
        void read_slice(const Output& output, i64 start, bool clamp = true) {
            check(is_open(), "The file should be opened");
            NOA_ASSERT(m_header);
            using value_t = nt::value_type_t<Output>;

            if (output.is_dereferenceable()) {
                m_header->read_slice(
                        output.eval().get(), output.strides(), output.shape(),
                        io::dtype<value_t>(), start, clamp);
            } else {
                Array<value_t> tmp(output.shape());
                m_header->read_slice(
                        tmp.get(), tmp.strides(), tmp.shape(),
                        io::dtype<value_t>(), start, clamp);
                tmp.to(output);
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
        template<typename Input, typename = std::enable_if_t<nt::is_varray_of_numeric_v<Input>>>
        void write(const Input& input, bool clamp = true) {
            check(is_open(), "The file should be opened");
            NOA_ASSERT(m_header);
            using value_t = nt::value_type_t<Input>;

            if (!input.is_dereferenceable()) {
                Array tmp = input.to_cpu();
                m_header->write_all(tmp.eval().get(), tmp.strides(), tmp.shape(), io::dtype<value_t>(), clamp);
            } else {
                m_header->write_all(input.eval().get(), input.strides(), input.shape(), io::dtype<value_t>(), clamp);
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
        template<typename Input, typename = std::enable_if_t<nt::is_varray_of_numeric_v<Input>>>
        void write_slice(const Input& input, i64 start, bool clamp = true) {
            check(is_open(), "The file should be opened");
            NOA_ASSERT(m_header);
            using value_t = nt::value_type_t<Input>;

            if (!input.is_dereferenceable()) {
                Array tmp = input.to_cpu();
                m_header->write_slice(
                        tmp.eval().get(), tmp.strides(), tmp.shape(),
                        io::dtype<value_t>(), start, clamp);
            } else {
                m_header->write_slice(
                        input.eval().get(), input.strides(), input.shape(),
                        io::dtype<value_t>(), start, clamp);
            }
        }

    private:
        std::unique_ptr<guts::ImageFile> m_header{};
        Format m_header_format{};
        bool m_is_open{};

    private:
        void allocate_header_(const Path& filename, Format new_format) {
            switch (new_format) {
                case Format::MRC:
                    m_header = std::make_unique<MRCFile>();
                    break;
                case Format::TIFF:
                    #if NOA_ENABLE_TIFF
                    m_header = std::make_unique<TIFFFile>();
                    break;
                    #else
                    panic("File {}: TIFF files are not supported in this build. See CMake option NOA_ENABLE_TIFF");
                    #endif
                default:
                    panic("File: {}. File format {} is not supported", filename, new_format);
            }
        }

        static Format extension2format_(const Path& extension) noexcept {
            if (extension == ".mrc" || extension == ".st" || extension == ".rec" || extension == ".mrcs")
                return Format::MRC;
            else if (extension == ".tif" || extension == ".tiff")
                return Format::TIFF;
            else
                return Format::UNKNOWN;
        }

        void open_(const Path& filename, OpenMode mode) {
            NOA_ASSERT(m_header);
            m_header->open(filename, mode);
            m_is_open = true;
        }

        void close_() {
            if (!m_header)
                return;
            m_header->close();
            m_is_open = false;
        }
    };
}

namespace noa::io {
    struct ReadOption {
        bool enforce_2d_stack{};
        bool clamp{true};
    };

    struct WriteOption {
        DataType data_type{DataType::UNKNOWN};
        bool clamp{true};
    };

    /// Loads the file into a new array.
    /// \tparam T               Any numeric type (integer, floating-point, complex).
    ///                         Values in the file are clamped to this type's range.
    /// \param[in] filename     Path of the file to read.
    /// \param enforce_2d_stack Whether to enforce the output array to be a stack of 2D images,
    ///                         instead of a single 3D volume. This is useful for files that are
    ///                         not encoded properly.
    /// \param option           Options for the output array.
    /// \return BDHW C-contiguous output array containing the whole data array of the file, and its pixel size.
    template<typename T>
    [[nodiscard]] auto read(
            const Path& filename,
            ReadOption read_option = {},
            ArrayOption array_option = {}
    ) -> Pair<Array<T>, Vec3<f32>> {
        auto file = ImageFile(filename, OpenMode{.read=true});
        Array data = file.read<T>(array_option, read_option.clamp);
        Vec3<f32> pixel_size = file.pixel_size();
        const auto& shape = data.shape();
        if (read_option.enforce_2d_stack and (not shape.is_batched() and shape.height() > 1))
            data = data.reshape(shape.filter(1, 0, 2, 3)); // TODO Use to_batched()?
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
        Array data = ImageFile(filename, OpenMode{.read=true}).read<T>(array_option, read_option.clamp);
        const auto& shape = data.shape();
        if (read_option.enforce_2d_stack and (not shape.is_batched() and shape.height() > 1))
            data = data.reshape(shape.filter(1, 0, 2, 3)); // TODO Use to_batched()?
        return data;
    }

    /// Saves the input array into a new file.
    /// \param[in] input    Array to serialize.
    /// \param pixel_size   (D)HW pixel size \p input.
    /// \param[in] filename Path of the new file.
    /// \param data_type    Data type of the file. If DataType::UNKNOWN, let the file format
    ///                     decide the best data type given the input value type, so that no
    ///                     truncation or loss of precision happens.
    template<typename Input, typename PixelSize>
    requires (nt::is_varray_of_numeric_v<Input> and
              (nt::is_vec_real_size_v<PixelSize, 2> or nt::is_vec_real_size_v<PixelSize, 3>))
    void write(
            const Input& input,
            const PixelSize& pixel_size,
            const Path& filename,
            WriteOption write_option = {}
    ) {
        auto file = ImageFile(filename, OpenMode{.write=true});
        if constexpr (nt::is_vec_of_size_v<PixelSize, 2>)
            file.set_pixel_size(pixel_size.push_front(1).template as<f32>());
        else
            file.set_pixel_size(pixel_size.template as<f32>());
        if (write_option.data_type != DataType::UNKNOWN)
            file.set_dtype(write_option.data_type);
        file.write(input, write_option.clamp);
    }

    /// Saves the input array into a new file.
    /// Same as the above overload, but without setting a pixel size.
    template<typename Input> requires nt::is_varray_of_numeric_v<Input>
    void write(const Input& input, const Path& filename, WriteOption write_option = {}) {
        auto file = ImageFile(filename, OpenMode{.write=true});
        if (write_option.data_type != DataType::UNKNOWN)
            file.set_dtype(write_option.data_type);
        file.write(input);
    }
}
#endif
