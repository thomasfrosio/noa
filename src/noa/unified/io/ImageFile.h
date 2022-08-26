#pragma once

#include "noa/common/io/IO.h"
#include "noa/common/io/MRCFile.h"
#include "noa/common/io/TIFFFile.h"
#include "noa/common/io/Stats.h"
#include "noa/unified/Array.h"

// TODO(TF) Add JPEG, PNG and EER file format. The doc might need to be updated for the read/write functions.
// TODO(TF) TIFF format is not tested! Add tests!

namespace noa::io {
    /// Manipulate an image file.
    /// \details
    /// - \b Format: ImageFile is an unified interface to manipulate image file formats. Only MRC and TIFF files are
    ///   currently supported. TIFF files only support (stack of) 2D image(s).\n
    /// - \b Batches: The library, including this class, uses the batch-depth-height-width (BDHW) order.
    ///   As such, a stack of 2D images is {batch, 1, height, width} and a 3D volume is {1, depth, height, width}.\n
    /// - \b Ordering: Data is always writen into the file in the BDHW rightmost order (i.e. row-major). For reading
    ///   operations, it is either 1) a contiguous array is provided, we assume BDHW rightmost order and we might have
    ///   to reorder the data before returning it if the data in the file is not stored in BDHW rightmost order, or
    ///   2) a View (which contains strides) is provided and we will return the data in the same order as it is saved
    ///   in the file because the order is saved in the strides.\n
    class ImageFile {
    public:
        /// Creates an empty instance. Use open(), otherwise all other function calls will be ignored.
        ImageFile() = default;

        /// Opens the image file.
        /// \param filename Path of the image file to open. The file format is deduced from the extension.
        /// \param mode     Open mode. See open() for more details.
        ImageFile(const path_t& filename, open_mode_t mode);

        ~ImageFile() noexcept(false);
        ImageFile(const ImageFile&) = delete;
        ImageFile(ImageFile&&) = default;

    public:
        /// (Re)Opens the file.
        /// \param filename     Path of the file to open.
        /// \param mode         Open mode. Should be one of the following combination:
        ///                     1) READ:                File should exists.
        ///                     2) READ|WRITE:          File should exists.     Backup copy.
        ///                     3) WRITE or WRITE|TRUNC Overwrite the file.     Backup move.
        ///                     4) READ|WRITE|TRUNC:    Overwrite the file.     Backup move.
        ///
        /// \throws Exception   In any of the following cases:
        ///         - If the file does not exist and \p mode is io::READ or io::READ|io::WRITE.
        ///         - If the permissions do not match the \p mode.
        ///         - If the file format is not recognized, nor supported.
        ///         - If failed to close the file before starting (if any).
        ///         - If an underlying OS error was raised.
        ///         - If the file header could not be read.
        ///
        /// \note Internally, io::BINARY is always considered on.
        ///       On the other hand, io::APP and io::ATE are always considered off.
        ///       Changing any of these bits has no effect.
        /// \note TIFF files don't support modes 2 and 4.
        void open(const path_t& filename, uint mode);

        /// Closes the file. In writing mode and for some file format,
        /// there can be a write operation to save the buffered data.
        void close();

        /// Whether the file is open.
        [[nodiscard]] bool isOpen() const noexcept;
        explicit operator bool() const noexcept;

        /// Returns the file format.
        [[nodiscard]] Format format() const noexcept;
        [[nodiscard]] bool isMRC() const noexcept;
        [[nodiscard]] bool isTIFF() const noexcept;
        [[nodiscard]] bool isEER() const noexcept;
        [[nodiscard]] bool isJPEG() const noexcept;
        [[nodiscard]] bool isPNG() const noexcept;

        /// Gets the path.
        [[nodiscard]] path_t filename() const noexcept;

        /// Returns a (brief) description of the file data.
        [[nodiscard]] std::string info(bool brief) const noexcept;

        /// Gets the shape of the data.
        [[nodiscard]] size4_t shape() const noexcept;

        /// Sets the shape of the data. In pure read mode, this is usually not allowed
        /// and is likely to throw an exception because the file cannot be modified.
        void shape(size4_t shape);

        /// Gets the pixel size of the data (the batch dimension does not have a pixel size).
        [[nodiscard]] float3_t pixelSize() const noexcept;

        /// Sets the pixel size of the data. In pure read mode, this is usually not allowed
        /// and is likely to throw an exception. Passing 0 for one or more dimensions is allowed.
        void pixelSize(float3_t pixel_size);

        /// Gets the type of the serialized data.
        [[nodiscard]] DataType dtype() const noexcept;

        /// Sets the type of the serialized data. This will affect all future writing operation.
        /// In read mode, this is usually not allowed and is likely to throw an exception.
        void dtype(DataType data_type);

        /// Gets the statistics of the data.
        /// Some fields might be unset (one should use the has*() function of stats_t before getting the values).
        [[nodiscard]] stats_t stats() const noexcept;

        /// Sets the statistics of the data. Depending on the open mode and
        /// file format, this might have no effect on the file.
        void stats(stats_t stats);

    public: // Read
        /// Loads the entire file in \p output.
        /// \tparam T           Any data type (integer, floating-point, complex).
        /// \param[out] output  Output array where the deserialized values are saved.
        ///                     Should match the shape of the file.
        /// \param clamp        Whether the deserialized values should be clamped to fit the output type \p T.
        ///                     If false, out of range values are undefined.
        template<typename T>
        void read(const Array<T>& output, bool clamp = true);

        /// Returns the entire content of the file.
        /// \param option   Output options for the array where the deserialized values are saved.
        /// \param clamp    Whether the deserialized values should be clamped to fit the output type \p T.
        ///                 If false, out of range values are undefined.
        template<typename T>
        Array<T> read(ArrayOption option = {}, bool clamp = true);

        /// Loads the entire file in \p output.
        /// \tparam T           Any data type (integer, floating-point, complex).
        /// \param[out] output  View of the output array where the deserialized values are saved.
        ///                     Should match the shape of the file and be dereferenceable.
        /// \param clamp        Whether the deserialized values should be clamped to fit the output type \p T.
        ///                     If false, out of range values are undefined.
        template<typename T, typename I>
        void read(const View<T, I>& output, bool clamp = true);

        /// Loads some 2D slices from the file.
        /// The file should describe a (stack of) 2D images(s), or a single volume.
        /// \tparam T           Any data type (integer, floating-point, complex).
        /// \param[out] output  View of the output BDHW array where the deserialized slice(s) are saved.
        ///                     Should describe a (stack of) 2D images(s), i.e. the depth should be 1.
        /// \param start        Index of the slice, in the file, where the deserialization starts.
        /// \param clamp        Whether the deserialized values should be clamped to fit the output type \p T.
        ///                     If false, out of range values are undefined.
        template<typename T>
        void readSlice(const Array<T>& output, size_t start, bool clamp = true);

        /// Loads some 2D slices from the file.
        /// The file should describe a (stack of) 2D images(s), or a single volume.
        /// \tparam T           Any data type (integer, floating-point, complex).
        /// \param[out] output  View of the output BDHW array where the deserialized slice(s) are saved.
        ///                     Should describe a (stack of) 2D images(s), i.e. the depth should be 1.
        /// \param start        Index of the slice, in the file, where the deserialization starts.
        /// \param clamp        Whether the deserialized values should be clamped to fit the output type \p T.
        ///                     If false, out of range values are undefined.
        template<typename T, typename I>
        void readSlice(const View<T, I>& output, size_t start, bool clamp = true);

    public: // Write
        /// Saves \p input into the file.
        /// \details If the shape of the file is set, \p input should have the same shape.
        ///          Otherwise, the shape of the file is set to the shape of \p input.
        /// \tparam T           Any data type (integer, floating-point, complex). See traits::is_data.
        ///                     If the data type of the file is set, \p T should be compatible with it.
        ///                     Otherwise, \p T is set as the data type of the file (or the closest supported type).
        /// \param[in] input    Input array to serialize.
        /// \param clamp        Whether the input values should be clamped to fit the file data type.
        ///                     If false, out of range values are undefined.
        template<typename T>
        void write(const Array<T>& input, bool clamp = true);

        /// Saves \p input into the file.
        /// \details If the shape of the file is set, \p input should have the same shape.
        ///          Otherwise, the shape of the file is set to the shape of \p input.
        /// \tparam T           Any data type (integer, floating-point, complex). See traits::is_data.
        ///                     If the data type of the file is set, \p T should be compatible with it.
        ///                     Otherwise, \p T is set as the data type of the file (or the closest supported type).
        /// \param[in] input    Input array to serialize.
        /// \param clamp        Whether the input values should be clamped to fit the file data type.
        ///                     If false, out of range values are undefined.
        template<typename T, typename I>
        void write(const View<T, I>& input, bool clamp = true);

        /// Saves some 2D slices into the file.
        /// The file should describe a (stack of) 2D slice(s), or a single volume.
        /// \tparam T           Any data type (integer, floating-point, complex). See traits::is_data.
        ///                     If the data type of the file is set, \p T should be compatible with it.
        ///                     Otherwise, \p T is set as the data type of the file (or the closest supported type).
        /// \param[in] input    Input array to serialize. Should correspond to the file shape.
        ///                     Should describe a (stack of) 2D images(s), i.e. the depth should be 1.
        /// \param start        Index of the slice, in the file, where the serialization starts.
        /// \param clamp        Whether the input values should be clamped to fit the file data type.
        ///                     If false, out of range values are undefined.
        template<typename T>
        void writeSlice(const Array<T>& input, size_t start, bool clamp = true);

        /// Saves some 2D slices into the file.
        /// The file should describe a (stack of) 2D slice(s), or a single volume.
        /// \tparam T           Any data type (integer, floating-point, complex). See traits::is_data.
        ///                     If the data type of the file is set, \p T should be compatible with it.
        ///                     Otherwise, \p T is set as the data type of the file (or the closest supported type).
        /// \param[in] input    Input array to serialize. Should correspond to the file shape.
        ///                     Should describe a (stack of) 2D images(s), i.e. the depth should be 1.
        /// \param start        Index of the slice, in the file, where the serialization starts.
        /// \param clamp        Whether the input values should be clamped to fit the file data type.
        ///                     If false, out of range values are undefined.
        template<typename T, typename I>
        void writeSlice(const View<T, I>& input, size_t start, bool clamp = true);

    private:
        std::unique_ptr<details::ImageFile> m_header{};
        Format m_header_format{};
        bool m_is_open{};

    private:
        void setHeader_(const path_t& filename, Format new_format);
        static Format format_(const path_t& extension) noexcept;
        void open_(const path_t& filename, open_mode_t mode);
        void close_();
    };
}

#define NOA_IMAGEFILE_INL_
#include "noa/unified/io/ImageFile.inl"
#undef NOA_IMAGEFILE_INL_

namespace noa::io {
    /// Loads a file into a new array.
    /// \tparam T           Any data type (integer, floating-point, complex).
    /// \param[in] filename Path of the file to read.
    /// \param clamp        Whether the values in the file should be clamped to fit the output \p T type.
    ///                     If false, out of range values are undefined.
    /// \param option       Options for the output array.
    /// \return             BDHW C-contiguous output array containing the whole data array of the file.
    /// \note This is an utility function. For better flexibility, use ImageFile directly.
    template<typename T>
    Array<T> load(const path_t& filename, bool clamp = false, ArrayOption option = {}) {
        return ImageFile(filename, io::READ).read<T>(option, clamp);
    }

    /// Saves the input array into a file.
    /// \tparam T           Any data type (integer, floating-point, complex).
    /// \param[in] input    Array to serialize.
    /// \param[in] filename Path of the new file.
    /// \param dtype        Data type of the file. If DTYPE_UNKNOWN, let the file format decide the best data type
    ///                     for \p T values, so that no truncation or loss of precision happens.
    /// \param clamp        Whether the values of the array should be clamped to fit the file data type.
    ///                     If false, out of range values are undefined.
    /// \note This is an utility function. For better flexibility, use ImageFile directly.
    template<typename T>
    void save(const Array<T> input, const path_t& filename, DataType dtype = DTYPE_UNKNOWN, bool clamp = false) {
        ImageFile file(filename, io::WRITE);
        if (dtype != DTYPE_UNKNOWN)
            file.dtype(dtype);
        file.write(input, clamp);
    }
}
