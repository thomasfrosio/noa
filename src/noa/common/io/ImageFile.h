/// \file noa/common/files/ImageFile.h
/// \brief ImageFile class. The interface to specialized image files.
/// \author Thomas - ffyr2w
/// \date 19 Dec 2020

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Exception.h"
#include "noa/common/Types.h"
#include "noa/common/io/IO.h"
#include "noa/common/io/header/Header.h"

// TODO(TF) Add JPEG, PNG and EER file format. The doc might need to be updated for the read/write functions.
// TODO(TF) TIFF format is not tested! Add tests!

namespace noa::io {
    /// Manipulate an image file.
    /// \details
    /// - \b Format: ImageFile is an unified interface to manipulate image file formats. Only MRC and TIFF files are
    ///   currently supported. TIFF files only support (stack of) 2D image(s).\n
    /// - \b Batches: The library, including this class, uses the batch-depth-height-width order. As such, a stack of
    ///   2D images is {batch, 1, height, width} and a 3D volume is {1, depth, height, width}.\n
    /// - \b Ordering: Functions without strides assume rightmost contiguity (C-contiguous/row-major). Serialization
    ///   writes the data assuming it is C-contiguous. Deserialization might have to reorder the data before returning
    ///   it if the data in the file is not stored in C-contiguous order. On the other hand, functions taking a View
    ///   (which contains strides) will return the data in the same order as it is saved in the file.
    /// \example
    /// \code
    /// const size4_t shape = file.shape(); // {1,60,124,124}
    /// const size_t end = at(0, 0, shape[2] - 1, shape[3] - 1, shape.strides()); // end of first YX slice
    /// assert(end == 15375);
    /// file.read(output, 0, end); // which is equivalent to:
    /// file.readSlice(output, 0, 1);
    /// \endcode
    class ImageFile {
    public:
        /// Creates an empty instance. Use open(), otherwise all other function calls will be ignored.
        ImageFile() = default;

        /// Opens the image file.
        /// \param filename Path of the image file to open. The file format is deduced from the extension.
        /// \param mode     Open mode. See open() for more details.
        template<typename T>
        ImageFile(T&& filename, open_mode_t mode);

        /// Opens the image file.
        /// \param filename     Path of the image file to open.
        /// \param file_format  File format used for this file.
        /// \param mode         Open mode. See open() for more details.
        template<typename T>
        ImageFile(T&& filename, Format file_format, open_mode_t mode);

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
        template<typename T>
        void open(T&& filename, uint mode);

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
        [[nodiscard]] const path_t& path() const noexcept;

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

        /// Deserializes some elements from the file.
        /// \tparam T           Any data type (integer, floating-point, complex).
        /// \param[out] output  Output array where the deserialized elements are saved.
        ///                     Should be able to hold at least \p end - \p start elements.
        /// \param start        Position, in the file, where the deserialization starts, in \p T elements.
        /// \param end          Position, in the file, where the deserialization stops, in \p T elements.
        /// \param clamp        Whether the deserialized values should be clamped to fit the output type \p T.
        ///                     If false, out of range values are undefined.
        /// \warning This is only currently supported for MRC files.
        template<typename T>
        void read(T* output, size_t start, size_t end, bool clamp = true);

        /// Deserializes some slices from the file.
        /// \tparam T           Any data type (integer, floating-point, complex).
        /// \param[out] output  Output array where the deserialized slices are saved.
        ///                     Should be able to hold at least \p end - \p start slices.
        /// \param start        Slice, in the file, where the deserialization starts, in \p T elements.
        /// \param end          Slice, in the file, where the deserialization stops, in \p T elements.
        /// \param clamp        Whether the deserialized values should be clamped to fit the output type \p T.
        ///                     If false, out of range values are undefined.
        template<typename T>
        void readSlice(T* output, size_t start, size_t end, bool clamp = true);

        template<typename T, typename I>
        void readSlice(const View<T, I>& view, size_t start = 0, bool clamp = true);

        /// Deserializes the entire file.
        /// \tparam T           Any data type (integer, floating-point, complex).
        /// \param[out] output  Output array where the deserialized values are saved.
        ///                     Should be able to hold the entire shape.
        /// \param clamp        Whether the deserialized values should be clamped to fit the output type \p T.
        ///                     If false, out of range values are undefined.
        template<typename T>
        void readAll(T* output, bool clamp = true);

        /// Deserializes the entire file.
        /// \tparam T           Any data type (integer, floating-point, complex).
        /// \param[out] output  View of the output array where the deserialized values are saved.
        ///                     Should be able to hold the entire shape.
        /// \param clamp        Whether the deserialized values should be clamped to fit the output type \p T.
        ///                     If false, out of range values are undefined.
        template<typename T, typename I>
        void readAll(const View<T, I>& output, bool clamp = true);

        /// Serializes some elements into the file.
        /// \tparam T           Any data type (integer, floating-point, complex). See traits::is_data.
        ///                     If the file data type is UINT4, \p T should not be complex.
        ///                     If the file data type is complex, \p T should be complex.
        /// \param[in] input    Input array to serialize. Read from index 0 to index (\p end - start).
        /// \param start        Position, in the file, where the serialization starts, in \p T elements.
        /// \param end          Position, in the file, where the serialization stops, in \p T elements.
        /// \param clamp        Whether the input values should be clamped to fit the file data type.
        ///                     If false, out of range values are undefined.
        /// \warning This is only supported for MRC files.
        template<typename T>
        void write(const T* input, size_t start, size_t end, bool clamp = true);

        /// Serializes some slices into the file.
        /// \tparam T           Any data type (integer, floating-point, complex). See traits::is_data.
        ///                     If the file data type is UINT4, \p T should not be complex.
        ///                     If the file data type is complex, \p T should be complex.
        /// \param[in] input    Input array to serialize. Read from index 0 to index (\p end - start).
        /// \param start        Slice, in the file, where the serialization starts, in \p T elements.
        /// \param end          Slice, in the file, where the serialization stops, in \p T elements.
        /// \param clamp        Whether the input values should be clamped to fit the file data type.
        ///                     If false, out of range values are undefined.
        template<typename T>
        void writeSlice(const T* input, size_t start, size_t end, bool clamp = true);

        template<typename T>
        void writeSlice(const View<T>& input, size_t start, bool clamp = true);

        /// Serializes the entire file.
        /// \tparam T           Any data type (integer, floating-point, complex). See traits::is_data.
        ///                     If the file data type is UINT4, \p T should not be complex.
        ///                     If the file data type is complex, \p T should be complex.
        /// \param[in] input    Input array to serialize. An entire shape is read from this array.
        /// \param clamp        Whether the input values should be clamped to fit the file data type.
        ///                     If false, out of range values are undefined.
        template<typename T>
        void writeAll(const T* input, bool clamp = true);

        /// Serializes \p input into the file.
        /// \tparam T           Any data type (integer, floating-point, complex). See traits::is_data.
        ///                     If the file data type is UINT4, \p T should not be complex.
        ///                     If the file data type is complex, \p T should be complex.
        /// \param[in] input    Input array to serialize.
        ///                     If the shape of the file is set, this view should have the same shape.
        ///                     If the shape of the file is not set, it will be set to the shape of this view.
        /// \param clamp        Whether the input values should be clamped to fit the file data type.
        ///                     If false, out of range values are undefined.
        template<typename T>
        void writeAll(const View<T>& input, bool clamp = true);

    private:
        path_t m_path{};
        std::unique_ptr<details::Header> m_header{};
        Format m_header_format{};
        bool m_is_open{};

    private:
        void setHeader_(Format new_format);
        static Format getFormat_(const path_t& extension) noexcept;
        void open_(open_mode_t mode);
        void close_();
    };
}

#define NOA_IMAGEFILE_INL_
#include "noa/common/io/ImageFile.inl"
#undef NOA_IMAGEFILE_INL_
