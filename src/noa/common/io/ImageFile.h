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

namespace noa::io {
    /// Manipulate image file.
    /// \note Only MRC and TIFF files are currently supported.
    /// \note Whether dealing with an image, a stack, or a volume, this object will always enforce the following
    ///       order on the deserialized data: shape{X:fast, Y:medium, Z:slow}, i.e. the innermost dimension is X,
    ///       the middlemost is Y and the outermost is Z, regardless of the actual order (in the file) of the
    ///       serialized data. As such, reading operations might flip the data around before returning it, and
    ///       writing operations will always write the data in the contiguous order assuming it matches the
    ///       aforementioned {X:fast, Y:medium, Z:slow} shape.
    /// \note The shape is always 3D, that is, for single images, z=1 and for stack of images, z=slices. A slice
    ///       is a 2D section/view of the data.
    /// \note Elements and lines are relative to the file shape, whether the file describes an image, a stack of
    ///       images or a volume. As such, the indexing is the same as the linear indexing for arrays.
    ///       For instance, if the shape of the file is {124,124,60}, the line 124*124*2=30752 is the first line
    ///       of the third slice. The element (124*124*4)+(19*124)+102=63962 is the element at the indexes
    ///       x=102, y=19, z=4.
    class ImageFile {
    public:
        /// Creates an empty instance. Use open(), otherwise all other function calls will be ignored.
        ImageFile() = default;

        /// Opens the image file.
        /// \param filename     Path of the image file to open. The file format is deduced from the extension.
        /// \param mode         Open mode. See open() for more details.
        template<typename T>
        NOA_HOST ImageFile(T&& filename, open_mode_t mode);

        /// Opens the image file.
        /// \param filename     Path of the image file to open.
        /// \param file_format  File format used for this file.
        /// \param mode         Open mode. See open() for more details.
        template<typename T>
        NOA_HOST ImageFile(T&& filename, Format file_format, open_mode_t mode);

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
        /// \throws Exception   If any of the following cases:
        ///         - If the file does not exist and \p mode is \c io::READ or \c io::READ|io::WRITE.
        ///         - If the permissions do not match the \p mode.
        ///         - If the file format is not recognized, nor supported.
        ///         - If failed to close the file before starting (if any).
        ///         - If an underlying OS error was raised.
        ///         - If the file header could not be read.
        ///
        /// \note Internally, BINARY is always considered on.
        ///       On the other hand, APP and ATE are always considered off.
        ///       Changing any of these bits has no effect.
        /// \note TIFF files don't support modes 2 and 4.
        template<typename T>
        NOA_HOST void open(T&& filename, uint mode);

        /// Closes the file. In writing mode and for some file format,
        /// there can be write operation to save buffered data.
        NOA_HOST void close();

        /// Whether the file is open.
        [[nodiscard]] NOA_HOST bool isOpen() const noexcept;
        NOA_HOST explicit operator bool() const noexcept;

        /// Returns the file format.
        [[nodiscard]] NOA_HOST Format format() const noexcept;
        [[nodiscard]] NOA_HOST bool isMRC() const noexcept;
        [[nodiscard]] NOA_HOST bool isTIFF() const noexcept;
        [[nodiscard]] NOA_HOST bool isEER() const noexcept;
        [[nodiscard]] NOA_HOST bool isJPEG() const noexcept;
        [[nodiscard]] NOA_HOST bool isPNG() const noexcept;

        /// Gets the path.
        [[nodiscard]] NOA_HOST const path_t& path() const noexcept;

        /// Returns a (brief) description of the file data.
        [[nodiscard]] NOA_HOST std::string info(bool brief) const noexcept;

        /// Gets the {X:fast, Y:medium, Z:slow} shape of the data.
        [[nodiscard]] NOA_HOST size3_t shape() const noexcept;

        /// Sets the {X:fast, Y:medium, Z:slow} shape of the data. In pure read mode,
        /// this is usually not allowed and will throw an exception.
        NOA_HOST void shape(size3_t shape);

        /// Gets the pixel size of the data.
        [[nodiscard]] NOA_HOST float3_t pixelSize() const noexcept;

        /// Sets the pixel size of the data. In pure read mode,
        /// this is usually not allowed and will throw an exception.
        NOA_HOST void pixelSize(float3_t pixel_size);

        /// Gets the type of the serialized data.
        [[nodiscard]] NOA_HOST DataType dataType() const noexcept;

        /// Sets the type of the serialized data. This will affect all future writing operation.
        /// In read mode, this is usually not allowed and will throw an exception.
        NOA_HOST void dataType(DataType data_type);

        /// Gets the statistics of the data. Only supported for MRC file formats.
        /// \note the sum and variance are not computed and set to 0.
        [[nodiscard]] NOA_HOST stats_t stats() const noexcept;

        /// Sets the statistics of the data. Depending on the open mode and
        /// file format, this might have no effect on the file.
        NOA_HOST void stats(stats_t stats);

        /// Deserializes some elements from the file.
        /// \tparam T           Any data type (integer, floating-point, complex). See traits::is_data.
        /// \param[out] output  Output array where the deserialized elements are saved.
        ///                     Should be able to hold at least \p end - \p start elements.
        /// \param start        Position, in the file, where the deserialization starts, in \p T elements.
        /// \param end          Position, in the file, where the deserialization stops, in \p T elements.
        /// \param clamp        Whether the deserialized values should be clamped to fit the output type \p T.
        ///                     If false, out of range values are undefined.
        /// \warning This is only currently supported for MRC files.
        template<typename T>
        NOA_HOST void read(T* output, size_t start, size_t end, bool clamp = true);

        /// Deserializes some lines from the file.
        /// \tparam T           Any data type (integer, floating-point, complex). See traits::is_data.
        /// \param[out] output  Output array where the deserialized lines are saved.
        ///                     Should be able to hold at least \p end - \p start lines.
        /// \param start        Line, in the file, where the deserialization starts, in \p T elements.
        /// \param end          Line, in the file, where the deserialization stops, in \p T elements.
        /// \param clamp        Whether the deserialized values should be clamped to fit the output type \p T.
        ///                     If false, out of range values are undefined.
        /// \warning This is only currently supported for MRC files.
        template<typename T>
        NOA_HOST void readLine(T* output, size_t start, size_t end, bool clamp = true);

        /// Deserializes some shape (i.e. 3D tile) from the file.
        /// \tparam T           Any data type (integer, floating-point, complex). See traits::is_data.
        /// \param[out] output  Output array where the deserialized shape is saved.
        ///                     Should be able to hold at least a \p shape.
        /// \param offset       Offset, in the file, the deserialization starts. Corresponds to \p shape.
        /// \param shape        {fast, medium, slow} shape, in \p T elements, to deserialize.
        /// \param clamp        Whether the deserialized values should be clamped to fit the output type \p T.
        ///                     If false, out of range values are undefined.
        /// \todo This is currently not supported.
        template<typename T>
        NOA_HOST void readShape(T* output, size3_t offset, size3_t shape, bool clamp = true);

        /// Deserializes some slices from the file.
        /// \tparam T           Any data type (integer, floating-point, complex). See traits::is_data.
        /// \param[out] output  Output array where the deserialized slices are saved.
        ///                     Should be able to hold at least \p end - \p start slices.
        /// \param start        Slice, in the file, where the deserialization starts, in \p T elements.
        /// \param end          Slice, in the file, where the deserialization stops, in \p T elements.
        /// \param clamp        Whether the deserialized values should be clamped to fit the output type \p T.
        ///                     If false, out of range values are undefined.
        template<typename T>
        NOA_HOST void readSlice(T* output, size_t start, size_t end, bool clamp = true);

        /// Deserializes the entire file.
        /// \tparam T           Any data type (integer, floating-point, complex). See traits::is_data.
        /// \param[out] output  Output array where the deserialized values are saved.
        ///                     Should be able to hold the entire shape.
        /// \param clamp        Whether the deserialized values should be clamped to fit the output type \p T.
        ///                     If false, out of range values are undefined.
        template<typename T>
        NOA_HOST void readAll(T* output, bool clamp = true);

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
        NOA_HOST void write(const T* input, size_t start, size_t end, bool clamp = true);

        /// Serializes some lines into the file.
        /// \tparam T           Any data type (integer, floating-point, complex). See traits::is_data.
        ///                     If the file data type is UINT4, \p T should not be complex.
        ///                     If the file data type is complex, \p T should be complex.
        /// \param[in] input    Input array to serialize. Read from index 0 to index (\p end - start).
        /// \param start        Line, in the file, where the serialization starts, in \p T elements.
        /// \param end          Line, in the file, where the serialization stops, in \p T elements.
        /// \param clamp        Whether the input values should be clamped to fit the file data type.
        ///                     If false, out of range values are undefined.
        template<typename T>
        NOA_HOST void writeLine(const T* input, size_t start, size_t end, bool clamp = true);

        /// Serializes some lines into the file.
        /// \tparam T           Any data type (integer, floating-point, complex). See traits::is_data.
        ///                     If the file data type is UINT4, \p T should not be complex.
        ///                     If the file data type is complex, \p T should be complex.
        /// \param[in] input    Input array to serialize. An entire contiguous \p shape is read from this array.
        /// \param offset       Offset, in the file, the serialization starts. Corresponds to \p shape.
        /// \param shape        {fast, medium, slow} shape, in \p T elements, to serialize.
        /// \param clamp        Whether the input values should be clamped to fit the file data type.
        ///                     If false, out of range values are undefined.
        /// \todo This is currently not supported.
        template<typename T>
        NOA_HOST void writeShape(const T* input, size3_t offset, size3_t shape, bool clamp = true);

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
        NOA_HOST void writeSlice(const T* input, size_t start, size_t end, bool clamp = true);

        /// Serializes the entire file.
        /// \tparam T           Any data type (integer, floating-point, complex). See traits::is_data.
        ///                     If the file data type is UINT4, \p T should not be complex.
        ///                     If the file data type is complex, \p T should be complex.
        /// \param[in] input    Input array to serialize. An entire shape is read from this array.
        /// \param clamp        Whether the input values should be clamped to fit the file data type.
        ///                     If false, out of range values are undefined.
        template<typename T>
        NOA_HOST void writeAll(const T* input, bool clamp = true);

    private:
        path_t m_path{};
        std::unique_ptr<details::Header> m_header{};
        Format m_header_format{};
        bool m_is_open{};

    private:
        NOA_HOST void setHeader_(Format new_format);
        NOA_HOST static Format getFormat_(const path_t& extension) noexcept;
        NOA_HOST void open_(open_mode_t mode);
        NOA_HOST void close_();
    };
}

#define NOA_IMAGEFILE_INL_
#include "noa/common/io/ImageFile.inl"
#undef NOA_IMAGEFILE_INL_
