/// \file ImageFile.h
/// \brief ImageFile class. The interface to specialized image files.
/// \author Thomas - ffyr2w
/// \date 19 Dec 2020

#pragma once

#include <memory>
#include <type_traits>
#include <utility>
#include <ios>
#include <filesystem>
#include <string>

#include "noa/Definitions.h"
#include "noa/Exception.h"
#include "noa/Types.h"
#include "noa/io/IO.h"
#include "noa/util/IntX.h"
#include "noa/util/FloatX.h"

namespace noa {
    /// The ImageFile tries to offer an uniform API to work with all image files
    /// (e.g. \a MRCFile, \a TIFFile, \a EERFile, etc.).
    class ImageFile {
    public:
        ImageFile() = default;
        virtual ~ImageFile() = default;

        /// Returns an ImageFile with a path, but the file is NOT opened.
        /// \return One of the derived ImageFile.
        /// \throws Exception if the extension is not recognized.
        ///
        /// \note    The previous implementation was using an opaque pointer, but this is much simpler
        ///          and probably safer since it is obvious that we are dealing with a pointer.
        [[nodiscard]] NOA_HOST static std::unique_ptr<ImageFile> get(const std::string& extension);

        [[nodiscard]] NOA_HOST static std::unique_ptr<ImageFile> get(const path_t& extension) {
            return get(extension.string());
        }

        /// Saves \a data to disk, under \a filename.
        /// \param filename     Filename. The file format is deduced from the extension.
        /// \param data         Data to write.
        /// \param dtype        Data type to convert \a data into. Should be a real dtype.
        /// \param shape        {fast, medium, slow} shape of \a data.
        /// \param pixel_size   Pixel size (corresponding to \a shape).
        NOA_HOST static void save(const path_t& filename, const float* data,
                                  size3_t shape, io::DataType dtype, float3_t ps);

        NOA_HOST static void save(const path_t& filename, const cfloat_t* data,
                                  size3_t shape, io::DataType dtype, float3_t ps);

        /// Saves \a data to disk, under \a filename. Data type is defaulted FLOAT32 and the pixel size is 1.
        NOA_HOST static void save(const path_t& filename, const float* data, size3_t shape) {
            save(filename, data, shape, io::DataType::FLOAT32, float3_t{1.0f});
        }

        NOA_HOST static void save(const path_t& filename, const cfloat_t* data, size3_t shape) {
            save(filename, data, shape, io::DataType::CFLOAT32, float3_t{1.0f});
        }

        // Below are the functions that derived classes should override.
        //  ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓
    public:
        /// (Re)Opens the file.
        /// \param open_mode    io::OpenMode bit mask. Should be one or a combination of the following:
        ///                     \c READ:                File should exists.
        ///                     \c READ|WRITE:          File should exists.     Backup copy.
        ///                     \c WRITE, WRITE|TRUNC:  Overwrite the file.     Backup move.
        ///                     \c READ|WRITE|TRUNC:    Overwrite the file.     Backup move.
        ///
        /// \throws If any of the following cases:
        ///         - If the file does not exist and \a mode is \c io::READ or \c io::READ|io::WRITE.
        ///         - If the permissions do not match the \a open_mode.
        ///         - If the image file type is not recognized, nor supported.
        ///         - If failed to close the file before starting (if any).
        ///         - If an underlying OS error was raised.
        ///         - If the file header could not be read.
        ///
        /// \note   Internally, the \c io::BINARY is always considered on. On the other hand, \c io::APP and
        ///         \c io::ATE are always considered off. Changing any of these bits has no effect.
        NOA_HOST virtual void open(uint open_mode) = 0;

        /// Resets the path and opens the file.
        NOA_HOST virtual void open(const path_t&, uint) = 0;

        /// Resets the path and opens the file.
        NOA_HOST virtual void open(path_t&&, uint) = 0;

        /// Whether or not the file is open.
        [[nodiscard]] NOA_HOST virtual bool isOpen() const = 0;

        /// Closes the file. For some file format, there can be write operation to save buffered data.
        NOA_HOST virtual void close() = 0;

        /// Gets the path.
        [[nodiscard]] NOA_HOST virtual const path_t* path() const noexcept = 0;

        /// Whether or not the file is in a good state.
        NOA_HOST virtual explicit operator bool() const noexcept = 0;

        /// Clears the underlying state to a good state.
        NOA_HOST virtual void clear() noexcept = 0;

        [[nodiscard]] NOA_HOST virtual size3_t getShape() const = 0;
        NOA_HOST virtual void setShape(size3_t) = 0;

        [[nodiscard]] NOA_HOST virtual Float3<float> getPixelSize() const = 0;
        NOA_HOST virtual void setPixelSize(Float3<float>) = 0;

        [[nodiscard]] NOA_HOST virtual std::string describe(bool) const = 0;

        /// Returns the underlying data type.
        NOA_HOST virtual io::DataType getDataType() const = 0;

        /// Sets the data type for all future writing operation.
        NOA_HOST virtual void setDataType(io::DataType) = 0;

        /// Reads the entire data into \a data.
        /// \param[out] data    Output array. Should be at least equal to \c getElements(getRandomShape()) * 4.
        /// \throw If io::readFloat fails.
        /// \note The underlying data should be a real (as opposed to complex) type.
        NOA_HOST virtual void readAll(float* data) = 0;

        /// Reads the entire data into \a data.
        /// \param[out] data    Output array. Should be at least equal to \c getElements(getRandomShape()) * 8.
        /// \throw If io::readComplexFloat fails or if the file format does not support complex data.
        /// \note The underlying data should be a complex type.
        NOA_HOST virtual void readAll(cfloat_t* data) = 0;

        /// Reads slices (z sections) from the file data and stores the data into \a data.
        /// \note Slices refer to whatever dimension is in the last order (the slowest dimension),
        ///       but since the only data ordering allowed is (x=fast, y=medium, z=slow), slices
        ///       are effectively z sections.
        /// \note The underlying data should be a real (as opposed to complex) type.
        ///
        /// \param[out] data    Output float array. It should be large enough to contain the desired
        ///                     data, that is `getElementsSlice(getRandomShape()) * 4 * z_count` bytes.
        /// \param z_pos        Slice to start reading from.
        /// \param z_count      Number of slices to read.
        /// \throws If io::readFloat fails.
        NOA_HOST virtual void readSlice(float* data, size_t z_pos, size_t z_count) = 0;

        /// Reads slices (z sections) from the file data and stores the data into \a out.
        /// \note The underlying data should be a complex type.
        ///
        /// \param[out] ptr_out Output float array. It should be large enough to contain the desired
        ///                     data, that is `getElements(getRandomShape()) * 8 * z_count` bytes.
        /// \param z_pos        Slice to start reading from.
        /// \param z_count      Number of slices to read.
        /// \throw If io::readComplexFloat fails or if the file format does not support complex data.
        NOA_HOST virtual void readSlice(cfloat_t* ptr_out, size_t z_pos, size_t z_count) = 0;

        /// Writes the entire file. The ordering is expected to be (x=1, y=2, z=3).
        /// \param[in] ptr_in   Array to write. Should be at least `getElements(getRandomShape()) * 4` bytes.
        /// \throw If io::writeFloat fails.
        /// \note The underlying data should be a real (as opposed to complex) type.
        NOA_HOST virtual void writeAll(const float* ptr_in) = 0;

        /// Writes the entire file. The ordering is expected to be (x=1, y=2, z=3).
        /// \param[in] ptr_in   Array to serialize. Should be at least `getElements(getRandomShape()) * 8` bytes.
        /// \throw If io::writeComplexFloat fails or if the file format does not support complex data.
        /// \note The underlying data should be a complex type.
        NOA_HOST virtual void writeAll(const cfloat_t* ptr_in) = 0;

        /// Writes slices (z sections) from \a out into the file.
        /// \param[out] ptr_out     Array to serialize.
        /// \param z_pos            Slice to start.
        /// \param z_count          Number of slices to serialize.
        /// \throw If io::writeFloat fails.
        /// \note The underlying data should be a real (as opposed to complex) type.
        NOA_HOST virtual void writeSlice(const float* ptr_out, size_t z_pos, size_t z_count) = 0;

        /// Writes slices (z sections) from \a out into the file.
        /// \param[out] ptr_out     Array to serialize.
        /// \param z_pos            Slice to start.
        /// \param z_count          Number of slices to serialize.
        /// \throw If io::writeComplexFloat fails or if the file format does not support complex data.
        /// \note The underlying data should be a complex type.
        NOA_HOST virtual void writeSlice(const cfloat_t* ptr_out, size_t z_pos, size_t z_count) = 0;
    };
}
