/**
 * @file ImageFile.h
 * @brief ImageFile class. The interface to specialized image files.
 * @author Thomas - ffyr2w
 * @date 19 Dec 2020
 */
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
#include "noa/util/IO.h"
#include "noa/util/IntX.h"
#include "noa/util/FloatX.h"

namespace Noa {
    /**
     * The ImageFile tries to offer an uniform API to work with all image files
     * (e.g. @a MRCFile, @a TIFFile, @a EERFile, etc.).
     */
    class ImageFile {
    public:
        ImageFile() = default;
        virtual ~ImageFile() = default;

        /**
         * Returns an ImageFile with a path, but the file is NOT opened.
         * @return One of the derived ImageFile.
         * @throws Exception if the extension is not recognized.
         *
         * @note    The previous implementation was using an opaque pointer, but this is much simpler
         *          and probably safer since it is obvious that we are dealing with a pointer.
         */
        [[nodiscard]] static std::unique_ptr<ImageFile> get(const std::string& extension);

        [[nodiscard]] inline static std::unique_ptr<ImageFile> get(const path_t& extension) {
            return get(extension.string());
        }

        // Below are the functions that derived classes should override.
        //  ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓
    public:
        /**
         * (Re)Opens the file.
         * @param mode  Should be one or a combination of the following:
         *              @c in:              Read.           File should exists.
         *              @c in|out:          Read & Write.   File should exists.     Backup copy.
         *              @c out, out|trunc:  Write.          Overwrite the file.     Backup move.
         *              @c in|out|trunc:    Read & Write.   Overwrite the file.     Backup move.
         * @param wait  Wait for the file to exist for 10*3s, otherwise wait for 5*10ms.
         *
         * @throws Exception if any of the following cases:
         *          - If the image file type is not recognized, nor supported.
         *          - If failed to close the file before starting (if any).
         *          - If an underlying OS error was raised.
         *          - If the file header could not be read.
         *
         * @note    Internally, the @c std::ios::binary is always considered on. On the other hand,
         *          @c std::ios::app and @c std::ios::ate are always considered off. Changing any
         *          of these bits has no effect.
         */
        virtual void open(openmode_t mode, bool wait) = 0;

        /** Resets the path and opens the file. */
        virtual void open(const path_t&, openmode_t, bool) = 0;

        /** Resets the path and opens the file. */
        virtual void open(path_t&&, openmode_t, bool) = 0;

        /** Whether or not the file is open. */
        [[nodiscard]] virtual bool isOpen() const = 0;

        /** Closes the file. For some file format, there can be write operation to save buffered data. */
        virtual void close() = 0;

        /** Gets the path. */
        [[nodiscard]] virtual const path_t* path() const noexcept = 0;

        /** Whether or not the file is in a good state. */
        virtual explicit operator bool() const noexcept = 0;

        /** Clears the underlying state to a good state. */
        virtual void clear() noexcept = 0;

        [[nodiscard]] virtual shape_t getShape() const = 0;
        virtual void setShape(shape_t) = 0;

        [[nodiscard]] virtual Float3<float> getPixelSize() const = 0;
        virtual void setPixelSize(Float3<float>) = 0;

        [[nodiscard]] virtual std::string toString(bool) const = 0;

        /** Returns the underlying data type. */
        virtual IO::DataType getDataType() const = 0;

        /** Sets the data type for all future writing operation. */
        virtual void setDataType(IO::DataType) = 0;

        /**
         * Reads the entire data into @a data.
         * @param[out] data     Output array. Should be at least equal to @c Math::elements(getShape()) * 4.
         * @throw Exception     If IO::readFloat fails.
         * @note The underlying data should be a real (as opposed to complex) type.
         */
        virtual void readAll(float* data) = 0;

        /**
         * Reads the entire data into @a data.
         * @param[out] data     Output array. Should be at least equal to @c Math::elements(getShape()) * 8.
         * @throw Exception     If IO::readComplexFloat fails or if the file format does not support complex data.
         * @note The underlying data should be a complex type.
         */
        virtual void readAll(cfloat_t* data) = 0;

        /**
         * Reads slices (z sections) from the file data and stores the data into @a data.
         * @note Slices refer to whatever dimension is in the last order (the slowest dimension),
         *       but since the only data ordering allowed is (x=fast, y=medium, z=slow), slices
         *       are effectively z sections.
         * @note The underlying data should be a real (as opposed to complex) type.
         *
         * @param[out] data     Output float array. It should be large enough to contain the desired
         *                      data, that is `Math::elementsSlice(getShape()) * 4 * z_count` bytes.
         * @param z_pos         Slice to start reading from.
         * @param z_count       Number of slices to read.
         * @throws Exception    If IO::readFloat fails.
         */
        virtual void readSlice(float* data, size_t z_pos, size_t z_count) = 0;

        /**
         * Reads slices (z sections) from the file data and stores the data into @a out.
         * @note The underlying data should be a complex type.
         *
         * @param[out] ptr_out  Output float array. It should be large enough to contain the desired
         *                      data, that is `Math::elements(getShape()) * 8 * z_count` bytes.
         * @param z_pos         Slice to start reading from.
         * @param z_count       Number of slices to read.
         * @throw Exception     If IO::readComplexFloat fails or if the file format does not support complex data.
         */
        virtual void readSlice(cfloat_t* ptr_out, size_t z_pos, size_t z_count) = 0;

        /**
         * Writes the entire file. The ordering is expected to be (x=1, y=2, z=3).
         * @param[in] ptr_in    Array to write. Should be at least `Math::elements(getShape()) * 4` bytes.
         * @throw Exception     If IO::writeFloat fails.
         * @note The underlying data should be a real (as opposed to complex) type.
         */
        virtual void writeAll(const float* ptr_in) = 0;

        /**
         * Writes the entire file. The ordering is expected to be (x=1, y=2, z=3).
         * @param[in] ptr_in    Array to serialize. Should be at least `Math::elements(getShape()) * 8` bytes.
         * @throw Exception     If IO::writeComplexFloat fails or if the file format does not support complex data.
         * @note The underlying data should be a complex type.
         */
        virtual void writeAll(const cfloat_t* ptr_in) = 0;

        /**
         * Writes slices (z sections) from @a out into the file.
         * @param[out] ptr_out  Array to serialize.
         * @param z_pos         Slice to start.
         * @param z_count       Number of slices to serialize.
         * @throw Exception     If IO::writeFloat fails.
         * @note The underlying data should be a real (as opposed to complex) type.
         */
        virtual void writeSlice(const float* ptr_out, size_t z_pos, size_t z_count) = 0;

        /**
         * Writes slices (z sections) from @a out into the file.
         * @param[out] ptr_out  Array to serialize.
         * @param z_pos         Slice to start.
         * @param z_count       Number of slices to serialize.
         * @throw Exception     If IO::writeComplexFloat fails or if the file format does not support complex data.
         * @note The underlying data should be a complex type.
         */
        virtual void writeSlice(const cfloat_t* ptr_out, size_t z_pos, size_t z_count) = 0;
    };
}
