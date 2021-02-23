/**
 * @file MRCFile.h
 * @brief MRCFile class.
 * @author Thomas - ffyr2w
 * @date 19 Dec 2020
 */
#pragma once

#include <memory>
#include <type_traits>
#include <utility>
#include <ios>
#include <filesystem>

#include "noa/Definitions.h"
#include "noa/Exception.h"
#include "noa/Types.h"
#include "noa/util/IO.h"
#include "noa/util/IntX.h"
#include "noa/util/FloatX.h"
#include "noa/util/files/ImageFile.h"

namespace Noa {
    /**
     * Handles MRC files.
     *
     * @warning Known limitation:
     *  - Grid size:    mx, my and mz must be equal to nx, ny and nz (i.e. the number of columns,
     *                  rows, and sections. Otherwise, the file will not be supported and the state
     *                  will be switched to Errno::not_supported.
     *  - Endianness:   It is not possible to change the endianness of the data. As such, in the
     *                  rare case of writing to an existing file (i.e. in|out), if the endianness is
     *                  not the same as the CPU, the data being written will be swapped, which
     *                  considerably slows down the execution.
     *  - Data type:    In the rare case of writing to an existing file (i.e. in|out), the data
     *                  type cannot be changed. In other cases, the user should only set the data
     *                  type once and before writing anything to the file.
     *  - Order:        The map ordering should be mapc=1, mapr=2 and maps=3. Anything else is not
     *                  supported and the state will be switched to Errno::not_supported.
     *  - Space group:  It is not used, but it is checked for validation. It should be 0 or 1.
     *                  Anything else (notably 401: stack of volumes) is not supported.
     *  - Unused flags: The extended header, the origin (xorg, yorg, zorg), nversion and other
     *                  parts of the header are ignored (see full detail on what is ignored in
     *                  writeHeader_()). These are set to 0 or to the expected default value, or are
     *                  left unchanged in non-overwriting mode.
     *  - Setters:      In reading mode, using the "setters" (i.e. set(PixelSize|Statistics|Shape|DataType)),
     *                  won't return any error and the metadata will be correctly set, but this will
     *                  have no effect on the actual file.
     *
     * @see     https://bio3d.colorado.edu/imod/doc/mrc_format.txt or
     *          https://www.ccpem.ac.uk/mrc_format/mrc2014.php
     *
     * @note    29Dec20 - ffyr2w: The previous implementation of the header was based on a
     *          reinterpretation of the serialized data (a buffer of char[1024]). You can find such
     *          implementation in the MRCFile implementation in cisTEM. Turns out, this break the
     *          strict aliasing rule and is therefore UB. In this case, I decided to follow the
     *          standard and change the implementation. This should be "defined behavior" now.
     *
     * @note    12Jan21 - ffyr2w: In most cases, it was not useful to save the m_header.buffer.
     *          Now, the buffer is only saved in in|out mode, which is a very rare case.
     */
    class MRCFile : public ImageFile {
    private:
        std::fstream m_fstream{};
        path_t m_path{};
        openmode_t m_open_mode{};

        struct Header {
            // Buffer containing the 1024 bytes of the header.
            // Only used if the header needs to be saved, that is, in in|out mode.
            std::unique_ptr<char[]> buffer{nullptr};
            IO::DataType data_type{IO::DataType::float32};
            Int3<int32_t> shape{0};                 // Number of columns (x), rows (y) and sections (z).
            Float3<float> pixel_size{0.f};          // Pixel spacing (x, y and z) = cell_size / shape.

            float min{0};                           // Minimum pixel value.
            float max{-1};                          // Maximum pixel value.
            float mean{-2};                         // Mean pixel value.
            float rms{0};                           // Std of densities from mean. Negative if not computed.

            int32_t extended_bytes_nb{0};           // Number of bytes in extended header.

            bool is_endian_swapped{false};          // Whether or not the endianness of the data is swapped.
            int32_t nb_labels{0};                   // Number of labels with useful data.
        } m_header{};

    public:
        /** Creates an empty instance. Use open() to open a file. */
        MRCFile() = default;

        /** Stores the path. The file is not opened. Use open() to open the associated file. */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, path_t>>>
        NOA_HOST explicit MRCFile(T&& path) : m_path(std::forward<T>(path)) {}

        /** Stores the path and opens the file. */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, path_t>>>
        NOA_HOST explicit MRCFile(T&& path, openmode_t mode, bool wait) : m_path(std::forward<T>(path)) {
            open_(mode, wait);
        }

        /** The file is closed before destruction. In writing mode, the header is saved before closing. */
        NOA_HOST ~MRCFile() override {
            close_();
        }

        NOA_HOST std::tuple<float, float, float, float> getStatistics() {
            return {m_header.min, m_header.max, m_header.mean, m_header.rms};
        }

        /** Sets the statistics in the header. */
        NOA_HOST void setStatistics(float min, float max, float mean, float rms) {
            m_header.min = min;
            m_header.max = max;
            m_header.mean = mean;
            m_header.rms = rms;
        }

        // Below are the overridden functions.
        // See the corresponding virtual function in @a ImageFile.
        //  ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓

        NOA_HOST void open(openmode_t mode, bool long_wait) override {
            open_(mode, long_wait);
        }

        NOA_HOST void open(const path_t& path, openmode_t mode, bool long_wait) override {
            m_path = path;
            open_(mode, long_wait);
        }

        NOA_HOST void open(path_t&& path, openmode_t mode, bool long_wait) override {
            m_path = std::move(path);
            open_(mode, long_wait);
        }

        [[nodiscard]] NOA_HOST std::string toString(bool brief) const override;
        [[nodiscard]] NOA_HOST const path_t* path() const noexcept override { return &m_path; }
        [[nodiscard]] NOA_HOST bool isOpen() const override { return m_fstream.is_open(); }
        NOA_HOST void close() override { close_(); }

        NOA_HOST explicit operator bool() const noexcept override { return !m_fstream.fail(); }
        NOA_HOST void clear() noexcept override { m_fstream.clear(); }

        NOA_HOST void setDataType(IO::DataType) override;
        NOA_HOST IO::DataType getDataType() const override { return m_header.data_type; }

        NOA_HOST void readAll(float* to_write) override;
        NOA_HOST void readAll(cfloat_t* to_write) override;
        NOA_HOST void readSlice(float* to_write, size_t z_pos, size_t z_count) override;
        NOA_HOST void readSlice(cfloat_t* to_write, size_t z_pos, size_t z_count) override;

        NOA_HOST void writeAll(const float* to_read) override;
        NOA_HOST void writeAll(const cfloat_t* to_read) override;
        NOA_HOST void writeSlice(const float* to_read, size_t z_pos, size_t z_count) override;
        NOA_HOST void writeSlice(const cfloat_t* to_read, size_t z_pos, size_t z_count) override;

        [[nodiscard]] NOA_HOST shape_t getShape() const override { return shape_t(m_header.shape); }
        NOA_HOST void setShape(shape_t new_shape) override { m_header.shape = new_shape; }

        [[nodiscard]] NOA_HOST Float3<float> getPixelSize() const override { return m_header.pixel_size; }

        NOA_HOST void setPixelSize(Float3<float> new_pixel_size) override {
            if (new_pixel_size > 0.f)
                m_header.pixel_size = new_pixel_size;
            else
                NOA_THROW("File: \"{}\". Could not save a negative or zero pixel size. Got ({:.3f},{:.3f},{:.3f})",
                          m_path.filename(), new_pixel_size.x, new_pixel_size.y, new_pixel_size.z);
        }

    private:
        /** Tries to open the file in @a m_path. See ImageFile::open() for more details. */
        NOA_HOST void open_(openmode_t mode, bool wait);

        /**
         * Reads and checks the header of an existing file.
         * @throw   Exception if the header doesn't look like a MRC header or if the MRC file is not supported.
         * @note    In the rare read & write (i.e. in|out) case, the header is saved in
         *          m_header.buffer. This will be used before closing the file. See close_().
         */
        NOA_HOST void readHeader_();

        /**
         * Swap the endianness of the header.
         * @param[in] buffer    At least the first 224 bytes of the MRC header.
         *
         * @note In read or in|out mode, the data of the header should be swapped if the endianness of the file is
         *       swapped. This function should be called just after reading the header AND just before writing it.
         * @note All used flags are swapped. Some unused flags are left unchanged.
         */
        NOA_HOST static void swapHeader_(char* buffer) {
            IO::swapEndian<4>(buffer, 24); // from 0 (nx) to 96 (next, included).
            IO::swapEndian<4>(buffer + 152, 2); // imodStamp, imodFlags
            IO::swapEndian<4>(buffer + 216, 2); // rms, nlabl
        }

        /**
         * Closes the stream. Separate function so that the destructor can call close().
         * @note    In writing mode, the header flags will be writing to the file's header.
         */
        NOA_HOST void close_();

        /**
        * Sets the header to default values.
        * @warning This function should only be called to initialize the header before closing
        *          a (overwritten or new) file.
        */
        NOA_HOST static void defaultHeader_(char* buffer);

        /**
         * Writes the header to the file's header. Only called before closing a file.
         * @param[in] buffer    Buffer containing the header. Only the used values (see m_header)
         *                      are going to be modified before write @a buffer to the file.
         *                      As such, unused values should either be set to the defaults
         *                      (overwrite mode, see defaultHeader_()) OR taken from an existing
         *                      file (in|out mode, see readHeader_()).
         */
        NOA_HOST void writeHeader_(char* buffer);

        /** Gets the offset to the data: header size (1024) + the extended header. */
        [[nodiscard]] NOA_HOST long getOffset_() const {
            return 1024 + m_header.extended_bytes_nb;
        }
    };
}
