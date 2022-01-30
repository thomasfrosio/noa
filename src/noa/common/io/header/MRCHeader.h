#pragma once

#include <memory>
#include <utility>
#include <ios>
#include <filesystem>
#include <fstream>

#include "noa/common/Definitions.h"
#include "noa/common/Exception.h"
#include "noa/common/Types.h"
#include "noa/common/io/IO.h"
#include "noa/common/io/header/Header.h"

namespace noa::io::details {
    // Handles MRC files.
    //
    // Current implementation:
    //  - Grid size:    mx, my and mz must be equal to nx, ny and nz (i.e. the number of columns,
    //                  rows, and sections. Otherwise, an exception is thrown when opening the file.
    //  - Pixel size:   If the cell size is equal to zero, the pixel size will be 0. This is allowed
    //                  since in some cases the pixel size is ignored. Thus the user should check the
    //                  returned value of getPixelSize() before using it. Similarly, setPixelSize()
    //                  can save a pixel size equal to 0, which is the default if a new file is closed
    //                  without calling this setPixelSize().
    //  - Endianness:   It is not possible to change the endianness of existing data. As such, in the
    //                  rare case of writing to an existing file (i.e. READ|WRITE), if the endianness is
    //                  not the same as the CPU, the data being written will be swapped, which is slower.
    //  - Data type:    In the rare case of writing to an existing file (i.e. READ|WRITE), the data
    //                  type cannot be changed. In other cases, the user should only set the data
    //                  type once and before writing anything to the file. By default, no conversions
    //                  are performed, i.e. DataType::FLOAT32.
    //  - Order:        The map ordering should be mapc=1, mapr=2 and maps=3. Anything else is not
    //                  supported and an exception is thrown when opening the file.
    //  - Space group:  It is not used, but it is checked for validation. It should be 0 or 1.
    //                  Anything else (notably 401: stack of volumes) is not supported.
    //  - Unused flags: The extended header, the origin (xorg, yorg, zorg), nversion and other
    //                  parts of the header are ignored (see full detail on what is ignored in
    //                  writeHeader_()). These are set to 0 or to the expected default value, or are
    //                  left unchanged in non-overwriting mode.
    //  - Setters:      In reading mode, using the "setters" (i.e. set(PixelSize|Shape|DataType)),
    //                  is not allowed. Use read|write mode to modify a corrupted header.
    //
    // see      https://bio3d.colorado.edu/imod/doc/mrc_format.txt or
    //          https://www.ccpem.ac.uk/mrc_format/mrc2014.php
    class MRCHeader : public Header {
    private:
        std::fstream m_fstream{};
        open_mode_t m_open_mode{};

        struct Header_impl {
            // Buffer containing the 1024 bytes of the header.
            // Only used if the header needs to be saved, that is, in in|out mode.
            std::unique_ptr<char[]> buffer{nullptr};
            DataType data_type{DataType::FLOAT32};

            // The MRC header is XYZ, so save the shape and pixel size in this order, for internal use.
            Int4<int32_t> shape{1};                 // Number of columns (x), rows (y) and sections (z).
            float3_t pixel_size{0.f};               // Pixel spacing (x, y and z) = cell_size / shape.

            float min{0};                           // Minimum pixel value.
            float max{-1};                          // Maximum pixel value.
            float mean{-2};                         // Mean pixel value.
            float std{-1};                          // Stdev. Negative if not computed.

            int32_t extended_bytes_nb{0};           // Number of bytes in extended header.

            bool is_endian_swapped{false};          // Whether the endianness of the data is swapped.
            int32_t nb_labels{0};                   // Number of labels with useful data.
        } m_header{};

    public:
        MRCHeader() = default;
        ~MRCHeader() override { close_(); }

        void open(const path_t& path, open_mode_t open_mode) override { open_(path, open_mode); }
        void close() override { close_(); }

        [[nodiscard]] Format getFormat() const noexcept override { return Format::MRC; }

        [[nodiscard]] std::string infoString(bool brief) const noexcept override;

        [[nodiscard]] size4_t getShape() const noexcept override {
            return size4_t{m_header.shape.flip()};
        }

        [[nodiscard]] stats_t getStats() const noexcept override {
            stats_t out;
            if (m_header.min != 0 || m_header.max != -1 || m_header.mean != -2) {
                // all or nothing...
                out.min(m_header.min);
                out.max(m_header.max);
                out.mean(m_header.mean);
            }
            if (m_header.std >= 0)
                out.std(m_header.std);
            return out;
        }

        [[nodiscard]] float3_t getPixelSize() const noexcept override {
            return m_header.pixel_size.flip();
        }

        [[nodiscard]] DataType getDataType() const noexcept override {
            return m_header.data_type;
        }

        void setShape(size4_t new_shape) override;
        void setDataType(io::DataType data_type) override;
        void setPixelSize(float3_t new_pixel_size) override;
        void setStats(stats_t stats) override {
            // In reading mode, this will have no effect.
            if (stats.hasMin())
                m_header.min = stats.min();
            if (stats.hasMax())
                m_header.max = stats.max();
            if (stats.hasMean())
                m_header.mean = stats.mean();
            if (stats.hasStd())
                m_header.std = stats.std();
        }

        void read(void* output, DataType data_type, size_t start, size_t end, bool clamp) override;
        void readLine(void* output, DataType data_type, size_t start, size_t end, bool clamp) override;
        void readShape(void* output, DataType data_type, size4_t offset, size4_t shape, bool clamp) override;
        void readSlice(void* output, DataType data_type, size_t start, size_t end, bool clamp) override;
        void readAll(void* output, DataType data_type, bool clamp) override;

        void write(const void* input, DataType data_type, size_t start, size_t end, bool clamp) override;
        void writeLine(const void* input, DataType data_type, size_t start, size_t end, bool clamp) override;
        void writeShape(const void* input, DataType data_type, size4_t offset, size4_t shape, bool clamp) override;
        void writeSlice(const void* input, DataType data_type, size_t start, size_t end, bool clamp) override;
        void writeAll(const void* input, DataType data_type, bool clamp) override;

    private:
        void open_(const path_t& filename, open_mode_t mode);

        // Reads and checks the header of an existing file.
        // Throws if the header doesn't look like a MRC header or if the MRC file is not supported.
        // If read|write mode, the header is saved in m_header.buffer. This will be used before closing the file.
        void readHeader_();

        // Swap the endianness of the header.
        // The buffer is at least the first 224 bytes of the MRC header.
        //
        // In read or read|write mode, the data of the header should be swapped if the endianness of the file is
        // swapped. This function should be called just after reading the header AND just before writing it.
        // All used flags are swapped. Some unused flags are left unchanged.
        static void swapHeader_(char* buffer) {
            io::swapEndian(buffer, 24, 4); // from 0 (nx) to 96 (next, included).
            io::swapEndian(buffer + 152, 2, 4); // imodStamp, imodFlags
            io::swapEndian(buffer + 216, 2, 4); // rms, nlabl
        }

        // Closes the stream. Separate function so that the destructor can call close().
        // In writing mode, the header flags will be writing to the file's header.
        void close_();

        // Sets the header to default values.
        // This function should only be called to initialize the header before closing a (overwritten or new) file.
        static void defaultHeader_(char* buffer);

        // Writes the header to the file's header. Only called before closing a file.
        // Buffer containing the header. Only the used values (see m_header) are going to be modified before write
        // buffer to the file. As such, unused values should either be set to the defaults (overwrite mode, see
        // defaultHeader_()) OR taken from an existing file (read|write mode, see readHeader_()).
        void writeHeader_(char* buffer);

        // Gets the offset to the data.
        [[nodiscard]] long offset_() const {
            return 1024 + m_header.extended_bytes_nb;
        }
    };
}
