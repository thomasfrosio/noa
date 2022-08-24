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
    // Notes about the current implementation:
    //  - Pixel size:   If the cell size is equal to zero, the pixel size will be 0. This is allowed
    //                  since in some cases the pixel size is ignored. Thus, one should check the
    //                  returned value of getPixelSize() before using it. Similarly, setPixelSize()
    //                  can save a pixel size equal to 0, which is the default if a new file is closed
    //                  without calling setPixelSize().
    //  - Endianness:   It is not possible to change the endianness of existing data. As such, in the
    //                  rare case of writing to an existing file (i.e. READ|WRITE), if the endianness is
    //                  not the same as the CPU, the data being written will be swapped, which is slower.
    //  - Data type:    In the rare case of writing to an existing file (i.e. READ|WRITE), the data
    //                  type cannot be changed. In other cases, the user should only set the data
    //                  type once and before writing anything to the file.
    //  - Order:        The map ordering should be mapc=1, mapr=2 and maps=3. Anything else is not
    //                  supported and an exception is thrown when opening the file.
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
    public:
        MRCHeader() = default;
        ~MRCHeader() override { close_(); }

    public:
        void reset() override {
            close();
            m_open_mode = open_mode_t{};
            m_header = HeaderImp{};
        };

        void open(const path_t& path, open_mode_t open_mode) override { open_(path, open_mode); }
        void close() override { close_(); }

    public:
        [[nodiscard]] bool isOpen() const noexcept override { return m_fstream.is_open(); }
        [[nodiscard]] const path_t& filename() const noexcept override { return m_filename; }
        [[nodiscard]] std::string infoString(bool brief) const noexcept override;
        [[nodiscard]] Format format() const noexcept override { return Format::MRC; }

    public:
        [[nodiscard]] size4_t shape() const noexcept  {
            return m_header.shape;
        }

        void shape(size4_t new_shape) override;

        [[nodiscard]] stats_t stats() const noexcept override {
            stats_t out;
            if (m_header.min != 0 || m_header.max != -1 || m_header.mean != -2) {
                out.min(m_header.min);
                out.max(m_header.max);
                out.mean(m_header.mean);
            }
            if (m_header.std >= 0)
                out.std(m_header.std);
            return out;
        }

        void stats(stats_t stats) override {
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

        [[nodiscard]] float3_t pixelSize() const noexcept override {
            return m_header.pixel_size;
        }

        void pixelSize(float3_t new_pixel_size) override;

        [[nodiscard]] DataType dtype() const noexcept override {
            return m_header.data_type;
        }

        void dtype(io::DataType data_type) override;

    public:
        void read(void* output, DataType data_type, size_t start, size_t end, bool clamp) override;
        void readSlice(void* output, DataType data_type, size_t start, size_t end, bool clamp) override;
        void readSlice(void* output, size4_t strides, size4_t shape, DataType data_type, size_t start, bool clamp) override;
        void readAll(void* output, DataType data_type, bool clamp) override;
        void readAll(void* output, size4_t strides, size4_t shape, DataType data_type, bool clamp) override;

        void write(const void* input, DataType data_type, size_t start, size_t end, bool clamp) override;
        void writeSlice(const void* input, size4_t strides, size4_t shape, DataType data_type, size_t start, bool clamp) override;
        void writeSlice(const void* input, DataType data_type, size_t start, size_t end, bool clamp) override;
        void writeAll(const void* input, size4_t strides, size4_t shape, DataType data_type, bool clamp) override;
        void writeAll(const void* input, DataType data_type, bool clamp) override;

    private:
        void open_(const path_t& filename, open_mode_t mode);

        // Reads and checks the header of an existing file.
        // Throws if the header doesn't look like a MRC header or if the MRC file is not supported.
        // If read|write mode, the header is saved in m_header.buffer. This will be used before closing the file.
        void readHeader_(const path_t& filename);

        // Swap the endianness of the header.
        // The buffer is at least the first 224 bytes of the MRC header.
        //
        // In read or read|write mode, the data of the header should be swapped if the endianness of the file is
        // swapped. This function should be called just after reading the header AND just before writing it.
        // All used flags are swapped. Some unused flags are left unchanged.
        static void swapHeader_(byte_t* buffer) {
            io::swapEndian(buffer, 24, 4); // from 0 (nx) to 96 (next, included).
            io::swapEndian(buffer + 152, 2, 4); // imodStamp, imodFlags
            io::swapEndian(buffer + 216, 2, 4); // rms, nlabl
        }

        // Closes the stream. Separate function so that the destructor can call close().
        // In writing mode, the header flags will be writing to the file's header.
        void close_();

        // Sets the header to default values.
        // This function should only be called to initialize the header before closing a (overwritten or new) file.
        static void defaultHeader_(byte_t* buffer);

        // Writes the header to the file's header. Only called before closing a file.
        // Buffer containing the header. Only the used values (see m_header) are going to be modified before write
        // buffer to the file. As such, unused values should either be set to the defaults (overwrite mode, see
        // defaultHeader_()) OR taken from an existing file (read|write mode, see readHeader_()).
        void writeHeader_(byte_t* buffer);

        // Gets the offset to the data.
        [[nodiscard]] constexpr long offset_() const noexcept {
            return 1024 + m_header.extended_bytes_nb;
        }

        // This is to set the default data type of the file in the first write operation of a new file in writing mode.
        // data_type is the type of the real data that is passed in, and we return the "closest" supported type.
        static DataType closestSupportedDataType_(DataType data_type);

    private:
        std::fstream m_fstream{};
        path_t m_filename{};
        open_mode_t m_open_mode{};

        struct HeaderImp {
            // Buffer containing the 1024 bytes of the header.
            // Only used if the header needs to be saved, that is, in in|out mode.
            std::unique_ptr<byte_t[]> buffer{nullptr};
            DataType data_type{DataType::DTYPE_UNKNOWN};

            size4_t shape{0};               // BDHW order.
            float3_t pixel_size{0.f};       // Pixel spacing (DHW order) = cell_size / shape.

            float min{0};                   // Minimum pixel value.
            float max{-1};                  // Maximum pixel value.
            float mean{-2};                 // Mean pixel value.
            float std{-1};                  // Stdev. Negative if not computed.

            int32_t extended_bytes_nb{0};   // Number of bytes in extended header.

            bool is_endian_swapped{false};  // Whether the endianness of the data is swapped.
            int32_t nb_labels{0};           // Number of labels with useful data.
        } m_header{};
    };
}
