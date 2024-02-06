#pragma once

#include "noa/core/Config.hpp"

#if defined(NOA_IS_OFFLINE)
#include <fstream>
#include "noa/core/Traits.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/io/IO.hpp"
#include "noa/core/io/Stats.hpp"
#include "noa/core/io/ImageFile.hpp"

namespace noa::io {
    // File supporting the MRC format.
    // Notes about the current implementation:
    //  - Pixel size:   If the cell size is equal to zero, the pixel size will be 0. This is allowed
    //                  since in some cases the pixel size is ignored. Thus, one should check the
    //                  returned value of pixel_size() before using it. Similarly, set_pixel_size()
    //                  can save a pixel size equal to 0, which is the default if a new file is closed
    //                  without calling set_pixel_size().
    //  - Endianness:   It is not possible to change the endianness of existing data. As such, in the
    //                  rare case of writing to an existing file (i.e. READ|WRITE mode), the written
    //                  data may have to be swapped to match the file's endianness.
    //  - Data type:    In the rare case of writing to an existing file (i.e. READ|WRITE mode), the data
    //                  type cannot be changed. In other cases, the user should only set the data
    //                  type once before writing anything to the file.
    //  - Order:        The map ordering should be mapc=1, mapr=2 and maps=3. Anything else is not
    //                  supported and an exception is thrown when opening a file with a different ordering.
    //  - Unused flags: The extended header, the origin (xorg, yorg, zorg), nversion and other
    //                  parts of the header are ignored (see full detail on what is ignored in
    //                  write_header_()). These are set to 0 or to the expected default value, or are
    //                  left unchanged in non-overwriting mode.
    //
    // see      https://bio3d.colorado.edu/imod/doc/mrc_format.txt or
    //          https://www.ccpem.ac.uk/mrc_format/mrc2014.php
    class MRCFile : public guts::ImageFile {
    public:
        MRCFile() = default;
        MRCFile(const Path& path, OpenMode open_mode) { open_(path, open_mode); }
        ~MRCFile() override { close_(); }

    public:
        void reset() override {
            close();
            m_open_mode = OpenMode{};
            m_header = HeaderData{};
        };

        void open(const Path& path, OpenMode open_mode) override { open_(path, open_mode); }
        void close() override { close_(); }

    public:
        [[nodiscard]] bool is_open() const noexcept override { return m_fstream.is_open(); }
        [[nodiscard]] const Path& filename() const noexcept override { return m_filename; }
        [[nodiscard]] std::string info_string(bool brief) const noexcept override;
        [[nodiscard]] Format format() const noexcept override { return Format::MRC; }

    public:
        [[nodiscard]] Shape4<i64> shape() const noexcept override {
            return m_header.shape;
        }

        void set_shape(const Shape4<i64>& new_shape) override;

        [[nodiscard]] Stats<f32> stats() const noexcept override {
            Stats<f32> out;
            if (m_header.min != 0 || m_header.max != -1 || m_header.mean != -2) {
                out.set_min(m_header.min);
                out.set_max(m_header.max);
                out.set_mean(m_header.mean);
            }
            if (m_header.std >= 0)
                out.set_std(m_header.std);
            return out;
        }

        void set_stats(Stats<f32> stats) override {
            // In reading mode, this will have no effect.
            if (stats.has_min())
                m_header.min = stats.min();
            if (stats.has_max())
                m_header.max = stats.max();
            if (stats.has_mean())
                m_header.mean = stats.mean();
            if (stats.has_std())
                m_header.std = stats.std();
        }

        [[nodiscard]] Vec3<f32> pixel_size() const noexcept override {
            return m_header.pixel_size;
        }

        void set_pixel_size(Vec3<f32> new_pixel_size) override;

        [[nodiscard]] DataType dtype() const noexcept override {
            return m_header.data_type;
        }

        void set_dtype(DataType data_type) override;

    public:
        void read_elements(void* output, DataType data_type, i64 start, i64 end, bool clamp) override;
        void read_slice(void* output, const Strides4<i64>& strides, const Shape4<i64>& shape, DataType data_type, i64 start, bool clamp) override;
        void read_slice(void* output, DataType data_type, i64 start, i64 end, bool clamp) override;
        void read_all(void* output, const Strides4<i64>& strides, const Shape4<i64>& shape, DataType data_type, bool clamp) override;
        void read_all(void* output, DataType data_type, bool clamp) override;

        void write_elements(const void* input, DataType data_type, i64 start, i64 end, bool clamp) override;
        void write_slice(const void* input, const Strides4<i64>& strides, const Shape4<i64>& shape, DataType data_type, i64 start, bool clamp) override;
        void write_slice(const void* input, DataType data_type, i64 start, i64 end, bool clamp) override;
        void write_all(const void* input, const Strides4<i64>& strides, const Shape4<i64>& shape, DataType data_type, bool clamp) override;
        void write_all(const void* input, DataType data_type, bool clamp) override;

    public:
        template<typename T>
        void read_elements(T* output, i64 start, i64 end, bool clamp = false) {
            read_elements(output, noa::io::dtype<T>(), start, end, clamp);
        }

        template<typename T>
        void read_all(T* output, bool clamp = false) {
            read_all(output, noa::io::dtype<T>(), clamp);
        }

        template<typename T>
        void read_slice(T* output, i64 start, i64 end, bool clamp = false) {
            read_slice(output, noa::io::dtype<T>(), start, end, clamp);
        }

        template<typename T>
        void write_all(T* output, bool clamp = false) {
            write_all(output, noa::io::dtype<T>(), clamp);
        }

        template<typename T>
        void write_slice(T* output, i64 start, i64 end, bool clamp = false) {
            write_slice(output, noa::io::dtype<T>(), start, end, clamp);
        }

        [[nodiscard]] explicit operator bool() const noexcept { return is_open(); }

    private:
        void open_(const Path& filename, OpenMode mode,
                   const std::source_location& location = std::source_location::current());

        // Reads and checks the header of an existing file.
        // Throws if the header doesn't look like a MRC header or if the MRC file is not supported.
        // If read|write mode, the header is saved in m_header.buffer. This will be used before closing the file.
        void read_header_(const Path& filename);

        // Swap the endianness of the header.
        // The buffer is at least the first 224 bytes of the MRC header.
        //
        // In read or read|write mode, the data of the header should be swapped if the endianness of the file is
        // swapped. This function should be called just after reading the header AND just before writing it.
        // All used flags are swapped. Some unused flags are left unchanged.
        static void swap_header_(Byte* buffer) {
            swap_endian(buffer, 24, 4); // from 0 (nx) to 96 (next, included).
            swap_endian(buffer + 152, 2, 4); // imodStamp, imodFlags
            swap_endian(buffer + 216, 2, 4); // rms, nlabl
        }

        // Closes the stream. Separate function so that the destructor can call close().
        // In writing mode, the header flags will be writing to the file's header.
        void close_();

        // Sets the header to default values.
        // This function should only be called to initialize the header before closing a (overwritten or new) file.
        static void default_header_(Byte* buffer);

        // Writes the header to the file's header. Only called before closing a file.
        // Buffer containing the header. Only the used values (see m_header) are going to be modified before write
        // buffer to the file. As such, unused values should either be set to the defaults (overwrite mode, see
        // default_header_()) OR taken from an existing file (read|write mode, see readHeader_()).
        void write_header_(Byte* buffer);

        // Gets the offset to the data.
        [[nodiscard]] constexpr i64 header_offset_() const noexcept {
            return 1024 + m_header.extended_bytes_nb;
        }

        // This is to set the default data type of the file in the first write operation of a new file in writing mode.
        // data_type is the type of the real data that is passed in, and we return the "closest" supported type.
        static DataType closest_supported_dtype_(DataType data_type);

    private:
        std::fstream m_fstream{};
        Path m_filename{};
        OpenMode m_open_mode{};

        struct HeaderData {
            // Buffer containing the 1024 bytes of the header.
            // Only used if the header needs to be saved, that is, in "in|out" mode.
            std::unique_ptr<Byte[]> buffer{nullptr};
            DataType data_type{DataType::UNKNOWN};

            Shape4<i64> shape{};            // BDHW order.
            Vec3<f32> pixel_size{0.f};      // Pixel spacing (DHW order) = cell_size / shape.

            f32 min{0};                     // Minimum pixel value.
            f32 max{-1};                    // Maximum pixel value.
            f32 mean{-2};                   // Mean pixel value.
            f32 std{-1};                    // Stdev. Negative if not computed.

            i32 extended_bytes_nb{0};       // Number of bytes in extended header.

            bool is_endian_swapped{false};  // Whether the endianness of the data is swapped.
            i32 nb_labels{0};               // Number of labels with useful data.
        } m_header{};
    };
}
#endif
