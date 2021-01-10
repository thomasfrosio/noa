/**
 * @file MRCFile.h
 * @brief MRCFile class.
 * @author Thomas - ffyr2w
 * @date 19 Dec 2020
 */
#pragma once

#include "noa/Base.h"
#include "noa/util/IO.h"
#include "noa/util/OS.h"
#include "noa/util/IntX.h"
#include "noa/util/FloatX.h"

#include "noa/files/AbstractImageFile.h"


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
     *  - Ignored bits: The extended header, the origin (xorg, yorg, zorg), nversion and other
     *                  parts of the header are ignored (see full detail on what is ignored in
     *                  writeHeader_()). These are set to 0 or to the expected default value, or are
     *                  left unchanged in non-overwriting mode.
     *
     * @see     https://bio3d.colorado.edu/imod/doc/mrc_format.txt or
     *          https://www.ccpem.ac.uk/mrc_format/mrc2014.php
     *
     * @note    29/12/20 - TF: The previous implementation of the header was based on a
     *          reinterpretation of the serialized data (a buffer of char[1024]). You can find such
     *          implementation in the MRCFile implementation in cisTEM. Turns out, this break the
     *          strict aliasing rule and is therefore UB. In this case, I decided to follow the
     *          standard and change the implementation. This should be "defined behavior" now.
     */
    class MRCFile : public AbstractImageFile {
    private:
        using openmode_t = std::ios_base::openmode;
        std::unique_ptr<std::fstream> m_fstream;
        openmode_t m_open_mode{};

        struct Header {
            std::unique_ptr<char[]> buffer{std::make_unique<char[]>(1024)};

            IO::DataType data_type{IO::DataType::float32};  // Data type.
            Int3<int32_t> shape{0};         // Number of columns (x), rows (y) and sections (z).
            Float3<float> pixel_size{0.f};  // Pixel spacing (x, y and z) = cell_size / shape.

            float min{0};                   // Minimum pixel value.
            float max{-1};                  // Maximum pixel value.
            float mean{-2};                 // Mean pixel value.
            float rms{0};                   // Std of densities from mean. Negative if not computed.

            int32_t extended_bytes_nb{0};   // Number of bytes in extended header.

            bool is_endian_swapped{false};  // Whether or not the endianness of the data is swapped.
            int32_t nb_labels{0};           // Number of labels with useful data.
        } m_header{};

    public:
        /** Allocates the file stream and file header. */
        inline MRCFile()
                : AbstractImageFile(),
                  m_fstream(std::make_unique<std::fstream>()) {}


        /** Stores the path and allocates the file stream and file header. */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        explicit inline MRCFile(T&& path)
                : AbstractImageFile(std::forward<T>(path)),
                  m_fstream(std::make_unique<std::fstream>()) {}


        /** Stores the path and allocates the file stream and file header. */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        explicit inline MRCFile(T&& path, openmode_t mode, bool wait = false)
                : AbstractImageFile(std::forward<T>(path)),
                  m_fstream(std::make_unique<std::fstream>()) { open_(mode, wait); }


        ~MRCFile() override {
            close_();
        }


        /** See the corresponding virtual function in @a AbstractImageFile. */
        inline Flag<Errno> open(openmode_t mode, bool wait) override {
            return open_(mode, wait);
        }


        /** See the corresponding virtual function in @a AbstractImageFile. */
        inline Flag<Errno> open(const fs::path& path, openmode_t mode, bool wait) override {
            m_path = path;
            return open_(mode, wait);
        }


        /** See the corresponding virtual function in @a AbstractImageFile. */
        inline Flag<Errno> open(fs::path&& path, openmode_t mode, bool wait) override {
            m_path = std::move(path);
            return open_(mode, wait);
        }


        /** See the corresponding virtual function in @a AbstractImageFile. */
        inline Flag<Errno> open(openmode_t mode) {
            return open_(mode, false);
        }


        /** See the corresponding virtual function in @a AbstractImageFile. */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, fs::path>>>
        inline Flag<Errno> open(T&& path, openmode_t mode) {
            m_path = std::forward<T>(path);
            return open_(mode, false);
        }


        [[nodiscard]] inline bool isOpen() const override { return m_fstream->is_open(); }


        /** See the corresponding virtual function in @a AbstractImageFile. */
        inline Flag<Errno> close() override { return close_(); }


        /** Whether or not the instance is in a "good" state. */
        [[nodiscard]] inline explicit operator bool() const noexcept override {
            return !m_state && !m_fstream->fail();
        }


        Flag<Errno> readAll(float* data) override;
        Flag<Errno> readSlice(float* data, size_t z_pos, size_t z_count) override;

        Flag<Errno> setDataType(IO::DataType) override;
        Flag<Errno> writeAll(float* data) override;
        Flag<Errno> writeSlice(float* data, size_t z_pos, size_t z_count) override;


        /** See the corresponding virtual function in @a AbstractImageFile. */
        [[nodiscard]] inline Int3<size_t> getShape() const override {
            return Int3<size_t>(m_header.shape);
        }


        /** See the corresponding virtual function in @a AbstractImageFile. */
        inline Flag<Errno> setShape(Int3<size_t> new_shape) override {
            m_header.shape = new_shape;
            return m_state;
        }


        /** See the corresponding virtual function in @a AbstractImageFile. */
        [[nodiscard]] inline Float3<float> getPixelSize() const override {
            return m_header.pixel_size;
        }


        /** See the corresponding virtual function in @a AbstractImageFile. */
        inline Flag<Errno> setPixelSize(Float3<float> new_pixel_size) override {
            if (new_pixel_size > 0)
                m_header.pixel_size = new_pixel_size;
            else
                m_state.update(Errno::invalid_argument);
            return m_state;
        }


        /** See the corresponding virtual function in @a AbstractImageFile. */
        [[nodiscard]] std::string toString(bool brief) const override;


        /** Sets the statistics in the header. */
        inline Flag<Errno> setStatistics(float min, float max, float mean, float rms) {
            m_header.min = min;
            m_header.max = max;
            m_header.mean = mean;
            m_header.rms = rms;
            return m_state;
        }

    private:
        Flag<Errno> open_(openmode_t mode, bool wait);


        /**
         * Reads and checks the header of an existing file.
         * @note    The state can be changed such as:
         *          @c Errno::invalid_data, if the header doesn't look like a MRC header.
         *          @c Errno::not_supported, if the MRC file is not supported.
         *          @c Errno::good, otherwise.
         */
        Flag<Errno> readHeader_();


        /**
         * Sets the header to default values.
         * @warning This function should only be called to initialize the header after opening a
         *          (overwritten or new) file.
         */
        void initHeader_() const;


        /** Closes the stream. Separate function so that the destructor can call close(). */
        Flag<Errno> close_();


        /** Writes the header to a file. Only called before closing a file. */
        void writeHeader_();


        /** Gets the offset to the data: header size (1024) + the extended header. */
        [[nodiscard]] inline long getOffset_() const {
            return 1024 + m_header.extended_bytes_nb;
        }
    };
}
