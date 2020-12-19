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
#include "noa/util/Vectors.h"

#include "noa/files/AbstractImageFile.h"


namespace Noa {
    class MRCFile : public AbstractImageFile {
    private:
        using openmode_t = std::ios_base::openmode;
        std::unique_ptr<std::fstream> m_fstream;
        std::unique_ptr<char> m_header;
        std::ios_base::openmode m_open_mode{};

        IO::DataType m_data_type{};
        bool m_is_endian_swapped{};

        // Reinterpretation of the underlying header @a m_header.
        static constexpr int m_header_size = 1024;
        Int3<int>* m_shape{};           // Number of columns (x), rows (y) and sections (z).
        int* m_mode{};                  // Types of pixel in image. See MRCFile::Type.
        Int3<int>* m_shape_sub{};       // Starting point of sub image (x, y and z).
        Int3<int>* m_shape_grid{};      // Grid size (x, y and z).
        Float3<float>* m_shape_cell{};  // Cell size. Pixel spacing (x, y and z) = cell_size / size.
        Float3<float>* m_angles{};      // Cell angles: alpha, beta, gamma.
        Int3<int>* m_map_order{};       // Map order (x, y and z).

        float* m_min{};                 // Minimum pixel value.
        float* m_max{};                 // Maximum pixel value.
        float* m_mean{};                // Mean pixel value.

        int* m_space_group{};           // Space group number: 0 = stack, 1 = volume.
        int* m_extended_bytes_nb{};     // Number of bytes in extended header.
        char* m_extra00{};              // Not used. creator_id + extra = 8 bytes
        char* m_extended_type{};        // Type of extended header. "SERI", "FEI1", "AGAR", "CCP4" or "MRCO"
        int* m_nversion{};              // MRC version

        int* m_imod_stamp{};            // 1146047817 indicates IMOD flags are used.
        int* m_imod_flags{};            // Bit flags. 1=signed, the rest is ignored.

        Float3<float>* m_origin{};      // Origin of image.
        char* m_cmap{};                 // Not used.
        char* m_stamp{};                // First 2 bytes have 17 and 17 for big-endian or 68 and 65 for little-endian.
        float* m_rms{};                 // Stddev of densities from mean. Negative if not computed.
        int* m_nb_labels{};             // Number of labels with useful data.
        char* m_labels{};               // 10 labels of 80 characters, blank-padded to end.

    public:
        /** Allocates the file stream and file header. */
        inline MRCFile()
                : AbstractImageFile(),
                  m_fstream(std::make_unique<std::fstream>()),
                  m_header(std::make_unique<char>(m_header_size)) { syncHeader_(); }


        /** Stores the path and allocates the file stream and file header. */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        explicit inline MRCFile(T&& path)
                : AbstractImageFile(std::forward<T>(path)),
                  m_fstream(std::make_unique<std::fstream>()),
                  m_header(std::make_unique<char>(m_header_size)) { syncHeader_(); }


        /** Move constructor. */
        inline MRCFile(MRCFile&& file) noexcept
                : AbstractImageFile(std::move(file.m_path)),
                  m_fstream(std::move(file.m_fstream)),
                  m_header(std::move(file.m_header)),
                  m_open_mode(file.m_open_mode),
                  m_data_type(file.m_data_type),
                  m_is_endian_swapped(file.m_is_endian_swapped) { syncHeader_(); }


        /** Move operator assignment. */
        inline MRCFile& operator=(MRCFile&& file) noexcept {
            if (this != &file) {
                m_path = std::move(file.m_path);
                m_state = file.m_state;
                m_fstream = std::move(file.m_fstream);
                m_header = std::move(file.m_header);
                m_open_mode = file.m_open_mode;
                m_data_type = file.m_data_type;
                m_is_endian_swapped = file.m_is_endian_swapped;
                syncHeader_();
            }
            return *this;
        }

        ~MRCFile() override {
            close_();
        }

        /** See the corresponding virtual function in @a AbstractImageFile. */
        errno_t open(openmode_t mode, bool wait) override;

        /** See the corresponding virtual function in @a AbstractImageFile. */
        inline errno_t open(const fs::path& path, openmode_t mode, bool wait) override {
            m_path = path;
            return open(mode, wait);
        }

        /** See the corresponding virtual function in @a AbstractImageFile. */
        inline errno_t open(fs::path&& path, openmode_t mode, bool wait) override {
            m_path = std::move(path);
            return open(mode, wait);
        }


        [[nodiscard]] inline bool isOpen() const override { return m_fstream->is_open(); }


        /** See the corresponding virtual function in @a AbstractImageFile. */
        inline errno_t close() override { return close_(); }


        errno_t readAll(float* data) override;
        errno_t readSlice(float* data, size_t z_pos, size_t z_count) override;
        errno_t writeAll(float* data) override;
        errno_t writeSlice(float* data, size_t z_pos, size_t z_count) override;


        /** See the corresponding virtual function in @a AbstractImageFile. */
        [[nodiscard]] inline Int3<size_t> getShape() const override {
            return Int3<size_t>(*m_shape);
        }


        /** See the corresponding virtual function in @a AbstractImageFile. */
        inline errno_t setShape(Int3<size_t> new_shape) override {
            *m_shape = new_shape;
            return m_state;
        }


        /** See the corresponding virtual function in @a AbstractImageFile. */
        [[nodiscard]] inline Float3<float> getPixelSize() const override {
            return *m_shape_cell / Float3(m_shape_grid->data());
        }


        /** See the corresponding virtual function in @a AbstractImageFile. */
        inline errno_t setPixelSize(Float3<float> new_pixel_size) override {
            if (new_pixel_size > 0)
                *m_shape_cell = Float3(m_shape_grid->data()) * new_pixel_size;
            else
                Errno::set(m_state, Errno::invalid_argument);
            return m_state;
        }


        /** See the corresponding virtual function in @a AbstractImageFile. */
        [[nodiscard]] std::string toString(bool brief) const override;


        /** Sets the statistics in the header. */
        inline errno_t setStatistics(float min, float max, float mean, float rms) {
            *m_min = min;
            *m_max = max;
            *m_mean = mean;
            *m_rms = rms;
            return m_state;
        }

    private:
        errno_t close_();

        /** Links the underlying buffer to the higher level pointers. */
        void syncHeader_();


        /** Reads and validates the header from a file. The file stream should be opened. */
        inline void readHeader_() {
            m_fstream->seekg(0);
            m_fstream->read(m_header.get(), m_header_size);
            if (m_fstream->fail())
                Errno::set(m_state, Errno::fail_read);
            validate_();
        }


        /** Writes the header to a file. The file stream should be opened. */
        inline void writeHeader_() {
            m_fstream->seekp(0);
            m_fstream->write(m_header.get(), m_header_size);
            if (m_fstream->fail())
                Errno::set(m_state, Errno::fail_write);
        }


        /**
         * Sets the default values.
         * @warning This function should only be called to initialize a new file header, not to
         *          overwrite an existing one.
         */
        void initHeader_();


        /**
         * Checks that the header complies to the MRC header standard.
         * @warning This is a brief validation and only checks the basics that are likely to be
         *          used systematically: shapes, layout, offset and endianness.
         * @return  @c Errno::invalid_data, if the header doesn't look like a MRC header.
         *          @c Errno::not_supported, if the MRC file is not supported.
         *          @c Errno::good, otherwise.
         */
        errno_t validate_();


        /** Gets the offset to the data: header size (1024) + the extended header. */
        [[nodiscard]] inline long getOffset_() const {
            return m_header_size + *m_extended_bytes_nb;
        }


        errno_t setDataType_(IO::DataType layout);


        void setEndianness_();
    };
}
