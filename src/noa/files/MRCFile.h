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
        static constexpr int m_header_size = 1024;

        std::unique_ptr<std::fstream> m_fstream;
        std::unique_ptr<char> m_buffer;
        std::ios_base::openmode m_open_mode{};

        IO::DataType m_data_type{};
        bool m_is_endian_swapped{};

        // Reinterpretation of the underlying buffer.
        struct Header {
            Int3<int>* shape{};           // Number of columns (x), rows (y) and sections (z).
            int* mode{};                  // Types of pixel in image. See MRCFile::Type.
            Int3<int>* shape_sub{};       // Starting point of sub image (x, y and z).
            Int3<int>* shape_grid{};      // Grid size (x, y and z).
            Float3<float>* shape_cell{};  // Cell size. Pixel spacing (x, y and z) = cell_size / size.
            Float3<float>* angles{};      // Cell angles: alpha, beta, gamma.
            Int3<int>* map_order{};       // Map order (x, y and z).

            float* min{};                 // Minimum pixel value.
            float* max{};                 // Maximum pixel value.
            float* mean{};                // Mean pixel value.

            int* space_group{};           // Space group number: 0 = stack, 1 = volume.
            int* extended_bytes_nb{};     // Number of bytes in extended header.
            char* extra00{};              // Not used. creator_id + extra = 8 bytes
            char* extended_type{};        // Type of extended header. "SERI", "FEI1", "AGAR", "CCP4" or "MRCO"
            int* nversion{};              // MRC version

            int* imod_stamp{};            // 1146047817 indicates IMOD flags are used.
            int* imod_flags{};            // Bit flags. 1=signed, the rest is ignored.

            Float3<float>* origin{};      // Origin of image.
            char* cmap{};                 // Not used.
            char* stamp{};                // First 2 bytes have 17 and 17 for big-endian or 68 and 65 for little-endian.
            float* rms{};                 // Stddev of densities from mean. Negative if not computed.
            int* nb_labels{};             // Number of labels with useful data.
            char* labels{};               // 10 labels of 80 characters, blank-padded to end.
        } m_header{};


    public:
        /** Allocates the file stream and file header. */
        inline MRCFile()
                : AbstractImageFile(),
                  m_fstream(std::make_unique<std::fstream>()),
                  m_buffer(std::make_unique<char>(m_header_size)) { syncHeader_(); }


        /** Stores the path and allocates the file stream and file header. */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        explicit inline MRCFile(T&& path)
                : AbstractImageFile(std::forward<T>(path)),
                  m_fstream(std::make_unique<std::fstream>()),
                  m_buffer(std::make_unique<char>(m_header_size)) { syncHeader_(); }


        /** Stores the path and allocates the file stream and file header. */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        explicit inline MRCFile(T&& path, openmode_t mode, bool wait = false)
                : AbstractImageFile(std::forward<T>(path)),
                  m_fstream(std::make_unique<std::fstream>()),
                  m_buffer(std::make_unique<char>(m_header_size)) {
            syncHeader_();
            open_(mode, wait);
        }


        /** Move constructor. */
        inline MRCFile(MRCFile&& file) noexcept
                : AbstractImageFile(std::move(file.m_path)),
                  m_fstream(std::move(file.m_fstream)),
                  m_buffer(std::move(file.m_buffer)),
                  m_open_mode(file.m_open_mode),
                  m_data_type(file.m_data_type),
                  m_is_endian_swapped(file.m_is_endian_swapped) { syncHeader_(); }


        /** Move operator assignment. */
        inline MRCFile& operator=(MRCFile&& file) noexcept {
            if (this != &file) {
                m_path = std::move(file.m_path);
                m_state = file.m_state;
                m_fstream = std::move(file.m_fstream);
                m_buffer = std::move(file.m_buffer);
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
        inline errno_t open(openmode_t mode, bool wait) override {
            return open_(mode, wait);
        }


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


        /** See the corresponding virtual function in @a AbstractImageFile. */
        inline errno_t open(openmode_t mode) {
            return open_(mode, false);
        }


        /** See the corresponding virtual function in @a AbstractImageFile. */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, fs::path>>>
        inline errno_t open(T&& path, openmode_t mode) {
            m_path = std::forward<T>(path);
            return open_(mode, false);
        }


        [[nodiscard]] inline bool isOpen() const override { return m_fstream->is_open(); }


        /** See the corresponding virtual function in @a AbstractImageFile. */
        inline errno_t close() override { return close_(); }

        /** Whether or not the instance is in a "good" state. */
        [[nodiscard]] inline explicit operator bool() const noexcept override {
            return !m_state && !m_fstream->fail();
        }


        errno_t readAll(float* data) override;
        errno_t readSlice(float* data, size_t z_pos, size_t z_count) override;

        errno_t setDataType(IO::DataType) override;
        errno_t writeAll(float* data) override;
        errno_t writeSlice(float* data, size_t z_pos, size_t z_count) override;


        /** See the corresponding virtual function in @a AbstractImageFile. */
        [[nodiscard]] inline Int3<size_t> getShape() const override {
            return Int3<size_t>(*m_header.shape);
        }


        /** See the corresponding virtual function in @a AbstractImageFile. */
        inline errno_t setShape(Int3<size_t> new_shape) override {
            *m_header.shape = new_shape;
            return m_state;
        }


        /** See the corresponding virtual function in @a AbstractImageFile. */
        [[nodiscard]] inline Float3<float> getPixelSize() const override {
            return *m_header.shape_cell / Float3(m_header.shape_grid->data());
        }


        /** See the corresponding virtual function in @a AbstractImageFile. */
        inline errno_t setPixelSize(Float3<float> new_pixel_size) override {
            if (new_pixel_size > 0)
                *m_header.shape_cell = Float3(m_header.shape_grid->data()) * new_pixel_size;
            else
                Errno::set(m_state, Errno::invalid_argument);
            return m_state;
        }


        /** See the corresponding virtual function in @a AbstractImageFile. */
        [[nodiscard]] std::string toString(bool brief) const override;


        /** Sets the statistics in the header. */
        inline errno_t setStatistics(float min, float max, float mean, float rms) {
            *m_header.min = min;
            *m_header.max = max;
            *m_header.mean = mean;
            *m_header.rms = rms;
            return m_state;
        }

    private:
        /**
         * Links the underlying buffer to the higher level pointers.
         * Called by constructors or move assignment operator.
         */
        void syncHeader_();

        errno_t open_(openmode_t mode, bool wait);


        /**
         * Reads and validates the header from a file. The file ifstream should be opened.
         * Only called after opening an existing file.
         */
        inline void readHeader_() {
            m_fstream->seekg(0);
            m_fstream->read(m_buffer.get(), m_header_size);
            if (m_fstream->fail())
                Errno::set(m_state, Errno::fail_read);
            else
                validate_();
        }


        /**
         * Checks that the header complies to (some of) the MRC header standard.
         * @warning This is a brief validation and only checks the basics that are likely to be
         *          used systematically: shapes, layout, offset and endianness.
         * @return  @c Errno::invalid_data, if the header doesn't look like a MRC header.
         *          @c Errno::not_supported, if the MRC file is not supported.
         *          @c Errno::good, otherwise.
         */
        errno_t validate_();


        /**
         * Sets the default values.
         * Only called after opening a (overwritten or new) file.
         * @warning This function should only be called to initialize a new file header, not to
         *          overwrite an existing one.
         */
        void initHeader_();


        /**
         * Sets the endianness of the machine into the file header.
         * No data conversion performed, which is fine since it is only called to initialize new files.
         */
        void setEndianness_();


        /** Closes the stream. Separate function so that the destructor can call close(). */
        errno_t close_();


        /**
         * Writes the header to a file. The file stream should be opened.
         * Only called before closing a file.
         */
        inline void writeHeader_() {
            m_fstream->seekp(0);
            m_fstream->write(m_buffer.get(), m_header_size);
            if (m_fstream->fail())
                Errno::set(m_state, Errno::fail_write);
        }


        /** Gets the offset to the data: header size (1024) + the extended header. */
        [[nodiscard]] inline long getOffset_() const {
            return m_header_size + *m_header.extended_bytes_nb;
        }
    };
}
