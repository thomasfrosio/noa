/**
 * @file MRCHeader.h
 * @brief
 * @author Thomas - ffyr2w
 * @date 30/10/2020
 */
#pragma once

#include "noa/Base.h"
#include "noa/util/OS.h"
#include "noa/util/Arrays.h"
#include "noa/files/headers/Header.h"


namespace Noa::Header {

    /** Handles the MRC file format. */
    class NOA_API MRCHeader : public Header {
    private:
        char* m_data; /// The underlying data.
        static constexpr int m_mrc_header_size = 1024;

        // Reinterpretation of the underlying header @a m_data.
        Int3<int>* m_shape{};           // Number of columns (x), rows (y) and sections (z).
        int* m_mode{};                  // Types of pixel in image. See MRCHeader::Type.
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
        /** Default constructor. */
        inline MRCHeader() : Header(), m_data(new char[m_mrc_header_size]) { link_(); }


        /** Copy constructor. */
        inline MRCHeader(const MRCHeader& to_copy)
                : Header(to_copy), m_data(new char[m_mrc_header_size]) {
            std::memcpy(m_data, to_copy.m_data, m_mrc_header_size);
            link_();
        }


        /** Move constructor. */
        inline MRCHeader(MRCHeader&& to_move) noexcept
                : Header(to_move), m_data(std::exchange(to_move.m_data, nullptr)) {
            link_();
        }


        /** Copy operator assignment. */
        inline MRCHeader& operator=(const MRCHeader& to_copy) noexcept {
            if (this != &to_copy) {
                m_layout = to_copy.m_layout;
                m_is_big_endian = to_copy.m_is_big_endian;
                std::memcpy(m_data, to_copy.m_data, m_mrc_header_size);
            }
            return *this;
        }


        /** Move operator assignment. */
        inline MRCHeader& operator=(MRCHeader&& to_move) noexcept {
            if (this != &to_move) {
                m_layout = to_move.m_layout;
                m_is_big_endian = to_move.m_is_big_endian;
                delete[] m_data;
                m_data = std::exchange(to_move.m_data, nullptr);
                link_();
            }
            return *this;
        }


        /** Destructor. */
        inline ~MRCHeader() override { delete[] m_data; }


        /** Reads in and validates the header from a file. The file stream should be opened. */
        inline errno_t read(std::fstream& fstream) override {
            fstream.seekg(0);
            fstream.read(m_data, m_mrc_header_size);
            if (fstream.fail())
                return Errno::fail_read;
            return validate_();
        }


        /** Writes the header to a file. The file stream should be opened. */
        inline errno_t write(std::fstream& fstream) override {
            fstream.seekp(0);
            fstream.write(m_data, m_mrc_header_size);
            return fstream.fail() ? Errno::fail_write : Errno::good;
        }


        /** (Re)set default values. Sets the endianness of the local machine. */
        void reset() override;


        /** Gets the shape by value */
        [[nodiscard]] inline Int3<size_t> getShape() const override {
            return Int3<size_t>(*m_shape);
        }


        /** Sets the shape - must be positive values */
        inline errno_t setShape(Int3<size_t> new_shape) override {
            *m_shape = new_shape;
            return Errno::good;
        }


        /** Gets the pixel size (in x, y and z) by value */
        [[nodiscard]] inline Float3<float> getPixelSize() const override {
            return *m_shape_cell / toFloat3(*m_shape_grid);
        }


        /** Sets the shape - must be positive values */
        inline errno_t setPixelSize(Float3<float> new_pixel_size) override {
            if (new_pixel_size < 0)
                return Errno::invalid_argument;
            *m_shape_cell = toFloat3(*m_shape_grid) * new_pixel_size;
            return Errno::good;
        }


        /** Gets the offset to the data: header size (1024) + the extended header. */
        [[nodiscard]] inline size_t getOffset() const override {
            return static_cast<size_t>(m_mrc_header_size) +
                   static_cast<size_t>(*m_extended_bytes_nb);
        }


        errno_t setLayout(iolayout_t layout) override;

        void setEndianness(bool big_endian) override;

        [[nodiscard]] std::string toString(bool brief) const override;

        inline void setStatistics(float min, float max, float mean, float rms) {
            *m_min = min;
            *m_max = max;
            *m_mean = mean;
            *m_rms = rms;
        }

    private:
        /** Links the underlying buffer to the higher level pointers. */
        void link_();


        /**
         * Checks that the header complies to the MRC header standard.
         * @note    This is a brief validation and only checks the basics that are likely to be
         *          used systematically: shapes, layout, offset and endianness.
         * @note    @a m_is_big_endian is updated.
         * @return  @c Errno::invalid_data, if the header doesn't look like a MRC header.
         *          @c Errno::not_supported, if the MRC file is not supported.
         *          @c Errno::good, otherwise.
         */
        [[nodiscard]] errno_t validate_();
    };
}
