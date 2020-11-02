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
        static constexpr int m_header_size = 1024;

        // Reinterpretation of the underlying header @a m_data.
        Int3<int>* shape{};           /// Number of columns (x), rows (y) and sections (z).
        int* mode{};                  /// Types of pixel in image. See MRCHeader::Type.
        Int3<int>* shape_sub{};       /// Starting point of sub image (x, y and z).
        Int3<int>* shape_grid{};      /// Grid size (x, y and z).
        Float3<float>* shape_cell{};  /// Cell size. Pixel spacing (x, y and z) = cell_size / size.
        Float3<float>* angles{};      /// Cell angles: alpha, beta, gamma.
        Int3<int>* map_order{};       /// Map order (x, y and z).

        float* min{};                 /// Minimum pixel value.
        float* max{};                 /// Maximum pixel value.
        float* mean{};                /// Mean pixel value.

        int* space_group{};           /// Space group number: 0 = stack, 1 = volume.
        int* extended_bytes_nb{};     /// Number of bytes in extended header.
        char* extra00{};              /// Not used. creator_id + extra = 8 bytes
        char* extended_type{};        /// Type of extended header. "SERI", "FEI1", "AGAR", "CCP4" or "MRCO"
        int* nversion{};              /// MRC version

        int* imod_stamp{};            /// 1146047817 indicates IMOD flags are used.
        int* imod_flags{};            /// Bit flags. 1=signed, the rest is ignored.

        Float3<float>* origin{};      /// Origin of image.
        char* cmap{};                 /// Not used.
        char* stamp{};                /// First 2 bytes have 17 and 17 for big-endian or 68 and 65 for little-endian.
        float* rms{};                 /// Stddev of densities from mean. Negative if not computed.
        int* nb_labels{};             /// Number of labels with useful data.
        char* labels{};               /// 10 labels of 80 characters, blank-padded to end.

    public:
        /** Default constructor. */
        inline MRCHeader() : Header(), m_data(new char[m_header_size]) {
            link_();
        }


        /** Copy constructor. */
        inline MRCHeader(const MRCHeader& to_copy)
                : Header(to_copy), m_data(new char[m_header_size]) {
            std::memcpy(m_data, to_copy.m_data, m_header_size);
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
                m_io_layout = to_copy.m_io_layout;
                m_io_option = to_copy.m_io_option;
                std::memcpy(m_data, to_copy.m_data, m_header_size);
            }
            return *this;
        }


        /** Move operator assignment. */
        inline MRCHeader& operator=(MRCHeader&& to_move) noexcept {
            if (this != &to_move) {
                m_io_layout = to_move.m_io_layout;
                m_io_option = to_move.m_io_option;
                delete[] m_data;
                m_data = std::exchange(to_move.m_data, nullptr);
                link_();
            }
            return *this;
        }


        /** Destructor. */
        inline ~MRCHeader() override  {
            delete[] m_data;
        }


        /** Read the header from @a fstream into @a m_data. @see Header::read(). */
        inline errno_t read(std::fstream& fstream) override {
            fstream.seekg(0);
            fstream.read(m_data, m_header_size);
            if (fstream.fail())
                return Errno::fail_read;
            return validate_();
        }


        /** Write @a m_data into @a fstream. @see Header::write(). */
        inline errno_t write(std::fstream& fstream) override {
            fstream.seekp(0);
            fstream.write(m_data, m_header_size);
            return fstream.fail() ? Errno::fail_write : Errno::good;
        }


        /** (Re)set default values. */
        void reset() override;


        /** Get the shape by value */
        [[nodiscard]] inline Int3<size_t> getShape() const override {
            return Int3<size_t>(*shape);
        }


        /** Set the shape - must be positive values */
        inline errno_t setShape(Int3<size_t> new_shape) override {
            *shape = new_shape;
            return Errno::good;
        }


        /** Get the pixel size (in x, y and z) by value */
        [[nodiscard]] inline Float3<float> getPixelSize() const override {
            return *shape_cell / toFloat3(*shape_grid);
        }


        /** Set the shape - must be positive values */
        inline errno_t setPixelSize(Float3<float> new_pixel_size) override {
            if (new_pixel_size < 0)
                return Errno::invalid_argument;
            *shape_cell = toFloat3(*shape_grid) * new_pixel_size;
            return Errno::good;
        }


        /** The offset to the data: header size (1024) + the extended header. */
        [[nodiscard]] inline size_t getOffset() const override {
            return static_cast<size_t>(m_header_size) + static_cast<size_t>(*extended_bytes_nb);
        }


        errno_t setLayout(ioflag_t layout) override;


        /** Print a nice header. */
        void print(bool brief = true) const;


    private:

        /**
         * Link the underlying buffer to the higher level pointers to reinterpret the MRC header
         * into the standard layout.
         * @note The content of @a m_data does not matter but it is excepted to point to an array
         *       of @a m_header_size elements.
         */
        void link_();


        /**
         * Check that the header complies to the MRC header standard.
         * @note    This is a brief validation and only checks the basics that are likely to be
         *          used systematically: shapes, offset and endianness.
         * @note    @a m_is_big_endian is updated.
         * @return  @c Errno::invalid_data if the header doesn't look like a MRC header.
         *          @c Errno::good (0) otherwise.
         */
        [[nodiscard]] uint8_t validate_();


        /** Store the endianness of the local machine into the header (in @a stamp). */
        inline void setEndianness_();

        // setStatistics_()


    };


}
