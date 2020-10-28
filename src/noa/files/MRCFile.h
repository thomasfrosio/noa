/**
 * @file MRCFile.h
 * @brief MRC file class.
 * @author Thomas - ffyr2w
 * @date 24/10/2020
 */
#pragma once

#include "noa/Base.h"
#include "noa/util/Arrays.h"
#include "noa/files/File.h"


namespace Noa {

    /**
     *
     */
    class NOA_API MRCHeader {
    private:
        char* m_header;                    /// The underlying data.

    protected:
        struct Type {
            static constexpr int byte = 0;
            static constexpr int int16 = 1;
            static constexpr int float32 = 2;
            static constexpr int complex32 = 3;
            static constexpr int complex64 = 4;
            static constexpr int uint16 = 6;
            static constexpr int uchar3 = 16;
            static constexpr int bit4 = 101;
        };

        static constexpr int header_size = 1024;

        // Reinterpretation of the underlying header.
        Int3<int>* size{};            /// Number of columns (x), rows (y) and sections (z).
        int* mode{};                  /// Types of pixel in image. See MRCHeader::Type.
        Int3<int>* size_sub{};        /// Starting point of sub image (x, y and z).
        Int3<int>* size_grid{};       /// Grid size (x, y and z).
        Float3<float>* size_cell{};   /// Cell size. Pixel spacing (x, y and z) = cell_size / size.
        Float3<float>* angles{};      /// Cell angles: alpha, beta, gamma.
        Int3<int>* map_order{};       /// Map layout (x, y and z).

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
        inline MRCHeader() : m_header(new char[header_size]) {
            link_();
        }


        /** Destructor. */
        inline ~MRCHeader() {
            delete[] m_header;
        }


        /** Copy constructor. */
        inline MRCHeader(const MRCHeader& to_copy) : m_header(new char[header_size]) {
            std::memcpy(m_header, to_copy.m_header, header_size);
            link_();
        }


        /** Move constructor. */
        inline MRCHeader(MRCHeader&& to_move) noexcept
                : m_header(std::exchange(to_move.m_header, nullptr)) {
            link_();
        }


        /** Copy operator assignment. */
        inline MRCHeader& operator=(const MRCHeader& to_copy) noexcept {
            if (this != &to_copy)
                std::memcpy(m_header, to_copy.m_header, header_size);
            return *this;
        }


        /** Move operator assignment. */
        inline MRCHeader& operator=(MRCHeader&& to_move) noexcept {
            if (this != &to_move) {
                delete[] m_header;
                m_header = std::exchange(to_move.m_header, nullptr);
                link_();
            }
            return *this;
        }


        /**
         * Read the header from @a fstream. The member pointers are already linked to @a m_header.
         * @param fstream       File stream to read from. Position is reset from 0.
         * @return              Errno::fail if the function wasn't able to read the header. 0 otherwise.
         */
        inline uint8_t read(std::fstream& fstream) {
            fstream.seekg(0);
            fstream.read(m_header, header_size);
            return fstream.fail() ? Errno::fail : 0U;
        }

        /**
         *
         * @param fstream
         * @return
         */
        inline uint8_t write(std::fstream& fstream) {
            fstream.seekp(0);
            fstream.write(m_header, header_size);
            return fstream.fail() ? Errno::fail : 0U;
        }


        /** Set default value - create blank header. */
        void setDefault();

        // print

        inline void setStamp();

    private:
        void link_();
    };


    /**
     *
     */
    class NOA_API MRCFile {
    private:
        std::filesystem::path m_path{};
        std::unique_ptr<std::fstream> m_fstream{nullptr};
        MRCHeader header{};

    public:

        // open
        // reopen
        // write
        // close
        // read
        // remove
        // rename
    };
}
