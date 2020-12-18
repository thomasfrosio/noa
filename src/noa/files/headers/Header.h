/**
 * @file Header.h
 * @brief File header class. Headers handle the metadata of image files.
 * @author Thomas - ffyr2w
 * @date 31/10/2020
 */
#pragma once

#include "noa/Base.h"
#include "noa/files/headers/MRCHeader.h"


namespace Noa {
    /** Communicates with file header. Used by @a ImageFile. */
    NOA_API class Header {
    public:
        friend class ImageFile;

        /** Type of header. @c unknown is used when the file extension is not recognized. */
        enum class Type {
            unknown, MRC, TIF, EER
        };

    private:
        /** Opaque pointer. At runtime, any derived @a AbstractHeader class is accepted. */
        std::unique_ptr<AbstractHeader> m_handle{nullptr};

    public:
        /** Default constructor. The header is not allocated. */
        Header() = default;

        /** Sets the desired header. */
        inline explicit Header(Type type) { setHandle_(type); }

        /** Gets and sets the desired header. */
        inline explicit Header(const fs::path& path) { setHandle_(getHeaderType_(path)); }

        /** Retrieves the IO layout, specifying the data layout. */
        [[nodiscard]] inline IO::Layout getLayout() const { return m_handle->getLayout(); }

        /** Sets the IO layout and updates whatever value it corresponds in the header. */
        inline errno_t setLayout(IO::Layout layout) { return m_handle->setLayout(layout); }

        /** Whether or not the data is big endian. */
        [[nodiscard]] inline bool isBigEndian() const { return m_handle->isBigEndian(); }

        /** Whether or not the endianness of the data is swapped. */
        [[nodiscard]] inline bool isSwapped() const { return isBigEndian() != OS::isBigEndian(); }

        /** Sets the endianness and updates whatever value it corresponds in the header file. */
        inline void setEndianness(bool is_big_endian) {
            return m_handle->setEndianness(is_big_endian);
        }

        /** Gets the position, in bytes, where the data starts, relative to the beginning of the file. */
        [[nodiscard]] inline size_t getOffset() const { return m_handle->getOffset(); }

        /** (Re)sets the metadata to the default values, i.e. creates a new header. */
        inline void reset() { return m_handle->reset(); }

        /** Gets the (x, y, z) size dimensions of the image. */
        [[nodiscard]] inline Int3<size_t> getShape() const { return m_handle->getShape(); }

        /** Sets the (x, y, z) size dimensions of the image. */
        inline errno_t setShape(Int3<size_t> shape) { return m_handle->setShape(shape); }

        /** Gets the (x, y, z) pixel size of the image. */
        [[nodiscard]] inline Float3<float> getPixelSize() const { return m_handle->getPixelSize(); }

        /** Sets the (x, y, z) pixel size of the image. */
        inline errno_t setPixelSize(Float3<float> ps) { return m_handle->setPixelSize(ps); }

        /** Prints the header. If @a brief is true, it should only print the shape and pixel size. */
        [[nodiscard]] inline std::string toString(bool brief) const {
            return m_handle->toString(brief);
        }


        /**
         * Reads the header from a file.
         * @param[in] fstream   File stream to read from. Should be opened. Current position should not matter.
         * @return              @c Errno::fail_read, if the function wasn't able to read @a fstream.
         *                      @c Errno::invalid_data, if the header is not recognized.
         *                      @c Errno::not_supported, if the data is not supported.
         *                      @c Errno::good, otherwise.
         * @warning The position at which the stream is left does not necessarily correspond to the
         *          beginning of the data. Use getOffset() instead.
         */
        inline errno_t read(std::fstream& fstream) { return m_handle->read(fstream); }


        /**
         * Writes the header into a file.
         * @param[in] fstream   File stream to write into. Should be opened. Current position should not matter.
         * @return              @c Errno::fail_write, if the function wasn't able to write into @a fstream.
         *                      @c Errno::good, otherwise.
         */
        inline errno_t write(std::fstream& fstream) { return m_handle->write(fstream); }


    private:
        /** Whether or not the handle is set. */
        [[nodiscard]] inline bool isHandled_() const { return m_handle == nullptr; }


        /** Sets the desired handle (i.e. the opaque pointer). */
        inline void setHandle_(Header::Type type) {
            if (type == Header::Type::MRC)
                m_handle = std::make_unique<MRCHeader>();
        }


        [[nodiscard]] static inline Header::Type getHeaderType_(const fs::path& path) {
            std::string extension = path.extension().string();
            if (extension == ".mrc" || extension == ".st" ||
                extension == ".rec" || extension == ".mrcs")
                return Header::Type::MRC;
            else if (extension == ".tif" || extension == ".tiff")
                return Header::Type::TIF;
            else if (extension == ".eer")
                return Header::Type::EER;
            else
                return Header::Type::unknown;
        }
    };
}
