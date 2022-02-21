#ifndef NOA_IMAGEFILE_INL_
#error "This is an internal header; it should not be included."
#endif

#include <memory>
#include <utility>

#include "noa/common/io/header/MRCHeader.h"
#include "noa/common/io/header/TIFFHeader.h"

#define NOA_IMAGEFILE_THROW_STRING_ "File {}: header failed"
namespace noa::io {
    template<typename T>
    ImageFile::ImageFile(T&& filename, open_mode_t mode)
            : m_path(std::forward<T>(filename)), m_header_format(getFormat_(m_path.extension())) {
        setHeader_(m_header_format);
        open_(mode);
    }

    template<typename T>
    ImageFile::ImageFile(T&& filename, Format file_format, open_mode_t mode)
            : m_path(std::forward<T>(filename)), m_header_format(file_format) {
        setHeader_(m_header_format);
        open_(mode);
    }

    template<typename T>
    void ImageFile::open(T&& filename, open_mode_t mode) {
        close();
        Format old_format = m_header_format;
        m_path = std::forward<T>(filename);
        m_header_format = getFormat_(m_path.extension());
        if (!m_header || m_header_format != old_format)
            setHeader_(m_header_format);
        open_(mode);
    }

    inline void ImageFile::close() {
        close_();
    }

    inline bool ImageFile::isOpen() const noexcept {
        return m_is_open;
    }

    inline ImageFile::operator bool() const noexcept {
        return m_is_open;
    }

    inline Format ImageFile::format() const noexcept {
        return m_header_format;
    }

    inline bool ImageFile::isMRC() const noexcept {
        return m_header_format == Format::MRC;
    }

    inline bool ImageFile::isTIFF() const noexcept {
        return m_header_format == Format::TIFF;
    }

    inline bool ImageFile::isEER() const noexcept {
        return m_header_format == Format::EER;
    }

    inline bool ImageFile::isJPEG() const noexcept {
        return m_header_format == Format::JPEG;
    }

    inline bool ImageFile::isPNG() const noexcept {
        return m_header_format == Format::PNG;
    }

    inline const path_t& ImageFile::path() const noexcept {
        return m_path;
    }

    inline std::string ImageFile::info(bool brief) const noexcept {
        return m_header ? m_header->infoString(brief) : "";
    }

    inline size4_t ImageFile::shape() const noexcept {
        return m_header ? m_header->getShape() : size4_t{};
    }

    inline float3_t ImageFile::pixelSize() const noexcept {
        return m_header ? m_header->getPixelSize() : float3_t{};
    }

    inline DataType ImageFile::dtype() const noexcept {
        return m_header ? m_header->getDataType() : DATA_UNKNOWN;
    }

    inline stats_t ImageFile::stats() const noexcept {
        return m_header ? m_header->getStats() : stats_t{};
    }

    inline void ImageFile::shape(size4_t shape) {
        try {
            if (m_header)
                m_header->setShape(shape);
        } catch (...) {
            NOA_THROW(NOA_IMAGEFILE_THROW_STRING_, m_path);
        }
    }

    inline void ImageFile::pixelSize(float3_t pixel_size) {
        try {
            if (m_header)
                m_header->setPixelSize(pixel_size);
        } catch (...) {
            NOA_THROW(NOA_IMAGEFILE_THROW_STRING_, m_path);
        }
    }

    inline void ImageFile::dtype(io::DataType data_type) {
        try {
            if (m_header)
                m_header->setDataType(data_type);
        } catch (...) {
            NOA_THROW(NOA_IMAGEFILE_THROW_STRING_, m_path);
        }
    }

    inline void ImageFile::stats(stats_t stats) {
        try {
            if (m_header)
                m_header->setStats(stats);
        } catch (...) {
            NOA_THROW(NOA_IMAGEFILE_THROW_STRING_, m_path);
        }
    }

    template<typename T>
    inline void ImageFile::read(T* output, size_t start, size_t end, bool clamp) {
        try {
            if (m_header)
                m_header->read(output, getDataType<T>(), start, end, clamp);
        } catch (...) {
            NOA_THROW(NOA_IMAGEFILE_THROW_STRING_, m_path);
        }
    }

    template<typename T>
    inline void ImageFile::readLine(T* output, size_t start, size_t end, bool clamp) {
        try {
            if (m_header)
                m_header->readLine(output, getDataType<T>(), start, end, clamp);
        } catch (...) {
            NOA_THROW(NOA_IMAGEFILE_THROW_STRING_, m_path);
        }
    }

    template<typename T>
    inline void ImageFile::readShape(T* output, size4_t offset, size4_t shape, bool clamp) {
        try {
            if (m_header)
                m_header->readShape(output, getDataType<T>(), offset, shape, clamp);
        } catch (...) {
            NOA_THROW(NOA_IMAGEFILE_THROW_STRING_, m_path);
        }
    }

    template<typename T>
    inline void ImageFile::readSlice(T* output, size_t start, size_t end, bool clamp) {
        try {
            if (m_header)
                m_header->readSlice(output, getDataType<T>(), start, end, clamp);
        } catch (...) {
            NOA_THROW(NOA_IMAGEFILE_THROW_STRING_, m_path);
        }
    }

    template<typename T>
    inline void ImageFile::readAll(T* output, bool clamp) {
        try {
            if (m_header)
                m_header->readAll(output, getDataType<T>(), clamp);
        } catch (...) {
            NOA_THROW(NOA_IMAGEFILE_THROW_STRING_, m_path);
        }
    }

    template<typename T>
    inline void ImageFile::write(const T* input, size_t start, size_t end, bool clamp) {
        try {
            if (m_header)
                m_header->write(input, getDataType<T>(), start, end, clamp);
        } catch (...) {
            NOA_THROW(NOA_IMAGEFILE_THROW_STRING_, m_path);
        }
    }

    template<typename T>
    inline void ImageFile::writeLine(const T* input, size_t start, size_t end, bool clamp) {
        try {
            if (m_header)
                m_header->writeLine(input, getDataType<T>(), start, end, clamp);
        } catch (...) {
            NOA_THROW(NOA_IMAGEFILE_THROW_STRING_, m_path);
        }
    }

    template<typename T>
    inline void ImageFile::writeShape(const T* input, size4_t offset, size4_t shape, bool clamp) {
        try {
            if (m_header)
                m_header->writeShape(input, getDataType<T>(), offset, shape, clamp);
        } catch (...) {
            NOA_THROW(NOA_IMAGEFILE_THROW_STRING_, m_path);
        }
    }

    template<typename T>
    inline void ImageFile::writeSlice(const T* input, size_t start, size_t end, bool clamp) {
        try {
            if (m_header)
                m_header->writeSlice(input, getDataType<T>(), start, end, clamp);
        } catch (...) {
            NOA_THROW(NOA_IMAGEFILE_THROW_STRING_, m_path);
        }
    }

    template<typename T>
    inline void ImageFile::writeAll(const T* input, bool clamp) {
        try {
            if (m_header)
                m_header->writeAll(input, getDataType<T>(), clamp);
        } catch (...) {
            NOA_THROW(NOA_IMAGEFILE_THROW_STRING_, m_path);
        }
    }

    inline void ImageFile::setHeader_(Format new_format) {
        switch (new_format) {
            case Format::MRC:
                m_header = std::make_unique<details::MRCHeader>();
                break;
            case Format::TIFF:
                #if NOA_ENABLE_TIFF
                m_header = std::make_unique<details::TIFFHeader>();
                #else
                NOA_THROW("File {}: TIFF files are not supported in this build. See CMake option NOA_ENABLE_TIFF");
                #endif
                break;
            default:
                NOA_THROW("File {}: format {} is currently not supported", m_path, new_format);
        }
    }

    inline Format ImageFile::getFormat_(const path_t& extension) noexcept {
        if (extension == ".mrc" || extension == ".st" || extension == ".rec" || extension == ".mrcs")
            return Format::MRC;
        else
            return Format::FORMAT_UNKNOWN;
    }


    inline void ImageFile::open_(open_mode_t mode) {
        if (!m_header)
            return;
        try {
            m_header->open(m_path, mode);
        } catch (...) {
            NOA_THROW(NOA_IMAGEFILE_THROW_STRING_, m_path);
        }
        m_is_open = true;
    }

    inline void ImageFile::close_() {
        if (!m_header)
            return;
        try {
            m_header->close();
        } catch (...) {
            NOA_THROW(NOA_IMAGEFILE_THROW_STRING_, m_path);
        }
        m_is_open = false;
    }

    inline ImageFile::~ImageFile() noexcept(false) {
        try {
            if (m_header)
                m_header->close();
        } catch (...) {
            if (!std::uncaught_exceptions()) {
                NOA_THROW(NOA_IMAGEFILE_THROW_STRING_, m_path);
            }
        }
    }
}
#undef NOA_IMAGEFILE_THROW_STRING_
