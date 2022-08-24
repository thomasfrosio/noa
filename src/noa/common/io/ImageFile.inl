#ifndef NOA_IMAGEFILE_INL_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

#include <memory>
#include <utility>

#include "noa/common/io/header/MRCHeader.h"
#include "noa/common/io/header/TIFFHeader.h"

namespace noa::io {
    inline ImageFile::ImageFile(const path_t& filename, open_mode_t mode)
            : m_header_format(format_(filename.extension())) {
        setHeader_(filename, m_header_format);
        open_(filename, mode);
    }

    inline void ImageFile::open(const path_t& filename, open_mode_t mode) {
        close();
        Format old_format = m_header_format;
        m_header_format = format_(filename.extension());
        if (!m_header || m_header_format != old_format) {
            setHeader_(filename, m_header_format);
        } else {
            m_header->reset();
        }
        open_(filename, mode);
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

    inline path_t ImageFile::filename() const noexcept {
        return m_header ? m_header->filename() : "";
    }

    inline std::string ImageFile::info(bool brief) const noexcept {
        return m_header ? m_header->infoString(brief) : "";
    }

    inline size4_t ImageFile::shape() const noexcept {
        return m_header ? m_header->shape() : size4_t{};
    }

    inline float3_t ImageFile::pixelSize() const noexcept {
        return m_header ? m_header->pixelSize() : float3_t{};
    }

    inline DataType ImageFile::dtype() const noexcept {
        return m_header ? m_header->dtype() : DTYPE_UNKNOWN;
    }

    inline stats_t ImageFile::stats() const noexcept {
        return m_header ? m_header->stats() : stats_t{};
    }

    inline void ImageFile::shape(size4_t shape) {
        NOA_CHECK(isOpen(), "The file should be opened");
        NOA_ASSERT(m_header);
        m_header->shape(shape);
    }

    inline void ImageFile::pixelSize(float3_t pixel_size) {
        NOA_CHECK(isOpen(), "The file should be opened");
        NOA_ASSERT(m_header);
        m_header->pixelSize(pixel_size);
    }

    inline void ImageFile::dtype(io::DataType data_type) {
        NOA_CHECK(isOpen(), "The file should be opened");
        NOA_ASSERT(m_header);
        m_header->dtype(data_type);
    }

    inline void ImageFile::stats(stats_t stats) {
        NOA_CHECK(isOpen(), "The file should be opened");
        NOA_ASSERT(m_header);
        m_header->stats(stats);
    }

    template<typename T>
    inline void ImageFile::read(T* output, size_t start, size_t end, bool clamp) {
        NOA_CHECK(isOpen(), "The file should be opened");
        NOA_ASSERT(m_header);
        m_header->read(output, io::dtype<T>(), start, end, clamp);
    }

    template<typename T>
    inline void ImageFile::readSlice(T* output, size_t start, size_t end, bool clamp) {
        NOA_CHECK(isOpen(), "The file should be opened");
        NOA_ASSERT(m_header);
        m_header->readSlice(output, io::dtype<T>(), start, end, clamp);
    }

    template<typename T, typename I>
    inline void ImageFile::readSlice(const View<T, I>& output, size_t start, bool clamp) {
        NOA_CHECK(isOpen(), "The file should be opened");
        NOA_ASSERT(m_header);
        m_header->readSlice(output.get(), output.strides(), output.shape(), io::dtype<T>(), start, clamp);
    }

    template<typename T>
    inline void ImageFile::readAll(T* output, bool clamp) {
        NOA_CHECK(isOpen(), "The file should be opened");
        NOA_ASSERT(m_header);
        m_header->readAll(output, io::dtype<T>(), clamp);
    }

    template<typename T, typename I>
    inline void ImageFile::readAll(const View<T, I>& output, bool clamp) {
        NOA_CHECK(isOpen(), "The file should be opened");
        NOA_ASSERT(m_header);
        m_header->readAll(output.get(), output.strides(), output.shape(), io::dtype<T>(), clamp);
    }

    template<typename T>
    inline void ImageFile::write(const T* input, size_t start, size_t end, bool clamp) {
        NOA_CHECK(isOpen(), "The file should be opened");
        NOA_ASSERT(m_header);
        m_header->write(input, io::dtype<T>(), start, end, clamp);
    }

    template<typename T>
    inline void ImageFile::writeSlice(const T* input, size_t start, size_t end, bool clamp) {
        NOA_CHECK(isOpen(), "The file should be opened");
        NOA_ASSERT(m_header);
        m_header->writeSlice(input, io::dtype<T>(), start, end, clamp);
    }

    template<typename T, typename I>
    void ImageFile::writeSlice(const View<T, I>& input, size_t start, bool clamp) {
        NOA_CHECK(isOpen(), "The file should be opened");
        NOA_ASSERT(m_header);
        m_header->writeSlice(input.get(), input.strides(), input.shape(), io::dtype<T>(), start, clamp);
    }

    template<typename T>
    inline void ImageFile::writeAll(const T* input, bool clamp) {
        NOA_CHECK(isOpen(), "The file should be opened");
        NOA_ASSERT(m_header);
        m_header->writeAll(input, io::dtype<T>(), clamp);
    }

    template<typename T, typename I>
    void ImageFile::writeAll(const View<T, I>& input, bool clamp) {
        NOA_CHECK(isOpen(), "The file should be opened");
        NOA_ASSERT(m_header);
        m_header->writeAll(input.get(), input.strides(), input.shape(), io::dtype<T>(), clamp);
    }

    inline void ImageFile::setHeader_(const path_t& filename, Format new_format) {
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
                NOA_THROW("File: {}. File format {} is not supported", filename, new_format);
        }
    }

    inline Format ImageFile::format_(const path_t& extension) noexcept {
        if (extension == ".mrc" || extension == ".st" || extension == ".rec" || extension == ".mrcs")
            return Format::MRC;
        else
            return Format::FORMAT_UNKNOWN;
    }


    inline void ImageFile::open_(const path_t& filename, open_mode_t mode) {
        NOA_ASSERT(m_header);
        m_header->open(filename, mode);
        m_is_open = true;
    }

    inline void ImageFile::close_() {
        if (!m_header)
            return;
        m_header->close();
        m_is_open = false;
    }

    inline ImageFile::~ImageFile() noexcept(false) {
        try {
            if (m_header)
                m_header->close();
        } catch (...) {
            if (!std::uncaught_exceptions()) {
                std::rethrow_exception(std::current_exception());
            }
        }
    }
}
