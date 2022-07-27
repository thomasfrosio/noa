#ifndef NOA_IMAGEFILE_INL_
#error "This is an internal header; it should not be included."
#endif

#include <memory>
#include <utility>

#include "noa/common/io/header/MRCHeader.h"
#include "noa/common/io/header/TIFFHeader.h"

#define NOA_IMAGEFILE_THROW_STRING_ "File {}: header failed"
#define NOA_IMAGEFILE_TRY_HEADER_(func, ...)        \
try {                                               \
    if (m_header)                                   \
        m_header->func(__VA_ARGS__);                \
} catch (...) {                                     \
    NOA_THROW(NOA_IMAGEFILE_THROW_STRING_, m_path); \
}

namespace noa::io {
    template<typename T>
    ImageFile::ImageFile(T&& filename, open_mode_t mode)
            : m_path(std::forward<T>(filename)), m_header_format(format_(m_path.extension())) {
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
        m_header_format = format_(m_path.extension());
        if (!m_header || m_header_format != old_format) {
            setHeader_(m_header_format);
        } else {
            try {
                m_header->reset();
            } catch (...) {
                NOA_THROW(NOA_IMAGEFILE_THROW_STRING_, m_path);
            }
        }
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
        return m_header ? m_header->shape() : size4_t{};
    }

    inline float3_t ImageFile::pixelSize() const noexcept {
        return m_header ? m_header->pixelSize() : float3_t{};
    }

    inline DataType ImageFile::dtype() const noexcept {
        return m_header ? m_header->dtype() : DATA_UNKNOWN;
    }

    inline stats_t ImageFile::stats() const noexcept {
        return m_header ? m_header->stats() : stats_t{};
    }

    inline void ImageFile::shape(size4_t shape) {
        NOA_IMAGEFILE_TRY_HEADER_(shape, shape)
    }

    inline void ImageFile::pixelSize(float3_t pixel_size) {
        NOA_IMAGEFILE_TRY_HEADER_(pixelSize, pixel_size)
    }

    inline void ImageFile::dtype(io::DataType data_type) {
        NOA_IMAGEFILE_TRY_HEADER_(dtype, data_type)
    }

    inline void ImageFile::stats(stats_t stats) {
        NOA_IMAGEFILE_TRY_HEADER_(stats, stats)
    }

    template<typename T>
    inline void ImageFile::read(T* output, size_t start, size_t end, bool clamp) {
        NOA_IMAGEFILE_TRY_HEADER_(read, output, io::dtype<T>(), start, end, clamp)
    }

    template<typename T>
    inline void ImageFile::readSlice(T* output, size_t start, size_t end, bool clamp) {
        NOA_IMAGEFILE_TRY_HEADER_(readSlice, output, io::dtype<T>(), start, end, clamp)
    }

    template<typename T, typename I>
    inline void ImageFile::readSlice(const View<T, I>& output, size_t start, bool clamp) {
        NOA_IMAGEFILE_TRY_HEADER_(readSlice,
                                  output.get(), output.strides(), output.shape(), io::dtype<T>(), start, clamp)
    }

    template<typename T>
    inline void ImageFile::readAll(T* output, bool clamp) {
        NOA_IMAGEFILE_TRY_HEADER_(readAll, output, io::dtype<T>(), clamp)
    }

    template<typename T, typename I>
    inline void ImageFile::readAll(const View<T, I>& output, bool clamp) {
        NOA_IMAGEFILE_TRY_HEADER_(readAll, output.get(), output.strides(), output.shape(), io::dtype<T>(), clamp)
    }

    template<typename T>
    inline void ImageFile::write(const T* input, size_t start, size_t end, bool clamp) {
        NOA_IMAGEFILE_TRY_HEADER_(write, input, io::dtype<T>(), start, end, clamp)
    }

    template<typename T>
    inline void ImageFile::writeSlice(const T* input, size_t start, size_t end, bool clamp) {
        NOA_IMAGEFILE_TRY_HEADER_(writeSlice, input, io::dtype<T>(), start, end, clamp)
    }

    template<typename T, typename I>
    void ImageFile::writeSlice(const View<T, I>& input, size_t start, bool clamp) {
        NOA_IMAGEFILE_TRY_HEADER_(writeSlice,
                                  input.get(), input.strides(), input.shape(), io::dtype<T>(), start, clamp)
    }

    template<typename T>
    inline void ImageFile::writeAll(const T* input, bool clamp) {
        NOA_IMAGEFILE_TRY_HEADER_(writeAll, input, io::dtype<T>(), clamp)
    }

    template<typename T, typename I>
    void ImageFile::writeAll(const View<T, I>& input, bool clamp) {
        NOA_IMAGEFILE_TRY_HEADER_(writeAll, input.get(), input.strides(), input.shape(), io::dtype<T>(), clamp)
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

    inline Format ImageFile::format_(const path_t& extension) noexcept {
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
