#pragma once

#ifndef NOA_IMAGEFILE_INL_
#error "This is an internal header. Include the corresponding .h file instead"
#endif


#include <memory>
#include <utility>

#include "noa/common/io/MRCFile.h"
#include "noa/common/io/TIFFFile.h"

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

    inline dim4_t ImageFile::shape() const noexcept {
        return m_header ? m_header->shape() : dim4_t{};
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

    inline void ImageFile::shape(dim4_t shape) {
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
    inline void ImageFile::read(const Array<T>& output, bool clamp) {
        NOA_CHECK(isOpen(), "The file should be opened");
        NOA_ASSERT(m_header);
        if (output.dereferenceable()) {
            m_header->readAll(output.eval().get(), output.strides(), output.shape(), io::dtype<T>(), clamp);
        } else {
            Array<T> tmp(output.shape());
            m_header->readAll(tmp.get(), tmp.strides(), tmp.shape(), io::dtype<T>(), clamp);
            tmp.to(output);
        }
    }

    template<typename T>
    inline Array<T> ImageFile::read(ArrayOption option, bool clamp) {
        NOA_CHECK(isOpen(), "The file should be opened");
        NOA_ASSERT(m_header);
        if (option.dereferenceable()) {
            Array<T> out(this->shape(), option);
            m_header->readAll(out.eval().get(), out.strides(), out.shape(), io::dtype<T>(), clamp);
            return out;
        } else {
            Array<T> tmp(this->shape());
            m_header->readAll(tmp.get(), tmp.strides(), tmp.shape(), io::dtype<T>(), clamp);
            return tmp.to(option);
        }
    }

    template<typename T, typename I>
    inline void ImageFile::read(const View<T, I>& output, bool clamp) {
        NOA_CHECK(isOpen(), "The file should be opened");
        NOA_ASSERT(m_header);
        m_header->readAll(output.get(), output.strides(), output.shape(), io::dtype<T>(), clamp);
    }

    template<typename T>
    inline void ImageFile::readSlice(const Array<T>& output, dim_t start, bool clamp) {
        NOA_CHECK(isOpen(), "The file should be opened");
        NOA_ASSERT(m_header);
        if (output.dereferenceable()) {
            m_header->readSlice(output.eval().get(), output.strides(), output.shape(), io::dtype<T>(), start, clamp);
        } else {
            Array<T> tmp(output.shape());
            m_header->readSlice(tmp.get(), tmp.strides(), tmp.shape(), io::dtype<T>(), start, clamp);
            tmp.to(output);
        }
    }

    template<typename T, typename I>
    inline void ImageFile::readSlice(const View<T, I>& output, dim_t start, bool clamp) {
        NOA_CHECK(isOpen(), "The file should be opened");
        NOA_ASSERT(m_header);
        m_header->readSlice(output.get(), output.strides(), output.shape(), io::dtype<T>(), start, clamp);
    }

    template<typename T>
    inline void ImageFile::write(const Array<T>& input, bool clamp) {
        NOA_CHECK(isOpen(), "The file should be opened");
        NOA_ASSERT(m_header);
        if (!input.dereferenceable()) {
            Array tmp = input.to(Device{}).release();
            m_header->writeAll(tmp.eval().get(), tmp.strides(), tmp.shape(), io::dtype<T>(), clamp);
        } else {
            m_header->writeAll(input.eval().get(), input.strides(), input.shape(), io::dtype<T>(), clamp);
        }
    }

    template<typename T, typename I>
    inline void ImageFile::write(const View<T, I>& input, bool clamp) {
        NOA_CHECK(isOpen(), "The file should be opened");
        NOA_ASSERT(m_header);
        m_header->writeAll(input.get(), input.strides(), input.shape(), io::dtype<T>(), clamp);
    }

    template<typename T>
    inline void ImageFile::writeSlice(const Array<T>& input, dim_t start, bool clamp) {
        NOA_CHECK(isOpen(), "The file should be opened");
        NOA_ASSERT(m_header);
        if (!input.dereferenceable()) {
            Array tmp = input.to(Device{}).release();
            m_header->writeSlice(tmp.eval().get(), tmp.strides(), tmp.shape(), io::dtype<T>(), start, clamp);
        } else {
            m_header->writeSlice(input.eval().get(), input.strides(), input.shape(), io::dtype<T>(), start, clamp);
        }
    }

    template<typename T, typename I>
    inline void ImageFile::writeSlice(const View<T, I>& input, dim_t start, bool clamp) {
        NOA_CHECK(isOpen(), "The file should be opened");
        NOA_ASSERT(m_header);
        m_header->writeSlice(input.get(), input.strides(), input.shape(), io::dtype<T>(), start, clamp);
    }

    inline void ImageFile::setHeader_(const path_t& filename, Format new_format) {
        switch (new_format) {
            case Format::MRC:
                m_header = std::make_unique<MRCFile>();
                break;
            case Format::TIFF:
                #if NOA_ENABLE_TIFF
                m_header = std::make_unique<TIFFFile>();
                break;
                #else
                NOA_THROW("File {}: TIFF files are not supported in this build. See CMake option NOA_ENABLE_TIFF");
                #endif
            default:
                NOA_THROW("File: {}. File format {} is not supported", filename, new_format);
        }
    }

    inline Format ImageFile::format_(const path_t& extension) noexcept {
        if (extension == ".mrc" || extension == ".st" || extension == ".rec" || extension == ".mrcs")
            return Format::MRC;
        else if (extension == ".tif" || extension == ".tiff")
            return Format::TIFF;
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
