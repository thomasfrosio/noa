#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/io/ImageFile.hpp"

#if NOA_ENABLE_TIFF
#ifdef NOA_IS_OFFLINE

namespace noa::io {
    class TIFFFile : public guts::ImageFile {
    public:
        TIFFFile();
        TIFFFile(const Path& filename, OpenMode open_mode) : TIFFFile() { open_(filename, open_mode); }
        ~TIFFFile() override { close_(); }

    public:
        void reset() override {
            close();
            m_shape = 0;
            m_pixel_size = 0.f;
            m_data_type = DataType::UNKNOWN;
            m_is_read = false;
        }

        void open(const Path& filename, OpenMode open_mode) override { open_(filename, open_mode); }
        void close() override { close_(); }

    public:
        [[nodiscard]] bool is_open() const noexcept override { return m_tiff; }
        [[nodiscard]] const Path& filename() const noexcept override { return m_filename; }
        [[nodiscard]] std::string info_string(bool brief) const noexcept override;
        [[nodiscard]] Format format() const noexcept override { return Format::TIFF; }

    public:
        [[nodiscard]] Shape4<i64> shape() const noexcept override { return {m_shape[0], 1, m_shape[1], m_shape[2]}; }
        [[nodiscard]] Stats<f32> stats() const noexcept override { return {}; }
        [[nodiscard]] Vec3<f32> pixel_size() const noexcept override { return {0.f, m_pixel_size[0], m_pixel_size[1]}; }
        [[nodiscard]] DataType dtype() const noexcept override { return m_data_type; }

        void set_shape(const Shape4<i64>& shape) override;
        void set_stats(Stats<f32>) override { /* Ignore for now */ }
        void set_pixel_size(Vec3<f32>) override;
        void set_dtype(DataType) override;

    public:
        void read_elements(void* output, DataType data_type, i64 start, i64 end, bool clamp) override;
        void read_slice(void* output, const Strides4<i64>& strides, const Shape4<i64>& shape, DataType data_type, i64 start, bool clamp) override;
        void read_slice(void* output, DataType data_type, i64 start, i64 end, bool clamp) override;
        void read_all(void* output, const Strides4<i64>& strides, const Shape4<i64>& shape, DataType data_type, bool clamp) override;
        void read_all(void* output, DataType data_type, bool clamp) override;

        void write_elements(const void* input, DataType data_type, i64 start, i64 end, bool clamp) override;
        void write_slice(const void* input, const Strides4<i64>& strides, const Shape4<i64>& shape, DataType data_type, i64 start, bool clamp) override;
        void write_slice(const void* input, DataType data_type, i64 start, i64 end, bool clamp) override;
        void write_all(const void* input, const Strides4<i64>& strides, const Shape4<i64>& shape, DataType data_type, bool clamp) override;
        void write_all(const void* input, DataType data_type, bool clamp) override;

    private:
        void open_(const Path& filename, OpenMode mode);
        void close_();
        void read_header_();
        static DataType get_dtype_(u16 sample_format, u16 bits_per_sample);
        static void set_dtype_(DataType data_type, u16* sample_format, u16* bits_per_sample);
        static void flip_y_(Byte* slice, i64 bytes_per_row, i64 rows, Byte* buffer);

    private:
        void* m_tiff{};
        Path m_filename{};
        Shape3<u32> m_shape{0}; // BHW
        Vec2<f32> m_pixel_size{0.f}; // HW
        DataType m_data_type{DataType::UNKNOWN};
        bool m_is_read{};
    };
}

#endif
#endif
