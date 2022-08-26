#pragma once
#if NOA_ENABLE_TIFF

#include "noa/common/io/ImageFile.h"

namespace noa::io {
    class TIFFFile : public details::ImageFile {
    public:
        TIFFFile();
        TIFFFile(const path_t& filename, open_mode_t open_mode) : TIFFFile() { open_(filename, open_mode); }
        ~TIFFFile() override { close(); }

    public:
        void reset() override {
            close();
            m_shape = 0;
            m_pixel_size = 0.f;
            m_data_type = DataType::DTYPE_UNKNOWN;
            m_is_read = false;
        }

        void open(const path_t& filename, open_mode_t open_mode) override { open_(filename, open_mode); }
        void close() override;

    public:
        [[nodiscard]] bool isOpen() const noexcept override { return m_tiff; }
        [[nodiscard]] const path_t& filename() const noexcept override { return m_filename; }
        [[nodiscard]]  std::string infoString(bool brief) const noexcept override;
        [[nodiscard]] Format format() const noexcept override { return Format::TIFF; }

    public:
        [[nodiscard]] size4_t shape() const noexcept override { return size4_t{m_shape[0], 1, m_shape[1], m_shape[2]}; }
        [[nodiscard]] stats_t stats() const noexcept override { return stats_t{}; }
        [[nodiscard]] float3_t pixelSize() const noexcept override { return {0.f, m_pixel_size[0], m_pixel_size[1]}; }
        [[nodiscard]] DataType dtype() const noexcept override { return m_data_type; }

        void shape(size4_t shape) override;
        void stats(stats_t) override { /* Ignore for now */ }
        void pixelSize(float3_t) override;
        void dtype(DataType) override;

    public:
        void read(void* output, DataType data_type, size_t start, size_t end, bool clamp) override;
        void readSlice(void* output, size4_t strides, size4_t shape, DataType data_type, size_t start, bool clamp) override;
        void readSlice(void* output, DataType data_type, size_t start, size_t end, bool clamp) override;
        void readAll(void* output, size4_t strides, size4_t shape, DataType data_type, bool clamp) override;
        void readAll(void* output, DataType data_type, bool clamp) override;

        void write(const void* input, DataType data_type, size_t start, size_t end, bool clamp) override;
        void writeSlice(const void* input, size4_t strides, size4_t shape, DataType data_type, size_t start, bool clamp) override;
        void writeSlice(const void* input, DataType data_type, size_t start, size_t end, bool clamp) override;
        void writeAll(const void* input, size4_t strides, size4_t shape, DataType data_type, bool clamp) override;
        void writeAll(const void* input, DataType data_type, bool clamp) override;

    private:
        void open_(const path_t& filename, open_mode_t mode);
        void readHeader_();
        static DataType getDataType_(uint16_t sample_format, uint16_t bits_per_sample);
        static void setDataType_(DataType data_type, uint16_t* sample_format, uint16_t* bits_per_sample);
        static void flipY_(byte_t* slice, size_t bytes_per_row, size_t rows, byte_t* buffer);

    private:
        void* m_tiff{};
        path_t m_filename{};
        uint3_t m_shape{0}; // BHW
        float2_t m_pixel_size{0.f}; // HW
        DataType m_data_type{DataType::DTYPE_UNKNOWN};
        bool m_is_read{};
    };
}

#endif // NOA_ENABLE_TIFF