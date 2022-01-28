///

#pragma once
#if NOA_ENABLE_TIFF

#include <tiffio.h>

#include "noa/common/io/header/Header.h"

namespace noa::io::details {
    class TIFFHeader : public Header {
    private:
        ::TIFF* m_tiff{};
        uint3_t m_shape{1};
        float2_t m_pixel_size{0.f};
        DataType m_data_type{DataType::FLOAT32};
        uint16_t m_min{}, m_max{};
        bool m_is_read{};

    public:
        TIFFHeader();

        [[nodiscard]] Format getFormat() const noexcept override { return Format::TIFF; }

        void open(const path_t& filename, open_mode_t mode) override;
        void close() override;

        [[nodiscard]] size4_t getShape() const noexcept override {
            return size4_t{m_shape.flip().get()};
        }

        [[nodiscard]] stats_t getStats() const noexcept override {
            return stats_t{static_cast<float>(m_min), static_cast<float>(m_max), 0.f, 0.f, 0.f, 0.f};
        }

        [[nodiscard]] float3_t getPixelSize() const noexcept override {
            return {0.f, m_pixel_size[0], m_pixel_size[1]};
        }

        [[nodiscard]] DataType getDataType() const noexcept override {
            return m_data_type;
        }

        void setShape(size4_t shape) override;
        void setStats(stats_t) override;
        void setPixelSize(float3_t) override;
        void setDataType(DataType) override;

        [[nodiscard]]  std::string infoString(bool brief) const noexcept override;

        void read(void* output, DataType data_type, size_t start, size_t end, bool clamp) override;
        void readLine(void* output, DataType data_type, size_t start, size_t end, bool clamp) override;
        void readShape(void* output, DataType data_type, size4_t offset, size4_t shape, bool clamp) override;
        void readSlice(void* output, DataType data_type, size_t start, size_t end, bool clamp) override;
        void readAll(void* output, DataType data_type, bool clamp) override;

        void write(const void* input, DataType data_type, size_t start, size_t end, bool clamp) override;
        void writeLine(const void* input, DataType data_type, size_t start, size_t end, bool clamp) override;
        void writeShape(const void* input, DataType data_type, size4_t offset, size4_t shape, bool clamp) override;
        void writeSlice(const void* input, DataType data_type, size_t start, size_t end, bool clamp) override;
        void writeAll(const void* input, DataType data_type, bool clamp) override;

    private:
        void readHeader_();
        static DataType getDataType_(uint16_t sample_format, uint16_t bits_per_sample);
        static void setDataType_(DataType data_type, uint16_t* sample_format, uint16_t* bits_per_sample);
        static void flipY_(char* slice, size_t bytes_per_row, size_t rows, char* buffer);
    };
}

#endif // NOA_ENABLE_TIFF
