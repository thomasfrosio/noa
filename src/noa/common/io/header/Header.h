#pragma once

#include "noa/common/Types.h"
#include "noa/common/io/IO.h"
#include "noa/common/io/Stats.h"

namespace noa::io::details {
    class Header {
    public:
        Header() = default;
        virtual ~Header() = default;

        [[nodiscard]] virtual Format format() const noexcept { return Format::FORMAT_UNKNOWN; }

        virtual void reset() = 0;
        virtual void open(const path_t&, open_mode_t) = 0;
        virtual void close() = 0;

        [[nodiscard]] virtual size4_t shape() const noexcept = 0;
        virtual void shape(size4_t) = 0;

        [[nodiscard]] virtual stats_t stats() const noexcept = 0;
        virtual void stats(stats_t) = 0;

        [[nodiscard]] virtual float3_t pixelSize() const noexcept = 0;
        virtual void pixelSize(float3_t) = 0;

        [[nodiscard]] virtual DataType dtype() const noexcept = 0;
        virtual void dtype(DataType) = 0;

        [[nodiscard]] virtual std::string infoString(bool) const noexcept = 0;

        virtual void read(void* output, DataType data_type, size_t start, size_t end, bool clamp) = 0;
        virtual void readSlice(void* output, size4_t strides, size4_t shape, DataType data_type, size_t start, bool clamp) = 0;
        virtual void readSlice(void* output, DataType data_type, size_t start, size_t end, bool clamp) = 0;
        virtual void readAll(void* output, size4_t strides, size4_t shape, DataType data_type, bool clamp) = 0;
        virtual void readAll(void* output, DataType data_type, bool clamp) = 0;

        virtual void write(const void* input, DataType data_type, size_t start, size_t end, bool clamp) = 0;
        virtual void writeSlice(const void* input, size4_t strides, size4_t shape, DataType data_type, size_t start, bool clamp) = 0;
        virtual void writeSlice(const void* input, DataType data_type, size_t start, size_t end, bool clamp) = 0;
        virtual void writeAll(const void* input, size4_t strides, size4_t shape, DataType data_type, bool clamp) = 0;
        virtual void writeAll(const void* input, DataType data_type, bool clamp) = 0;
    };
}
