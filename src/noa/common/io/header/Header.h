#pragma once

#include "noa/common/Types.h"
#include "noa/common/io/IO.h"
#include "noa/common/io/Stats.h"

namespace noa::io::details {
    // Only meant to be accessed by ImageFile.
    class Header {
    public:
        Header() = default;
        virtual ~Header() = default;

        [[nodiscard]] virtual Format getFormat() const noexcept { return Format::FORMAT_UNKNOWN; }

        virtual void reset() = 0;
        virtual void open(const path_t&, open_mode_t) = 0;
        virtual void close() = 0;

        [[nodiscard]]  virtual size4_t getShape() const noexcept = 0;
        virtual void setShape(size4_t) = 0;

        [[nodiscard]]  virtual stats_t getStats() const noexcept = 0;
        virtual void setStats(stats_t) = 0;

        [[nodiscard]]  virtual float3_t getPixelSize() const noexcept = 0;
        virtual void setPixelSize(float3_t) = 0;

        [[nodiscard]] virtual DataType getDataType() const noexcept = 0;
        virtual void setDataType(DataType) = 0;

        [[nodiscard]]  virtual std::string infoString(bool) const noexcept = 0;

        virtual void read(void* output, DataType data_type, size_t start, size_t end, bool clamp) = 0;
        virtual void readLine(void* output, DataType data_type, size_t start, size_t end, bool clamp) = 0;
        virtual void readShape(void* output, DataType data_type, size4_t offset, size4_t shape, bool clamp) = 0;
        virtual void readSlice(void* output, DataType data_type, size_t start, size_t end, bool clamp) = 0;
        virtual void readAll(void* output, DataType data_type, bool clamp) = 0;

        virtual void write(const void* input, DataType data_type, size_t start, size_t end, bool clamp) = 0;
        virtual void writeLine(const void* input, DataType data_type, size_t start, size_t end, bool clamp) = 0;
        virtual void writeShape(const void* input, DataType data_type, size4_t offset, size4_t shape, bool clamp) = 0;
        virtual void writeSlice(const void* input, DataType data_type, size_t start, size_t end, bool clamp) = 0;
        virtual void writeAll(const void* input, DataType data_type, bool clamp) = 0;
    };
}
