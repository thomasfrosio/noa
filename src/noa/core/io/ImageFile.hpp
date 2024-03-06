#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/io/IO.hpp"
#include "noa/core/io/Stats.hpp"

#ifdef NOA_IS_OFFLINE
namespace noa::io::guts {
    class ImageFile {
    public:
        ImageFile() = default;
        virtual ~ImageFile() = default;

    public:
        virtual void reset() = 0;
        virtual void open(const Path&, OpenMode) = 0;
        virtual void close() = 0;

    public: // Getters
        [[nodiscard]] virtual bool is_open() const noexcept = 0;
        [[nodiscard]] virtual const Path& filename() const noexcept = 0;
        [[nodiscard]] virtual std::string info_string(bool) const noexcept = 0;
        [[nodiscard]] virtual Format format() const noexcept { return Format::UNKNOWN; }

    public: // Getters and setters
        [[nodiscard]] virtual Shape4<i64> shape() const noexcept = 0;
        virtual void set_shape(const Shape4<i64>&) = 0;

        [[nodiscard]] virtual Stats<f32> stats() const noexcept = 0;
        virtual void set_stats(Stats<f32>) = 0;

        [[nodiscard]] virtual Vec3<f32> pixel_size() const noexcept = 0;
        virtual void set_pixel_size(Vec3<f32>) = 0;

        [[nodiscard]] virtual DataType dtype() const noexcept = 0;
        virtual void set_dtype(DataType) = 0;

    public:
        virtual void read_elements(void* output, DataType data_type, i64 start, i64 end, bool clamp) = 0;
        virtual void read_slice(void* output, const Strides4<i64>& strides, const Shape4<i64>& shape, DataType data_type, i64 start, bool clamp) = 0;
        virtual void read_slice(void* output, DataType data_type, i64 start, i64 end, bool clamp) = 0;
        virtual void read_all(void* output, const Strides4<i64>& strides, const Shape4<i64>& shape, DataType data_type, bool clamp) = 0;
        virtual void read_all(void* output, DataType data_type, bool clamp) = 0;

        virtual void write_elements(const void* input, DataType data_type, i64 start, i64 end, bool clamp) = 0;
        virtual void write_slice(const void* input, const Strides4<i64>& strides, const Shape4<i64>& shape, DataType data_type, i64 start, bool clamp) = 0;
        virtual void write_slice(const void* input, DataType data_type, i64 start, i64 end, bool clamp) = 0;
        virtual void write_all(const void* input, const Strides4<i64>& strides, const Shape4<i64>& shape, DataType data_type, bool clamp) = 0;
        virtual void write_all(const void* input, DataType data_type, bool clamp) = 0;
    };
}
#endif
