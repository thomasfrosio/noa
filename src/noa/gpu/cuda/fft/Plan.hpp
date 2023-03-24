#pragma once

#include <cufft.h>

#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Stream.hpp"
#include "noa/gpu/cuda/fft/Exception.hpp"

// TODO Add f16/c16 support. https://docs.nvidia.com/cuda/cufft/index.html#half-precision-transforms

namespace noa::cuda::fft::details {
    [[nodiscard]] Shared<cufftHandle> get_plan(cufftType_t type, const Shape4<i64>& shape, i32 device);

    [[nodiscard]] Shared<cufftHandle> get_plan(
            cufftType_t type, Strides4<i64> input_stride, Strides4<i64> output_stride,
            const Shape4<i64>& shape, i32 device);

    void cache_clear(i32 device) noexcept;
    void cache_set_limit(i32 device, i32 count) noexcept;
}

namespace noa::cuda::fft {
    // Returns the optimum even size, greater or equal than "size".
    // An optimum size is an even integer satisfying (2^a)*(3^b)*(5^c)*(7^d).
    // If \p size is >16800, this function will simply return the next even number and will not necessarily
    // satisfy the aforementioned requirements.
    i64 fast_size(i64 size);

    // Returns the optimum BDHW logical shape.
    template<typename T, size_t N>
    Shape<T, N> fast_shape(Shape<T, N> shape) noexcept {
        for (size_t i = 1; i < N; ++i) // ignore batch dimension
            if (shape[i] > 1)
                shape[i] = static_cast<T>(fast_size(static_cast<i64>(shape[i])));
        return shape;
    }

    // Type of transform to plan for.
    enum Type : i32 {
        R2C = CUFFT_R2C,
        C2R = CUFFT_C2R,
        C2C = CUFFT_C2C
    };

    class PlanCache {
    public:
        static void cleanup(Device device = Device::current()) {
            details::cache_clear(device.id());
        }
        static void set_limit(i32 count, Device device = Device::current()) {
            details::cache_set_limit(device.id(), count);
        }
    };

    // Template class managing FFT plans in CUDA using cuFFT.
    // NOTE: For R2C/C2R transforms, the 2D/3D arrays should be in the rightmost order for best performance since the
    //       library currently always assumes the non-redundant dimension is the width, i.e. the rightmost dimension.
    //       The current exception is for column vectors, which are explicitly detected and supported.
    // NOTE: In-place R2C/C2R transforms are allowed. In this case, the input requires extra padding, like in FFTW.
    //       If strides are provided, this padding should be reflected in the strides.
    // NOTE: Plan creation (and the cuFFT APIs in general) is thread safe. However, plans and output data
    //       should only be accessed by one (host) thread at a time.
    // NOTE: For C2C, column-major is also supported.
    //       If strides are not provided, arrays should be C-contiguous.
    template<typename Real, typename = std::enable_if_t<traits::is_any_v<Real, float, double>>>
    class Plan {
    public:
        Plan(Type type, const Shape4<i64>& shape, Device device = Device::current())
                : m_plan(details::get_plan(to_cufft_type_(type), shape, device.id())) {}

        Plan(Type type, const Strides4<i64>& input_strides, const Strides4<i64>& output_strides,
             const Shape4<i64>& shape, Device device = Device::current()) {

            const auto input_shape = type == Type::C2R ? shape.fft() : shape;
            const auto output_shape = type == Type::R2C ? shape.fft() : shape;
            if (noa::indexing::are_contiguous(input_strides, input_shape) &&
                noa::indexing::are_contiguous(output_strides, output_shape)) {
                m_plan = details::get_plan(to_cufft_type_(type), shape, device.id());
            } else {
                m_plan = details::get_plan(to_cufft_type_(type), input_strides, output_strides, shape, device.id());
            }
        }

        // Gets the underlying cuFFT plan.
        [[nodiscard]] cufftHandle get() const noexcept { return *m_plan; }
        [[nodiscard]] const Shared<cufftHandle>& share() const noexcept { return m_plan; }

    private:
        // Offset the type if double precision.
        static cufftType_t to_cufft_type_(i32 type) noexcept {
            static_assert(CUFFT_Z2Z - CUFFT_C2C == 64 && CUFFT_Z2D - CUFFT_C2R == 64 && CUFFT_D2Z - CUFFT_R2C == 64);
            return std::is_same_v<Real, float> ? static_cast<cufftType_t>(type) : static_cast<cufftType_t>(type + 64);
        }

    private:
        Shared<cufftHandle> m_plan{};
    };
}
