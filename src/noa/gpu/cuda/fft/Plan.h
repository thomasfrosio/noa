#pragma once

#include <cufft.h>

#include "noa/common/Definitions.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"
#include "noa/gpu/cuda/fft/Exception.h"

// TODO Add half_t/chalf_t support. https://docs.nvidia.com/cuda/cufft/index.html#half-precision-transforms

namespace noa::cuda::fft::details {
    std::shared_ptr<cufftHandle> getPlan(cufftType_t type, size4_t shape, int device);
    std::shared_ptr<cufftHandle> getPlan(cufftType_t type, size4_t input_stride, size4_t output_stride,
                                         size4_t shape, int device);
    void cacheClear(int device) noexcept;
    void cacheLimit(int device, size_t count) noexcept;
}

namespace noa::cuda::fft {
    // Returns the optimum even size, greater or equal than "size".
    // An optimum size is an even integer satisfying (2^a)*(3^b)*(5^c)*(7^d).
    // If \p size is >16800, this function will simply return the next even number and will not necessarily
    // satisfy the aforementioned requirements.
    size_t fastSize(size_t size);

    // Returns the optimum DHW logical shape.
    template<typename T>
    inline Int3<T> fastShape(Int3<T> shape) {
        return {shape[0] > 1 ? static_cast<T>(fastSize(static_cast<size_t>(shape[0]))) : shape[0],
                shape[1] > 1 ? static_cast<T>(fastSize(static_cast<size_t>(shape[1]))) : shape[1],
                shape[2] > 1 ? static_cast<T>(fastSize(static_cast<size_t>(shape[2]))) : shape[2]};
    }

    // Returns the optimum BDHW logical shape.
    template<typename T>
    inline Int4<T> fastShape(Int4<T> shape) {
        return {shape[0],
                shape[1] > 1 ? static_cast<T>(fastSize(static_cast<size_t>(shape[1]))) : shape[1],
                shape[2] > 1 ? static_cast<T>(fastSize(static_cast<size_t>(shape[2]))) : shape[2],
                shape[3] > 1 ? static_cast<T>(fastSize(static_cast<size_t>(shape[3]))) : shape[3]};
    }

    // Type of transform to plan for.
    enum Type : int {
        R2C = CUFFT_R2C,
        C2R = CUFFT_C2R,
        C2C = CUFFT_C2C
    };

    class PlanCache {
    public:
        static void cleanup(Device device = Device::current()) {
            details::cacheClear(device.id());
        }
        static void limit(size_t count, Device device = Device::current()) {
            details::cacheLimit(device.id(), count);
        }
    };

    // Template class managing FFT plans in CUDA using cuFFT.
    // NOTE: For R2C/C2R transforms, the 2D/3D arrays should be in the rightmost order for best performance since the
    //       library currently always assumes the non-redundant dimension is the rows, i.e. the rightmost dimension.
    //       The current exception is for column vectors, which are explicitly detected and supported.
    // NOTE: In-place R2C/C2R transforms are allowed. In this case, the input requires extra padding, like in FFTW.
    //       If strides are provided, this padding should be reflected in the strides.
    // NOTE: Plan creation (and the cuFFT APIs in general) is thread safe. However, plans and output data
    //       should only be accessed by one (host) thread at a time.
    // NOTE: For C2C, column-major is also supported.
    //       If strides are not provided, arrays should be C-contiguous.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, double>>>
    class Plan {
    public:
        Plan(Type type, size4_t shape, Device device = Device::current()) {
            m_plan = details::getPlan(getCufftType_(type), shape, device.id());
        }

        Plan(Type type, size4_t input_strides, size4_t output_strides, size4_t shape,
             Device device = Device::current()) {

            const size4_t input_shape = type == Type::C2R ? shape.fft() : shape;
            const size4_t output_shape = type == Type::R2C ? shape.fft() : shape;
            if (indexing::areContiguous(input_strides, input_shape) &&
                indexing::areContiguous(output_strides, output_shape)) {
                m_plan = details::getPlan(getCufftType_(type), shape, device.id());
                return;
            }

            m_plan = details::getPlan(getCufftType_(type), input_strides, output_strides,
                                      shape, device.id());
        }

        // Gets the underlying cuFFT plan.
        [[nodiscard]] cufftHandle get() const noexcept { return *m_plan; }
        [[nodiscard]] const std::shared_ptr<cufftHandle>& share() const noexcept { return m_plan; }

    private:
        std::shared_ptr<cufftHandle> m_plan{};

        // Offset the type if double precision.
        static cufftType_t getCufftType_(int type) noexcept {
            static_assert(CUFFT_Z2Z - CUFFT_C2C == 64 && CUFFT_Z2D - CUFFT_C2R == 64 && CUFFT_D2Z - CUFFT_R2C == 64);
            return std::is_same_v<T, float> ? static_cast<cufftType_t>(type) : static_cast<cufftType_t>(type + 64);
        }
    };
}
