#pragma once

#include <cufft.h>

#include "noa/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/fourier/Exception.h"
#include "noa/gpu/cuda/util/Stream.h"

#include <cufft.h>

namespace Noa::CUDA::Fourier {
    /// Type of transform. @note Double precision uses the same @a Type.
    enum class Type : int {
        R2C = CUFFT_R2C,
        C2R = CUFFT_C2R,
        C2C = CUFFT_C2C
    };

    /**
     * Templated class managing FFT plans in CUDA.
     * @tparam R    Real type of the transforms, i.e. float or double.
     */
    template<typename R, typename = std::enable_if_t<Noa::Traits::is_float_v<R>>>
    class Plan {
    private:
        cufftHandle m_plan{};

    private:
        // Offset the type if double precision.
        static cufftType_t getType_(Type type) noexcept {
            // In case cuFFT changes this in future version...
            static_assert(CUFFT_Z2Z - CUFFT_C2C == 64 && CUFFT_Z2D - CUFFT_C2R == 64 && CUFFT_D2Z - CUFFT_R2C == 64);
            return std::is_same_v<R, float> ?
                   static_cast<cufftType_t>(type) :
                   static_cast<cufftType_t>(static_cast<int>(type) + 64);
        }

    public:
        /**
         * Creates a plan for a transform of a given @a type, @a shape and @a batch.
         * @param shape     Logical {fast, medium, slow} shape, in number of elements.
         * @param batch     Batch size, in number of batches. Batches should be contiguous.
         * @param type      One of @a Type, i.e. Type::R2C, Type::C2R, Type::C2C.
         * @note Plan creation (and the cuFFT APIs in general) is thread safe. However, plans and output data
         *       should only be access one (host) thread at a time.
         */
        NOA_HOST Plan(size3_t shape, uint batch, Type type) {
            int n[3] = {static_cast<int>(shape.z), static_cast<int>(shape.y), static_cast<int>(shape.x)};
            int rank = static_cast<int>(getRank(shape));
            NOA_THROW_IF(cufftPlanMany(&m_plan, rank, n + 3 - rank, nullptr, 0, 0, nullptr, 0, 0,
                                       getType_(type), static_cast<int>(batch)));
        }

        /**
         * Creates a plan for a transform of a given @a type, @a shape and @a batch.
         * @param shape     Logical {fast, medium, slow} shape, in number of elements.
         * @param pitch_in  Pitch in the fast (contiguous) dimension of the input, in number of elements.
         * @param pitch_out Pitch in the fast (contiguous) dimension of the output, in number of elements.
         * @param batch     Batch size, in number of batches. Batches should be contiguous.
         * @param type      One of @a Type, i.e. Type::R2C, Type::C2R, Type::C2C.
         * @note Plan creation (and the cuFFT APIs in general) is thread safe. However, plans and output data
         *       should only be access one (host) thread at a time.
         * @warning With real transforms, @a pitch_in and @a pitch_out do not refer to the same type. For instance,
         *          if @a type is Type::R2C, @a pitch_in is in number of real elements (i.e. float or double) and
         *          @a pitch_out is in number of complex elements (i.e. cfloat_t or cdouble_t). In most cases,
         *          this is used with PtrDevicePadded and these pitches can be computed using the member function
         *          PtrDevicePadded::pitch().
         */
        NOA_HOST Plan(size3_t shape, uint batch, size_t pitch_in, size_t pitch_out, Type type) {
            int n[3] = {static_cast<int>(shape.z), static_cast<int>(shape.y), static_cast<int>(shape.x)};
            int rank = static_cast<int>(getRank(shape));
            int inembed[3] = {n[0], n[1], static_cast<int>(pitch_in)};
            int onembed[3] = {n[0], n[1], static_cast<int>(pitch_out)};
            NOA_THROW_IF(cufftPlanMany(&m_plan, rank, n + 3 - rank,
                                       inembed + 3 - rank, 1, *(inembed + 3 - rank),
                                       onembed + 3 - rank, 1, *(onembed + 3 - rank),
                                       getType_(type), static_cast<int>(batch)));
        }

        NOA_IH void setStream(Stream& stream) const { NOA_THROW_IF(cufftSetStream(m_plan, stream.get())); }

        NOA_HOST ~Plan() { NOA_THROW_IF(cufftDestroy(m_plan)); }

        NOA_IH cufftHandle get() const noexcept { return m_plan; }
    };
}
