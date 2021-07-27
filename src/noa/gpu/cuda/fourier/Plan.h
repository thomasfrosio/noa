/// \file noa/gpu/cuda/fourier/Plan.h
/// \brief cuFFT plans.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include <cufft.h>

#include "noa/common/Definitions.h"
#include "noa/common/Profiler.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/fourier/Exception.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace noa::cuda::fourier {
    /// Type of transform to plan for.
    enum Type: int {
        R2C = CUFFT_R2C,
        C2R = CUFFT_C2R,
        C2C = CUFFT_C2C
    };

    /// Templated class managing FFT plans in CUDA using cuFFT.
    /// \tparam R   Precision of the transforms, i.e. float or double.
    template<typename R, typename = std::enable_if_t<noa::traits::is_float_v<R>>>
    class Plan {
    private:
        cufftHandle m_plan{};

    private:
        // Offset the type if double precision.
        static cufftType_t getType_(int type) noexcept {
            // In case cuFFT changes this in future version...
            static_assert(CUFFT_Z2Z - CUFFT_C2C == 64 && CUFFT_Z2D - CUFFT_C2R == 64 && CUFFT_D2Z - CUFFT_R2C == 64);
            return std::is_same_v<R, float> ? static_cast<cufftType_t>(type) : static_cast<cufftType_t>(type + 64);
        }

    public:
        /// Creates a plan for a transform of a given \p type, \p shape and \p batch.
        /// \param shape        Logical {fast, medium, slow} shape, in number of elements.
        /// \param batches      Batch size, in number of batches. Batches should be contiguous.
        /// \param type         One of \c R2C, \c C2R or \c C2C.
        /// \param[in] stream   All transforms will be enqueued to this stream.
        ///
        /// \note Plan creation (and the cuFFT APIs in general) is thread safe. However, plans and output data
        ///       should only be access one (host) thread at a time.
        /// \note In-place transforms are allowed. In this case, the execution functions will requires extra
        ///       padding: each row (the fastest dimension) should have an extra float if the dimension is odd, or
        ///       two extra float if it is even. This is the same layout used for the CPU (i.e. FFTW) backend.
        NOA_HOST Plan(size3_t shape, uint batches, Type type, Stream& stream) {
            NOA_PROFILE_FUNCTION();
            int n[3] = {static_cast<int>(shape.z), static_cast<int>(shape.y), static_cast<int>(shape.x)};
            int rank = static_cast<int>(getRank(shape));
            NOA_THROW_IF(cufftPlanMany(&m_plan, rank, n + 3 - rank, nullptr, 1, 0, nullptr, 1, 0,
                                       getType_(type), static_cast<int>(batches)));
            setStream(stream);
        }

        /// Creates a plan for a transform of a given \p type, \p shape and \p batch.
        /// \param shape        Logical {fast, medium, slow} shape, in number of elements.
        /// \param batches      The number of transforms to compute. Data should be contiguous.
        /// \param pitch_in     Pitch in the fast (contiguous) dimension of the input, in number of elements.
        /// \param pitch_out    Pitch in the fast (contiguous) dimension of the output, in number of elements.
        /// \param type         One of \c R2C, \c C2R or \c C2C.
        /// \param[in] stream   All transforms will be enqueued to this stream.
        ///
        /// \note Plan creation (and the cuFFT APIs in general) is thread safe. However, plans and output data
        ///       should only be access one (host) thread at a time.
        /// \note With real transforms, \p pitch_in and \p pitch_out do not refer to the same type. For instance,
        ///       if \p type is PLAN_R2C, \p pitch_in is in number of real elements (i.e. float or double) and
        ///       \p pitch_out is in number of complex elements (i.e. cfloat_t or cdouble_t). In most cases,
        ///       this is used with PtrDevicePadded and these can be computed with PtrDevicePadded::pitch().
        NOA_HOST Plan(size3_t shape, uint batches, size_t pitch_in, size_t pitch_out, Type type, Stream& stream) {
            NOA_PROFILE_FUNCTION();
            int n[3] = {static_cast<int>(shape.z), static_cast<int>(shape.y), static_cast<int>(shape.x)};
            int rank = static_cast<int>(getRank(shape));
            int inembed[3] = {n[0], n[1], static_cast<int>(pitch_in)};
            int onembed[3] = {n[0], n[1], static_cast<int>(pitch_out)};
            NOA_THROW_IF(cufftPlanMany(&m_plan, rank, n + 3 - rank,
                                       inembed + 3 - rank, 1, *(inembed + 3 - rank),
                                       onembed + 3 - rank, 1, *(onembed + 3 - rank),
                                       getType_(type), static_cast<int>(batches)));
            setStream(stream);
        }

        NOA_HOST ~Plan() {
            cufftResult_t err = cufftDestroy(m_plan);
            if (err != CUFFT_SUCCESS && std::uncaught_exceptions() == 0)
                NOA_THROW(toString(err));
        }

        /// Enqueues all future executions of the plan to \p stream.
        NOA_HOST void setStream(Stream& stream) const { NOA_THROW_IF(cufftSetStream(m_plan, stream.get())); }

        /// Gets the underlying cuFFT plan.
        NOA_HOST cufftHandle get() const noexcept { return m_plan; }
    };
}
