/// \file noa/gpu/cuda/fft/Plan.h
/// \brief cuFFT plans.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include <cufft.h>

#include "noa/common/Definitions.h"
#include "noa/common/Profiler.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"
#include "noa/gpu/cuda/fft/Exception.h"

namespace noa::cuda::fft {
    /// Returns the optimum even size, greater or equal than \p size.
    /// \note A optimum size is an even integer satisfying (2^a)*(3^b)*(5^c)*(7^d).
    /// \note If \p size is >16800, this function will simply return the next even number and will not necessarily
    ///       satisfy the aforementioned requirements.
    NOA_HOST size_t fastSize(size_t size);

    /// Returns a "nice" shape.
    /// \note Dimensions of size 0 or 1 are ignored, e.g. {51,51,1} is rounded up to {52,52,1}.
    NOA_IH size3_t getNiceShape(size3_t shape) {
        return {shape.x > 1 ? fastSize(shape.x) : shape.x,
                shape.y > 1 ? fastSize(shape.y) : shape.y,
                shape.z > 1 ? fastSize(shape.z) : shape.z};
    }

    /// Type of transform to plan for.
    enum Type : int {
        R2C = CUFFT_R2C,
        C2R = CUFFT_C2R,
        C2C = CUFFT_C2C
    };

    /// Templated class managing FFT plans in CUDA using cuFFT.
    /// \tparam T   Precision of the transforms, i.e. float or double.
    template<typename T, typename = std::enable_if_t<noa::traits::is_float_v<T>>>
    class Plan {
    public:
        /// Creates a plan for a transform of a given \p type, \p shape and \p batch.
        /// \param type             One of \c R2C, \c C2R or \c C2C.
        /// \param shape            Logical {fast, medium, slow} shape, in number of elements.
        /// \param batches          Number of contiguous transforms to compute.
        /// \param[in,out] stream   All transforms will be enqueued to this stream.
        ///
        /// \note Plan creation (and the cuFFT APIs in general) is thread safe. However, plans and output data
        ///       should only be accessed by one (host) thread at a time.
        /// \note In-place transforms are allowed. In this case, the input of the transform will requires extra
        ///       padding: each row (the fastest dimension) should have an extra float if the dimension is odd, or
        ///       two extra float if it is even. This is the same layout used for the CPU backend using FFTW3.
        NOA_HOST Plan(Type type, size3_t shape, size_t batches, Stream& stream) {
            NOA_PROFILE_FUNCTION();
            int3_t s_shape(shape);
            int n[3] = {s_shape.z, s_shape.y, s_shape.x};
            int rank = ndim(s_shape);
            NOA_THROW_IF(cufftPlanMany(&m_plan, rank, n + 3 - rank, nullptr, 1, 0, nullptr, 1, 0,
                                       getType_(type), static_cast<int>(batches)));
            setStream(stream);
        }

        /// Creates a plan for a transform of a given \p type, \p shape and \p batch.
        /// \param type             One of \c R2C, \c C2R or \c C2C.
        /// \param shape            Logical {fast, medium, slow} shape, in number of elements.
        /// \param batches          Number of contiguous transforms to compute.
        /// \param pitch_in         Pitch of the input, in number of elements.
        /// \param pitch_out        Pitch of the output, in number of elements.
        /// \param[in,out] stream   All transforms will be enqueued to this stream.
        ///
        /// \note Plan creation (and the cuFFT APIs in general) is thread safe. However, plans and output data
        ///       should only be accessed by one (host) thread at a time.
        /// \note With real transforms, \p pitch_in and \p pitch_out do not refer to the same type.
        ///       With \c fft::R2C transforms, \p pitch_in is in number of real elements (i.e. float or double) and
        ///       \p pitch_out is in number of complex elements (i.e. cfloat_t or cdouble_t). With \c fft::C2R, it
        ///       is the opposite.
        NOA_HOST Plan(Type type, size3_t shape, size_t batches, size_t pitch_in, size_t pitch_out, Stream& stream) {
            NOA_PROFILE_FUNCTION();
            int3_t s_shape(shape);
            int n[3] = {s_shape.z, s_shape.y, s_shape.x};
            int rank = ndim(s_shape);
            int inembed[3] = {n[0], n[1], static_cast<int>(pitch_in)};
            int onembed[3] = {n[0], n[1], static_cast<int>(pitch_out)};
            NOA_THROW_IF(cufftPlanMany(&m_plan, rank, n + 3 - rank,
                                       inembed + 3 - rank, 1, *(inembed + 3 - rank),
                                       onembed + 3 - rank, 1, *(onembed + 3 - rank),
                                       getType_(type), static_cast<int>(batches)));
            setStream(stream);
        }

        NOA_HOST ~Plan() noexcept(false) {
            cufftResult_t err = cufftDestroy(m_plan);
            if (err != CUFFT_SUCCESS && std::uncaught_exceptions() == 0)
                NOA_THROW(toString(err));
        }

        /// Enqueues all future executions of the plan to \p stream.
        NOA_HOST void setStream(Stream& stream) const { NOA_THROW_IF(cufftSetStream(m_plan, stream.get())); }

        /// Gets the underlying cuFFT plan.
        NOA_HOST [[nodiscard]] cufftHandle get() const noexcept { return m_plan; }

    private:
        cufftHandle m_plan{};

        // Offset the type if double precision.
        static cufftType_t getType_(int type) noexcept {
            // In case cuFFT changes this in future version...
            static_assert(CUFFT_Z2Z - CUFFT_C2C == 64 && CUFFT_Z2D - CUFFT_C2R == 64 && CUFFT_D2Z - CUFFT_R2C == 64);
            return std::is_same_v<T, float> ? static_cast<cufftType_t>(type) : static_cast<cufftType_t>(type + 64);
        }
    };
}
