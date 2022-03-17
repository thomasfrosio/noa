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

// TODO Add half_t/chalf_t support. https://docs.nvidia.com/cuda/cufft/index.html#half-precision-transforms

namespace noa::cuda::fft {
    /// Returns the optimum even size, greater or equal than \p size.
    /// \note A optimum size is an even integer satisfying (2^a)*(3^b)*(5^c)*(7^d).
    /// \note If \p size is >16800, this function will simply return the next even number and will not necessarily
    ///       satisfy the aforementioned requirements.
    NOA_HOST size_t fastSize(size_t size);

    /// Returns the optimum rightmost shape.
    /// \note Dimensions of size 0 or 1 are ignored, e.g. {1,51,51} is rounded up to {1,52,52}.
    template<typename T>
    NOA_IH Int3<T> fastShape(Int3<T> shape) {
        return {shape[0] > 1 ? static_cast<T>(fastSize(static_cast<size_t>(shape[0]))) : shape[0],
                shape[1] > 1 ? static_cast<T>(fastSize(static_cast<size_t>(shape[1]))) : shape[1],
                shape[2] > 1 ? static_cast<T>(fastSize(static_cast<size_t>(shape[2]))) : shape[2]};
    }

    /// Returns the optimum rightmost shape.
    /// \note Dimensions of size 0 or 1 are ignored as well as the leftmost dimension, e.g. {1,1,51,51}
    ///       is rounded up to {1,1,52,52}.
    template<typename T>
    NOA_IH Int4<T> fastShape(Int4<T> shape) {
        return {shape[0],
                shape[1] > 1 ? static_cast<T>(fastSize(static_cast<size_t>(shape[1]))) : shape[1],
                shape[2] > 1 ? static_cast<T>(fastSize(static_cast<size_t>(shape[2]))) : shape[2],
                shape[3] > 1 ? static_cast<T>(fastSize(static_cast<size_t>(shape[3]))) : shape[3]};
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
        /// Creates a plan for a transform of a given \p type and \p shape.
        /// \param type             One of \c R2C, \c C2R or \c C2C.
        /// \param shape            Rightmost shape, in number of elements.
        /// \param[in,out] stream   All transforms will be enqueued to this stream, except if bind() is used.
        ///
        /// \note Plan creation (and the cuFFT APIs in general) is thread safe. However, plans and output data
        ///       should only be accessed by one (host) thread at a time.
        /// \note In-place transforms are allowed. In this case and with R2C transforms, the real input requires extra
        ///       padding: the innermost dimension should have an extra real element if the dimension is odd, or
        ///       two extra float if it is even. This is the same layout used for the CPU backend using FFTW3.
        NOA_HOST Plan(Type type, size4_t shape, Stream& stream) {
            NOA_PROFILE_FUNCTION();
            int3_t s_shape(shape.get() + 1);
            int rank = s_shape.ndim();
            NOA_THROW_IF(cufftPlanMany(&m_plan, rank, s_shape.get() + 3 - rank, nullptr, 1, 0, nullptr, 1, 0,
                                       getType_(type), static_cast<int>(shape[0])));
            bind(stream);
        }

        /// Creates a plan for a transform of a given \p type, \p shape and strides.
        /// \param type             One of \c R2C, \c C2R or \c C2C.
        /// \param input_stride     Rightmost strides of the input, in number of elements.
        /// \param output_stride    Rightmost strides of the output, in number of elements.
        /// \param shape            Rightmost shape, in number of elements.
        /// \param[in,out] stream   All transforms will be enqueued to this stream, except if bind() is used.
        ///
        /// \note Plan creation (and the cuFFT APIs in general) is thread safe. However, plans and output data
        ///       should only be accessed by one (host) thread at a time.
        /// \note With real transforms, \p input_stride and \p output_stride do not refer to the same type.
        ///       With \c fft::R2C transforms, \p input_stride is in number of real elements (i.e. float or double) and
        ///       \p output_stride is in number of complex elements (i.e. cfloat_t or cdouble_t). With \c fft::C2R, it
        ///       is the opposite.
        NOA_HOST Plan(Type type, size4_t input_stride, size4_t output_stride, size4_t shape, Stream& stream) {
            NOA_PROFILE_FUNCTION();
            int3_t s_shape(shape.get() + 1);
            const int4_t i_stride(input_stride);
            const int4_t o_stride(output_stride);
            int3_t i_pitch(i_stride.pitch());
            int3_t o_pitch(o_stride.pitch());
            const int rank = s_shape.ndim();
            const int how_many = static_cast<int>(shape[0]);
            const int offset = 3 - rank;
            NOA_THROW_IF(cufftPlanMany(&m_plan, rank, s_shape.get() + offset,
                                       i_pitch.get() + offset, i_stride[3], i_stride[0],
                                       o_pitch.get() + offset, o_stride[3], o_stride[0],
                                       getType_(type), how_many));
            bind(stream);
        }

        NOA_HOST ~Plan() noexcept(false) {
            cufftResult_t err = cufftDestroy(m_plan);
            if (err != CUFFT_SUCCESS && std::uncaught_exceptions() == 0)
                NOA_THROW(toString(err));
        }

        /// Enqueues all future executions of the plan to \p stream.
        NOA_HOST void bind(Stream& stream) const { NOA_THROW_IF(cufftSetStream(m_plan, stream.get())); }

        /// Gets the underlying cuFFT plan.
        NOA_HOST [[nodiscard]] cufftHandle get() const noexcept { return m_plan; }

    private:
        cufftHandle m_plan{};

        // Offset the type if double precision.
        static cufftType_t getType_(int type) noexcept {
            static_assert(CUFFT_Z2Z - CUFFT_C2C == 64 && CUFFT_Z2D - CUFFT_C2R == 64 && CUFFT_D2Z - CUFFT_R2C == 64);
            return std::is_same_v<T, float> ? static_cast<cufftType_t>(type) : static_cast<cufftType_t>(type + 64);
        }
    };
}
