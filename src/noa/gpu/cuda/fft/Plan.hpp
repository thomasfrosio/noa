#pragma once

#include "noa/core/Config.hpp"

#ifdef NOA_IS_OFFLINE
#include "noa/core/indexing/Layout.hpp"
#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Stream.hpp"
#include "noa/gpu/cuda/fft/Exception.hpp"

// TODO Add f16/c16 support. https://docs.nvidia.com/cuda/cufft/index.html#half-precision-transforms

namespace noa::cuda::fft {
    /// Returns the optimum even size, greater or equal than "size".
    /// An optimum size is an even integer satisfying (2^a)*(3^b)*(5^c)*(7^d).
    /// If \p size is >16800, this function will simply return the next even number and will not necessarily
    /// satisfy the aforementioned requirements.
    i64 fast_size(i64 size);

    /// Type of transform to plan for.
    enum class Type : i32 {
        R2C,
        C2R,
        C2C,
    };

    /// Manages FFT plans.
    i32 clear_caches(i32 device) noexcept;
    void set_cache_limit(i32 device, i32 count) noexcept;
}

namespace noa::cuda::fft::guts {
    [[nodiscard]] std::shared_ptr<void> get_plan(
            Type type,
            bool is_single_precision,
            const Shape4<i64>& shape,
            i32 device,
            bool save_in_cache);

    [[nodiscard]] std::shared_ptr<void> get_plan(
            Type type,
            bool is_single_precision,
            Strides4<i64> input_stride,
            Strides4<i64> output_stride,
            const Shape4<i64>& shape,
            i32 device,
            bool save_in_cache);
}

namespace noa::cuda::fft {
    /// Template class managing FFT plans in CUDA using cuFFT.
    /// \note For r2c/c2r transforms, the 2d/3d arrays should be in the rightmost order for best performance since the
    ///       library currently always assumes the non-redundant dimension is the width, i.e. the rightmost dimension.
    ///       The current exception is for column vectors, which are explicitly detected and converted to row vectors.
    /// \note In-place r2c/c2r transforms are allowed. In this case, the input requires extra padding, like in FFTW.
    ///       If strides are provided, this padding should be reflected in the strides.
    /// \note Plan creation (and the cuFFT APIs in general) is thread safe. However, plans and output data
    ///       should only be accessed by one (host) thread at a time. As such and for simplicity, we hold a per-host-
    ///      thread cache (and each GPU has its own cache of course).
    /// \note For c2c, column-major is also supported.
    ///       If strides are not provided, arrays are assumed to be C-contiguous.
    template<typename Real>
    class Plan {
        static_assert(nt::is_any_v<Real, f32, f64>);
        static constexpr bool is_single_precision = std::is_same_v<Real, f32>;

    public:
        Plan(Type type,
             const Shape4<i64>& shape,
             Device device = Device::current(),
             bool save_to_cache = true
        ) : m_plan(guts::get_plan(type, is_single_precision, shape, device.id(), save_to_cache)) {}

        Plan(Type type,
             const Strides4<i64>& input_strides,
             const Strides4<i64>& output_strides,
             const Shape4<i64>& shape,
             Device device = Device::current(),
             bool save_to_cache = true
        ) {
            const auto input_shape = type == Type::C2R ? shape.rfft() : shape;
            const auto output_shape = type == Type::R2C ? shape.rfft() : shape;
            if (ni::are_contiguous(input_strides, input_shape) and
                ni::are_contiguous(output_strides, output_shape)) {
                m_plan = guts::get_plan(
                        type, is_single_precision, shape, device.id(), save_to_cache);
            } else {
                m_plan = guts::get_plan(
                        type, is_single_precision, input_strides, output_strides, shape, device.id(), save_to_cache);
            }
        }

        /// The shape, strides, alignment and type (that includes in-place or not) should match the plan.
        void execute(Real* input, Complex<Real>* output, Stream& stream);
        void execute(Complex<Real>* input, Real* output, Stream& stream);
        void execute(Complex<Real>* input, Complex<Real>* output, noa::fft::Sign sign, Stream& stream);

    private:
        std::shared_ptr<void> m_plan{};
    };
}
#endif
