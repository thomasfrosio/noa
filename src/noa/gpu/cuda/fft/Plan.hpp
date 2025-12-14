#pragma once

#include <memory>

#include "noa/core/indexing/Layout.hpp"
#include "noa/core/types/Complex.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/Enums.hpp"
#include "noa/gpu/cuda/Device.hpp"
#include "noa/gpu/cuda/Stream.hpp"

// TODO Add f16/c16 support. https://docs.nvidia.com/cuda/cufft/index.html#half-precision-transforms

namespace noa::cuda::fft {
    /// Returns the optimum even size, greater or equal than "size".
    /// An optimum size is an even integer satisfying (2^a)*(3^b)*(5^c)*(7^d).
    /// If \p size is >16800, this function will simply return the next even number and will not necessarily
    /// satisfy the aforementioned requirements.
    auto fast_size(isize size) -> isize;

    /// Type of transform to plan for.
    enum class Type : i32 {
        R2C,
        C2R,
        C2C,
    };

    /// Manages FFT plans.
    auto clear_cache(Device device) noexcept -> i32;
    auto cache_limit(Device device) noexcept -> i32;
    auto cache_size(Device device) noexcept -> i32;
    auto set_cache_limit(Device device, i32 count) noexcept -> i32;
    auto workspace_left_to_allocate(Device device) noexcept -> isize;
    auto set_workspace(Device device, const std::shared_ptr<std::byte[]>& buffer, isize buffer_size) -> i32;
}

namespace noa::cuda::fft::details {
    [[nodiscard]] auto get_plan(
        Type type,
        bool is_single_precision,
        const Shape4& shape,
        Device device,
        bool save_in_cache,
        bool plan_only,
        bool record_workspace
    ) -> std::shared_ptr<void>;

    [[nodiscard]] auto get_plan(
        Type type,
        bool is_single_precision,
        Strides4 input_stride,
        Strides4 output_stride,
        const Shape4& shape,
        Device device,
        bool save_in_cache,
        bool plan_only,
        bool record_workspace
    ) -> std::shared_ptr<void>;
}

namespace noa::cuda::fft {
    /// Template class managing FFT plans in CUDA using cuFFT.
    /// \note For r2c/c2r transforms, the 2d/3d arrays should be in the rightmost order for best performance since the
    ///       library currently always assumes the non-redundant dimension is the width, i.e. the rightmost dimension.
    ///       The current exception is for column vectors, which are explicitly detected and converted to row vectors.
    /// \note In-place r2c/c2r transforms are allowed. In this case, the input requires extra padding, like in FFTW.
    ///       If strides are provided, this padding should be reflected in the strides.
    /// \note Plan creation (and the cuFFT APIs in general) is thread safe. However, plans and output data
    ///       should only be accessed by one (host) thread at a time. As such and for simplicity, we hold a per host-
    ///       thread cache (and each GPU has its own cache, of course).
    /// \note For c2c, column-major is also supported.
    ///       If strides are not provided, arrays are assumed to be C-contiguous.
    template<typename T>
    class Plan {
        static_assert(nt::any_of<T, f32, f64>);
        static constexpr bool is_single_precision = std::same_as<T, f32>;

    public:
        Plan(Type type,
             const Strides4& input_strides,
             const Strides4& output_strides,
             const Shape4& shape, Device device,
             bool save_to_cache, bool plan_only, bool record_workspace
        ) {
            const auto input_shape = type == Type::C2R ? shape.rfft() : shape;
            const auto output_shape = type == Type::R2C ? shape.rfft() : shape;
            if (ni::are_contiguous(input_strides, input_shape) and
                ni::are_contiguous(output_strides, output_shape)) {
                m_plan = details::get_plan(
                    type, is_single_precision, shape, device,
                    save_to_cache, plan_only, record_workspace);
            } else {
                m_plan = details::get_plan(
                    type, is_single_precision, input_strides, output_strides, shape, device,
                    save_to_cache, plan_only, record_workspace);
            }
        }

        /// The shape, strides, alignment and type (that includes in-place or not) should match the plan.
        void execute(T* input, Complex<T>* output, Stream& stream) &&;
        void execute(Complex<T>* input, T* output, Stream& stream) &&;
        void execute(Complex<T>* input, Complex<T>* output, nf::Sign sign, Stream& stream) &&;

    private:
        std::shared_ptr<void> m_plan{};
    };
}
