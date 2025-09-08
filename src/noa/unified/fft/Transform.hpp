#pragma once

#include "noa/unified/Array.hpp"
#include "noa/unified/Factory.hpp"
#include "noa/unified/Ewise.hpp"
#include "noa/unified/Session.hpp"

#include "noa/cpu/fft/Transforms.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/fft/Transforms.hpp"
#endif

namespace noa::fft {
    static constexpr Norm NORM_DEFAULT = Norm::FORWARD;

    struct FFTOptions {
        /// Normalization mode.
        Norm norm = NORM_DEFAULT;

        /// Whether this transform should be cached.
        /// On the CPU (built with FFTW3), this does nothing. See Session for more details.
        bool cache_plan = true;

        /// If true, the transform isn't computed; only the plan is created.
        /// This is intended for caching plans early. As such, using this option with cache_plan=false does nothing.
        bool plan_only = false;

        /// Whether the plan workspace (the memory buffer allocated for the plan) should be postponed and
        /// shared with later transforms also using this option.
        /// \details On CPU (built with FFTW3), this is equivalent to plan_only=true since the workspace cannot be set.
        /// However, note that FFTW3 is very memory efficiency, and it shouldn't be any issue storing many plans in
        /// the first place.
        /// \details On CUDA, calling a transform with record_and_share_workspace=true creates the plan, records the
        /// workspace size, but postpones the allocation to the next transform call. If a suitable plan already exists
        /// in the cache, nothing is done. If multiple transforms are called sequentially with record_workspace=true,
        /// they will share the same workspace and should therefore only be executed sequentially (e.g., on the same
        /// stream). When a transform is later called with record_and_share_workspace=false (the default), the workspace
        /// of previous cached plans is allocated; then the transform is run as usual.
        /// \example
        /// \code
        /// // Prepare for many transforms of different sizes and make them share the same workspace.
        /// noa::fft::clear_cache(); // optional
        /// noa::fft::set_cache_limit(100); // len(inputs) <= 100; cache everything
        /// for (auto&& [input, output]: noa::zip(inputs, outputs))
        ///     noa::fft::r2c(input, output, {.record_and_share_workspace = true}); // plan only
        /// ...
        /// // Execute the transforms sequentially. The transforms are picked from the cache and
        /// // use the same workspace, potentially saving a lot of memory.
        /// for (auto&& [input, output]: noa::zip(inputs, outputs))
        ///     noa::fft::r2c(input, output);
        /// \endcode
        bool record_and_share_workspace = false;
    };

    /// Clears the FFT cache for a given device.
    /// Returns how many "plans" were cleared from the cache, or returns zero. See Session for more details.
    /// \warning This function doesn't synchronize before clearing the cache, so the caller should make sure
    ///          that none of the plans are being used. This can be easily done by synchronizing the relevant
    ///          streams or the device.
    inline auto clear_cache(Device device) -> i64 { return Session::clear_fft_cache(device); }

    /// Sets the maximum number of plans the FFT cache can hold on a given device.
    /// Returns how many plans were cleared from the resizing of the cache. See Session for more details.
    /// \note Some backends may not allow setting this parameter, in which case -1 is returned.
    inline auto set_cache_limit(i64 count, Device device) -> i64 { return Session::set_fft_cache_limit(count, device); }

    /// Gets the maximum number of plans the FFT cache can hold on a given device.
    /// \note Some backends may not support retrieving this parameter or may have a dynamic cache without a
    ///       fixed capacity. In these cases, -1 is returned. See Session for more details.
    [[nodiscard]] inline auto cache_limit(Device device) -> i64 { return Session::fft_cache_limit(device); }

    /// Returns the current cache size, i.e. how many plans are currently cached.
    /// \note This number may be different from the number of transforms that have been launched.
    ///       Indeed, FFT planners (like FFTW) may create and cache many plans for a single transformation.
    ///       Similarly, transforms may be divided internally into multiple transforms. See Session for more details.
    /// \note Some backends may not support retrieving this parameter, in which case, -1 is returned.
    [[nodiscard]] inline auto cache_size(Device device) -> i64 { return Session::fft_cache_size(device); }
}

namespace noa::fft::guts {
    template<typename T>
    void normalize(T&& array, const Shape4<i64>& shape, Sign sign, Norm norm) {
        using real_t = nt::mutable_value_type_twice_t<T>;
        const auto count = static_cast<real_t>(product(shape.pop_front()));
        const auto scale = norm == Norm::ORTHO ? sqrt(count) : count;
        if ((sign == Sign::FORWARD and (norm == Norm::FORWARD or norm == Norm::ORTHO)) or
            (sign == Sign::BACKWARD and (norm == Norm::BACKWARD or norm == Norm::ORTHO))) {
            ewise({}, std::forward<T>(array), Scale{1 / scale});
        }
    }
}

namespace noa::fft {
    /// Computes the forward r2c transform of (batched) 2d/3d array(s) or column/row vector(s).
    /// \param[in] input    Real space array.
    /// \param[out] output  Non-redundant non-centered, aka "h" layout, FFT(s).
    /// \note In-place transforms are allowed if the \p input is appropriately padded to account
    ///       for the extra one (if odd) or two (if even) real element along the width dimension.
    template<nt::varray_decay_of_almost_any<f32, f64> Input,
             nt::varray_decay_of_any<Complex<nt::mutable_value_type_t<Input>>> Output>
    void r2c(Input&& input, Output&& output, FFTOptions options = {}) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        const auto logical_shape = input.shape();
        check(vall(Equal{}, output.shape(), logical_shape.rfft()),
              "Given the real input with a shape of {}, the non-redundant shape of the complex output "
              "should be {}, but got {}", logical_shape, logical_shape.rfft(), output.shape());

        const Device device = output.device();
        check(device == input.device(),
              "The input and output arrays must be on the same device, but got input:device={}, output:device={}",
              input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto n_threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=, real = std::forward<Input>(input)] {
                constexpr auto flags = noa::cpu::fft::ESTIMATE | noa::cpu::fft::PRESERVE_INPUT;
                noa::cpu::fft::r2c(
                    real.get(), real.strides(),
                    output.get(), output.strides(),
                    real.shape(), flags, options.plan_only, n_threads
                );
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            noa::cuda::fft::r2c(
                input.get(), input.strides(),
                output.get(), output.strides(), input.shape(),
                options.cache_plan, options.plan_only, options.record_and_share_workspace,
                cuda_stream
            );
            cuda_stream.enqueue_attach(std::forward<Input>(input), output);
            #else
            panic_no_gpu_backend();
            #endif
        }
        if (not options.plan_only and not options.record_and_share_workspace)
            guts::normalize(std::forward<Output>(output), logical_shape, Sign::FORWARD, options.norm);
    }

    /// Computes the forward r2c transform of (batched) 2d/3d array(s) or column/row vector(s).
    /// \param[in] input    Real space array.
    /// \return Non-redundant non-centered, aka "h" layout, FFT(s).
    template<nt::varray_decay_of_almost_any<f32, f64> Input>
    [[nodiscard]] auto r2c(Input&& input, FFTOptions options = {}) {
        using real_t = nt::mutable_value_type_twice_t<Input>;
        Array<Complex<real_t>> output(input.shape().rfft(), input.options());
        r2c(std::forward<Input>(input), output, options);
        return output;
    }

    /// Computes the backward c2r transform.
    /// \param[in,out] input    Non-redundant non-centered, aka "h" layout, FFT(s).
    /// \param[out] output      Real space array.
    /// \note In-place transforms are allowed if the \p output is appropriately padded to account
    ///       for the extra one (if odd) or two (if even) real element in the width dimension.
    /// \note For multidimensional c2r transforms, the input is not preserved.
    template<nt::varray_decay_of_almost_any<c32, c64> Input,
             nt::varray_decay_of_any<nt::mutable_value_type_twice_t<Input>> Output>
    void c2r(Input&& input, Output&& output, FFTOptions options = {}) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        const auto logical_shape = output.shape();
        check(vall(Equal{}, input.shape(), logical_shape.rfft()),
              "Given the real output with a shape of {}, the non-redundant shape of the complex input "
              "should be {}, but got {}", logical_shape, logical_shape.rfft(), input.shape());

        const Device device = output.device();
        check(device == input.device(),
              "The input and output arrays must be on the same device, but got input:device={}, output:device={}",
              input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=, complex = std::forward<Input>(input)] {
                constexpr auto flags = noa::cpu::fft::ESTIMATE;
                noa::cpu::fft::c2r(
                    complex.get(), complex.strides(),
                    output.get(), output.strides(),
                    output.shape(), flags, options.plan_only, threads
                );
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            noa::cuda::fft::c2r(
                input.get(), input.strides(),
                output.get(), output.strides(), output.shape(),
                options.cache_plan, options.plan_only, options.record_and_share_workspace,
                cuda_stream
            );
            cuda_stream.enqueue_attach(std::forward<Input>(input), output);
            #else
            panic_no_gpu_backend();
            #endif
        }
        if (not options.plan_only and not options.record_and_share_workspace)
            guts::normalize(std::forward<Output>(output), logical_shape, Sign::BACKWARD, options.norm);
    }

    /// Computes the backward c2r transform.
    /// \param[in,out] input    Non-redundant non-centered, aka "h" layout, FFT(s).
    /// \param shape            BDHW logical shape of \p input.
    /// \return Real space array.
    /// \note For multidimensional c2r transforms, the input is not preserved.
    template<nt::varray_decay_of_almost_any<c32, c64> Input>
    [[nodiscard]] auto c2r(Input&& input, const Shape4<i64> shape, FFTOptions options = {}) {
        using real_t = nt::mutable_value_type_twice_t<Input>;
        Array<real_t> output(shape, input.options());
        c2r(std::forward<Input>(input), output, options);
        return output;
    }

    /// Computes the c2c transform.
    /// \param[in] input    Input complex data.
    /// \param[out] output  Non-centered, aka "f" layout, FFT(s).
    /// \param sign         Sign of the exponent in the formula that defines the Fourier transform.
    ///                     It can be −1 (\c Sign::FORWARD) or +1 (\c Sign::BACKWARD).
    /// \note In-place transforms are allowed.
    template<nt::varray_decay_of_almost_any<c32, c64> Input,
             nt::varray_decay_of_any<nt::mutable_value_type_t<Input>> Output>
    void c2c(Input&& input, Output&& output, Sign sign, FFTOptions options = {}) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        const auto logical_shape = input.shape();
        check(vall(Equal{}, logical_shape, output.shape()),
              "The input and output shape should match (no broadcasting allowed), but got input {} and output {}",
              logical_shape, output.shape());

        const Device device = output.device();
        check(device == input.device(),
              "The input and output arrays must be on the same device, but got input:device={}, output:device={}",
              input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=, i = std::forward<Input>(input)] {
                constexpr auto flags = noa::cpu::fft::ESTIMATE | noa::cpu::fft::PRESERVE_INPUT;
                noa::cpu::fft::c2c(
                    i.get(), i.strides(),
                    output.get(), output.strides(),
                    i.shape(), sign, flags, options.plan_only, threads
                    );
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            noa::cuda::fft::c2c(
                input.get(), input.strides(),
                output.get(), output.strides(), input.shape(), sign,
                options.cache_plan, options.plan_only, options.record_and_share_workspace,
                cuda_stream
            );
            cuda_stream.enqueue_attach(std::forward<Input>(input), output);
            #else
            panic_no_gpu_backend();
            #endif
        }
        if (not options.plan_only and not options.record_and_share_workspace)
            guts::normalize(std::forward<Output>(output), logical_shape, sign, options.norm);
    }

    /// Computes the c2c transform.
    /// \param[in] input    Input complex data.
    /// \param sign         Sign of the exponent in the formula that defines the Fourier transform.
    ///                     It can be −1 (\c Sign::FORWARD) or +1 (\c Sign::BACKWARD).
    /// \return Non-centered, aka "f" layout, FFT(s).
    template<nt::varray_decay_of_almost_any<c32, c64> Input>
    [[nodiscard]] auto c2c(Input&& input, Sign sign, FFTOptions options = {}) {
        using complex_t = nt::mutable_value_type_t<Input>;
        Array<complex_t> output(input.shape(), input.options());
        c2c(std::forward<Input>(input), output, sign, options);
        return output;
    }
}
