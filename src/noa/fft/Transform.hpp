#pragma once

#include "noa/runtime/Array.hpp"
#include "noa/runtime/Ewise.hpp"

#include "noa/fft/core/Transform.hpp"
#include "noa/fft/cpu/Transform.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/fft/cuda/Transform.hpp"
#endif

namespace noa::fft {
    static constexpr Norm NORM_DEFAULT = Norm::FORWARD;

    /// Options for FFTs.
    ///
    /// \details FFT plans:
    /// Backends guarantee that FFT plans are cached so that the transform functions (e.g. noa::fft::r2c)
    /// are efficient. However, since these plans can take a lot of memory, we provide ways for users to control and
    /// query this cache. Some backends (and the libraries they use) may be more flexible than others.
    /// CUDA:
    ///     - We manage the caching system explicitly, which offers maximum flexibility. The API below allows turning
    ///       on and off the cache, resizing it, and querying its current state. The free-functions we provide in
    ///       noa::fft also allow turning on and off the cache temporarily for a particular transform (see
    ///       noa::fft::FFTOptions.cache_plan).
    ///     - Plan creation (and the cuFFT APIs in general) is thread safe. However, plans and output data should only
    ///       be accessed by one (host) thread at a time. As such and for simplicity, we hold a per host-thread and
    ///       per-device cache. In cuFFT, a single plan encodes both the forward and backward transform. For instance,
    ///       noa::r2c and noa::c2r on the same arrays only require (and cache) a single plan.
    ///     - Plans often allocate a workspace on the device. Holding many plans in the cache quite often results
    ///       in significant memory usage. If plans are to be executed sequentially (e.g., on the same stream), the
    ///       workspace can be shared across multiple plans (see noa::fft::FFTOptions.record_and_share_workspace).
    /// CPU-FFTW3:
    ///     - The cache is handled by FFTW3's wisdom. We don't manage it, and its flexibility is limited.
    ///       The user can only clear the cache or query its size.
    ///     - The cache is per device (CPU), globally shared within the application, and has no fixed capacity.
    ///       Forward and backward transforms are considered different plans in FFTW3. In fact, a single transform
    ///       can generate many (>16) entries in the cache.
    ///     - FFTW3's wisdom is well-optimized and doesn't require a lot of memory.
    struct FFTOptions {
        /// The rank of the transform.
        /// See ranked_shape for more details.
        i32 rank = -1;

        /// Normalization mode.
        Norm norm = NORM_DEFAULT;

        /// Whether this transform should be cached.
        /// On the CPU (built with FFTW3), this does nothing.
        bool cache_plan = true;

        /// If true, the transform isn't computed; only the plan is created.
        /// This is intended for caching plans early. As such, using this option with cache_plan=false does nothing.
        bool plan_only = false;

        /// Whether the plan workspace (the memory buffer allocated for the plan) should be postponed and
        /// shared with later transforms also using this option.
        /// \details On CPU (built with FFTW3), this is equivalent to plan_only=true since the workspace cannot be set.
        /// However, note that FFTW3 is very memory efficient, and there shouldn't be any issue storing many plans in
        /// the first place.
        /// \details In CUDA, calling a transform with record_and_share_workspace=true creates the plan, records the
        /// workspace size, but postpones the allocation to the next transform call. If a suitable plan already exists
        /// in the cache, nothing is done. If multiple transforms are called sequentially with record_workspace=true,
        /// they will share the same workspace and should therefore only be executed sequentially (e.g., on the same
        /// stream). When a transform is later called with record_and_share_workspace=false (the default), the workspace
        /// of previous cached plans is allocated; then the transform is run as usual. The allocation can also be
        /// bypassed using the set_workspace function.
        /// \example
        /// \code
        /// // Prepare for many transforms of different sizes and make them share the same workspace.
        /// noa::fft::clear_cache(); // optional
        /// noa::fft::set_cache_limit(100); // len(inputs) <= 100; cache everything
        /// for (auto&& [input, output]: noa::zip(inputs, outputs))
        ///     noa::fft::r2c(input, output, {.record_and_share_workspace = true}); // plan only
        /// ...
        /// // Optional: if called, set_workspace uses my_workspace directly and
        /// // no further allocation will be required to execute the plans below.
        /// noa::fft::set_workspace(input.device(), my_workspace);
        /// ...
        /// // Execute the transforms sequentially.
        /// // If set_workspace wasn't called, the first r2c allocates
        /// // and shares the workspace with every plan missing their workspace.
        /// for (auto&& [input, output]: noa::zip(inputs, outputs))
        ///     noa::fft::r2c(input, output);
        /// \endcode
        bool record_and_share_workspace = false;
    };

    /// Clears the FFT cache for a given device.
    /// Returns how many "plans" were cleared from the cache, or returns zero.
    /// \warning This function doesn't synchronize before clearing the cache, so the caller should make sure
    ///          that none of the plans are being used. This can be easily done by synchronizing the relevant
    ///          streams or the device.
    auto clear_cache(Device device) -> i32;

    /// Sets the maximum number of plans the FFT cache can hold on a given device.
    /// Returns how many plans were cleared from the resizing of the cache.
    /// \note Some backends may not allow setting this parameter, in which case -1 is returned.
    auto set_cache_limit(i32 count, Device device) -> i32;

    /// Gets the maximum number of plans the FFT cache can hold on a given device.
    /// \note Some backends may not support retrieving this parameter or may have a dynamic cache without a
    ///       fixed capacity. In these cases, -1 is returned.
    [[nodiscard]] auto cache_limit(Device device) -> i32;

    /// Returns the current cache size, i.e. how many plans are currently cached.
    /// \note This number may be different from the number of transforms that have been launched.
    ///       Indeed, FFT planners (like FFTW) may create and cache many plans for a single transformation.
    ///       Similarly, transforms may be divided internally into multiple transforms.
    /// \note Some backends may not support retrieving this parameter, in which case, -1 is returned.
    [[nodiscard]] auto cache_size(Device device) -> i32;

    /// Returns the number of bytes that are left to allocate from previous plan creations.
    /// \see noa::fft::FFTOptions.record_and_share_workspace for more details.
    /// \note Some backends (e.g., FFTW) do not support explicit workspace management and always return 0.
    [[nodiscard]] auto workspace_left_to_allocate(Device device) -> isize;

    namespace details {
        auto set_workspace(Device device, std::shared_ptr<std::byte[]>&& buffer, isize buffer_bytes) -> i32;
        auto set_workspace(Device device, const std::shared_ptr<std::byte[]>& buffer, isize buffer_bytes) -> i32;
    }

    /// Assigns cached plans without a workspace to this buffer.
    /// \details The buffer should be on the device and contiguous, otherwise an error will be thrown. The number of
    ///          plans that have been assigned to the given buffer is returned. If the size of the buffer is less than
    ///          workspace_left_to_allocate(device), the buffer cannot be used and 0 is returned.
    /// \see noa::fft::FFTOptions.record_and_share_workspace for more details.
    /// \note Some backends (e.g., FFTW) do not support explicit workspace management and always return 0.
    template<nt::array_decay T>
    auto set_workspace(Device device, T&& buffer) -> i32 {
        check(buffer.device() == device,
              "The buffer should be on the device, but got device={} and buffer:device={}",
              device, buffer.device());
        check(buffer.is_contiguous(), "The workspace should be a contiguous array");
        return details::set_workspace(
            device,
            std::reinterpret_pointer_cast<std::byte[]>(std::forward<T>(buffer).share()),
            buffer.ssize() * static_cast<isize>(sizeof(nt::value_type_t<T>))
        );
    }

    /// Assigns cached plans without a workspace to a newly allocated buffer.
    /// \see set_workspace for more details.
    inline auto allocate_workspace(Device device, Allocator allocator) -> i32 {
        if (const auto n_bytes = workspace_left_to_allocate(device); n_bytes > 0)
            return set_workspace(device, Array<std::byte>(n_bytes, {.device = device, .allocator = allocator}));
        return 0;
    }
}

namespace noa::fft::details {
    template<typename T>
    void normalize(T&& array, const Shape4& shape, Sign sign, Norm norm) {
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
    /// Computes the forward r2c transform of (batched) 1d/2d/3d array(s).
    /// \param[in] input:
    ///     Real space array.
    ///     Should be reshapeable to 4D.
    /// \param[out] output:
    ///     Non-redundant non-centered, aka "h" layout, FFT(s).
    ///     Should be reshapeable to 4D.
    /// \param options:
    ///     FFT options.
    /// \note
    ///     In-place transforms are allowed if the input is appropriately padded to account
    ///     for the extra one (if odd) or two (if even) real element along the width dimension.
    template<nt::array_decay_of_almost_any<f32, f64> Input,
             nt::array_decay_of_any<Complex<nt::mutable_value_type_t<Input>>> Output>
        requires nt::array_decay_with_same_nd<Input, Output>
    void r2c(Input&& input, Output&& output, FFTOptions options = {}) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        check(output.shape() == input.shape().rfft(),
              "Given the real input with a shape of {}, the non-redundant shape of the complex output "
              "should be {}, but got {}", input.shape(), input.shape().rfft(), output.shape());

        const Device device = output.device();
        check(device == input.device(),
              "The input and output arrays must be on the same device, but got input:device={}, output:device={}",
              input.device(), device);

        auto input_4d = input.span().template as_nd<4>();
        auto output_4d = output.span().template as_nd<4>();
        auto shape_4d = input_4d.shape();
        details::prepare_ranked_spans(input_4d, output_4d, shape_4d, options.rank);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto n_threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=, handles =
                nd::extract_shared_handle_from_arrays(
                    noa::forward_as_tuple(
                        std::forward<Input>(input), std::forward<Output>(output)))] {
                constexpr auto flags = noa::fft::cpu::ESTIMATE | noa::fft::cpu::PRESERVE_INPUT;
                noa::fft::cpu::r2c(
                    input_4d.get(), input_4d.strides(),
                    output_4d.get(), output_4d.strides(),
                    shape_4d, flags, options.plan_only or options.record_and_share_workspace,
                    n_threads
                );
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            noa::fft::cuda::r2c(
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
            details::normalize(std::forward<Output>(output), shape_4d, Sign::FORWARD, options.norm);
    }

    /// Computes the forward r2c transform of (batched) 1d/2d/3d array(s).
    /// \param[in] input:   Real space array.
    /// \return Non-redundant non-centered, aka "h" layout, FFT(s).
    template<nt::array_decay_of_almost_any<f32, f64> Input>
    [[nodiscard]] auto r2c(Input&& input, FFTOptions options = {}) {
        using real_t = nt::mutable_value_type_twice_t<Input>;
        auto output = Array<Complex<real_t>, nt::array_size_v<Input>>(input.shape().rfft(), input.options());
        r2c(std::forward<Input>(input), output, options);
        return output;
    }

    /// Computes the backward c2r transform of (batched) 1d/2d/3d array(s).
    /// \param[in,out] input:
    ///     Non-redundant non-centered, aka "h" layout, FFT(s).
    ///     Should be reshapeable to 4D.
    /// \param[out] output:
    ///     Real space array.
    ///     Should be reshapeable to 4D.
    /// \note
    ///     In-place transforms are allowed if the \p output is appropriately padded to account
    ///     for the extra one (if odd) or two (if even) real element in the width dimension.
    /// \note
    ///     For multidimensional c2r transforms, the input is not preserved.
    template<nt::array_decay_of_almost_any<c32, c64> Input,
             nt::array_decay_of_any<nt::mutable_value_type_twice_t<Input>> Output>
        requires nt::array_decay_with_same_nd<Input, Output>
    void c2r(Input&& input, Output&& output, FFTOptions options = {}) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        check(input.shape() == output.shape().rfft(),
              "Given the real output with a shape of {}, the non-redundant shape of the complex input "
              "should be {}, but got {}", output.shape(), output.shape().rfft(), input.shape());

        const Device device = output.device();
        check(device == input.device(),
              "The input and output arrays must be on the same device, but got input:device={}, output:device={}",
              input.device(), device);

        auto input_4d = input.span().template as_nd<4>();
        auto output_4d = output.span().template as_nd<4>();
        auto shape_4d = output_4d.shape();
        details::prepare_ranked_spans(input_4d, output_4d, shape_4d, options.rank);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=, handles =
                nd::extract_shared_handle_from_arrays(
                    noa::forward_as_tuple(
                        std::forward<Input>(input), std::forward<Output>(output)))] {
                constexpr auto flags = noa::fft::cpu::ESTIMATE;
                noa::fft::cpu::c2r(
                    input_4d.get(), input_4d.strides(),
                    output_4d.get(), output_4d.strides(),
                    shape_4d, flags, options.plan_only or options.record_and_share_workspace,
                    threads
                );
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            noa::fft::cuda::c2r(
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
            details::normalize(std::forward<Output>(output), shape_4d, Sign::BACKWARD, options.norm);
    }

    /// Computes and returns the backward c2r transform of (batched) 1d/2d/3d array(s).
    /// \param[in,out] input:
    ///     Non-redundant non-centered, aka "h" layout, FFT(s).
    ///     Should be reshapeable to 4D.
    /// \param shape:
    ///     Logical shape of the input.
    /// \note
    ///     For multidimensional c2r transforms, the input is not preserved.
    template<nt::array_decay_of_almost_any<c32, c64> Input, usize N>
        requires (nt::array_decay_nd<Input, N>)
    [[nodiscard]] auto c2r(Input&& input, const Shape<isize, N> shape, FFTOptions options = {}) {
        using real_t = nt::mutable_value_type_twice_t<Input>;
        auto output = Array<real_t, nt::array_size_v<Input>>(shape, input.options());
        c2r(std::forward<Input>(input), output, options);
        return output;
    }

    /// Computes the c2c transform of (batched) 1d/2d/3d array(s).
    /// \param[in] input:
    ///     Input complex data.
    ///     Should be reshapeable to 4D.
    /// \param[out] output:
    ///     Non-centered, aka "f" layout, FFT(s).
    ///     Should be reshapeable to 4D.
    /// \param sign:
    ///     Sign of the exponent in the formula that defines the Fourier transform.
    ///     It can be −1 (Sign::FORWARD) or +1 (Sign::BACKWARD).
    /// \note
    ///     In-place transforms are allowed.
    template<nt::array_decay_of_almost_any<c32, c64> Input,
             nt::array_decay_of_any<nt::mutable_value_type_t<Input>> Output>
        requires nt::array_decay_with_same_nd<Input, Output>
    void c2c(Input&& input, Output&& output, Sign sign, FFTOptions options = {}) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        check(input.shape() == output.shape(),
              "The input and output shape should match (no broadcasting allowed), but got input {} and output {}",
              input.shape(), output.shape());

        const Device device = output.device();
        check(device == input.device(),
              "The input and output arrays must be on the same device, but got input:device={}, output:device={}",
              input.device(), device);

        auto input_4d = input.span().template as_nd<4>();
        auto output_4d = output.span().template as_nd<4>();
        auto shape_4d = input_4d.shape();
        details::prepare_ranked_spans(input_4d, output_4d, shape_4d, options.rank);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=, handles =
                nd::extract_shared_handle_from_arrays(
                    noa::forward_as_tuple(
                        std::forward<Input>(input), std::forward<Output>(output)))] {
                constexpr auto flags = noa::fft::cpu::ESTIMATE | noa::fft::cpu::PRESERVE_INPUT;
                noa::fft::cpu::c2c(
                    input_4d.get(), input_4d.strides(),
                    output_4d.get(), output_4d.strides(),
                    shape_4d, sign, flags, options.plan_only or options.record_and_share_workspace,
                    threads
                    );
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            noa::fft::cuda::c2c(
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
            details::normalize(std::forward<Output>(output), shape_4d, sign, options.norm);
    }

    /// Computes and returns the c2c transform 1d/2d/3d array(s).
    /// \param[in] input:
    ///     Input complex data.
    ///     Should be reshapeable to 4D.
    /// \param sign:
    ///     Sign of the exponent in the formula that defines the Fourier transform.
    ///     It can be −1 (Sign::FORWARD) or +1 (Sign::BACKWARD).
    template<nt::array_decay_of_almost_any<c32, c64> Input>
    [[nodiscard]] auto c2c(Input&& input, Sign sign, FFTOptions options = {}) {
        using complex_t = nt::mutable_value_type_t<Input>;
        auto output = Array<complex_t, nt::array_size_v<Input>>(input.shape(), input.options());
        c2c(std::forward<Input>(input), output, sign, options);
        return output;
    }
}
