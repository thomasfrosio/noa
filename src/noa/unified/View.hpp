#pragma once

#include "noa/core/Ewise.hpp"
#include "noa/core/Error.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/indexing/Subregion.hpp"

#include "noa/unified/ArrayOption.hpp"
#include "noa/unified/Stream.hpp"
#include "noa/unified/Traits.hpp"
#include "noa/unified/Indexing.hpp"
#include "noa/unified/Utilities.hpp"
#include "noa/unified/Ewise.hpp"

#include "noa/cpu/Copy.hpp"
#include "noa/cpu/Permute.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/Copy.cuh"
#include "noa/gpu/cuda/Permute.cuh"
#include "noa/gpu/cuda/Allocators.hpp"
#endif

namespace noa::inline types {
    template<typename T>
    class Array;
}

namespace noa {
    struct CopyOptions {
        /// When transferring from a GPU to the CPU, the copy is enqueued to the input's (GPU) current stream.
        /// By default, this stream is synchronized before returning to guarantee that the copy is completed so
        /// that the output array can be safely accessed by the CPU. This option allows not waiting for the copy
        /// to finish.
        bool sync_gpu_to_cpu{true};
    };

    /// (Deep-)Copies arrays.
    /// \param[in] input    Source.
    /// \param[out] output  Destination. It should not overlap with \p input.
    /// \param options      Copy options.
    /// \note Contiguous regions of memory have no copy restrictions and can be copied to any device. This is
    ///       also true for pitched layouts and colum or row vectors. However, other non-contiguous memory
    ///       layouts can only be copied if the source and destination are both on the same GPU or on the CPU.
    template<nt::readable_varray_decay Input,
             nt::writable_varray_decay_of_any<nt::mutable_value_type_t<Input>> Output>
    void copy(Input&& input, Output&& output, const CopyOptions& options = {}) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        check(not ni::are_overlapped(input, output), "The input and output should not overlap");

        const auto input_strides = ng::broadcast_strides(input, output);
        const Device input_device = input.device();
        const Device output_device = output.device();
        if (input_device.is_cpu() and output_device.is_cpu()) {
            auto& cpu_stream = Stream::current(input_device).cpu();
            const auto n_threads = cpu_stream.thread_limit();
            if ((nt::array_decay<Input> or nt::array_decay<Output>) and cpu_stream.is_async()) {
                cpu_stream.enqueue(
                    [=,
                     input_ = std::forward<Input>(input),
                     output_ = std::forward<Output>(output)
                    ] {
                        noa::cpu::copy(
                            input_.get(), input_strides,
                            output_.get(), output_.strides(),
                            output_.shape(), n_threads);
                    });
            } else {
                noa::cpu::copy(
                    input.get(), input_strides,
                    output.get(), output.strides(),
                    output.shape(), n_threads);
            }
        } else if (output_device.is_cpu()) { // gpu -> cpu
            #ifdef NOA_ENABLE_CUDA
            Stream::current(output_device).synchronize();
            auto& cuda_stream = Stream::current(input_device).cuda();
            noa::cuda::copy(input.get(), input_strides,
                            output.get(), output.strides(),
                            output.shape(), cuda_stream);
            cuda_stream.enqueue_attach(std::forward<Input>(input), std::forward<Output>(output));
            if (options.sync_gpu_to_cpu)
                cuda_stream.synchronize();
            #else
            (void) options;
            panic_no_gpu_backend();
            #endif
        } else { // gpu -> gpu or cpu -> gpu
            #ifdef NOA_ENABLE_CUDA
            if (input_device != output_device)
                Stream::current(input_device).synchronize(); // wait for the input
            auto& cuda_stream = Stream::current(output_device).cuda();
            noa::cuda::copy(input.get(), input_strides,
                            output.get(), output.strides(),
                            output.shape(), cuda_stream);
            cuda_stream.enqueue_attach(std::forward<Input>(input), std::forward<Output>(output));
            #else
            panic_no_gpu_backend();
            #endif
        }
    }

    /// Permutes the input by reordering its dimensions. The returned object points to the same data.
    template<nt::varray_decay Input>
    auto permute(Input&& input, const Vec<i64, 4>& permutation) {
        check(all(permutation <= 3) and sum(permutation) == 6, "Permutation {} is not valid", permutation);
        auto permuted_shape = ni::reorder(input.shape(), permutation);
        auto permuted_strides = ni::reorder(input.strides(), permutation);
        using output_t = std::decay_t<Input>;
        return output_t(std::forward<Input>(input).share(), permuted_shape, permuted_strides, input.options());
    }

    /// Permutes, in memory, the axes of an array.
    /// \tparam T           Any numeric type.
    /// \param[in] input    Array to permute.
    /// \param[out] output  Permuted array. Its shape and strides should be already permuted.
    /// \param permutation  Permutation. Axes are numbered from 0 to 3.
    /// \note For in-place permutations, only 0123, 0213, 0132, and 0321 are supported. Anything else throws an error.
    /// \note The in-place 0213 permutation requires the axis 1 and 2 to have the same size.
    ///       The in-place 0132 permutation requires the axis 3 and 2 to have the same size.
    ///       The in-place 0321 permutation requires the axis 3 and 1 to have the same size.
    /// \note On the GPU, the following permutations are optimized: 0123, 0132, 0312, 0321, 0213, 0231.
    ///       Anything else calls copy(), which is slower.
    template<nt::readable_varray_decay Input,
             nt::writable_varray_decay_of_any<nt::mutable_value_type_t<Input>> Output>
    void permute_copy(Input&& input, Output&& output, const Vec4<i64>& permutation) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        check(all(permutation <= 3) and sum(permutation) == 6, "Permutation {} is not valid", permutation);

        // To enable broadcasting, we need to permute the input.
        auto input_strides = input.strides();
        auto input_shape = input.shape();
        for (i64 i{}; i < 4; ++i) {
            const i64 d = permutation[i];
            if (input.shape()[d] == 1 and output.shape()[i] != 1) {
                input_strides[d] = 0; // broadcast this dimension
                input_shape[d] = output.shape()[i];
            } else if (input.shape()[d] != output.shape()[i]) {
                panic("Cannot broadcast an array of shape {} into an array of shape {}",
                      ni::reorder(input.shape(), permutation), output.shape());
            }
        }

        const Device device = output.device();
        check(device == input.device(),
              "The input and output arrays must be on the same device, but got input:{} and output:{}",
              input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto n_threads = cpu_stream.thread_limit();
            if ((nt::array_decay<Input> or nt::array_decay<Output>) and cpu_stream.is_async()) {
                cpu_stream.enqueue(
                    [=,
                     input_ = std::forward<Input>(input),
                     output_ = std::forward<Output>(output)
                    ] {
                        noa::cpu::permute_copy(
                            input_.get(), input_strides, input_shape,
                            output_.get(), output_.strides(),
                            permutation, n_threads);
                    });
            } else {
                noa::cpu::permute_copy(
                    input.get(), input_strides, input_shape,
                    output.get(), output.strides(),
                    permutation, n_threads);
            }
        } else {
            #ifdef NOA_ENABLE_CUDA
            noa::cuda::Stream& cuda_stream = stream.cuda();
            noa::cuda::permute_copy(
                input.get(), input_strides, input_shape,
                output.get(), output.strides(),
                permutation, cuda_stream);
            cuda_stream.enqueue_attach(std::forward<Input>(input), std::forward<Output>(output));
            #else
            panic_no_gpu_backend();
            #endif
        }
    }

    /// Permutes the input by performing a deep-copy. The returned Array is a new C-contiguous array.
    /// \param[in] input    VArray to permute.
    /// \param permutation  Permutation with the axes numbered from 0 to 3.
    template<nt::varray_decay Input>
    auto permute_copy(Input&& input, const Vec4<i64>& permutation) {
        auto permuted_shape = ni::reorder(input.shape(), permutation);
        auto output = Array<nt::mutable_value_type_t<Input>>(permuted_shape, input.options());
        permute_copy(std::forward<Input>(input), output, permutation);
        return output;
    }

    /// Casts an array.
    /// \param[in] input    Array to convert.
    /// \param[out] output  Array with the cast values.
    /// \param clamp        Whether the input values should be clamped to the output range before casting.
    /// \note If the input is complex and the output is real, the power spectrum is first computed.
    template<nt::readable_varray_decay Input, nt::writable_varray_decay Output>
    requires nt::varray_decay_with_compatible_or_spectrum_types<Input, Output>
    void cast(Input&& input, Output&& output, bool clamp = false) {
        check(not ni::are_overlapped(input, output), "The input and output arrays should not overlap");
        ewise(std::forward<Input>(input), std::forward<Output>(output), Cast{clamp});
    }

    struct ReinterpretAsOptions {
        /// Whether to prefetch the memory to the target device. Used as an optimization hint for the driver.
        /// In CUDA, this only affects MANAGED(_GLOBAL) memory and can result in moving the memory from the original
        /// to the target device.
        /// To follow the stream-ordering access rule, prefetching may require synchronizing the original
        /// and/or the target stream.
        bool prefetch{true};
    };

    /// Changes the device type (CPU<->GPU) on which the memory should be accessed.
    /// \details If the memory resource can be accessed by the CPU and/or a GPU, this function returns a new view
    ///          with the new device. This is used to control whether PINNED or MANAGED memory should be accessed
    ///          by the CPU or the GPU. MANAGED_GLOBAL memory is not attached to any particular GPU, so the
    ///          current GPU is used in that case.
    ///
    /// \note Note that while this allows having CPU and GPU arrays pointing to the same memory, concurrent access
    ///       from both CPU and GPU is always illegal (see Allocator). As such, when the source and destination device
    ///       are different (CPU->GPU or GPU->CPU), the input stream is synchronized when this function returns, so
    ///       that the output can be used right away.
    template<nt::varray_decay Input>
    auto reinterpret_as(
        Input&& input,
        Device::Type type,
        [[maybe_unused]] ReinterpretAsOptions parameters = {}
    ) {
        using input_t = std::decay_t<Input>;
        ArrayOption options = input.options();
        const bool change_device = options.device.type() != type;

        if (options.device.is_gpu()) {
            check(not change_device or
                  options.allocator.is_any(Allocator::PINNED, Allocator::MANAGED, Allocator::MANAGED_GLOBAL, Allocator::PITCHED_MANAGED),
                  "GPU memory {} cannot be reinterpreted as a CPU memory-region. "
                  "This is only supported for pinned and managed memory-regions",
                  options.allocator);

            Stream& input_stream = Stream::current(options.device);

            #ifdef NOA_ENABLE_CUDA
            if (parameters.prefetch and options.allocator.is_any(Allocator::MANAGED, Allocator::MANAGED_GLOBAL, Allocator::PITCHED_MANAGED)) {
                const auto n_elements = ni::offset_at(input.strides(), input.shape().vec - 1) + 1;
                if (change_device)
                    noa::cuda::AllocatorManaged::prefetch_to_cpu(input.get(), n_elements, input_stream.cuda());
                else
                    noa::cuda::AllocatorManaged::prefetch_to_gpu(input.get(), n_elements, input_stream.cuda());
            }
            #endif

            if (change_device) { // gpu -> cpu
                input_stream.synchronize();
                options.device = Device{};
            }

        } else if (options.device.is_cpu() and (change_device or parameters.prefetch)) {
            if (change_device) {
                check(Device::is_any_gpu(), "No GPU detected");
                check(options.allocator.is_any(Allocator::PINNED, Allocator::MANAGED, Allocator::MANAGED_GLOBAL, Allocator::PITCHED_MANAGED),
                      "CPU memory-region with the allocator {} cannot be reinterpreted as a GPU memory-region. "
                      "This is only supported for pinned and managed memory-regions",
                      options.allocator);
            }

            #ifdef NOA_ENABLE_CUDA
            Device gpu_device;
            if (options.allocator.is_any(Allocator::PINNED, Allocator::MANAGED, Allocator::PITCHED_MANAGED)) {
                // CUDA doesn't document what the attr.device is for managed memory. Hopefully this is the device
                // against which the allocation was performed and not the current device. With "stream-attached"
                // managed memory, it is up to the user to know what stream was used to perform the allocation.
                const cudaPointerAttributes attr = noa::cuda::pointer_attributes(input.get());
                gpu_device = Device(Device::GPU, attr.device, Unchecked{});
                check((options.allocator == Allocator::PINNED and attr.type == cudaMemoryTypeHost) or
                      (options.allocator.is_any(Allocator::MANAGED, Allocator::PITCHED_MANAGED) and attr.type == cudaMemoryTypeManaged));

            } else if (options.allocator == Allocator::MANAGED_GLOBAL) {
                // This can be accessed from any stream and any GPU. It seems better to return the current
                // device and not the original device against which the allocation was performed.
                gpu_device = Device::current(Device::GPU);
            }

            if (change_device) {
                input.eval();
                options.device = gpu_device;
            }

            if (parameters.prefetch and options.allocator.is_any(Allocator::MANAGED, Allocator::MANAGED_GLOBAL, Allocator::PITCHED_MANAGED)) {
                const auto n_elements = ni::offset_at(input.strides(), input.shape().vec - 1) + 1;
                auto& gpu_stream = Stream::current(gpu_device).cuda();
                using value_t = nt::mutable_value_type_t<input_t>;
                if (change_device) {
                    noa::cuda::AllocatorManaged::prefetch_to_gpu<value_t>(input.get(), n_elements, gpu_stream);
                    // We return a GPU array, so no need to synchronize the GPU stream.
                } else {
                    input.eval();
                    noa::cuda::AllocatorManaged::prefetch_to_cpu<value_t>(input.get(), n_elements, gpu_stream);
                    gpu_stream.synchronize();
                }
            }
            #endif
        }

        return input_t(std::forward<Input>(input).share(), input.shape(), input.strides(), options);
    }
}

namespace noa::inline types {
    /// Views a memory region.
    /// \details
    ///   Views have a similar interface as the Array's, but their value type can be const-qualified.
    ///   More importantly, Views do not own the memory region they point to. This makes it more lightweight
    ///   than an Array, but is slightly more difficult to use correctly.\n
    /// - \b Deferred-destruction: As opposed to Arrays, when passing Views to a function, the stream cannot keep
    ///   a reference of the input/output memory-regions, because Views are not reference-counted (this is what
    ///   makes them more lightweight than Arrays). As such, users should make sure that the memory-regions
    ///   are not destructed while being used. Fortunately, this can be easily achieved by synchronizing the stream
    ///   (e.g. using View::eval()) before destructing the original Arrays. Note that if an exception is thrown,
    ///   Arrays will synchronize the current stream if needed, making sure any potential use of their managed region
    ///   via Views is completed before destruction.
    template<typename T>
    class View {
    public:
        using pointer_type = T*;
        using value_type = T;
        using mutable_value_type = std::remove_const_t<T>;
        using const_value_type = std::add_const_t<mutable_value_type>;
        using index_type = i64;
        using strides_type = Strides<index_type, 4>;
        using view_type = View;
        using shape_type = Shape<index_type, 4>;
        using span_type = Span<value_type, 4, index_type>;

        static constexpr StridesTraits STRIDES_TRAIT = span_type::STRIDES_TRAIT;
        static constexpr PointerTraits POINTER_TRAIT = span_type::POINTER_TRAIT;
        static constexpr bool IS_CONTIGUOUS = span_type::IS_CONTIGUOUS;
        static constexpr bool IS_RESTRICT = span_type::IS_RESTRICT;
        static constexpr size_t SIZE = 4;
        static constexpr int64_t SSIZE = 4;

    public: // Constructors
        /// Creates an empty view.
        constexpr View() = default;

        /// Creates a view of a contiguous row-vector.
        constexpr explicit View(pointer_type data, i64 n_elements = 1, ArrayOption options = {}) noexcept :
            m_shape{1, 1, 1, n_elements},
            m_strides{n_elements, n_elements, n_elements, 1},
            m_ptr{data},
            m_options{options}
        {
            allocator().validate(data, device());
        }

        /// Creates a view of a 1d (contiguous) span.
        template<StridesTraits S>
        constexpr explicit View(const Span<value_type, 1, index_type, S>& span, ArrayOption options = {}) :
            m_ptr{span.data()}, m_options{options}
        {
            allocator().validate(span.data(), device());
            auto span_4d = span.as_4d(); // row vector
            m_shape = span_4d.shape();
            m_strides = span_4d.strides_full();
        }

        /// Creates a view of a 4d (contiguous) span.
        template<StridesTraits S>
        constexpr explicit View(const Span<value_type, 4, index_type, S>& span, ArrayOption options = {}) noexcept :
            m_shape{span.shape()},
            m_strides{span.strides_full()},
            m_ptr{span.data()},
            m_options{options}
        {
            allocator().validate(span.data(), device());
        }

        /// Creates a view of a strided 4d memory region.
        constexpr View(T* data, const shape_type& shape, const strides_type& strides, ArrayOption options = {}) noexcept :
            m_shape{shape},
            m_strides{strides},
            m_ptr{data},
            m_options{options}
        {
            allocator().validate(data, device());
        }

        /// Creates a view of a strided 4d memory region, assuming the data is C-contiguous.
        constexpr View(T* data, const shape_type& shape, ArrayOption options = {}) noexcept :
            m_shape{shape},
            m_strides{shape.strides()},
            m_ptr{data},
            m_options{options}
        {
            allocator().validate(data, device());
        }

        /// Creates a const view from an existing non-const view.
        template<nt::mutable_of<value_type> U>
        constexpr /*implicit*/ View(const View<U>& view) noexcept :
            m_shape{view.shape()},
            m_strides{view.strides()},
            m_ptr{view.data()},
            m_options{view.options()} {}

        /// Creates a view of an Array.
        template<nt::almost_same_as<value_type> U>
        constexpr explicit View(const Array<U>& array) noexcept :
            m_shape{array.shape()},
            m_strides{array.strides()},
            m_ptr{array.data()},
            m_options{array.options()} {}

        /// Creates a view of a strided 4d memory region, but don't validate the data pointer.
        constexpr View(T* data, const shape_type& shape, const strides_type& strides, ArrayOption options, Unchecked) noexcept :
            m_shape{shape},
            m_strides{strides},
            m_ptr{data},
            m_options{options} {}

    public: // Getters
        [[nodiscard]] constexpr auto options() const noexcept -> const ArrayOption& { return m_options; }
        [[nodiscard]] constexpr auto device() const noexcept -> Device { return options().device; }
        [[nodiscard]] constexpr auto allocator() const noexcept -> Allocator { return options().allocator; }
        [[nodiscard]] constexpr auto shape() const noexcept -> const shape_type& { return m_shape; }
        [[nodiscard]] constexpr auto strides() const noexcept -> const strides_type& { return m_strides; }
        [[nodiscard]] constexpr auto strides_full() const noexcept -> const strides_type& { return m_strides; }
        [[nodiscard]] constexpr auto n_elements() const noexcept -> index_type { return shape().n_elements(); }
        [[nodiscard]] constexpr auto ssize() const noexcept -> index_type { return n_elements(); }
        [[nodiscard]] constexpr auto size() const noexcept -> size_t { return static_cast<size_t>(n_elements()); }

        template<char ORDER = 'C'>
        [[nodiscard]] constexpr bool are_contiguous() const noexcept {
            return ni::are_contiguous<ORDER>(strides(), shape());
        }

        template<char ORDER = 'C'>
        [[nodiscard]] constexpr auto is_contiguous() const noexcept {
            return ni::is_contiguous<ORDER>(strides(), shape());
        }

        /// Whether the view is empty. A View is empty if not initialized,
        /// or if the viewed data is null, or if one of its dimensions is 0.
        [[nodiscard]] constexpr bool is_empty() const noexcept { return not get() or shape().is_empty(); }

    public:
        /// Synchronizes the current stream of the view's device.
        /// \details It guarantees safe access to the memory region. Note that stream-ordered access (i.e. passing
        ///          this to the library API) is safe and doesn't need synchronization.
        auto& eval() const {
            Stream::current(device()).synchronize();
            return *this;
        }

        /// Returns the pointer to the data.
        /// \warning Depending on the current stream of this view's device,
        ///          reading/writing to this pointer may be illegal or create a data race.
        [[nodiscard]] constexpr auto get() const noexcept -> pointer_type { return m_ptr; }
        [[nodiscard]] constexpr auto data() const noexcept -> pointer_type { return get(); }
        [[nodiscard]] constexpr auto share() const noexcept -> pointer_type { return get(); }

        /// Returns a span of the view.
        /// \warning Depending on the current stream of this view's device,
        ///          reading/writing through this Span may be illegal or create a data race.
        [[nodiscard]] constexpr auto span() const -> span_type {
            return span_type(get(), shape(), strides());
        }

        template<typename U, size_t N = 4, typename I = index_type, StridesTraits STRIDES_TRAIT = STRIDES_TRAIT>
        [[nodiscard]] constexpr auto span() const {
            return span().template span<U, N, I, STRIDES_TRAIT>();
        }

        template<typename U = value_type, size_t N = 4, typename I = index_type>
        [[nodiscard]] constexpr auto span_contiguous() const {
            return span<U, N, I, StridesTraits::CONTIGUOUS>();
        }

        template<typename U = value_type, typename I = index_type, StridesTraits STRIDES_TRAIT = StridesTraits::CONTIGUOUS>
        [[nodiscard]] constexpr auto span_1d() const {
            return span<U, 1, I, STRIDES_TRAIT>();
        }

        template<typename U = value_type, typename I = index_type>
        [[nodiscard]] constexpr auto span_1d_contiguous() const {
            return span<U, 1, I, StridesTraits::CONTIGUOUS>();
        }

        template<typename U = value_type, typename I = index_type>
        [[nodiscard]] constexpr auto span_1d_strided() const {
            return span<U, 1, I, StridesTraits::STRIDED>();
        }

        /// Returns a (const-)view of the view.
        template<nt::almost_same_as<value_type> U = value_type>
        [[nodiscard]] constexpr auto view() const noexcept -> View<U> {
            return *this;
        }

        [[nodiscard]] constexpr auto as_const() const noexcept -> View<const_value_type> {
            return *this;
        }

    public: // Deep copy
        /// Performs a deep copy of the view to \p output.
        /// \details Contiguous regions of memory have no copy restrictions and can be copied to any device. This is
        ///          also true for pitched layouts, colum or row vectors, or any layout that can be reordered and/or
        ///          reshaped to the aforementioned layouts. However, other non-contiguous memory layouts can only
        ///          be copied if the source and destination are both on the same GPU or on the CPU.
        /// \param[out] output  Destination. It should not overlap with this view.
        template<nt::writable_varray_decay Output> requires nt::varray_decay_of_almost_same_type<View, Output>
        auto to(Output&& output) const -> decltype(auto) {
            noa::copy(*this, output);
            return std::forward<Output>(output);
        }

        /// Performs a deep copy of the view according \p option.
        /// \details The returned array is completely independent of the original one and is C-contiguous.
        ///          Contiguous regions of memory have no copy restrictions and can be copied to any device. This is
        ///          also true for pitched layouts, colum or row vectors, or any layout that can be reordered and/or
        ///          reshaped to the aforementioned layouts. However, other non-contiguous memory layouts can only
        ///          be copied if the source and destination are both on the same GPU or on the CPU.
        /// \param option   Output device and resource to perform the allocation of the new array.
        ///                 The current stream for that device is used to perform the copy.
        [[nodiscard]] auto to(ArrayOption option) const -> Array<mutable_value_type> {
            auto out = Array<mutable_value_type>(shape(), option);
            to(out);
            return out;
        }

        /// Performs a deep copy of the array to the CPU.
        [[nodiscard]] auto to_cpu() const -> Array<mutable_value_type> {
            return to(ArrayOption{});
        }

        /// Performs a deep copy of the view preserving the view's options.
        [[nodiscard]] auto copy() const -> Array<mutable_value_type> {
            return to(options());
        }

        /// Returns a copy of the first value in the array.
        /// Note that the stream of the array's device is synchronized when this functions returns.
        [[nodiscard]] auto first() const -> mutable_value_type {
            check(not is_empty());
            if (is_dereferenceable()) {
                eval();
                return get()[0];
            } else {
                mutable_value_type output;
                auto one_shape = shape_type::from_value(1);
                auto one_strides = strides_type::from_value(1);
                View(get(), one_shape, one_strides, options(), Unchecked{}).to(
                    View(&output, one_shape, one_strides, ArrayOption{}, Unchecked{}));
                eval(); // protect against async cpu stream
                return output;
            }
        }

    public: // Data reinterpretation
        [[nodiscard]] constexpr auto is_reinterpretable_as(Device::Type type) const noexcept -> bool {
            return options().is_reinterpretable(type);
        }
        [[nodiscard]] constexpr auto is_reinterpretable_as_cpu() const noexcept -> bool {
            return options().is_reinterpretable(Device::CPU);
        }
        [[nodiscard]] constexpr auto is_dereferenceable() const noexcept -> bool {
            return options().is_dereferenceable();
        }

        auto reinterpret_as(Device::Type type, ReinterpretAsOptions parameters = {}) const -> View {
            return noa::reinterpret_as(*this, type, parameters);
        }
        auto reinterpret_as_cpu(ReinterpretAsOptions parameters = {}) const -> View {
            return noa::reinterpret_as(*this, Device::CPU, parameters);
        }
        auto reinterpret_as_gpu(ReinterpretAsOptions parameters = {}) const -> View {
            return noa::reinterpret_as(*this, Device::GPU, parameters);
        }

        /// Reinterprets the value type.
        /// \note This is only well-defined in cases where reinterpret_cast<U*>(T*) is well-defined, for instance,
        ///       to represent any data type as an array of bytes, or to switch between complex and real floating-point
        ///       numbers with the same precision.
        template<typename U>
        [[nodiscard]] auto reinterpret_as() const -> View<U> {
            const auto out = ni::ReinterpretLayout(shape(), strides(), get()).template as<U>();
            return View<U>(out.ptr, out.shape, out.strides, options(), Unchecked{});
        }

        /// Reshapes the view (must have the same number of elements as the current view).
        [[nodiscard]] constexpr auto reshape(shape_type new_shape) const -> View {
            auto new_span = span().reshape(new_shape);
            return View(new_span.data(), new_span.shape(), new_span.strides(), options(), Unchecked{});
        }

        /// Reshapes the array in a vector along a particular axis.
        /// Returns a row vector by default (axis = 3).
        [[nodiscard]] constexpr auto flat(i32 axis = 3) const -> View {
            auto new_span = span().flat(axis);
            return View(new_span.data(), new_span.shape(), new_span.strides(), options(), Unchecked{});
        }

        /// Permutes the dimensions of the view.
        /// \param permutation  Permutation with the axes numbered from 0 to 3.
        [[nodiscard]] constexpr auto permute(const Vec4<i64>& permutation) const -> View {
            auto new_span = span().permute(permutation);
            return View(new_span.data(), new_span.shape(), new_span.strides(), options(), Unchecked{});
        }

        [[nodiscard]] auto permute_copy(const Vec4<i64>& permutation) const -> Array<mutable_value_type> {
            return noa::permute_copy(*this, permutation);
        }

    public:
        /// Clears the view. Equivalent to assigning *this with an empty view.
        View& operator=(std::nullptr_t) {
            *this = View{};
            return *this;
        }

        /// Returns the view and reset this view to an empty view.
        /// \note drop() is more meaningful for Array.
        View drop() noexcept {
            return std::exchange(*this, View{});
        }

    public:
        /// Element access (unsafe if not synchronized). For efficient access, prefer to use Span.
        template<typename... U> requires nt::iwise_indexing<SIZE, U...>
        [[nodiscard]] constexpr auto at(const U&... indices) const -> value_type& {
            check(is_dereferenceable(), "Memory buffer cannot be accessed from the CPU");
            return span().at(indices...);
        }

        /// Element access (unsafe). For efficient access, prefer to use Span.
        template<typename... U> requires nt::iwise_indexing<SIZE, U...>
        [[nodiscard]] constexpr auto operator()(const U&... indices) const -> value_type& {
            return span()(indices...);
        }

        /// Subregion indexing. Extracts a subregion from the current view.
        template<typename... U>
        [[nodiscard]] constexpr auto subregion(const ni::Subregion<4, U...>& subregion) const -> View {
            auto [new_shape, new_strides, offset] = subregion.extract_from(shape(), strides());
            return View(get() + offset, new_shape, new_strides, options(), Unchecked{});
        }

        /// Subregion indexing. Extracts a subregion from the current view.
        /// \see noa::indexing::Subregion for more details on the variadic parameters to enter.
        template<typename... U> requires nt::subregion_indexing<4, U...>
        [[nodiscard]] constexpr auto subregion(const U&... indices) const -> View {
            return subregion(ni::Subregion<4, U...>(indices...));
        }

    private:
        shape_type m_shape{};
        strides_type m_strides{};
        pointer_type m_ptr{};
        ArrayOption m_options{};
    };
}

namespace noa::traits {
    template<typename T> struct proclaim_is_view<View<T>> : std::true_type {};
}
