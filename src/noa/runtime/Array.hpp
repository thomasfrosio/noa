#pragma once

#include "noa/base/Complex.hpp"
#include "noa/base/Half.hpp"
#include "noa/runtime/core/Shape.hpp"
#include "noa/runtime/core/Span.hpp"
#include "noa/runtime/core/Traits.hpp"
#include "noa/runtime/core/Shareable.hpp"
#include "noa/runtime/core/Subregion.hpp"

#include "noa/runtime/cpu/Copy.hpp"
#include "noa/runtime/cpu/Permute.hpp"

#include "noa/runtime/ArrayOption.hpp"
#include "noa/runtime/Stream.hpp"
#include "noa/runtime/Traits.hpp"
#include "noa/runtime/Utils.hpp"
#include "noa/runtime/Ewise.hpp"
#include <type_traits>

#ifdef NOA_ENABLE_CUDA
#include "noa/runtime/cuda/Copy.cuh"
#include "noa/runtime/cuda/Permute.cuh"
#include "noa/runtime/cuda/Allocators.hpp"
#include "noa/runtime/cuda/Pointers.hpp"
#endif

namespace noa::inline types {
    enum class ArrayOwnership : usize {
        RC, VIEW
    };

    template<typename T, usize N = 4, ArrayOwnership O = ArrayOwnership::RC>
    class Array;
}

namespace noa {
    /// Whether the array is broadcastable to the given shape.
    template<typename T, usize N, ArrayOwnership O>
    [[nodiscard]] constexpr auto is_broadcastable(
        const Array<T, N, O>& array,
        const Shape<usize, N>& shape
    ) -> bool {
        return is_broadcastable(array.shape(), shape);
    }

    /// Broadcasts an array to a given shape.
    template<nt::array_decay T, usize N> // FIXME
    [[nodiscard]] auto broadcast(T&& input, const Shape<usize, N>& shape) {
        auto strides = input.strides();
        if (not broadcast(input.shape(), strides, shape))
            panic("Cannot broadcast shape={} into a shape={}", input.shape(), shape);
        return std::decay_t<T>(std::forward<T>(input).share(), shape, strides, input.options());
    }

    /// Whether lhs and rhs overlap in memory.
    [[nodiscard]] bool are_overlapped(nt::array auto const& lhs, nt::array auto const& rhs) {
        if (lhs.is_empty() or rhs.is_empty())
            return false;
        auto const lhs_start = reinterpret_cast<uintptr_t>(lhs.get());
        auto const rhs_start = reinterpret_cast<uintptr_t>(rhs.get());
        auto const lhs_end = reinterpret_cast<uintptr_t>(lhs.get() + offset_at(lhs.strides(), (lhs.shape() - 1).vec));
        auto const rhs_end = reinterpret_cast<uintptr_t>(rhs.get() + offset_at(rhs.strides(), (rhs.shape() - 1).vec));
        return details::are_overlapped(lhs_start, lhs_end, rhs_start, rhs_end);
    }

    /// Returns the multidimensional indexes of \p array corresponding to a memory \p offset.
    /// \note 0 indicates the beginning of the array. The array should not have any broadcast dimension.
    [[nodiscard]] constexpr auto offset2index(isize offset, nt::array auto const& array) -> Vec<isize, 4> {
        check(array.strides() > 0,
              "Cannot retrieve the 4d index from a broadcast dimension. Got strides={}",
              array.strides());
        return offset2index(offset, array.strides(), array.shape());
    }

    /// Whether the input is a contiguous vector.
    [[nodiscard]] constexpr bool is_contiguous_vector(nt::array auto const& input) {
        return input.shape().is_vector() and input.is_contiguous();
    }

    /// Whether the input is a contiguous vector or a contiguous batch of contiguous vectors.
    [[nodiscard]] constexpr bool is_contiguous_vector_batched(nt::array auto const& input) {
        return input.shape().is_vector(true) and input.is_contiguous();
    }

    /// Whether the input is a contiguous vector or a contiguous/strided batch of contiguous vectors.
    /// The batch stride doesn't have to be contiguous.
    [[nodiscard]] constexpr bool is_contiguous_vector_batched_strided(nt::array auto const& input) {
        return input.shape().is_vector(true) and input.contiguity().pop_front() == true;
    }

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
    template<nt::readable_array_decay Input,
             nt::writable_array_decay_of_any<nt::mutable_value_type_t<Input>> Output>
    void copy(Input&& input, Output&& output, const CopyOptions& options = {}) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        check(not noa::are_overlapped(input, output), "The input and output should not overlap");

        using input_t = std::remove_reference_t<Input>;
        using output_t = std::remove_reference_t<Output>;
        constexpr usize NDIM = std::max(input_t::SIZE, output_t::SIZE);
        const auto output_shape = output.shape().template push_front<NDIM - output_t::SIZE>(1);
        const auto output_strides = output.strides().template push_front<NDIM - output_t::SIZE>(0);
        const auto input_shape = input.shape().template push_front<NDIM - input_t::SIZE>(1);
        auto input_strides = input.strides().template push_front<NDIM - input_t::SIZE>(0);
        if (not noa::broadcast(input_shape, input_strides, output_shape)) {
            panic("Cannot broadcast an array of shape {} into an array of shape {}",
                  input_shape, output.shape());
        }

        const Device input_device = input.device();
        const Device output_device = output.device();
        if (input_device.is_cpu() and output_device.is_cpu()) {
            auto& cpu_stream = Stream::current(input_device).cpu();
            const auto n_threads = cpu_stream.thread_limit();
            if ((nt::array_rc<input_t> or nt::array_rc<output_t>) and cpu_stream.is_async()) {
                cpu_stream.enqueue(
                    [=,
                     input_ptr = input.data(),
                     output_ptr = output.data(),
                     input_handle = std::forward<Input>(input).share(),
                     output_handle = std::forward<Output>(output).share()
                    ] {
                        noa::cpu::copy(
                            input_ptr, input_strides,
                            output_ptr, output_strides,
                            output_shape, n_threads
                        );
                    });
            } else {
                noa::cpu::copy(
                    input.get(), input_strides,
                    output.get(), output_strides,
                    output_shape, n_threads
                );
            }
        } else if (output_device.is_cpu()) { // gpu -> cpu
            #ifdef NOA_ENABLE_CUDA
            Stream::current(output_device).synchronize();
            auto& cuda_stream = Stream::current(input_device).cuda();
            noa::cuda::copy(
                input.get(), input_strides,
                output.get(), output_strides,
                output_shape, cuda_stream
            );
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
            noa::cuda::copy(
                input.get(), input_strides,
                output.get(), output_strides,
                output_shape, cuda_stream
            );
            cuda_stream.enqueue_attach(std::forward<Input>(input), std::forward<Output>(output));
            #else
            panic_no_gpu_backend();
            #endif
        }
    }

    /// Permutes the input by reordering its dimensions. The returned object points to the same data.
    template<nt::array_decay Input>
    auto permute(Input&& input, const Vec<i32, std::remove_reference_t<Input>::SIZE>& permutation) {
        constexpr auto N = std::remove_reference_t<Input>::SIZE;
        check(permutation <= N - 1 and sum(permutation) == N * (N - 1) / 2,
              "Permutation {} is not valid for an array with {} dimension(s)", permutation, N);
        auto permuted_shape = input.shape().permute(permutation);
        auto permuted_strides = input.strides().permute(permutation);
        return std::decay_t<Input>(std::forward<Input>(input).share(), permuted_shape, permuted_strides, input.options());
    }

    /// Permutes, in memory, the axes of an array.
    /// \param[in] input    Array to permute.
    /// \param[out] output  Permuted array. Its shape and strides should be already permuted.
    /// \param permutation  Permutation. Axes are numbered from 0 to 3.
    /// \note For in-place permutations, only 0123, 0213, 0132, and 0321 are supported. Anything else throws an error.
    /// \note The in-place 0213 permutation requires the axis 1 and 2 to have the same size.
    ///       The in-place 0132 permutation requires the axis 3 and 2 to have the same size.
    ///       The in-place 0321 permutation requires the axis 3 and 1 to have the same size.
    /// \note On the GPU, the following permutations are optimized: 0123, 0132, 0312, 0321, 0213, 0231.
    ///       Anything else calls copy(), which is slower.
    template<typename Input, typename Output>
        requires (nt::readable_array_decay<Input> and
                  nt::writable_array_decay_of_any<Output, nt::mutable_value_type_t<Input>> and
                  nt::array_decay_with_same_nd<Input, Output>)
    void permute_copy(Input&& input, Output&& output, const Vec<i32, std::remove_reference_t<Input>::SIZE>& permutation) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        constexpr auto N = std::remove_reference_t<Input>::SIZE;
        check(permutation <= N - 1 and sum(permutation) == N * (N - 1) / 2,
              "Permutation {} is not valid for an array with {} dimension(s)", permutation, N);

        // To enable broadcasting, we need to permute the input.
        auto input_strides = input.strides();
        auto input_shape = input.shape();
        for (usize i{}; i < N; ++i) {
            const auto d = permutation[i];
            if (input.shape()[d] == 1 and output.shape()[i] != 1) {
                input_strides[d] = 0; // broadcast this dimension
                input_shape[d] = output.shape()[i];
            } else if (input.shape()[d] != output.shape()[i]) {
                panic("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape().permute(permutation), output.shape());
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
    template<nt::array_decay Input>
    auto permute_copy(Input&& input, const Vec<i32, std::remove_reference_t<Input>::SIZE>& permutation) {
        auto permuted_shape = input.shape().permute(permutation);
        using array_t = std::remove_reference_t<Input>::array_type;
        auto output = array_t(permuted_shape, input.options());
        permute_copy(std::forward<Input>(input), output, permutation);
        return output;
    }

    /// Casts an array.
    /// \param[in] input    Array to convert.
    /// \param[out] output  Array with the cast values.
    /// \param clamp        Whether the input values should be clamped to the output range before casting.
    /// \note If the input is complex and the output is real, the power spectrum is first computed.
    template<nt::readable_array_decay Input, nt::writable_array_decay Output>
        requires nt::array_decay_with_compatible_or_spectrum_types<Input, Output>
    void cast(Input&& input, Output&& output, bool clamp = false) {
        check(not noa::are_overlapped(input, output), "The input and output arrays should not overlap");
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
    template<nt::array_decay Input>
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
                const auto n_elements = noa::offset_at(input.strides(), input.shape().vec - 1) + 1;
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
                const auto n_elements = noa::offset_at(input.strides(), input.shape().vec - 1) + 1;
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
    /// N-dimensional array.
    ///
    /// - Memory ownership:
    /// Arrays can manage reference-counted memory regions (using a std::shared_ptr) or be simple views over existing memory.
    /// This behavior is controlled at compile time by the ArrayOwnership template parameter.
    ///
    /// - Memory resource:
    /// The memory resource and location depend on the ArrayOption used to create the array. Arrays are therefore
    /// attached to a device, either the CPU or a GPU. Depending on the resource, arrays can be interpreted as CPU or
    /// GPU arrays, and it is possible to create aliases of the same resource for the CPU and the GPU (see as()).
    ///
    /// - Value type:
    /// Arrays are usually managing "numeric" types, like integers or (complex) floating-points. However, any trivially
    /// destructible type is supported, such as Vec or Matrices, except for references, pointers, and extents.
    /// Arrays can have const-qualified value types, however owning arrays must have mutable value-types when
    /// using the allocating constructors.
    ///
    /// - Eager asynchronous execution:
    /// The library is stream-ordered and starts computation eagerly by enqueing backend calls to the current stream.
    /// The current depends on the array's device and might be an asynchronous stream (GPU streams are all asynchronous).
    /// If the compute stream executes calls asynchronously, library functions may return before completion, and
    /// the input/output arrays of these functions may end up being destructed before completion. To ensure safe
    /// and simple usage of this asynchronous execution, owning arrays share a copy of their reference-counted
    /// pointer (using array.share()) to the stream. The stream will keep these copies alive until it is done
    /// executing the enqueued call, ensuring the memory stays valid during computation. Non-owning arrays, aka views,
    /// cannot do this, and users should therefore make sure that the viewed memory regions are destructed only after
    /// the scheduled work is completed (e.g., using array.sync()).
    /// Note that when an exception is thrown and arrays are destructed during stack unwinding, owning arrays will
    /// synchronize the current stream if needed (synchronous streams don't need this). This makes sure any potential
    /// use of their managed memory region via views is completed before freeing the memory.
    ///
    /// - Direct access:
    /// Arrays are not intended for direct element access as the underlying elements may not be dereferanceable by
    /// the current thread and stream-ordered compution might not be completed. Use span() to perform direct access
    /// and iterate through the array. Use sync() to synchronize the stream used by the array before accessing the
    /// underlying array elements.
    ///
    /// - Shape and strides:
    /// Shape and strides are specified in number of elements. When allocating an array, C-contiguous arrays are created
    /// with strides specified in the rightmost order (from left to right). While column-major ordering is supported,
    /// or any other stride order, row-major ordering is recommended. Empty dimensions have a size of 1. If one
    /// dimension is 0, the entire array is considered empty. Arrays can be broadcast to other shapes according to the
    /// broadcasting rule (see noa::broadcast()). As such, some arrays can have dimensions with a stride of 0.
    /// Negative strides are currently not supported. The number of dimensions can easily be changed using the
    /// axes(...) or array<...>() member functions.
    template<typename T, usize N, ArrayOwnership O>
    class Array {
    public: // typedefs
        static constexpr ArrayOwnership ARRAY_OWNERSHIP = O;
        static constexpr bool IS_VIEW = ARRAY_OWNERSHIP == ArrayOwnership::VIEW;

        static_assert(not std::is_pointer_v<T>);
        static_assert(not std::is_reference_v<T>);
        static_assert(std::is_trivially_destructible_v<T>);

        using pointer_type = T*;
        using value_type = T;
        using mutable_value_type = std::remove_const_t<T>;
        using const_value_type = std::add_const_t<mutable_value_type>;
        using index_type = isize;
        using strides_type = Strides<index_type, N>;
        using view_type = Array<value_type, N, ArrayOwnership::VIEW>;
        using array_type = Array<mutable_value_type, N, ArrayOwnership::RC>;
        using shape_type = Shape<index_type, N>;
        using span_type = Span<value_type, N, index_type>;
        using shared_type = std::conditional_t<IS_VIEW, pointer_type, std::shared_ptr<value_type[]>>;

        static constexpr StridesTraits STRIDES_TRAIT = span_type::STRIDES_TRAIT;
        static constexpr PointerTraits POINTER_TRAIT = span_type::POINTER_TRAIT;
        static constexpr bool IS_CONTIGUOUS = span_type::IS_CONTIGUOUS;
        static constexpr bool IS_RESTRICT = span_type::IS_RESTRICT;
        static constexpr usize SIZE = N;
        static constexpr isize SSIZE = N;

    public: // Static factory functions
        template<std::convertible_to<value_type>... Ts> requires (not IS_VIEW)
        constexpr static auto from_values(Ts&&... a) -> Array {
            auto output = Array(sizeof...(Ts));
            [&]<usize...I>(std::index_sequence<I...>) {
                (std::construct_at(output.data() + I, std::forward<Ts>(a)), ...);
            }(std::make_index_sequence<sizeof...(Ts)>{});
            return output;
        }

    public: // Constructors
        /// Creates an empty array.
        constexpr Array() = default;

        /// Creates and allocates a contiguous row vector.
        /// \param n_elements   Number of elements.
        /// \param option       Options of the created array.
        /// \see Allocator for more details.
        constexpr explicit Array(isize n_elements, ArrayOption option = {}) requires (not IS_VIEW):
            m_options{option}
        {
            for (usize i = 0; i < N - 1; ++i) {
                m_shape[i] = 1;
                m_strides[i] = n_elements;
            }
            m_shape[N - 1] = n_elements;
            m_strides[N - 1] = 1;
            allocate_();
        }

        /// Creates and allocates a contiguous array.
        /// \param shape    Shape of the array.
        /// \param option   Options of the created array.
        /// \see Allocator for more details.
        constexpr explicit Array(const shape_type& shape, ArrayOption option = {}) requires (not IS_VIEW) :
            m_shape(shape),
            m_strides(shape.strides()),
            m_options{option} { allocate_(); }

        /// Encapsulates an existing 1d contiguous memory region.
        /// \param[in,out] data Data to encapsulate.
        /// \param n_elements   Number of elements in data.
        /// \param option       Options of data.
        constexpr Array(shared_type data, isize n_elements, ArrayOption option = {}) :
            m_shared(std::move(data)),
            m_options{option}
        {
            for (usize i = 0; i < N - 1; ++i) {
                m_shape[i] = 1;
                m_strides[i] = n_elements;
            }
            m_shape[N - 1] = n_elements;
            m_strides[N - 1] = 1;
            allocator().validate(get(), device());
        }

        /// Encapsulates an existing contiguous memory region.
        /// \param[in,out] data Data to encapsulate.
        /// \param shape        Shape of data.
        /// \param option       Options of data.
        constexpr Array(
            shared_type data,
            const shape_type& shape,
            ArrayOption option = {}
        ) :
            m_shape{shape},
            m_strides{shape.strides()},
            m_shared(std::move(data)),
            m_options{option}
        {
            allocator().validate(get(), device());
        }

        /// Encapsulates an existing strided memory region.
        /// \param[in,out] data Data to encapsulate.
        /// \param shape        Shape of data.
        /// \param strides      Strides of data.
        /// \param option       Options of data.
        constexpr Array(
            shared_type data,
            const shape_type& shape,
            const strides_type& strides,
            ArrayOption option = {}
        ) :
            m_shape{shape},
            m_strides{strides},
            m_shared(std::move(data)),
            m_options{option}
        {
            allocator().validate(get(), device());
        }

        /// Encapsulates an existing strided memory region, without validation.
        /// \param[in,out] data Data to encapsulate.
        /// \param shape        Shape of data.
        /// \param strides      Strides of data.
        /// \param option       Options of data.
        constexpr Array(
            shared_type data,
            const shape_type& shape,
            const strides_type& strides,
            ArrayOption option,
            Unchecked
        ) :
            m_shape{shape},
            m_strides{strides},
            m_shared(std::move(data)),
            m_options{option} {}

        /// Creates a view from a span.
        template<StridesTraits S> requires IS_VIEW
        constexpr explicit Array(const Span<value_type, N, index_type, S>& span, ArrayOption options = {}) noexcept:
            m_shape{span.shape()},
            m_strides{span.strides_full()},
            m_shared{span.data()},
            m_options{options}
        {
            allocator().validate(span.data(), device());
        }

        /// Creates a const array from an existing non-const array.
        template<nt::mutable_of<value_type> U> requires std::is_const_v<value_type>
        constexpr /*implicit*/ Array(Array<U, N, O> array) noexcept :
            m_shape{array.shape()},
            m_strides{array.strides()},
            m_shared{std::move(array).share()},
            m_options{array.options()} {}

        /// Creates a view of an owning array.
        template<nt::almost_same_as<value_type> U> requires IS_VIEW
        constexpr explicit Array(const Array<U, N>& array) noexcept :
            m_shape{array.shape()},
            m_strides{array.strides()},
            m_shared{array.data()},
            m_options{array.options()} {}

        ~Array() noexcept requires (IS_VIEW) = default;
        ~Array() noexcept requires (not IS_VIEW) {
            /// If an exception is thrown, make sure to synchronize the stream to guarantee that functions called
            /// with a view of that reference-counted array are done executing so that this array can release its memory.

            // This should slow down the exception path due to the extra synchronization, but given
            // that exceptions are used as non-recoverable errors that will ultimately end up terminating
            // the program, we don't really care about performance here.

            // While we could record the number of living exceptions at construction time to correctly
            // detect the case where an Array is created during stack unwinding, it doesn't seem worth it:
            // 1) we would need to keep track of a "count".
            // 2) creating an Array during stack unwinding is weird/rare.
            // 3) this will just end up saving a few stream synchronizations...

            try {
                // Freeing memory in CUDA GPUs will either synchronize the device, synchronize the stream or
                // enqueue an asynchronous free, all of which guarantees that the GPU is done using the memory
                // region of the Array. As such, only synchronize CPU arrays.
                if (std::uncaught_exceptions() > 0 and device().is_cpu())
                    eval();
            } catch (...) {} // ignore new exception thrown by eval()
        }

    public: // Queries
        [[nodiscard]] constexpr auto options() const noexcept -> ArrayOption { return m_options; }
        [[nodiscard]] constexpr auto device() const noexcept -> Device { return options().device; }
        [[nodiscard]] constexpr auto allocator() const noexcept -> Allocator { return options().allocator; }
        [[nodiscard]] constexpr auto shape() const noexcept -> const shape_type& { return m_shape; }
        [[nodiscard]] constexpr auto strides() const noexcept -> const strides_type& { return m_strides; }
        [[nodiscard]] constexpr auto strides_full() const noexcept -> const strides_type& { return m_strides; }
        [[nodiscard]] constexpr auto n_elements() const noexcept -> index_type { return shape().n_elements(); }
        [[nodiscard]] constexpr auto ssize() const noexcept -> isize { return static_cast<isize>(n_elements()); }
        [[nodiscard]] constexpr auto size() const noexcept -> usize { return static_cast<usize>(n_elements()); }

        template<char ORDER = 'C'>
        [[nodiscard]] constexpr auto contiguity() const noexcept -> Vec<bool, N> {
            return strides().template contiguity<ORDER>(shape());
        }

        template<char ORDER = 'C'>
        [[nodiscard]] constexpr bool is_contiguous() const noexcept {
            return strides().template is_contiguous<ORDER>(shape());
        }

        [[nodiscard]] constexpr auto broadcasting() const noexcept -> Vec<bool, N> {
            return strides().broadcasting(shape());
        }

        [[nodiscard]] constexpr bool is_broadcast() const noexcept {
            return strides().is_broadcast();
        }

        /// Whether the array is empty.
        /// An array is empty if not initialized, o if the viewed data is null, or if one of its dimensions is 0.
        [[nodiscard]] constexpr bool is_empty() const noexcept { return not get() or shape().is_empty(); }

    public: // Accessors
        /// Synchronizes the current stream of the array's device.
        /// \details It guarantees safe access to the memory region. Note that stream-ordered access
        ///          (i.e., passing this to the library API) is safe and doesn't need synchronization.
        auto eval() const& -> const Array& {
            Stream::current(device()).synchronize();
            return *this;
        }
        auto eval() & -> Array& {
            Stream::current(device()).synchronize();
            return *this;
        }
        auto eval() && -> Array&& {
            Stream::current(device()).synchronize();
            return std::move(*this);
        }

        /// Returns the pointer to the data.
        /// \warning Depending on the current stream of this array's device,
        ///          reading/writing to this pointer may be illegal or create a data race.
        [[nodiscard]] constexpr auto get() const noexcept -> pointer_type {
            if constexpr (IS_VIEW)
                return m_shared;
            else
                return m_shared.get();
        }
        [[nodiscard]] constexpr auto data() const noexcept -> pointer_type { return get(); }
        [[nodiscard]] constexpr auto share() const& noexcept -> const shared_type& { return m_shared; }
        [[nodiscard]] constexpr auto share() && noexcept -> shared_type&& { return std::move(m_shared); }

        /// Returns a span of the array.
        /// \warning Depending on the current stream of this view's device,
        ///          reading/writing through this Span may be illegal or create a data race.
        [[nodiscard]] constexpr auto span() const -> span_type {
            return span_type(get(), shape(), strides());
        }

        template<typename U, usize NewN = 4,
                 typename I = index_type,
                 StridesTraits NewStridesTrait = STRIDES_TRAIT,
                 PointerTraits NewPointerTrait = POINTER_TRAIT>
        [[nodiscard]] constexpr auto span() const {
            return span().template span<U, NewN, I, NewStridesTrait, NewPointerTrait>();
        }

        template<typename U = value_type, usize NewN = 4,
                 typename I = index_type,
                 PointerTraits NewPointerTrait = POINTER_TRAIT>
        [[nodiscard]] constexpr auto span_contiguous() const {
            return span<U, NewN, I, StridesTraits::CONTIGUOUS, NewPointerTrait>();
        }

        template<typename U = value_type,
                 typename I = index_type,
                 StridesTraits NewStridesTrait = StridesTraits::CONTIGUOUS,
                 PointerTraits NewPointerTrait = POINTER_TRAIT>
        [[nodiscard]] constexpr auto span_1d() const {
            return span<U, 1, I, NewStridesTrait, NewPointerTrait>();
        }

        template<typename U = value_type,
                 typename I = index_type,
                 PointerTraits NewPointerTrait = POINTER_TRAIT>
        [[nodiscard]] constexpr auto span_1d_contiguous() const {
            return span<U, 1, I, StridesTraits::CONTIGUOUS, NewPointerTrait>();
        }

        template<typename U = value_type,
                 typename I = index_type,
                 PointerTraits NewPointerTrait = POINTER_TRAIT>
        [[nodiscard]] constexpr auto span_1d_strided() const {
            return span<U, 1, I, StridesTraits::STRIDED, NewPointerTrait>();
        }

    public: // Deep copy
        /// Performs a deep copy of the array to \p output.
        /// \details Contiguous regions of memory have no copy restrictions and can be copied to any device. This is
        ///          also true for pitched layouts, colum or row vectors, or any layout that can be reordered and/or
        ///          reshaped to the aforementioned layouts. However, other non-contiguous memory layouts can only
        ///          be copied if the source and destination are both on the same GPU or on the CPU.
        /// \param[out] output  Destination. It should not overlap with this array.
        template<nt::writable_array_decay Output>
            requires nt::array_decay_of_almost_same_type<Array, Output>
        auto to(Output&& output) const& -> decltype(auto) {
            noa::copy(*this, output);
            return std::forward<Output>(output);
        }
        template<nt::writable_array_decay Output>
            requires nt::array_decay_of_almost_same_type<Array, Output>
        auto to(Output&& output) && -> decltype(auto) {
            noa::copy(std::move(*this), output);
            return std::forward<Output>(output);
        }

        /// Performs a deep copy of the array according \p option.
        /// \details The returned array is completely independent of the original one and is C-contiguous.
        ///          Contiguous regions of memory have no copy restrictions and can be copied to any device. This is
        ///          also true for pitched layouts, colum or row vectors, or any layout that can be reordered and/or
        ///          reshaped to the aforementioned layouts. However, other non-contiguous memory layouts can only
        ///          be copied if the source and destination are both on the same GPU or on the CPU.
        /// \param option   Output device and resource to perform the allocation of the new array.
        ///                 The current stream for that device is used to perform the copy.
        [[nodiscard]] auto to(ArrayOption option) const& -> array_type {
            auto out = array_type(shape(), option);
            to(out);
            return out;
        }
        [[nodiscard]] auto to(ArrayOption option) && -> array_type {
            auto out = array_type(shape(), option);
            std::move(*this).to(out);
            return out;
        }

        /// Performs a deep copy of the array to the CPU.
        [[nodiscard]] auto to_cpu() const& -> array_type {
            return to(ArrayOption{});
        }
        [[nodiscard]] auto to_cpu() && -> array_type {
            return std::move(*this).to(ArrayOption{});
        }

        /// Performs a deep copy of the array preserving the array's options.
        [[nodiscard]] auto copy() const& -> array_type {
            return to(options());
        }
        [[nodiscard]] auto copy() && -> array_type {
            return std::move(*this).to(options());
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
                view_type(get(), one_shape, one_strides, options(), Unchecked{}).to(
                    view_type(&output, one_shape, one_strides, ArrayOption{}, Unchecked{}));
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
        [[nodiscard]] constexpr auto is_reinterpretable_as_gpu() const noexcept -> bool {
            return options().is_reinterpretable(Device::GPU);
        }
        [[nodiscard]] constexpr auto is_dereferenceable() const noexcept -> bool {
            return options().is_dereferenceable();
        }

        /// Changes the device type (CPU<->GPU) on which the memory should be accessed. See reinterpret_as for more details.
        auto reinterpret_as(Device::Type type, ReinterpretAsOptions parameters = {}) const& -> Array {
            return noa::reinterpret_as(*this, type, parameters);
        }
        auto reinterpret_as_cpu(ReinterpretAsOptions parameters = {}) const& -> Array {
            return noa::reinterpret_as(*this, Device::CPU, parameters);
        }
        auto reinterpret_as_gpu(ReinterpretAsOptions parameters = {}) const& -> Array {
            return noa::reinterpret_as(*this, Device::GPU, parameters);
        }
        auto reinterpret_as(Device::Type type, ReinterpretAsOptions parameters = {}) && -> Array {
            return noa::reinterpret_as(std::move(*this), type, parameters);
        }
        auto reinterpret_as_cpu(ReinterpretAsOptions parameters = {}) && -> Array {
            return noa::reinterpret_as(std::move(*this), Device::CPU, parameters);
        }
        auto reinterpret_as_gpu(ReinterpretAsOptions parameters = {}) && -> Array {
            return noa::reinterpret_as(std::move(*this), Device::GPU, parameters);
        }

        /// Returns a new array.
        /// See ReinterpretLayout for more details.
        template<typename NewT = T, usize NewN = N, ArrayOwnership NewArrayOwnership = ARRAY_OWNERSHIP>
        [[nodiscard]] constexpr auto array() const& {
            static_assert((NewArrayOwnership == ArrayOwnership::RC and not IS_VIEW) or
                          NewArrayOwnership == ArrayOwnership::VIEW, "Cannot create an owning array from a view");

            using output_t = Array<NewT, NewN, ARRAY_OWNERSHIP>;
            using reinterpret_t = nd::ReinterpretLayoutStrided<value_type, N, index_type>;
            const auto reinterpreted = reinterpret_t(data(), shape(), strides()).template as<NewT, NewN>();

            if constexpr (IS_VIEW or NewArrayOwnership == ArrayOwnership::VIEW) {
                return output_t(reinterpreted.ptr, reinterpreted.shape, reinterpreted.strides, options(), Unchecked{});
            } else {
                return output_t(std::shared_ptr<NewT[]>(m_shared, reinterpreted.ptr),
                                reinterpreted.shape, reinterpreted.strides, options(), Unchecked{});
            }
        }
        template<typename NewT = T, usize NewN = N, ArrayOwnership NewArrayOwnership = ARRAY_OWNERSHIP>
        [[nodiscard]] constexpr auto array() && {
            static_assert((NewArrayOwnership == ArrayOwnership::RC and not IS_VIEW) or
                          NewArrayOwnership == ArrayOwnership::VIEW, "Cannot create an owning array from a view");

            using output_t = Array<NewT, NewN, ARRAY_OWNERSHIP>;
            using reinterpret_t = nd::ReinterpretLayoutStrided<value_type, N, index_type>;
            const auto reinterpreted = reinterpret_t(data(), shape(), strides()).template as<NewT, NewN>();

            if constexpr (IS_VIEW or NewArrayOwnership == ArrayOwnership::VIEW) {
                return output_t(reinterpreted.ptr, reinterpreted.shape, reinterpreted.strides, options(), Unchecked{});
            } else {
                return output_t(std::shared_ptr<NewT[]>(std::move(m_shared), reinterpreted.ptr),
                                reinterpreted.shape, reinterpreted.strides, options(), Unchecked{});
            }
        }

        template<typename NewT, usize NewN = N>
        [[nodiscard]] constexpr auto as() const& {
            return array<NewT, NewN>();
        }
        template<typename NewT, usize NewN = N>
        [[nodiscard]] constexpr auto as() && {
            return std::move(*this).template array<NewT, NewN>();
        }

        template<usize NewN = N>
        [[nodiscard]] constexpr auto as_const() const& {
            return array<const_value_type, NewN>();
        }
        template<usize NewN = N>
        [[nodiscard]] constexpr auto as_const() && {
            return std::move(*this).template array<const_value_type, NewN>();
        }

        template<usize NewN = N>
        [[nodiscard]] constexpr auto as_bytes() const& {
            using output_t = std::conditional_t<std::is_const_v<value_type>, const Byte, Byte>;
            return array<output_t, NewN>();
        }
        template<usize NewN = N>
        [[nodiscard]] constexpr auto as_bytes() && {
            using output_t = std::conditional_t<std::is_const_v<value_type>, const Byte, Byte>;
            return std::move(*this).template array<output_t, NewN>();
        }

        template<typename NewT = value_type>
        [[nodiscard]] constexpr auto as_1d() const& {
            return array<NewT, 1>();
        }
        template<typename NewT = value_type>
        [[nodiscard]] constexpr auto as_1d() && {
            return std::move(*this).template array<NewT, 1>();
        }

        template<usize NewN, typename NewT = value_type>
        [[nodiscard]] constexpr auto as_nd() const& {
            return array<NewT, NewN>();
        }
        template<usize NewN, typename NewT = value_type>
        [[nodiscard]] constexpr auto as_nd() && {
            return std::move(*this).template array<NewT, NewN>();
        }

        /// Returns a (const-)view of the array.
        template<nt::almost_same_as<value_type> NewT = value_type, usize NewN = N>
        [[nodiscard]] constexpr auto view() const noexcept {
            if constexpr (NewN == N)
                return Array<NewT, NewN, ArrayOwnership::VIEW>(*this);
            else
                return array<NewT, NewN, ArrayOwnership::VIEW>();
        }

        /// Reshapes the array, with size inference.
        template<usize NewN>
        [[nodiscard]] constexpr auto reshape(Shape<index_type, NewN> new_shape) const& {
            Strides<index_type, NewN> new_stride;
            check(noa::infer_size(new_shape, n_elements()),
                  "The desired shape {} is not compatible with the current shape {}, "
                  "or the size inference is invalid or ambiguous", new_shape, shape());
            check(noa::reshape(shape(), strides(), new_shape, new_stride),
                  "An memory region of shape {} and stride {} cannot be reshaped to a shape of {}",
                  shape(), strides(), new_shape);
            return Array<value_type, NewN, ARRAY_OWNERSHIP>(m_shared, new_shape, new_stride, options(), Unchecked{});
        }
        template<usize NewN>
        [[nodiscard]] constexpr auto reshape(Shape<index_type, NewN> new_shape) && {
            Strides<index_type, NewN> new_stride;
            check(noa::infer_size(new_shape, n_elements()),
                  "The desired shape {} is not compatible with the current shape {}, "
                  "or the size inference is invalid or ambiguous", new_shape, shape());
            check(noa::reshape(shape(), strides(), new_shape, new_stride),
                  "An memory region of shape {} and stride {} cannot be reshaped to a shape of {}",
                  shape(), strides(), new_shape);
            return Array<value_type, NewN, ARRAY_OWNERSHIP>(std::move(m_shared), new_shape, new_stride, options(), Unchecked{});
        }

        /// Reshapes the array in a vector along a particular axis.
        /// Returns a row vector by default (axis = N - 1).
        template<nt::integer I = usize>
        [[nodiscard]] constexpr auto flat(I axis = static_cast<I>(N - 1)) const & -> Array {
            bounds_check<true>(N, axis);
            return this->reshape(shape().flat(axis));
        }
        template<nt::integer I = usize>
        [[nodiscard]] constexpr auto flat(I axis = static_cast<I>(N - 1)) && -> Array {
            bounds_check<true>(N, axis);
            return std::move(*this).reshape(shape().flat(axis));
        }

        /// Permutes the dimensions of the view.
        /// \param permutation  Permutation with the axes numbered from 0 to N - 1.
        template<nt::integer... I>
        [[nodiscard]] constexpr auto permute(I... permutation) const& -> Array {
            return Array(m_shared, shape().permute(permutation...), strides().permute(permutation...), options(), Unchecked{});
        }
        template<nt::integer... I>
        [[nodiscard]] constexpr auto permute(I... permutation) && -> Array {
            return Array(std::move(m_shared), shape().permute(permutation...), strides().permute(permutation...), options(), Unchecked{});
        }
        template<nt::integer I = i32>
        [[nodiscard]] constexpr auto permute(const Vec<I, N>& permutation) const& -> Array {
            return Array(m_shared, shape().permute(permutation), strides().permute(permutation), options(), Unchecked{});
        }
        template<nt::integer I = i32>
        [[nodiscard]] constexpr auto permute(const Vec<I, N>& permutation) && -> Array {
            return Array(std::move(m_shared), shape().permute(permutation), strides().permute(permutation), options(), Unchecked{});
        }

        /// Permutes the array by performing a deep-copy. The returned Array is a new C-contiguous array.
        /// \param permutation  Permutation with the axes numbered from 0 to N - 1.
        template<nt::integer... I>
        [[nodiscard]] auto permute_copy(I... permutation) const& -> array_type {
            return noa::permute_copy(*this, Vec{static_cast<i32>(permutation)...});
        }
        template<nt::integer... I>
        [[nodiscard]] auto permute_copy(I... permutation) && -> array_type {
            return noa::permute_copy(std::move(*this), Vec{static_cast<i32>(permutation)...});
        }
        template<nt::integer I = isize>
        [[nodiscard]] auto permute_copy(const Vec<I, N>& permutation) const& -> array_type {
            return noa::permute_copy(*this, permutation.template as<i32>());
        }
        template<nt::integer I = isize>
        [[nodiscard]] auto permute_copy(const Vec<I, N>& permutation) && -> array_type {
            return noa::permute_copy(std::move(*this), permutation.template as<i32>());
        }

        /// Returns an array with the given axes.
        template<nt::integer... U>
        [[nodiscard]] constexpr auto filter(U... axes) const& {
            return Array<value_type, sizeof...(U), ARRAY_OWNERSHIP>(
                m_shared, shape().filter(axes...), strides().filter(axes...));
        }
        template<nt::integer... U>
        [[nodiscard]] constexpr auto filter(U... axes) && {
            return Array<value_type, sizeof...(U), ARRAY_OWNERSHIP>(
                std::move(m_shared), shape().filter(axes...), strides().filter(axes...));
        }
        template<nt::integer U, usize M>
        [[nodiscard]] constexpr auto filter(const Vec<U, M>& axes) const& {
            return Array<value_type, M, ARRAY_OWNERSHIP>(
                m_shared, shape().filter(axes), strides().filter(axes));
        }
        template<nt::integer U, usize M>
        [[nodiscard]] constexpr auto filter(const Vec<U, M>& axes) && {
            return Array<value_type, M, ARRAY_OWNERSHIP>(
                std::move(m_shared), shape().filter(axes), strides().filter(axes));
        }

    public:
        /// Clears the array. Equivalent to assigning *this with an empty array.
        Array& operator=(std::nullptr_t) {
            *this = Array{};
            return *this;
        }

        /// Drops the resource of this array into the returned array.
        auto drop() & -> Array {
            Array out = *this;
            *this = Array{};
            return out;
        }
        auto drop() && -> Array {
            Array out = std::move(*this);
            *this = Array{};
            return out;
        }

    public:
        /// Subregion indexing. Extracts a subregion from the current array.
        template<typename... U>
        [[nodiscard]] constexpr auto subregion(const Subregion<N, U...>& subregion) const& -> Array {
            auto [new_shape, new_strides, offset] = subregion.extract_from(shape(), strides());
            if constexpr (IS_VIEW)
                return Array(get() + offset, new_shape, new_strides, options(), Unchecked{});
            else
                return Array(shared_type(m_shared, get() + offset), new_shape, new_strides, options(), Unchecked{});
        }
        template<typename... U>
        [[nodiscard]] constexpr auto subregion(const Subregion<N, U...>& subregion) && -> Array {
            auto [new_shape, new_strides, offset] = subregion.extract_from(shape(), strides());
            if constexpr (IS_VIEW)
                return Array(get() + offset, new_shape, new_strides, options(), Unchecked{});
            else
                return Array(shared_type(std::move(m_shared), get() + offset), new_shape, new_strides, options(), Unchecked{});
        }

        /// Subregion indexing. Extracts a subregion from the current array.
        /// \see noa::Subregion for more details on the variadic parameters to enter.
        template<typename... U> requires nt::subregion_access_sequence<4, U...>
        [[nodiscard]] constexpr auto subregion(const U&... access_sequence) const& -> Array {
            return subregion(Subregion<N, U...>(access_sequence...));
        }
        template<typename... U> requires nt::subregion_access_sequence<4, U...>
        [[nodiscard]] constexpr auto subregion(const U&... access_sequence) && -> Array {
            return std::move(*this).subregion(Subregion<N, U...>(access_sequence...));
        }

    private:
        void allocate_() {
            if (allocator() == Allocator::PITCHED) {
                noa::tie(m_shared, m_strides) = allocator().template allocate_pitched<value_type>(shape(), device());
            } else {
                m_shared = allocator().template allocate<value_type>(n_elements(), device());
            }
        }

    private:
        shape_type m_shape{};
        strides_type m_strides{};
        shared_type m_shared{};
        ArrayOption m_options{};
    };

    /// Deduction guides for ctor encapsulating data.
    template<typename T, nt::integer I>
    Array(T*, I, ArrayOption a = {}) -> Array<T, 1, ArrayOwnership::VIEW>;
    template<typename T, nt::integer I>
    Array(std::shared_ptr<T[]>, I, ArrayOption a = {}) -> Array<T, 1, ArrayOwnership::RC>;

    template<typename T, nt::integer I, usize N>
    Array(T*, Shape<I, N>, ArrayOption a = {}) -> Array<T, N, ArrayOwnership::VIEW>;
    template<typename T, nt::integer I, usize N>
    Array(std::shared_ptr<T[]>, Shape<I, N>, ArrayOption a = {}) -> Array<T, N, ArrayOwnership::RC>;

    template<typename T, nt::integer I, usize N>
    Array(T*, Shape<I, N>, Strides<I, N>, ArrayOption a = {}) -> Array<T, N, ArrayOwnership::VIEW>;
    template<typename T, nt::integer I, usize N>
    Array(std::shared_ptr<T[]>, Shape<I, N>, Strides<I, N>, ArrayOption a = {}) -> Array<T, N, ArrayOwnership::RC>;

    template<typename T, usize N>
    using View = Array<T, N, ArrayOwnership::VIEW>;
}

namespace noa::traits {
    template<typename T, usize N, ArrayOwnership O> struct proclaim_is_array<Array<T, N, O>> : std::true_type {};
    template<typename T, usize N> struct proclaim_is_array_rc<Array<T, N>> : std::true_type {};
    template<typename T, usize N> struct proclaim_is_array_view<Array<T, N, ArrayOwnership::VIEW>> : std::true_type {};
}
