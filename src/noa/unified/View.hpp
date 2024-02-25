#pragma once

#include "noa/core/Config.hpp"

#if defined(NOA_IS_OFFLINE)
#include "noa/core/Exception.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/Types.hpp"
#include "noa/core/Indexing.hpp"

#include "noa/unified/ArrayOption.hpp"
#include "noa/unified/Stream.hpp"
#include "noa/unified/Traits.hpp"
#include "noa/unified/Indexing.hpp"

#include "noa/cpu/Copy.hpp"
#include "noa/cpu/Permute.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/Copy.cuh"
#include "noa/gpu/cuda/Permute.cuh"
#include "noa/gpu/cuda/AllocatorManaged.hpp"
#endif

namespace noa::inline types {
    template<typename T>
    class Array;
}

namespace noa {
    /// (Deep-)Copies arrays.
    /// \details Contiguous regions of memory have no copy restrictions and can be copied to any device. This is
    ///          also true for pitched layouts and colum or row vectors. However, other non-contiguous memory
    ///          layouts can only be copied if the source and destination are both on the same GPU or on the CPU.
    /// \param[in] input    Source.
    /// \param[out] output  Destination. It should not overlap with \p input.
    template<typename Input, typename Output>
    requires (nt::are_varray_v<Input, Output> and
              nt::are_almost_same_value_type_v<Input, Output>)
    void copy(const Input& input, const Output& output) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        check(not ni::are_overlapped(input, output), "The input and output should not overlap");

        auto input_strides = input.strides();
        if (not ni::broadcast(input.shape(), input_strides, output.shape())) {
            panic("Cannot broadcast an array of shape {} into an array of shape {}",
                  input.shape(), output.shape());
        }

        const Device input_device = input.device();
        const Device output_device = output.device();
        if (input_device.is_cpu() and output_device.is_cpu()) {
            auto& cpu_stream = Stream::current(input_device).cpu();
            const auto n_threads = cpu_stream.thread_limit();
            if ((nt::is_array_v<Input> or nt::is_array_v<Output>) and cpu_stream.is_async()) {
                cpu_stream.enqueue([=]() { // capture the arrays
                    noa::cpu::copy(
                            input.get(), input_strides,
                            output.get(), output.strides(),
                            output.shape(), n_threads);
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
            cuda_stream.enqueue_attach(input, output);
            cuda_stream.synchronize();
            #else
            panic("No GPU backend detected");
            #endif
        } else { // gpu -> gpu or cpu -> gpu
            #ifdef NOA_ENABLE_CUDA
            if (input_device != output_device)
                Stream::current(input_device).synchronize(); // wait for the input
            auto& cuda_stream = Stream::current(output_device).cuda();
            noa::cuda::copy(input.get(), input_strides,
                            output.get(), output.strides(),
                            output.shape(), cuda_stream);
            cuda_stream.enqueue_attach(input, output);
            #else
            panic("No GPU backend detected");
            #endif
        }
    }

    /// Permutes the input by reordering its dimensions. The returned object points to the same data.
    template<typename Input> requires nt::is_varray_v<Input>
    Input permute(const Input& input, const Vec4<i64>& permutation) {
        check(all(permutation <= 3) and sum(permutation) == 6, "Permutation {} is not valid", permutation);
        const auto permuted_shape = ni::reorder(input.shape(), permutation);
        const auto permuted_strides = ni::reorder(input.strides(), permutation);
        return Input(input.share(), permuted_shape, permuted_strides, input.options());
    }

    /// Permutes, in memory, the axes of an array.
    /// \tparam T           Any numeric type.
    /// \param[in] input    Array to permute.
    /// \param[out] output  Permuted array. Its shape and strides should be permuted already.
    /// \param permutation  Permutation. Axes are numbered from 0 to 3.
    /// \note For in-place permutations, only 0123, 0213, 0132, and 0321 are supported. Anything else throws an error.
    /// \note The in-place 0213 permutation requires the axis 1 and 2 to have the same size.
    ///       The in-place 0132 permutation requires the axis 3 and 2 to have the same size.
    ///       The in-place 0321 permutation requires the axis 3 and 1 to have the same size.
    /// \note On the GPU, the following permutations are optimized: 0123, 0132, 0312, 0321, 0213, 0231.
    ///       Anything else calls copy(), which is slower.
    template<typename Input, typename Output>
    requires (nt::are_varray_v<Input, Output> and nt::are_almost_same_value_type_v<Input, Output>)
    void permute_copy(const Input& input, const Output& output, const Vec4<i64>& permutation) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        check(all(permutation <= 3) and sum(permutation) == 6, "Permutation {} is not valid", permutation);

        // To enable broadcasting, we need to permute the input.
        auto input_strides = input.strides();
        auto input_shape = input.shape();
        for (i64 i = 0; i < 4; ++i) {
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
            if ((nt::is_array_v<Input> or nt::is_array_v<Output>) and cpu_stream.is_async()) {
                cpu_stream.enqueue([=]() { // capture the arrays
                    noa::cpu::permute_copy(
                            input.get(), input_strides, input_shape,
                            output.get(), output.strides(),
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
            cuda_stream.enqueue_attach(input, output);
            #else
            panic("No GPU backend detected");
            #endif
        }
    }

    /// Permutes the input by performing a deep-copy. The returned Array is a new C-contiguous array.
    /// \param[in] input    VArray to permute.
    /// \param permutation  Permutation with the axes numbered from 0 to 3.
    template<typename Input> requires nt::is_varray_v<Input>
    auto permute_copy(const Input& input, const Vec4<i64>& permutation) {
        using mutable_value_type = nt::mutable_value_type_t<Input>;
        const auto permuted_shape = ni::reorder(input.shape(), permutation);
        auto output = Array<mutable_value_type>(permuted_shape, input.options());
        permute_copy(input, output, permutation);
        return output;
    }
}

namespace noa::inline types {
    /// View of an Array or other memory region.
    /// \details
    ///   Views have a similar interface as the Array interface, but their value type can be const-qualified.
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
        using shape_type = Shape<i64, 4>;
        using accessor_type = Accessor<T, 4, i64>;
        using pointer_type = typename accessor_type::pointer_type;
        using value_type = typename accessor_type::value_type;
        using index_type = typename accessor_type::index_type;
        using strides_type = typename accessor_type::strides_type;
        using mutable_value_type = std::remove_const_t<value_type>;

        static_assert(not std::is_pointer_v<value_type>);
        static_assert(not std::is_reference_v<value_type>);

    public: // Constructors
        /// Creates an empty view.
        constexpr View() = default;

        /// Creates a view of a contiguous row-vector.
        constexpr View(T* data, i64 n_elements, ArrayOption options = {})
                : m_accessor(data, strides_type{n_elements, n_elements, n_elements, 1}),
                  m_shape{1, 1, 1, n_elements}, m_options(options) {}

        /// Creates a view of a span.
        template<i64 S, typename I>
        constexpr explicit View(const Span<T, S, I>& span, ArrayOption options = {})
                : View(span.data(), span.size(), options) {}

        /// Creates a view of a strided memory region.
        constexpr View(T* data, const shape_type& shape, const strides_type& strides, ArrayOption options = {})
                : m_accessor(data, strides), m_shape(shape), m_options(options) {}

        /// Creates a view, assuming the data is C-contiguous.
        constexpr View(T* data, const shape_type& shape, ArrayOption options = {})
                : m_accessor(data, shape.strides()), m_shape(shape), m_options(options) {}

        /// Creates a const view from an existing non-const view.
        template<typename U> requires nt::is_mutable_value_type_v<U, value_type>
        constexpr /*implicit*/ View(const View<U>& view)
                : m_accessor(view.data(), view.strides()), m_shape(view.shape()), m_options(view.options()) {}

        /// Creates a view of an Array.
        template<typename U> requires nt::is_almost_same_v<U, value_type>
        constexpr explicit View(const Array<U>& array)
                : m_accessor(array.get(), array.strides()), m_shape(array.shape()), m_options(array.options()) {}

    public: // Getters
        /// Returns the options used to create the viewed array.
        [[nodiscard]] constexpr const ArrayOption& options() const noexcept { return m_options; }

        /// Returns the device used to create the viewed array.
        [[nodiscard]] constexpr Device device() const noexcept { return options().device; }

        /// Returns the memory resource used to create the viewed array.
        [[nodiscard]] constexpr Allocator allocator() const noexcept { return options().allocator; }

        /// Whether the managed data can be accessed by CPU threads.
        [[nodiscard]] constexpr bool is_dereferenceable() const noexcept { return options().is_dereferenceable(); }

        /// Returns the BDHW shape of the viewed array.
        [[nodiscard]] const shape_type& shape() const noexcept { return m_shape; }

        /// Returns the BDHW strides of the viewed array.
        [[nodiscard]] const strides_type& strides() const noexcept { return m_accessor.strides(); }

        /// Returns the number of elements in the viewed array.
        [[nodiscard]] constexpr index_type elements() const noexcept { return shape().elements(); }
        [[nodiscard]] constexpr index_type ssize() const noexcept { return elements(); }
        [[nodiscard]] constexpr size_t size() const noexcept { return static_cast<size_t>(elements()); }

        template<char ORDER = 'C'>
        [[nodiscard]] constexpr bool are_contiguous() const noexcept {
            return are_contiguous<ORDER>(m_accessor.strides(), shape());
        }

        template<char ORDER = 'C'>
        [[nodiscard]] constexpr auto is_contiguous() const noexcept {
            return is_contiguous<ORDER>(m_accessor.strides(), shape());
        }

        /// Whether the view is empty. A View is empty if not initialized,
        /// or if the viewed data is null, or if one of its dimension is 0.
        [[nodiscard]] constexpr bool is_empty() const noexcept { return not get() or any(shape() == 0); }

    public: // Accessors
        /// Synchronizes the current stream of the view's device.
        /// \details It guarantees safe access to the memory region using get(), data(), operator(...), and accessor().
        ///          Note that stream-ordered access (i.e. passing this to the library API) is safe and doesn't need
        ///          synchronization.
        const View& eval() const {
            Stream::current(device()).synchronize();
            return *this;
        }

        /// Returns the pointer to the data.
        /// \warning Depending on the current stream of this array's device,
        ///          reading/writing to this pointer may be illegal or create a data race.
        [[nodiscard]] constexpr pointer_type get() const noexcept { return m_accessor.get(); }

        /// Returns the pointer to the data.
        /// \warning Depending on the current stream of this array's device,
        ///          reading/writing to this pointer may be illegal or create a data race.
        [[nodiscard]] constexpr pointer_type data() const noexcept { return m_accessor.data(); }

        /// Returns the pointer to the data.
        /// \warning Depending on the current stream of this array's device,
        ///          reading/writing to this pointer may be illegal or create a data race.
        [[nodiscard]] constexpr pointer_type share() const noexcept { return get(); }

        /// Returns a (const-)span of the array.
        /// \warning Depending on the current stream of this array's device,
        ///          reading/writing through this Span may be illegal or create a data race.
        template<typename U = value_type, i64 SIZE = -1, typename I = index_type>
        requires (nt::is_almost_same_v<U, value_type> and std::is_integral_v<I>)
        [[nodiscard]] constexpr auto span() const noexcept {
            check(are_contiguous(),
                  "Cannot create a Span from a non-contiguous view (shape={}, strides={})",
                  shape(), strides());
            if constexpr (SIZE >= 0) {
                check(elements() >= SIZE,
                      "Cannot create a Span with a static size of {} from a view with only {} elements",
                      SIZE, elements());
            }
            return Span<U, SIZE, I>(get(), safe_cast<I>(ssize()));
        }

        /// Returns a new Accessor and its corresponding shape.
        /// \details While constructing the accessor, this function can also reinterpret the current value type.
        ///          This is only well defined in cases where View::as<U>() is well defined.
        ///          If N < 4, the outer-dimensions are stacked together.
        template<typename U, size_t N = 4, typename I = index_type,
                 PointerTraits PointerTrait = PointerTraits::DEFAULT,
                 StridesTraits StridesTrait = StridesTraits::STRIDED>
        [[nodiscard]] constexpr auto accessor_and_shape() const {
            using output_shape_t = Shape<I, N>;
            using output_strides_t = Strides<I, N>;
            using output_accessor_t = Accessor<U, N, I, PointerTrait, StridesTrait>;
            using output_t = Pair<output_accessor_t, output_shape_t>;

            const auto reinterpreted = ni::ReinterpretLayout(shape(), strides(), get()).template as<U>();
            // FIXME If StridesTraits::CONTIGUOUS, assert(strides[3] == 1) ?

            if constexpr (N == 4) {
                return output_t{output_accessor_t(reinterpreted.ptr, reinterpreted.strides.template as_safe<I>()),
                                reinterpreted.shape.template as_safe<I>()};
            } else {
                // Construct the new shape by stacking the outer dimensions together.
                constexpr i64 OFFSET = 4 - N;
                auto new_shape = Shape4<i64>::filled_with(1);
                for (i64 i = 0; i < 4; ++i)
                    new_shape[max(i, OFFSET)] *= reinterpreted.shape[i];

                // Reshape.
                Strides4<i64> new_stride{};
                check(ni::reshape(reinterpreted.shape, reinterpreted.strides, new_shape, new_stride),
                      "A view of shape {} and strides {} cannot be reshaped to shape {}",
                      reinterpreted.shape, reinterpreted.strides, new_shape);

                // Ignore the outer empty dimensions.
                output_shape_t output_shape{new_shape.template pop_front<OFFSET>().template as_safe<I>()};
                output_strides_t output_strides{new_stride.template pop_front<OFFSET>().template as_safe<I>()};
                return output_t{output_accessor_t(reinterpreted.ptr, output_strides),
                                output_shape};
            }
        }

        [[nodiscard]] constexpr const accessor_type& accessor() const { return m_accessor; }

        template<typename U, size_t N = 4, typename I = index_type,
                 PointerTraits PointerTrait = PointerTraits::DEFAULT,
                 StridesTraits StridesTrait = StridesTraits::STRIDED>
        [[nodiscard]] constexpr auto accessor() const {
            return accessor_and_shape<U, N, I, PointerTrait, StridesTrait>().first;
        }

        template<typename U = value_type, size_t N = 4, typename I = index_type,
                 PointerTraits PointerTrait = PointerTraits::DEFAULT>
        [[nodiscard]] constexpr auto accessor_contiguous() const noexcept {
            return accessor<U, N, I, PointerTrait, StridesTraits::CONTIGUOUS>();
        }

        template<typename U = value_type, typename I = index_type,
                 PointerTraits PointerTrait = PointerTraits::DEFAULT>
        [[nodiscard]] constexpr auto accessor_contiguous_1d() const noexcept {
            return accessor_contiguous<U, 1, I, PointerTrait>();
        }

    public: // Deep copy
        /// Performs a deep copy of the view to \p output.
        /// \details Contiguous regions of memory have no copy restrictions and can be copied to any device. This is
        ///          also true for pitched layouts, colum or row vectors, or any layout that can be reordered and/or
        ///          reshaped to the aforementioned layouts. However, other non-contiguous memory layouts can only
        ///          be copied if the source and destination are both on the same GPU or on the CPU.
        /// \param[out] output  Destination. It should not overlap with this view.
        template<typename Output>
        requires (nt::is_varray_v<Output> and nt::are_almost_same_value_type_v<View, Output>)
        void to(const Output& output) const {
            noa::copy(*this, output);
        }

        /// Performs a deep copy of the view according \p option.
        /// \details The returned array is completely independent from the original one and is C-contiguous.
        ///          Contiguous regions of memory have no copy restrictions and can be copied to any device. This is
        ///          also true for pitched layouts, colum or row vectors, or any layout that can be reordered and/or
        ///          reshaped to the aforementioned layouts. However, other non-contiguous memory layouts can only
        ///          be copied if the source and destination are both on the same GPU or on the CPU.
        /// \param option   Output device and resource to perform the allocation of the new array.
        ///                 The current stream for that device is used to perform the copy.
        [[nodiscard]] Array<mutable_value_type> to(ArrayOption option) const {
            Array<mutable_value_type> out(shape(), option);
            to(out);
            return out;
        }

        /// Performs a deep copy of the array to the CPU.
        [[nodiscard]] Array<mutable_value_type> to_cpu() const {
            return to(ArrayOption{});
        }

        /// Performs a deep copy of the view preserving the view's options.
        [[nodiscard]] Array<mutable_value_type> copy() const {
            return to(options());
        }

    public: // Data reinterpretation
        /// Reinterprets the value type.
        /// \note This is only well defined in cases where reinterpret_cast<U*>(T*) is well defined, for instance,
        ///       when \p U is a unsigned char or std::byte to represent any data type as an array of bytes,
        ///       or to switch between complex and real floating-point numbers with the same precision.
        template<typename U>
        [[nodiscard]] View<U> as() const {
            const auto out = ni::ReinterpretLayout(shape(), strides(), get()).template as<U>();
            return View<U>(out.ptr, out.shape, out.strides, options());
        }

        /// Changes the device type (CPU<->GPU) on which the memory should be accessed.
        /// \details If the memory resource can be accessed by the CPU and/or a GPU, this function returns a new view
        ///          with the new device. This is used to control whether PINNED or MANAGED memory should be accessed
        ///          by the CPU or the GPU. MANAGED_GLOBAL memory is not attached to any particular GPU, so the
        ///          current GPU is used in that case.
        /// \param prefetch Whether to prefetch the memory to the target device. This only affects MANAGED(_GLOBAL)
        ///                 memory and should be used to anticipate access of that memory region by the target device,
        ///                 and/or to "move" the memory from the original to the target device. The prefetching is
        ///                 enqueued to the GPU stream, and as always, concurrent access from both CPU and GPU is illegal.
        [[nodiscard]] View as(DeviceType type, [[maybe_unused]] bool prefetch = false) const {
            const MemoryResource alloc = m_options.allocator.resource();
            if (device().is_gpu() and type == DeviceType::CPU) { // GPU -> CPU
                check(alloc == MemoryResource::PINNED or
                      alloc == MemoryResource::MANAGED or
                      alloc == MemoryResource::MANAGED_GLOBAL,
                      "GPU memory {} cannot be reinterpreted as a CPU memory-region. "
                      "This is only supported for pinned and managed memory-regions", alloc);
                #ifdef NOA_ENABLE_CUDA
                if (prefetch and (alloc == MemoryResource::MANAGED or alloc == MemoryResource::MANAGED_GLOBAL)) {
                    noa::cuda::AllocatorManaged<value_type>::prefetch_to_cpu(
                            get(), shape().elements(), Stream::current(device()).cuda());
                }
                #endif
                return View(get(), shape(), strides(), ArrayOption{.device=Device(type), .allocator=m_options.allocator});

            } else if (device().is_cpu() and type == DeviceType::GPU) { // CPU -> GPU
                check(Device::is_any(DeviceType::GPU), "No GPU detected");
                check(alloc == MemoryResource::PINNED ||
                      alloc == MemoryResource::MANAGED ||
                      alloc == MemoryResource::MANAGED_GLOBAL,
                      "CPU memory-region with the allocator {} cannot be reinterpreted as a GPU memory-region. "
                      "This is only supported for pinned and managed memory-regions", alloc);
                #ifdef NOA_ENABLE_CUDA
                Device gpu;
                if (alloc == MemoryResource::PINNED or alloc == MemoryResource::MANAGED) {
                    // NOTE: CUDA doesn't document what the attr.device is for managed memory.
                    //       Hopefully this is the device against which the allocation was performed
                    //       and not the current device.
                    // NOTE: With "stream-attached" managed memory, it is up to the user to know what
                    //       stream was used to perform the allocation.
                    const cudaPointerAttributes attr = noa::cuda::pointer_attributes(get());
                    gpu = Device(DeviceType::GPU, attr.device, Device::DeviceUnchecked{});
                    NOA_ASSERT((alloc == MemoryResource::PINNED and attr.type == cudaMemoryTypeHost) or
                               (alloc == MemoryResource::MANAGED and attr.type == cudaMemoryTypeManaged));

                } else if (alloc == MemoryResource::MANAGED_GLOBAL) {
                    // NOTE: This can be accessed from any stream and any GPU. It seems to be better to return the
                    //       current device and not the original device against which the allocation was performed.
                    gpu = Device::current(DeviceType::GPU);
                }
                if (prefetch && (alloc == MemoryResource::MANAGED or alloc == MemoryResource::MANAGED_GLOBAL)) {
                    noa::cuda::AllocatorManaged<value_type>::prefetch_to_gpu(
                            get(), shape().elements(), Stream::current(gpu).cuda());
                }
                return View(get(), shape(), strides(), ArrayOption{.device=gpu, .allocator=m_options.allocator});
                #else
                panic("No GPU backend detected");
                #endif
            } else {
                return *this;
            }
        }

        /// Reshapes the view (must have the same number of elements as the current view).
        [[nodiscard]] View reshape(shape_type new_shape) const {
            // Infer the size, if needed.
            check(ni::infer_size(new_shape, ssize()),
                  "The desired shape {} is not compatible with the current shape of the array {}, "
                  "or the size inference is invalid or ambiguous", new_shape, shape());

            // Then reshape.
            strides_type new_stride;
            check(ni::reshape(shape(), strides(), new_shape, new_stride),
                  "An array of shape {} and stride {} cannot be reshaped to an array of shape {}",
                  shape(), strides(), new_shape);

            return View(get(), new_shape, new_stride, options());
        }

        /// Reshapes the array in a vector along a particular axis.
        /// Returns a row vector by default (axis = 3).
        [[nodiscard]] View flat(i32 axis = 3) const {
            check(axis >= 0 and axis < 4);
            auto output_shape = shape_type::filled_with(1);
            output_shape[axis] = shape().elements();
            return reshape(output_shape);
        }

        /// Permutes the dimensions of the view.
        /// \param permutation  Permutation with the axes numbered from 0 to 3.
        [[nodiscard]] constexpr View permute(const Vec4<i64>& permutation) const {
            return View(get(),
                        ni::reorder(shape(), permutation),
                        ni::reorder(strides(), permutation),
                        options());
        }

        /// See permute().
        [[nodiscard]] constexpr View reorder(const Vec4<i64>& permutation) const {
            return permute(permutation);
        }

        [[nodiscard]] Array<mutable_value_type> permute_copy(const Vec4<i64>& permutation) const {
            return noa::permute_copy(*this, permutation);
        }

    public: // Assignment operators
        /// Clears the view. Equivalent to assigning *this with an empty view.
        View& operator=(std::nullptr_t) {
            *this = View{};
            return *this;
        }

    public:
        /// Element access. For efficient access, prefer to use Span or Accessor.
        [[nodiscard]] constexpr value_type& operator()(auto i0, auto i1, auto i2, auto i3) const noexcept {
            check(is_dereferenceable(), "Memory buffer cannot be accessed from the CPU");
            check(clamp_cast<i64>(i0) >= 0 and clamp_cast<i64>(i0) < shape()[0] and
                  clamp_cast<i64>(i1) >= 0 and clamp_cast<i64>(i1) < shape()[1] and
                  clamp_cast<i64>(i2) >= 0 and clamp_cast<i64>(i2) < shape()[2] and
                  clamp_cast<i64>(i3) >= 0 and clamp_cast<i64>(i3) < shape()[3],
                  "Indices are out of bound");
            return m_accessor(i0, i1, i2, i3);
        }

        /// Subregion indexing. Extracts a subregion from the current view.
        [[nodiscard]] constexpr View subregion(const ni::SubregionIndexer& indexer) const {
            return View(get() + indexer.offset, indexer.shape, indexer.strides, options());
        }

        /// Subregion indexing. Extracts a subregion from the current view.
        /// \see noa::indexing::Subregion for more details on the variadic parameters to enter.
        template<typename... Ts>
        [[nodiscard]] constexpr View subregion(Ts&&... indices) const {
            const auto indexer = ni::SubregionIndexer(shape(), strides())
                    .extract_subregion(std::forward<Ts>(indices)...);
            return subregion(indexer);
        }

    private:
        accessor_type m_accessor{};
        shape_type m_shape{};
        ArrayOption m_options{};
    };
}

namespace noa::traits {
    template<typename T> struct proclaim_is_view<View<T>> : std::true_type {};
}
#endif
