#pragma once

#include "noa/core/Config.hpp"

#if defined(NOA_IS_OFFLINE)
#include "noa/core/Exception.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/Types.hpp"
#include "noa/core/Indexing.hpp"

#include "noa/cpu/AllocatorHeap.hpp"
#if defined(NOA_ENABLE_CUDA)
#include "noa/gpu/cuda/AllocatorDevice.hpp"
#include "noa/gpu/cuda/AllocatorDevicePadded.hpp"
#include "noa/gpu/cuda/AllocatorManaged.hpp"
#include "noa/gpu/cuda/AllocatorPinned.hpp"
#include "noa/gpu/cuda/Pointers.hpp"
#endif

#include "noa/unified/ArrayOption.hpp"
#include "noa/unified/Stream.hpp"
#include "noa/unified/View.hpp"

namespace noa::inline types {
    /// 4-dimensional owning array.
    /// \details
    /// - \b Type: Arrays are usually managing "numeric" types, i.e. integers, (complex) floating-points.
    ///   However, other types are supported, such as, Vec or Matrices. Array of non-numeric types are limited to be
    ///   simple containers and most free functions in the library will not be available for these types.
    ///   The managed type cannot be const-qualified, nor can it be a reference, a pointer or an extent.\n
    /// - \b Resource: Arrays manage a reference-counted resource, which can be shared, moved and copied.
    ///   The resource type and location depends on the ArrayOption used to create the array. Arrays are therefore
    ///   attached to a device, either the CPU or a GPU. Depending on the resource (thus the allocator used to create
    ///   the array), arrays can be interpreted as CPU or GPU arrays and it is possible to create aliases of the same
    ///   resource for the CPU and the GPU (see as() for more details).\n
    /// - \b Accessors: Functions in the library taking or returning an Array enqueue backend calls to the current
    ///   stream of the output Array's device. As such, one does not and should not (for performance reasons)
    ///   synchronize the stream between each operation involving an Array, except if:\n
    ///     - The input and output Array(s) are used in a "unsafe" way between the function call and a synchronization
    ///       point. These unsafe ways of accessing the managed data of an Array are by get()/data()/operator(...),
    ///       share(). While these are often required, e.g. for efficient loop-like indexing, one must make sure the
    ///       current stream of the Array's device is synchronized.\n
    ///     - The input and output Array(s) are on the same device but used by different streams. The library will
    ///       always use the current stream of the device. If an array is used by a non-current stream (e.g. the current
    ///       stream was changed), one must make sure that stream-ordering is respected by synchronizing this stream
    ///       before calling the function. Note that if the arrays are on different devices, the implementation will
    ///       make sure that stream-ordering is respected.\n
    /// - \b Shape: Shape and strides are in number of elements and specified in the BDHW order (from left to right).
    ///   While column-major ordering is supported, row-major ordering is recommended. Empty dimensions have a size
    ///   of 1. If one dimension is 0, the entire array is considered empty. Arrays can be broadcasted to other shapes
    ///   according to the broadcasting rule (see noa::indexing::broadcast()). As such, some arrays can have dimensions
    ///   with a stride of 0. Negative strides are not supported.\n
    /// - \b Deferred-deletion: When passing Arrays to a function, the stream keeps a reference of the input/output
    ///   memory-regions. These references are only destructed when the stream is done executing the function. If the
    ///   stream executes asynchronously, functions may return before completion, thus the input/output Arrays may end
    ///   up being destructed before completion. Thanks to the references kept by the stream, destructing the Arrays
    ///   only defers the destruction of the managed memory-regions until the stream is done using them. This allows
    ///   for simple and safe asynchronous execution, e.g. using a GPU or concurrent CPU thread.\n
    template<typename T>
    class Array {
    public: // typedefs
        using shape_type = Shape<i64, 4>;
        using accessor_type = Accessor<T, 4, i64>;
        using accessor_reference_type = AccessorReference<T, 4, i64>;
        using pointer_type = typename accessor_type::pointer_type;
        using value_type = typename accessor_type::value_type;
        using mutable_value_type = value_type;
        using index_type = typename accessor_type::index_type;
        using strides_type = typename accessor_type::strides_type;
        using shared_type = std::shared_ptr<value_type[]>;

        static_assert(not std::is_const_v<value_type>);
        static_assert(not std::is_pointer_v<value_type>);
        static_assert(not std::is_reference_v<value_type>);

    public: // Constructors
        /// Creates an empty array.
        constexpr Array() = default;

        /// Creates a contiguous row vector.
        /// \param n_elements   Number of elements.
        /// \param option       Options of the created array.
        /// \see Allocator for more details.
        constexpr explicit Array(i64 n_elements, ArrayOption option = {})
                : m_shape{1, 1, 1, n_elements},
                  m_strides{n_elements, n_elements, n_elements, 1},
                  m_options(option) { allocate_(); }

        /// Creates a contiguous array.
        /// \param shape    BDHW shape of the array.
        /// \param option   Options of the created array.
        /// \see Allocator for more details.
        constexpr explicit Array(const shape_type& shape, ArrayOption option = {})
                : m_shape(shape), m_strides(shape.strides()), m_options(option) { allocate_(); }

        /// Creates a contiguous row vector from an existing allocated memory region.
        /// \param[in,out] data Data to encapsulate.
        /// \param n_elements   Number of elements in \p data.
        /// \param option       Options of \p data.
        constexpr Array(const shared_type& data, i64 n_elements, ArrayOption option = {})
                : m_shape{1, 1, 1, n_elements},
                  m_strides{n_elements, n_elements, n_elements, 1},
                  m_shared(data),
                  m_options(option) {
            validate_(m_shared.get(), option);
        }

        /// Creates an array from an existing allocated memory region.
        /// \param[in,out] data Data to encapsulate.
        /// \param shape        BDHW shape of \p data.
        /// \param strides      BDHW strides of \p data.
        /// \param option       Options of \p data.
        constexpr Array(const shared_type& data, const shape_type& shape,
                        const strides_type& strides, ArrayOption option = {})
                : m_shape(shape), m_strides(strides), m_shared(data), m_options(option) {
            validate_(m_shared.get(), option);
        }

        /// If an exception is thrown, make sure to synchronize the stream to guarantee that functions called
        /// with a View of that Array are done executing so that this Array can release its memory.
        ~Array() noexcept {
            // This should slow down the exception path due to the extra synchronization, but given
            // that exception are used as non-recoverable errors that will ultimately end up terminating
            // the program, we don't really care about performance in the exception path.

            // While we could record the number of alive exceptions at construction time, to correctly
            // detect the case where an Array is created during stack unwinding, it doesn't seem worth it:
            // 1) we would need to keep track of a "count".
            // 2) creating an Array during stack unwinding is weird/rare.
            // 3) this will just end up saving a few stream synchronizations...

            try {
                // Freeing memory in CUDA GPUs will either synchronize the device, synchronize the stream or
                // enqueue an asynchronous free, all of which guarantees that the GPU is done using the memory
                // region of the Array. As such, only synchronize CPU arrays.
                if (std::uncaught_exceptions() > 0 && device().is_cpu())
                    eval();
            } catch (...) {} // ignore new exception thrown by eval()
        }

    public: // Queries
        /// Returns the options used to create the array.
        [[nodiscard]] constexpr ArrayOption options() const noexcept { return m_options; }

        /// Returns the device used to create the array.
        [[nodiscard]] constexpr Device device() const noexcept { return options().device; }

        /// Returns the memory resource used to create the array.
        [[nodiscard]] constexpr Allocator allocator() const noexcept { return options().allocator; }

        /// Whether the managed data can be accessed by CPU threads.
        [[nodiscard]] constexpr bool is_dereferenceable() const noexcept { return options().is_dereferenceable(); }

        /// Returns the BDHW shape of the array.
        [[nodiscard]] constexpr const shape_type& shape() const noexcept { return m_shape; }

        /// Returns the BDHW strides of the array.
        [[nodiscard]] constexpr const strides_type& strides() const noexcept { return m_strides; }

        /// Returns the number of elements in the array.
        [[nodiscard]] constexpr index_type elements() const noexcept { return shape().elements(); }
        [[nodiscard]] constexpr index_type ssize() const noexcept { return shape().elements(); }
        [[nodiscard]] constexpr size_t size() const noexcept { return static_cast<size_t>(ssize()); }

        /// Whether the dimensions of the array are C or F contiguous.
        template<char ORDER = 'C'>
        [[nodiscard]] constexpr bool are_contiguous() const noexcept {
            return ni::are_contiguous<ORDER>(strides(), shape());
        }

        template<char ORDER = 'C'>
        [[nodiscard]] constexpr auto is_contiguous() const noexcept {
            return ni::is_contiguous<ORDER>(strides(), shape());
        }

        /// Whether the array is empty. An array is empty if not initialized or if one of its dimension is 0.
        [[nodiscard]] constexpr bool is_empty() const noexcept { return not get() or any(shape() == 0); }

    public: // Accessors
        /// Synchronizes the current stream of the Array's device.
        /// \details It guarantees safe access to the memory region using get(), data(), operator(...), and accessor().
        ///          Note that stream-ordered access (i.e. passing this to the library API) is safe and doesn't need
        ///          synchronization.
        const Array& eval() const {
            Stream::current(device()).synchronize();
            return *this;
        }

        /// Returns the pointer to the data.
        /// \warning Depending on the current stream of this array's device,
        ///          reading/writing to this pointer may be illegal or create a data race.
        [[nodiscard]] constexpr pointer_type get() const noexcept { return m_shared.get(); }

        /// Returns the pointer to the data.
        /// \warning Depending on the current stream of this array's device,
        ///          reading/writing to this pointer may be illegal or create a data race.
        [[nodiscard]] constexpr pointer_type data() const noexcept { return m_shared.get(); }

        /// Returns a reference of the managed resource.
        /// \warning Depending on the current stream of this array's device,
        ///          reading/writing to this pointer may be illegal or create a data race.
        [[nodiscard]] constexpr const shared_type& share() const noexcept { return m_shared; }

        /// Returns a (const-)span of the array.
        template<typename U = value_type, i64 SIZE = -1, typename I = index_type>
        requires (nt::is_almost_same_v<U, value_type> and std::is_integral_v<I>)
        [[nodiscard]] constexpr Span<U, SIZE, I> span() const noexcept {
            return view().template span<U, SIZE, I>();
        }

        /// Returns a new Accessor and its corresponding shape.
        /// \details While constructing the accessor, this function can also reinterpret the current value type.
        ///          This is only well defined in cases where View::as<U>() is well defined.
        ///          If N < 4, the outer-dimensions are stacked together.
        template<typename U, size_t N = 4, typename I = index_type,
                 PointerTraits PointerTrait = PointerTraits::DEFAULT,
                 StridesTraits StridesTrait = StridesTraits::STRIDED>
        [[nodiscard]] constexpr auto accessor_and_shape() const noexcept {
            return view().template accessor_and_shape<U, N, I, PointerTrait, StridesTrait>();
        }

        /// Returns a (const-)view of the array.
        template<typename U = value_type> requires nt::is_almost_same_v<U, value_type>
        [[nodiscard]] constexpr View<U> view() const noexcept {
            return View<U>(get(), shape(), strides(), options());
        }

        /// Releases the array. *this is left empty.
        /// \note Moving an array using std::move() is effectively equivalent.
        Array release() noexcept {
            return std::exchange(*this, Array{});
        }

        [[nodiscard]] constexpr accessor_type accessor() const noexcept {
            return accessor_type(get(), strides());
        }

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
        /// Performs a deep copy of the array to \p output.
        /// \details Contiguous regions of memory have no copy restrictions and can be copied to any device. This is
        ///          also true for pitched layouts, colum or row vectors, or any layout that can be reordered and/or
        ///          reshaped to the aforementioned layouts. However, other non-contiguous memory layouts can only
        ///          be copied if the source and destination are both on the same GPU or on the CPU.
        /// \param[out] output  Destination. It should not overlap with this array.
        template<typename Output>
        requires (nt::is_varray_v<Output> and nt::are_almost_same_value_type_v<Array, Output>)
        void to(const Output& output) const {
            noa::copy(*this, output);
        }

        /// Performs a deep copy of the array according \p option.
        /// \details The returned array is completely independent from the original one and is C-contiguous.
        ///          Contiguous regions of memory have no copy restrictions and can be copied to any device. This is
        ///          also true for pitched layouts, colum or row vectors, or any layout that can be reordered and/or
        ///          reshaped to the aforementioned layouts. However, other non-contiguous memory layouts can only
        ///          be copied if the source and destination are both on the same GPU or on the CPU.
        /// \param option   Output device and resource to perform the allocation of the new array.
        ///                 The current stream for that device is used to perform the copy.
        [[nodiscard]] Array to(ArrayOption option) const {
            Array out(shape(), option);
            to(out);
            return out;
        }

        /// Performs a deep copy of the array to the CPU.
        [[nodiscard]] Array to_cpu() const {
            return to(ArrayOption{});
        }

        /// Performs a deep copy of the array preserving the array's options.
        [[nodiscard]] Array copy() const {
            return to(options());
        }

    public: // Data reinterpretation
        /// Reinterprets the value type.
        /// \note This is only well defined in cases where reinterpret_cast<U*>(T*) is well defined, for instance,
        ///       when \p U is a unsigned char or std::byte to represent any data type as an array of bytes,
        ///       or to switch between complex and real floating-point numbers with the same precision.
        template<typename U>
        [[nodiscard]] Array<U> as() const {
            const auto out = ni::ReinterpretLayout(shape(), strides(), get()).template as<U>();
            return Array<U>(std::shared_ptr<U[]>(m_shared, out.ptr), out.shape, out.strides, options());
        }

        /// Changes the device type (CPU<->GPU) on which the memory should be accessed.
        /// \details If the memory resource can be accessed by the CPU and/or a GPU, this function returns an array
        ///          with the new. This is used to control whether PINNED or MANAGED memory should be accessed by
        ///          the CPU or the GPU. MANAGED_GLOBAL memory is not attached to any particular GPU, so the current
        ///         GPU is used in that case.
        /// \param prefetch Whether to prefetch the memory to the target device. This only affects MANAGED(_GLOBAL)
        ///                 memory and should be used to anticipate access of that memory region by the target device,
        ///                 and/or to "move" the memory from the original to the target device. The prefetching is
        ///                 enqueued to the GPU stream, and as always, concurrent access from both CPU and GPU is illegal.
        [[nodiscard]] Array as(DeviceType type, bool prefetch = false) const {
            return Array(share(), shape(), strides(), options().set_device(view().as(type, prefetch).device()));
        }

        /// Reshapes the array.
        /// \details This function performs a "safe" reshape by making sure the new shape contains the same number
        ///          of elements. The new shape can have one dimension to -1, meaning the size of this dimension
        ///          should be inferred from the other dimensions and the number of elements.
        ///          If one wants to assign an array to an arbitrary new shape and new strides, one can use the
        ///          alias Array constructor instead.
        /// \return An alias of the array with the new shape and strides.
        [[nodiscard]] Array reshape(const shape_type& new_shape) const {
            auto v = View<value_type>(nullptr, shape(), strides()).reshape(new_shape);
            return Array(share(), v.shape(), v.strides(), options());
        }

        /// Reshapes the array in a vector along a particular axis.
        /// Returns a row vector by default (axis = 3).
        [[nodiscard]] Array flat(i32 axis = 3) const {
            check(axis >= 0 and axis < 4);
            auto output_shape = shape_type::filled_with(1);
            output_shape[axis] = shape().elements();
            return reshape(output_shape);
        }

        /// Permutes the dimensions of the view.
        /// \param permutation  Permutation with the axes numbered from 0 to 3.
        [[nodiscard]] constexpr Array permute(const Vec4<i64>& permutation) const {
            return Array(share(),
                         ni::reorder(shape(), permutation),
                         ni::reorder(strides(), permutation),
                         options());
        }

        /// Permutes the array by performing a deep-copy. The returned Array is a new C-contiguous array.
        /// \param permutation  Permutation with the axes numbered from 0 to 3.
        [[nodiscard]] Array permute_copy(const Vec4<i64>& permutation) const {
            return noa::permute_copy(*this, permutation);
        }

    public: // Assignment operators
        /// Clears the array. Equivalent to assigning *this with an empty array.
        Array& operator=(std::nullptr_t) {
            *this = Array{};
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
            return accessor_reference_type(get(), strides().data())(i0, i1, i2, i3);
        }

        /// Subregion indexing. Extracts a subregion from the current array.
        [[nodiscard]] constexpr Array subregion(const ni::SubregionIndexer& indexer) const {
            return Array(shared_type(share(), get() + indexer.offset), indexer.shape, indexer.strides, options());
        }

        /// Subregion indexing. Extracts a subregion from the current array.
        /// \see noa::indexing::Subregion for more details on the variadic parameters to enter.
        template<typename... Ts>
        [[nodiscard]] constexpr Array subregion(Ts&&... indices) const {
            const auto indexer = ni::SubregionIndexer(shape(), strides())
                    .extract_subregion(std::forward<Ts>(indices)...);
            return subregion(indexer);
        }

    private:
        void allocate_() {
            const auto n_elements = m_shape.elements();
            if (!n_elements) {
                m_shape = 0;
                m_shared = nullptr;
                return;
            }

            const Device device = m_options.device;
            switch (m_options.allocator.resource()) {
                case MemoryResource::NONE:
                    break;
                case MemoryResource::DEFAULT:
                    if (device.is_cpu()) {
                        m_shared = noa::cpu::AllocatorHeap<value_type>::allocate(n_elements);
                    } else {
                        #ifdef NOA_ENABLE_CUDA
                        const DeviceGuard guard(device);
                        m_shared = noa::cuda::AllocatorDevice<value_type>::allocate(n_elements);
                        #endif
                    }
                    break;
                case MemoryResource::DEFAULT_ASYNC:
                    if (device.is_cpu()) {
                        m_shared = noa::cpu::AllocatorHeap<value_type>::allocate(n_elements);
                    } else {
                        #ifdef NOA_ENABLE_CUDA
                        m_shared = noa::cuda::AllocatorDevice<value_type>::allocate_async(
                                n_elements, Stream::current(device).cuda());
                        #endif
                    }
                    break;
                case MemoryResource::PITCHED:
                    if (device.is_cpu()) {
                        m_shared = noa::cpu::AllocatorHeap<value_type>::allocate(n_elements);
                    } else {
                        #ifdef NOA_ENABLE_CUDA
                        const DeviceGuard guard(device);
                        // AllocatorDevicePadded requires sizeof(T) <= 16 bytes.
                        if constexpr (nt::is_numeric_v<value_type>) {
                            auto [ptr, strides] = noa::cuda::AllocatorDevicePadded<value_type>::allocate(m_shape);
                            m_shared = std::move(ptr);
                            m_strides = strides;
                        } else {
                            m_shared = noa::cuda::AllocatorDevice<value_type>::allocate(n_elements);
                        }
                        #endif
                    }
                    break;
                case MemoryResource::PINNED: {
                    if (device.is_cpu() and not Device::is_any(DeviceType::GPU)) {
                        m_shared = noa::cpu::AllocatorHeap<value_type>::allocate(n_elements);
                    } else {
                        #ifdef NOA_ENABLE_CUDA
                        const DeviceGuard guard(device.is_gpu() ? device : Device::current(DeviceType::GPU));
                        m_shared = noa::cuda::AllocatorPinned<value_type>::allocate(n_elements);
                        #endif
                    }
                    break;
                }
                case MemoryResource::MANAGED: {
                    if (device.is_cpu() and not Device::is_any(DeviceType::GPU)) {
                        m_shared = noa::cpu::AllocatorHeap<value_type>::allocate(n_elements);
                    } else {
                        #ifdef NOA_ENABLE_CUDA
                        const Device gpu = device.is_gpu() ? device : Device::current(DeviceType::GPU);
                        const DeviceGuard guard(gpu); // could be helpful when retrieving device
                        auto& cuda_stream = Stream::current(gpu).cuda();
                        m_shared = noa::cuda::AllocatorManaged<value_type>::allocate(n_elements, cuda_stream);
                        #endif
                    }
                    break;
                }
                case MemoryResource::MANAGED_GLOBAL: {
                    if (device.is_cpu() and not Device::is_any(DeviceType::GPU)) {
                        m_shared = noa::cpu::AllocatorHeap<value_type>::allocate(n_elements);
                    } else {
                        #ifdef NOA_ENABLE_CUDA
                        const DeviceGuard guard(device.is_gpu() ? device : Device::current(DeviceType::GPU));
                        m_shared = noa::cuda::AllocatorManaged<value_type>::allocate_global(n_elements);
                        #endif
                    }
                    break;
                }
                case MemoryResource::CUDA_ARRAY:
                    panic("CUDA arrays are not supported by the Array's allocator. Use Texture instead");
                default:
                    panic("Allocator {} is not supported by the Array's allocator", m_options.allocator);
            }
        }

        static void validate_(void* ptr, ArrayOption option) {
            const MemoryResource memory_resource = option.allocator.resource();
            check(memory_resource != MemoryResource::CUDA_ARRAY,
                  "CUDA arrays are not supported by the Array class. Use a Texture instead");
            check(memory_resource != MemoryResource::NONE or ptr == nullptr,
                  "{} is for nullptr only", MemoryResource::NONE);

            if (option.device.is_cpu()) {
                if (not Device::is_any(DeviceType::GPU))
                    return; // Everything is allocated using AllocatorHeap
                #ifdef NOA_ENABLE_CUDA
                const cudaPointerAttributes attr = noa::cuda::pointer_attributes(ptr);
                switch (attr.type) {
                    case cudaMemoryTypeUnregistered:
                        if (memory_resource != MemoryResource::DEFAULT and
                            memory_resource != MemoryResource::DEFAULT_ASYNC and
                            memory_resource != MemoryResource::PITCHED) {
                            panic("Attempting to create a CPU array with {} from a CPU-only "
                                  "(CUDA unregistered) memory region", memory_resource);
                        }
                        break;
                    case cudaMemoryTypeHost:
                        if (memory_resource != MemoryResource::PINNED)
                            panic("Attempting to create a CPU array with {} from a pinned memory region",
                                  memory_resource);
                        break;
                    case cudaMemoryTypeDevice:
                        panic("Attempting to create an CPU array that points to a GPU-only memory region");
                    case cudaMemoryTypeManaged:
                        if (memory_resource != MemoryResource::DEFAULT and
                            memory_resource != MemoryResource::DEFAULT_ASYNC and
                            memory_resource != MemoryResource::PITCHED and
                            memory_resource != MemoryResource::MANAGED and
                            memory_resource != MemoryResource::MANAGED_GLOBAL)
                            panic("Attempting to create an CPU array with {} from a (CUDA) managed pointer",
                                  memory_resource);
                        break;
                }
                #endif

            } else if (option.device.is_gpu()) {
                #ifdef NOA_ENABLE_CUDA
                const cudaPointerAttributes attr = noa::cuda::pointer_attributes(ptr);
                switch (attr.type) {
                    case cudaMemoryTypeUnregistered:
                        panic("Attempting to create GPU array from a CPU-only (CUDA unregistered) memory region");
                    case cudaMemoryTypeHost:
                        if (memory_resource != MemoryResource::PINNED)
                            panic("Attempting to create a GPU array with {} from a pinned memory region",
                                  memory_resource);
                        break;
                    case cudaMemoryTypeDevice:
                        if (attr.device != option.device.id())
                            panic("Attempting to create a GPU array with a device ID of {} from a memory region "
                                  "located on another device (ID={})", option.device.id(), attr.device);
                        break;
                    case cudaMemoryTypeManaged:
                        if (memory_resource != MemoryResource::DEFAULT and
                            memory_resource != MemoryResource::DEFAULT_ASYNC and
                            memory_resource != MemoryResource::PITCHED and
                            memory_resource != MemoryResource::MANAGED and
                            memory_resource != MemoryResource::MANAGED_GLOBAL)
                            panic("Attempting to create a GPU array with {} from a (CUDA) managed pointer",
                                  memory_resource);
                        break;
                }
                #endif
            }
        }

    private:
        shape_type m_shape{};
        strides_type m_strides{};
        shared_type m_shared;
        ArrayOption m_options;
    };
}

namespace noa::traits {
    template<typename T> struct proclaim_is_array<Array<T>> : std::true_type {};
}
#endif
