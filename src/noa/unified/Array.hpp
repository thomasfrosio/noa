#pragma once

#include "noa/core/Config.hpp"

#ifdef NOA_IS_OFFLINE
#include "noa/core/Exception.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/Types.hpp"
#include "noa/core/utils/ShareHandles.hpp"

#if defined(NOA_ENABLE_CUDA)
#include "noa/gpu/cuda/Pointers.hpp"
#endif

#include "noa/unified/ArrayOption.hpp"
#include "noa/unified/Stream.hpp"
#include "noa/unified/View.hpp"

namespace noa::inline types {
    /// 4-dimensional owning array.
    /// \details
    /// - \b Type: Arrays are usually managing "numeric" types, i.e. integers, (complex) floating-points.
    ///   However, other types are supported, such as, Vec or Matrices. Arrays of non-numeric types are limited to be
    ///   simple containers, and most free functions in the library will not be available for these types.
    ///   The managed type cannot be const-qualified, nor can it be a reference, a pointer or an extent.\n
    /// - \b Resource: Arrays manage a reference-counted resource, which can be shared, moved and copied.
    ///   The resource type and location depend on the ArrayOption used to create the array. Arrays are therefore
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
        using pointer_type = T*;
        using value_type = T;
        using mutable_value_type = std::remove_const_t<T>;
        using const_value_type = std::add_const_t<mutable_value_type>;
        using index_type = i64;
        using strides_type = Strides<index_type, 4>;
        using view_type = View<value_type>;
        using shape_type = Shape<index_type, 4>;
        using span_type = Span<value_type, 4, index_type>;
        using shared_type = std::shared_ptr<value_type[]>;

        static constexpr StridesTraits STRIDES_TRAIT = view_type::STRIDES_TRAIT;
        static constexpr PointerTraits POINTER_TRAIT = view_type::POINTER_TRAIT;
        static constexpr bool IS_CONTIGUOUS = view_type::IS_CONTIGUOUS;
        static constexpr bool IS_RESTRICT = view_type::IS_RESTRICT;
        static constexpr size_t SIZE = 4;
        static constexpr int64_t SSIZE = 4;

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
        constexpr explicit Array(i64 n_elements, ArrayOption option = {}) :
            m_shape{1, 1, 1, n_elements},
            m_strides{n_elements, n_elements, n_elements, 1},
            m_options{option} { allocate_(); }

        /// Creates a contiguous array.
        /// \param shape    BDHW shape of the array.
        /// \param option   Options of the created array.
        /// \see Allocator for more details.
        constexpr explicit Array(const shape_type& shape, ArrayOption option = {}) :
            m_shape(shape),
            m_strides(shape.strides()),
            m_options{option} { allocate_(); }

        /// Creates a contiguous row vector from an existing allocated memory region.
        /// \param[in,out] data Data to encapsulate.
        /// \param n_elements   Number of elements in \p data.
        /// \param option       Options of \p data.
        template<nt::smart_ptr_decay SharedPtr>
        constexpr Array(SharedPtr&& data, i64 n_elements, ArrayOption option = {}) :
            m_shape{1, 1, 1, n_elements},
            m_strides{n_elements, n_elements, n_elements, 1},
            m_shared(std::forward<SharedPtr>(data)),
            m_options{option}
        {
            validate_(get(), option);
        }

        /// Creates an array from an existing allocated memory region.
        /// \param[in,out] data Data to encapsulate.
        /// \param shape        BDHW shape of \p data.
        /// \param strides      BDHW strides of \p data.
        /// \param option       Options of \p data.
        template<nt::smart_ptr_decay SharedPtr>
        constexpr Array(
            SharedPtr&& data,
            const shape_type& shape,
            const strides_type& strides,
            ArrayOption option = {}
        ) :
            m_shape{shape},
            m_strides{strides},
            m_shared(std::forward<SharedPtr>(data)),
            m_options{option}
        {
            validate_(get(), option);
        }

        /// If an exception is thrown, make sure to synchronize the stream to guarantee that functions called
        /// with a View of that Array are done executing so that this Array can release its memory.
        ~Array() noexcept {
            // This should slow down the exception path due to the extra synchronization, but given
            // that exception are used as non-recoverable errors that will ultimately end up terminating
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
        [[nodiscard]] constexpr auto is_dereferenceable() const noexcept -> bool { return options().is_dereferenceable(); }
        [[nodiscard]] constexpr auto shape() const noexcept -> const shape_type& { return m_shape; }
        [[nodiscard]] constexpr auto strides() const noexcept -> const strides_type& { return m_strides; }
        [[nodiscard]] constexpr auto strides_full() const noexcept -> const strides_type& { return m_strides; }
        [[nodiscard]] constexpr auto n_elements() const noexcept -> index_type { return shape().n_elements(); }
        [[nodiscard]] constexpr auto ssize() const noexcept -> index_type { return shape().n_elements(); }
        [[nodiscard]] constexpr auto size() const noexcept -> size_t { return static_cast<size_t>(ssize()); }

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
        [[nodiscard]] constexpr bool is_empty() const noexcept { return not get() or shape().is_empty(); }

    public: // Accessors
        /// Synchronizes the current stream of the array's device.
        /// \details It guarantees safe access to the memory region. Note that stream-ordered access (i.e. passing
        ///          this to the library API) is safe and doesn't need synchronization.
        auto& eval() const {
            Stream::current(device()).synchronize();
            return *this;
        }

        /// Returns the pointer to the data.
        /// \warning Depending on the current stream of this array's device,
        ///          reading/writing to this pointer may be illegal or create a data race.
        [[nodiscard]] constexpr auto get() const noexcept -> pointer_type { return m_shared.get(); }
        [[nodiscard]] constexpr auto data() const noexcept -> pointer_type { return m_shared.get(); }
        [[nodiscard]] constexpr auto share() const& noexcept -> const shared_type& { return m_shared; }
        [[nodiscard]] constexpr auto share() && noexcept -> shared_type&& { return std::move(m_shared); }

        /// Returns a span of the array.
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

        template<typename U = value_type, typename I = index_type, StridesTraits STRIDES_TRAIT = STRIDES_TRAIT>
        [[nodiscard]] constexpr auto span_1d() const {
            return span<U, 1, I, STRIDES_TRAIT>();
        }

        template<typename U = value_type, typename I = index_type>
        [[nodiscard]] constexpr auto span_1d_contiguous() const {
            return span<U, 1, I, StridesTraits::CONTIGUOUS>();
        }

        /// Returns a (const-)view of the array.
        template<nt::almost_same_as<value_type> U = value_type>
        [[nodiscard]] constexpr auto view() const noexcept -> View<U> {
            return View<U>(get(), shape(), strides(), options());
        }

    public: // Deep copy
        /// Performs a deep copy of the array to \p output.
        /// \details Contiguous regions of memory have no copy restrictions and can be copied to any device. This is
        ///          also true for pitched layouts, colum or row vectors, or any layout that can be reordered and/or
        ///          reshaped to the aforementioned layouts. However, other non-contiguous memory layouts can only
        ///          be copied if the source and destination are both on the same GPU or on the CPU.
        /// \param[out] output  Destination. It should not overlap with this array.
        template<nt::varray_decay_of_almost_same_type<Array> Output>
        void to(Output&& output) const& {
            noa::copy(*this, std::forward<Output>(output));
        }
        template<nt::varray_decay_of_almost_same_type<Array> Output>
        void to(Output&& output) && {
            noa::copy(std::move(*this), std::forward<Output>(output));
        }

        /// Performs a deep copy of the array according \p option.
        /// \details The returned array is completely independent from the original one and is C-contiguous.
        ///          Contiguous regions of memory have no copy restrictions and can be copied to any device. This is
        ///          also true for pitched layouts, colum or row vectors, or any layout that can be reordered and/or
        ///          reshaped to the aforementioned layouts. However, other non-contiguous memory layouts can only
        ///          be copied if the source and destination are both on the same GPU or on the CPU.
        /// \param option   Output device and resource to perform the allocation of the new array.
        ///                 The current stream for that device is used to perform the copy.
        [[nodiscard]] auto to(ArrayOption option) const& -> Array {
            Array out(shape(), option);
            to(out);
            return out;
        }
        [[nodiscard]] auto to(ArrayOption option) && -> Array {
            Array out(shape(), option);
            std::move(*this).to(out);
            return out;
        }

        /// Performs a deep copy of the array to the CPU.
        [[nodiscard]] auto to_cpu() const& -> Array {
            return to(ArrayOption{});
        }
        [[nodiscard]] auto to_cpu() && -> Array {
            return std::move(*this).to(ArrayOption{});
        }

        /// Performs a deep copy of the array preserving the array's options.
        [[nodiscard]] auto copy() const& -> Array {
            return to(options());
        }
        [[nodiscard]] auto copy() && -> Array {
            return std::move(*this).to(options());
        }

        /// Returns a copy of the first value in the array.
        /// Note that the stream of the array's device is synchronized when this functions returs.
        [[nodiscard]] auto first() const -> value_type {
            return view().first();
        }

    public: // Data reinterpretation
        /// Reinterprets the value type.
        /// \note This is only well-defined in cases where reinterpret_cast<U*>(T*) is well-defined, for instance,
        ///       when \p U is an unsigned char or std::byte to represent any data type as an array of bytes,
        ///       or to switch between complex and real floating-point numbers with the same precision.
        template<typename U>
        [[nodiscard]] auto as() const& -> Array<U> {
            const auto out = ni::ReinterpretLayout(shape(), strides(), get()).template as<U>();
            return Array<U>(std::shared_ptr<U[]>(m_shared, out.ptr), out.shape, out.strides, options());
        }
        template<typename U>
        [[nodiscard]] auto as() && -> Array<U> {
            const auto out = ni::ReinterpretLayout(shape(), strides(), get()).template as<U>();
            return Array<U>(std::shared_ptr<U[]>(std::move(m_shared), out.ptr), out.shape, out.strides, options());
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
        [[nodiscard]] auto as(Device::Type type, bool prefetch = false) const& -> Array {
            const auto new_device = view().as(type, prefetch).device();
            return Array(m_shared, shape(), strides(), options().set_device(new_device));
        }
        [[nodiscard]] auto as(Device::Type type, bool prefetch = false) && -> Array {
            const auto new_device = view().as(type, prefetch).device();
            return Array(std::move(m_shared), shape(), strides(), options().set_device(new_device));
        }

        /// Reshapes the array.
        /// \details This function performs a "safe" reshape by making sure the new shape contains the same number
        ///          of elements. The new shape can have one dimension to -1, meaning the size of this dimension
        ///          should be inferred from the other dimensions and the number of elements.
        ///          If one wants to assign an array to an arbitrary new shape and new strides, one can use the
        ///          alias Array constructor instead.
        /// \return An alias of the array with the new shape and strides.
        [[nodiscard]] constexpr auto reshape(const shape_type& new_shape) const& -> Array {
            auto new_span = span_type(nullptr, shape(), strides()).reshape(new_shape);
            return Array(m_shared, new_span.shape(), new_span.strides(), options());
        }
        [[nodiscard]] constexpr auto reshape(const shape_type& new_shape) && -> Array {
            auto new_span = span_type(nullptr, shape(), strides()).reshape(new_shape);
            return Array(std::move(m_shared), new_span.shape(), new_span.strides(), options());
        }

        /// Reshapes the array in a vector along a particular axis.
        /// Returns a row vector by default (axis = 3).
        [[nodiscard]] constexpr auto flat(i32 axis = 3) const& -> Array {
            check(axis >= 0 and axis < 4);
            auto output_shape = shape_type::filled_with(1);
            output_shape[axis] = shape().n_elements();
            return reshape(output_shape);
        }
        [[nodiscard]] constexpr auto flat(i32 axis = 3) && -> Array {
            check(axis >= 0 and axis < 4);
            auto output_shape = shape_type::filled_with(1);
            output_shape[axis] = shape().n_elements();
            return std::move(*this).reshape(output_shape);
        }

        /// Permutes the dimensions of the view.
        /// \param permutation  Permutation with the axes numbered from 0 to 3.
        [[nodiscard]] constexpr auto permute(const Vec4<i64>& permutation) const& -> Array {
            return Array(m_shared,
                         ni::reorder(shape(), permutation),
                         ni::reorder(strides(), permutation),
                         options());
        }
        [[nodiscard]] constexpr auto permute(const Vec4<i64>& permutation) && -> Array {
            return Array(std::move(m_shared),
                         ni::reorder(shape(), permutation),
                         ni::reorder(strides(), permutation),
                         options());
        }

        /// Permutes the array by performing a deep-copy. The returned Array is a new C-contiguous array.
        /// \param permutation  Permutation with the axes numbered from 0 to 3.
        [[nodiscard]] Array permute_copy(const Vec4<i64>& permutation) const& {
            return noa::permute_copy(*this, permutation);
        }
        [[nodiscard]] Array permute_copy(const Vec4<i64>& permutation) && {
            return noa::permute_copy(std::move(*this), permutation);
        }

    public:
        /// Clears the array. Equivalent to assigning *this with an empty array.
        Array& operator=(std::nullptr_t) {
            *this = Array{};
            return *this;
        }

        /// Releases the array. *this is left empty.
        /// \note This effectively does a move with a guaranteed reset of the array.
        Array release() noexcept {
            return std::exchange(*this, Array{});
        }

    public:
        /// Element access (unsafe if not synchronized). For efficient access, prefer to use Span.
        template<typename... U> requires nt::iwise_general_indexing<SIZE, U...>
        [[nodiscard]] constexpr auto at(const U&... indices) const -> value_type& {
            check(is_dereferenceable(), "Memory buffer cannot be accessed from the CPU");
            return span().at(indices...);
        }

        /// Element access (unsafe). For efficient access, prefer to use Span or Accessor.
        template<typename... U> requires nt::iwise_general_indexing<SIZE, U...>
        [[nodiscard]] constexpr auto operator()(const U&... indices) const -> value_type& {
            return span()(indices...);
        }

        /// Subregion indexing. Extracts a subregion from the current array.
        template<typename... U>
        [[nodiscard]] constexpr auto subregion(const ni::Subregion<4, U...>& subregion) const& -> Array {
            auto [new_shape, new_strides, offset] = subregion.extract_from(shape(), strides());
            return Array(shared_type(m_shared, get() + offset), new_shape, new_strides, options());
        }
        template<typename... U>
        [[nodiscard]] constexpr auto subregion(const ni::Subregion<4, U...>& subregion) && -> Array {
            auto [new_shape, new_strides, offset] = subregion.extract_from(shape(), strides());
            return Array(shared_type(std::move(m_shared), get() + offset), new_shape, new_strides, options());
        }

        /// Subregion indexing. Extracts a subregion from the current array.
        /// \see noa::indexing::Subregion for more details on the variadic parameters to enter.
        template<typename... U> requires nt::subregion_indexing<4, U...>
        [[nodiscard]] constexpr auto subregion(const U&... indices) const& -> Array {
            return subregion(ni::Subregion<4, U...>(indices...));
        }
        template<typename... U> requires nt::subregion_indexing<4, U...>
        [[nodiscard]] constexpr auto subregion(const U&... indices) && -> Array {
            return std::move(*this).subregion(ni::Subregion<4, U...>(indices...));
        }

    private:
        void allocate_() {
            if (allocator() == Allocator::PITCHED) {
                noa::tie(m_shared, m_strides) = allocator().template allocate_pitched<value_type>(shape(), device());
            } else {
                m_shared = allocator().template allocate<value_type>(n_elements(), device());
            }
        }

        static void validate_(const void* ptr, ArrayOption option) {
            check(option.allocator != Allocator::CUDA_ARRAY,
                  "CUDA arrays are not supported by the Array class. Use a Texture instead");
            check(option.allocator != Allocator::NONE or ptr == nullptr,
                  "{} is for nullptr only", Allocator::NONE);

            if (option.device.is_cpu()) {
                if (not Device::is_any_gpu())
                    return; // Everything is allocated using AllocatorHeap
                #ifdef NOA_ENABLE_CUDA
                const cudaPointerAttributes attr = noa::cuda::pointer_attributes(ptr);
                switch (attr.type) {
                    case cudaMemoryTypeUnregistered:
                        if (not option.allocator.is_any(
                            Allocator::DEFAULT, Allocator::DEFAULT_ASYNC, Allocator::PITCHED)) {
                            panic("Attempting to create a CPU array with {} from a CPU-only "
                                  "(CUDA unregistered) memory region", option.allocator);
                        }
                        break;
                    case cudaMemoryTypeHost:
                        if (option.allocator != Allocator::PINNED)
                            panic("Attempting to create a CPU array with {} from a pinned memory region",
                                  option.allocator);
                        break;
                    case cudaMemoryTypeDevice:
                        panic("Attempting to create an CPU array that points to a GPU-only memory region");
                    case cudaMemoryTypeManaged:
                        if (not option.allocator.is_any(
                            Allocator::DEFAULT, Allocator::DEFAULT_ASYNC, Allocator::PITCHED,
                            Allocator::MANAGED, Allocator::MANAGED_GLOBAL))
                            panic("Attempting to create an CPU array with {} from a (CUDA) managed pointer",
                                  option.allocator);
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
                        if (option.allocator != Allocator::PINNED)
                            panic("Attempting to create a GPU array with {} from a pinned memory region",
                                  option.allocator);
                        break;
                    case cudaMemoryTypeDevice:
                        if (attr.device != option.device.id())
                            panic("Attempting to create a GPU array with a device ID of {} from a memory region "
                                  "located on another device (ID={})", option.device.id(), attr.device);
                        break;
                    case cudaMemoryTypeManaged:
                        if (not option.allocator.is_any(
                            Allocator::DEFAULT, Allocator::DEFAULT_ASYNC, Allocator::PITCHED,
                            Allocator::MANAGED, Allocator::MANAGED_GLOBAL))
                            panic("Attempting to create a GPU array with {} from a (CUDA) managed pointer",
                                  option.allocator);
                        break;
                }
                #endif
            }
        }

    private:
        shape_type m_shape{};
        strides_type m_strides{};
        shared_type m_shared{};
        ArrayOption m_options{};
    };
}

namespace noa::traits {
    template<typename T> struct proclaim_is_array<Array<T>> : std::true_type {};
}
#endif
