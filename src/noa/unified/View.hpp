#pragma once

#include "noa/core/Exception.hpp"
#include "noa/core/Types.hpp"
#include "noa/core/utils/Indexing.hpp"

#include "noa/unified/ArrayOption.hpp"
#include "noa/unified/Stream.hpp"
#include "noa/unified/memory/Copy.hpp"
#include "noa/unified/memory/Permute.hpp"

namespace noa {
    template<typename T>
    class Array;

    /// View of an Array or memory region.
    /// \details
    ///   It behaves like an Array, but its value type can be const-qualified. More importantly, it does not own
    ///   the memory region it points to. This makes it more lightweight than an Array, but is slightly more
    ///   difficult to use correctly.\n
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

        static_assert(!std::is_pointer_v<value_type>);
        static_assert(!std::is_reference_v<value_type>);

    public: // Constructors
        // Creates an empty view.
        NOA_HD constexpr View() = default;

        // Creates a view of a contiguous row-vector.
        template<typename Index, typename = std::enable_if_t<std::is_integral_v<Index>>>
        NOA_HD constexpr View(T* data, Index elements, ArrayOption options = {})
                : m_accessor(data, strides_type{elements, elements, elements, 1}),
                  m_shape{1, 1, 1, elements}, m_options(options) {}

        // Creates a view.
        NOA_HD constexpr View(T* data, const shape_type& shape, const strides_type& strides, ArrayOption options = {})
                : m_accessor(data, strides), m_shape(shape), m_options(options) {}

        // Creates a view, assuming the data is C-contiguous.
        NOA_HD constexpr View(T* data, const shape_type& shape, ArrayOption options = {})
                : m_accessor(data, shape.strides()), m_shape(shape), m_options(options) {}

        // Creates a const view from an existing non-const view.
        template<typename U, typename = std::enable_if_t<details::is_mutable_value_type_v<U, value_type>>>
        NOA_HD constexpr /*implicit*/ View(const View<U>& view)
                : m_accessor(view.data(), view.strides()), m_shape(view.shape()), m_options(view.options()) {}

        // Creates a view from an Array.
        template<typename U, typename = std::enable_if_t<noa::traits::is_almost_same_v<U, value_type>>>
        constexpr explicit View(const Array<U>& array)
                : m_accessor(array.get(), array.strides()), m_shape(array.shape()), m_options(array.options()) {}

    public: // Getters
        /// Returns the options used to create the viewed array.
        [[nodiscard]] NOA_HD constexpr const ArrayOption& options() const noexcept { return m_options; }

        /// Returns the device used to create the viewed array.
        [[nodiscard]] NOA_HD constexpr Device device() const noexcept { return options().device(); }

        /// Returns the memory resource used to create the viewed array.
        [[nodiscard]] NOA_HD constexpr Allocator allocator() const noexcept { return options().allocator(); }

        /// Whether the managed data can be accessed by CPU threads.
        [[nodiscard]] NOA_HD constexpr bool is_dereferenceable() const noexcept { return options().is_dereferenceable(); }

        /// Returns the BDHW shape of the viewed array.
        [[nodiscard]] NOA_HD const shape_type& shape() const noexcept { return m_shape; }

        /// Returns the BDHW strides of the viewed array.
        [[nodiscard]] NOA_HD const strides_type& strides() const noexcept { return m_accessor.strides(); }

        /// Returns the number of elements in the viewed array.
        [[nodiscard]] NOA_HD constexpr index_type elements() const noexcept { return shape().elements(); }
        [[nodiscard]] NOA_HD constexpr index_type ssize() const noexcept { return elements(); }
        [[nodiscard]] NOA_HD constexpr size_t size() const noexcept { return static_cast<size_t>(elements()); }

        template<char ORDER = 'C'>
        [[nodiscard]] NOA_HD constexpr bool are_contiguous() const noexcept {
            return noa::indexing::are_contiguous<ORDER>(m_accessor.strides(), shape());
        }

        template<char ORDER = 'C'>
        [[nodiscard]] NOA_HD constexpr auto is_contiguous() const noexcept {
            return noa::indexing::is_contiguous<ORDER>(m_accessor.strides(), shape());
        }

        /// Whether the view is empty. A View is empty if not initialized,
        /// or if the viewed data is null, or if one of its dimension is 0.
        [[nodiscard]] NOA_HD constexpr bool is_empty() const noexcept { return !get() || noa::any(shape() == 0); }

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
        [[nodiscard]] NOA_HD constexpr pointer_type get() const noexcept { return m_accessor.get(); }

        /// Returns the pointer to the data.
        /// \warning Depending on the current stream of this array's device,
        ///          reading/writing to this pointer may be illegal or create a data race.
        [[nodiscard]] NOA_HD constexpr pointer_type data() const noexcept { return m_accessor.data(); }

        /// Returns the pointer to the data.
        /// \warning Depending on the current stream of this array's device,
        ///          reading/writing to this pointer may be illegal or create a data race.
        [[nodiscard]] NOA_HD constexpr pointer_type share() const noexcept { return get(); }

        /// Returns a (const-)span of the array.
        /// \warning Depending on the current stream of this array's device,
        ///          reading/writing through this Span may be illegal or create a data race.
        template<typename U = value_type, i64 SIZE = -1, typename I = index_type,
                 typename = std::enable_if_t<noa::traits::is_almost_same_v<U, value_type> && std::is_integral_v<T>>>
        [[nodiscard]] constexpr Span<U, SIZE, I> span() const noexcept {
            NOA_CHECK(are_contiguous(),
                      "Cannot create a Span from a non-contiguous view (shape={}, strides={})",
                      shape(), strides());
            if constexpr (SIZE >= 0) {
                NOA_CHECK(elements() == SIZE,
                          "Cannot create a Span with a static size of {} from a view with {} elements",
                          SIZE, elements());
            }
            return Span<U, SIZE, I>(get(), elements());
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
            using output_t = std::pair<output_accessor_t, output_shape_t>;

            const auto reinterpreted = noa::indexing::Reinterpret(shape(), strides(), get()).template as<U>();
            // FIXME If StridesTraits::CONTIGUOUS, assert(strides[3] == 1) ?

            if constexpr (N == 4) {
                return output_t{output_accessor_t(reinterpreted.ptr, reinterpreted.strides.template as_safe<I>()),
                                reinterpreted.shape.template as_safe<I>()};
            } else {
                // Construct the new shape by stacking the outer dimensions together.
                constexpr i64 OFFSET = 4 - N;
                Shape4<i64> new_shape{1};
                for (i64 i = 0; i < 4; ++i)
                    new_shape[noa::math::max(i, OFFSET)] *= reinterpreted.shape[i];

                // Reshape.
                Strides4<i64> new_stride{};
                if (!noa::indexing::reshape(reinterpreted.shape, reinterpreted.strides, new_shape, new_stride))
                    NOA_THROW("A view of shape {} and strides {} cannot be reshaped to shape {}",
                              reinterpreted.shape, reinterpreted.strides, new_shape);

                // Ignore the outer empty dimensions.
                output_shape_t output_shape(new_shape.template pop_front<OFFSET>().template as_safe<I>());
                output_strides_t output_strides(new_stride.template pop_front<OFFSET>().template as_safe<I>());
                return output_t{output_accessor_t(reinterpreted.ptr, output_strides),
                                output_shape};
            }
        }

        [[nodiscard]] NOA_HD constexpr const accessor_type& accessor() const { return m_accessor; }

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
        template<typename Output, typename = std::enable_if_t<
                 noa::traits::is_array_or_view_v<Output> &&
                 noa::traits::are_almost_same_value_type_v<View, Output>>>
        void to(const Output& output) const {
            noa::memory::copy(*this, output);
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
            return to(Device{});
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
            const auto out = noa::indexing::Reinterpret(shape(), strides(), get()).template as<U>();
            return View<U>(out.ptr, out.shape, out.strides, options());
        }

        /// Changes the device type (CPU<->GPU) on which the memory should be accessed.
        /// \details If the memory resource can be accessed by the CPU and/or a GPU, this function returns a new view
        ///          with the new device. This is used to control whether PINNED or MANAGED memory should be accessed
        ///          by the CPU or the GPU. MANAGED_GLOBAL memory is not attached to any particular GPU, so the
        ///          current GPU is used in that case.
        [[nodiscard]] View as(DeviceType type) const {
            const Allocator alloc = m_options.allocator();
            if (device().is_gpu() && type == DeviceType::CPU) { // GPU -> CPU
                NOA_CHECK(alloc == Allocator::PINNED ||
                          alloc == Allocator::MANAGED ||
                          alloc == Allocator::MANAGED_GLOBAL,
                          "GPU memory-region with the allocator {} cannot be reinterpreted as a CPU memory-region. "
                          "This is only supported for pinned and managed memory-regions", alloc);
                return View(get(), shape(), strides(), ArrayOption(m_options).set_device(Device(type)));

            } else if (device().is_cpu() && type == DeviceType::GPU) { // CPU -> GPU
                NOA_CHECK(Device::is_any(DeviceType::GPU), "No GPU detected");
                NOA_CHECK(alloc == Allocator::PINNED ||
                          alloc == Allocator::MANAGED ||
                          alloc == Allocator::MANAGED_GLOBAL,
                          "CPU memory-region with the allocator {} cannot be reinterpreted as a GPU memory-region. "
                          "This is only supported for pinned and managed memory-regions", alloc);
                Device gpu;
                #ifdef NOA_ENABLE_CUDA
                if (alloc == Allocator::PINNED || alloc == Allocator::MANAGED) {
                    // NOTE: CUDA doesn't document what the attr.device is for managed memory.
                    //       Hopefully this is the device against which the allocation was performed
                    //       and not the current device.
                    // NOTE: With "stream-attached" managed memory, it is up to the user to know what
                    //       stream was used to perform the allocation.
                    const cudaPointerAttributes attr = cuda::utils::pointer_attributes(get());
                    gpu = Device(DeviceType::GPU, attr.device, Device::DeviceUnchecked{});
                    NOA_ASSERT((alloc == Allocator::PINNED && attr.type == cudaMemoryTypeHost) ||
                               (alloc == Allocator::MANAGED && attr.type == cudaMemoryTypeManaged));
                    // TODO Add cudaPrefetchAsync when it is added to cuda::PtrManaged.

                } else if (alloc == Allocator::MANAGED_GLOBAL) {
                    // NOTE: This can be accessed from any stream and any GPU. It seems to be better to return the
                    //       current device and not the original device against which the allocation was performed.
                    gpu = Device::current(DeviceType::GPU);
                }
                #else
                NOA_THROW("No GPU backend detected");
                #endif
                return View(get(), shape(), strides(), ArrayOption(m_options).set_device(gpu));
            } else {
                return *this;
            }
        }

        /// Reshapes the view (must have the same number of elements as the current view).
        [[nodiscard]] View reshape(const shape_type& new_shape) const {
            strides_type new_stride;
            if (!noa::indexing::reshape(shape(), strides(), new_shape, new_stride)) {
                NOA_THROW("A view of shape {} and stride {} cannot be reshaped to a view of shape {}",
                          shape(), strides(), new_shape);
            }
            return View(get(), new_shape, new_stride, options());
        }

        /// Reshapes the array in a vector along a particular axis.
        /// Returns a row vector by default (axis = 3).
        [[nodiscard]] View flat(i32 axis = 3) const {
            shape_type output_shape(1);
            output_shape[axis] = shape().elements();
            return reshape(output_shape);
        }

        /// Permutes the dimensions of the view.
        /// \param permutation  Permutation with the axes numbered from 0 to 3.
        [[nodiscard]] NOA_HD constexpr View permute(const Vec4<i64>& permutation) const {
            return View(get(),
                        noa::indexing::reorder(shape(), permutation),
                        noa::indexing::reorder(strides(), permutation),
                        options());
        }

        [[nodiscard]] Array<mutable_value_type> permute_copy(const Vec4<i64>& permutation) const {
            return memory::permute_copy(*this, permutation);
        }

    public: // Assignment operators
        /// Clears the view. Equivalent to assigning *this with an empty view.
        View& operator=(std::nullptr_t) {
            *this = View{};
            return *this;
        }

    public: // Indexing & Subregion
        template<typename I0, typename I1, typename I2, typename I3>
        [[nodiscard]] NOA_HD constexpr value_type& operator()(I0 i0, I1 i1, I2 i2, I3 i3) const noexcept {
            NOA_ASSERT(is_dereferenceable());
            NOA_ASSERT(clamp_cast<i64>(i0) >= 0 && clamp_cast<i64>(i0) < shape()[0] &&
                       clamp_cast<i64>(i1) >= 0 && clamp_cast<i64>(i1) < shape()[1] &&
                       clamp_cast<i64>(i2) >= 0 && clamp_cast<i64>(i2) < shape()[2] &&
                       clamp_cast<i64>(i3) >= 0 && clamp_cast<i64>(i3) < shape()[3]);
            return m_accessor(i0, i1, i2, i3);
        }

    public: // Subregion
        template<typename A,
                 typename B = indexing::full_extent_t,
                 typename C = indexing::full_extent_t,
                 typename D = indexing::full_extent_t,
                 typename = std::enable_if_t<indexing::Subregion::are_indexer_v<A, B, C, D>>>
        [[nodiscard]] constexpr View subregion(A&& i0, B&& i1 = {}, C&& i2 = {}, D&& i3 = {}) const {
            const auto indexer = indexing::Subregion(shape(), strides()).extract(i0, i1, i2, i3);
            return View(get() + indexer.offset, indexer.shape, indexer.strides, options());
        }

        [[nodiscard]] constexpr View subregion(indexing::ellipsis_t) const {
            return *this;
        }

        template<typename A,
                 typename = std::enable_if_t<indexing::Subregion::is_indexer_v<A>>>
        [[nodiscard]] constexpr View subregion(indexing::ellipsis_t, A&& i3) const {
            return subregion(indexing::full_extent_t{}, indexing::full_extent_t{}, indexing::full_extent_t{}, i3);
        }

        template<typename A, typename B,
                 typename = std::enable_if_t<indexing::Subregion::are_indexer_v<A, B>>>
        [[nodiscard]] constexpr View subregion(indexing::ellipsis_t, A&& i2, B&& i3) const {
            return subregion(indexing::full_extent_t{}, indexing::full_extent_t{}, i2, i3);
        }

        template<typename A, typename B, typename C,
                 typename = std::enable_if_t<indexing::Subregion::are_indexer_v<A, B, C>>>
        [[nodiscard]] constexpr View subregion(indexing::ellipsis_t, A&& i1, B&& i2, C&& i3) const {
            return subregion(indexing::full_extent_t{}, i1, i2, i3);
        }

    private:
        accessor_type m_accessor;
        shape_type m_shape;
        ArrayOption m_options;
    };
}

namespace noa::traits {
    template<typename T> struct proclaim_is_view<View<T>> : std::true_type {};
}
