#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Indexing.h"
#include "noa/common/Types.h"
#include "noa/common/types/View.h"

#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/PtrHost.h"

#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/PtrDevicePadded.h"
#include "noa/gpu/cuda/memory/PtrManaged.h"
#include "noa/gpu/cuda/memory/PtrPinned.h"
#include "noa/gpu/cuda/util/Pointers.h"
#endif

#include "noa/unified/Allocator.h"
#include "noa/unified/ArrayOption.h"
#include "noa/unified/Device.h"
#include "noa/unified/Stream.h"

namespace noa {
    template<typename T>
    class Array;
}

namespace noa::details {
    template<typename T>
    void arrayTranspose(const Array<T>& src, const Array<T>& dst, uint4_t permutation);
}

namespace noa {
    /// 4-dimensional array of any data type.
    /// \details
    /// - \b Type: Arrays are usually managing data type, i.e. integers, floating-points or complex floating-points.
    ///   However, other types are supported, namely, small static vectors (e.g. float4_t) or static matrices
    ///   (e.g. float33_t). Array of such composite types are limited to be simple containers (e.g. arithmetics are
    ///   not supported). The managed type cannot be const-qualified, nor can it be a reference, a pointer or an extent.\n
    /// - \b Resource: Arrays manage a reference-counted resource, which can be shared, moved and copied.
    ///   The resource type and location depends on the ArrayOption used to create the array. Arrays are therefore
    ///   attached to a device, either the CPU or a GPU. Depending on the resource (thus the allocator used to create
    ///   the array), arrays can be interpreted as CPU or GPU and it is possible to create aliases of the same
    ///   resource for the CPU and the GPU (see as() for more details).\n
    /// - \b Accessors: When a backend call is needed, arrays will use the current stream of their device; they are
    ///   "stream-safe". In other words, one does not and should not synchronize the stream between each operation
    ///   involving an Array, except if the result of that operation is used in a "unsafe" way. These unsafe ways of
    ///   accessing the managed data are by get(), share() or view(). While these are often required for e.g. efficient
    ///   loop-like indexing, one must make sure the current stream of the Array's device is synchronized.\n
    /// - \b Shape: Shape and strides are in number of elements and specified in the rightmost order (from outermost
    ///   to innermost). Empty dimensions have a size of 1. If one dimension is 0, the entire array is considered empty.
    ///   Arrays can be broadcast to another shape and they follow the broadcasting rule (see indexing::broadcast()).
    ///   As such, some arrays can have dimensions with a stride of 0. Negative strides are not supported.
    template<typename T>
    class Array {
    public: // typedefs
        using value_t = T;
        using dim_t = size_t;
        using dim4_t = Int4<dim_t>;
        using ptr_t = T*;
        using ref_t = T&;

        static_assert(!std::is_const_v<T>);
        static_assert(!std::is_pointer_v<T>);
        static_assert(!std::is_reference_v<T>);
        static_assert(noa::traits::is_data_v<T> ||
                      noa::traits::is_intX_v<T> ||
                      noa::traits::is_floatX_v<T> ||
                      noa::traits::is_floatXX_v<T>);

        template<typename U>
        static constexpr bool is_indexable_v =
                std::bool_constant<noa::traits::is_int_v<U> ||
                                   noa::traits::is_almost_same_v<U, indexing::full_extent_t> ||
                                   noa::traits::is_almost_same_v<U, indexing::slice_t>>::value;

    public: // Constructors
        /// Creates an empty array.
        constexpr Array() = default;

        /// Creates a contiguous 1D array.
        /// \param elements Number of elements.
        /// \param option   Options of the created array.
        /// \see Allocator for more details.
        constexpr explicit Array(size_t elements, ArrayOption option = {})
                : m_shape(size4_t{1, 1, 1, elements}),
                  m_stride(size4_t{elements, elements, elements, 1}),
                  m_options(option) { alloc_(); }

        /// Creates a contiguous array.
        /// \param shape    Rightmost shape of the array.
        /// \param option   Options of the created array.
        /// \see Allocator for more details.
        constexpr explicit Array(size4_t shape, ArrayOption option = {})
                : m_shape(shape), m_stride(shape.stride()), m_options(option) { alloc_(); }

        /// Creates a non-owning array from an existing allocated memory region.
        /// \param[in,out] data Data to encapsulate.
        /// \param shape        Rightmost shape of \p data.
        /// \param stride       Rightmost stride of \p data.
        /// \param option       Options of \p data.
        constexpr Array(T* data, size4_t shape, size4_t stride, ArrayOption option)
                : m_shape(shape), m_stride(stride), m_ptr(data, [](void*) {}), m_options(option) {
            validate_(data, option);
        }

        /// Creates an array from an existing allocated memory region.
        /// \param[in,out] data Data to encapsulate.
        /// \param shape        Rightmost shape of \p data.
        /// \param stride       Rightmost stride of \p data.
        /// \param option       Options of \p data.
        constexpr Array(shared_t<T[]> data, size4_t shape, size4_t stride, ArrayOption option)
                : m_shape(shape), m_stride(stride), m_ptr(std::move(data)), m_options(option) {
            validate_(m_ptr.get(), option);
        }

    public: // Getters
        /// Returns the options used to create the array.
        [[nodiscard]] constexpr ArrayOption options() const noexcept { return m_options; }

        /// Returns the device used to create the array.
        [[nodiscard]] constexpr Device device() const noexcept { return m_options.device(); }

        /// Returns the memory resource used to create the array.
        [[nodiscard]] constexpr Allocator allocator() const noexcept { return m_options.allocator(); }

        /// Whether the managed data can be accessed by CPU threads.
        /// \note While it indicates whether the managed data can be dereferenced on the CPU, it does not indicate
        ///       if the Array will be seen as a CPU (or a GPU) array. This choice is purely made on the Array's
        ///       device type. For instance, pinned memory can be dereferenced by the CPU, so this function will
        ///       returned true, but if the Array's device is a GPU, the implementations will refer to this Array
        ///       as a GPU array and will therefore prioritizing GPU access.
        [[nodiscard]] constexpr bool dereferencable() const noexcept {
            return m_options.device().cpu() || allocator() == Allocator::PINNED ||
                   allocator() == Allocator::MANAGED || allocator() == Allocator::MANAGED_GLOBAL;
        }

        /// Whether the array is empty.
        [[nodiscard]] bool empty() const noexcept { return !m_ptr || !m_shape.elements(); }

        /// Returns the rightmost shape of the array.
        [[nodiscard]] const size4_t& shape() const noexcept { return m_shape; }

        /// Returns the rightmost stride of the array.
        [[nodiscard]] const size4_t& stride() const noexcept { return m_stride; }

        /// Whether the dimensions of the array are contiguous.
        [[nodiscard]] bool4_t contiguous() const noexcept {
            return indexing::isContiguous(m_stride, m_shape);
        }

    public: // Accessors
        /// Returns the pointer to the data.
        [[nodiscard]] constexpr T* get() noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr const T* get() const noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr T* data() noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr const T* data() const noexcept { return m_ptr.get(); }

        /// Returns a reference of the managed resource.
        [[nodiscard]] constexpr const std::shared_ptr<T[]>& share() const noexcept { return m_ptr; }

        /// Returns a view of the array.
        template<typename I = size_t>
        [[nodiscard]] constexpr View <T, I> view() noexcept {
            return {get(), Int4<I>{m_shape}, Int4<I>{m_stride}};
        }

        /// Returns a const view of the array.
        template<typename I = size_t>
        [[nodiscard]] constexpr View<const T, I> view() const noexcept {
            return {get(), Int4<I>{m_shape}, Int4<I>{m_stride}};
        }

        /// Returns the current stream of the Array's device. This is often used to synchronize
        /// the stream before accessing the managed data in a non-stream-ordered way.
        [[nodiscard]] Stream& stream() const {
            return Stream::current(device());
        }

    public: // Deep copy
        /// Performs a deep copy of the array. The returned array is completely independent from the original one.
        Array copy(bool as_contiguous = false) const {
            return to(m_options, as_contiguous);
        }

        /// Performs a deep copy of the array.
        /// \details The returned array is completely independent from the original one and is contiguous.
        ///          Contiguous regions of memory have no copy restrictions and can be copied to any device. This is
        ///          also true for pitched layouts. However, other non-contiguous memory layouts can only be copied
        ///          if the source and destination are both on the same GPU or on the CPU.
        /// \param option   Output device and resource to perform the allocation of the new array.
        ///                 The current stream for that device is used to perform the copy.
        Array to(ArrayOption option) const {
            Array out{m_shape, option};
            const Device input_device = this->device();
            const Device output_device = out.device();

            if (input_device.cpu() && output_device.cpu()) {
                cpu::memory::copy(this->share(), m_stride,
                                  out.share(), out.stride(),
                                  out.shape(), Stream::current(input_device).cpu());
            } else if (output_device.cpu()) { // gpu->cpu
                #ifdef NOA_ENABLE_CUDA
                Stream::current(output_device).synchronize();
                cuda::Stream& cuda_stream = Stream::current(input_device).cuda();
                cuda::memory::copy(this->share(), m_stride,
                                   out.share(), out.stride(),
                                   out.shape(), cuda_stream);
                cuda_stream.synchronize();
                #else
                NOA_THROW("No GPU backend detected");
                #endif
            } else { // gpu->gpu or cpu->gpu
                #ifdef NOA_ENABLE_CUDA
                if (input_device != output_device)
                    Stream::current(input_device).synchronize(); // wait for the input
                cuda::memory::copy(this->share(), m_stride,
                                   out.share(), out.stride(),
                                   out.shape(), Stream::current(output_device).cuda());
                #else
                NOA_THROW("No GPU backend detected");
                #endif
            }
            return out;
        }

        Array to(Device device) const {
            return to(ArrayOption{m_options}.device(device));
        }

    public: // Data reinterpretation
        /// Reinterprets the managed array of \p T as an array of \p U.
        /// \note This is only well defined in cases where reinterpret_cast<U*>(T*) is well defined, for instance,
        ///       when \p U is a unsigned char (to represent any data type as a array of bytes), or to switch between
        ///       complex and real floating-point numbers with the same precision.
        template<typename U>
        Array<U> as() const {
            const auto out = indexing::Reinterpret<T, size_t>{m_shape, m_stride, get()}.template as<U>();
            return {std::shared_ptr<U[]>{m_ptr, out.ptr}, out.shape, out.stride, options()};
        }

        /// Reinterprets the managed array of \p T as an array of \p U.
        /// \note This is only well defined in cases where reinterpret_cast<U*>(T*) is well defined, for instance,
        ///       when \p U is a unsigned char (to represent any data type as a array of bytes), or to switch between
        ///       complex and real floating-point numbers with the same precision.
        template<typename U>
        Array<U> as() {
            const auto out = indexing::Reinterpret<T, size_t>{m_shape, m_stride, get()}.template as<U>();
            return {std::shared_ptr<U[]>{m_ptr, out.ptr}, out.shape, out.stride, options()};
        }

        /// Changes the side (CPU<->GPU) on which the memory should be accessed.
        /// \details If the memory resource can be accessed by the CPU and/or a GPU, this function returns an array
        ///          with the device of the given \p type. This is used to control whether PINNED or MANAGED memory
        ///          should be accessed by the CPU or the GPU. MANAGED_GLOBAL memory is not attached to any particular
        ///          GPU, so the current GPU is used in that case.
        Array as(Device::Type type) const {
            const Allocator alloc = m_options.allocator();
            if (type == Device::CPU && device().gpu()) { // see as CPU array
                NOA_CHECK(alloc != Allocator::DEFAULT &&
                          alloc != Allocator::DEFAULT_ASYNC &&
                          alloc != Allocator::PITCHED,
                          "GPU arrays with the allocator {} cannot be reinterpreted as CPU arrays", alloc);
                return {m_ptr, m_shape, m_stride, ArrayOption{m_options}.device(Device{type})};

            } else if (type == Device::GPU && device().cpu()) { // see as GPU array
                NOA_CHECK(Device::any(Device::GPU), "No GPU detected");
                NOA_CHECK(alloc != Allocator::DEFAULT &&
                          alloc != Allocator::DEFAULT_ASYNC &&
                          alloc != Allocator::PITCHED,
                          "CPU arrays with the allocator {} cannot be reinterpreted as GPU arrays", alloc);
                Device gpu;
                #ifdef NOA_ENABLE_CUDA
                if (alloc == Allocator::PINNED || alloc == Allocator::MANAGED) {
                    // NOTE: CUDA doesn't document what the attr.device is for managed memory.
                    //       Hopefully this is the device against which the allocation was performed
                    //       and not the current device.
                    // NOTE: With "stream-attached" managed memory, it is up to the user to know what
                    //       stream was used to perform the allocation.
                    const cudaPointerAttributes attr = cuda::util::getAttributes(m_ptr.get());
                    gpu = Device{Device::GPU, attr.device, true};
                    NOA_ASSERT((alloc == Allocator::PINNED && attr.type == cudaMemoryTypeHost) ||
                               (alloc == Allocator::MANAGED && attr.type == cudaMemoryTypeManaged));
                    // TODO Add cudaPrefetchAsync when it is added to cuda::PtrManaged.

                } else if (alloc == Allocator::MANAGED_GLOBAL) {
                    // NOTE: This can be accessed from any stream and any GPU. It seems to be better to return the
                    //       current device and not the original device against which the allocation was performed.
                    gpu = Device::current(Device::GPU);
                }
                #else
                NOA_THROW("No GPU backend detected");
                #endif
                return {m_ptr, m_shape, m_stride, ArrayOption{m_options}.device(gpu)};
            } else {
                return *this;
            }
        }

        /// Reshapes the array.
        /// \details This function performs a "safe" reshape by making sure the new shape contains the same number
        ///          of elements. If one wants to assign an array to an arbitrary new shape and new stride, one
        ///          can use the alias Array constructor instead.
        /// \param shape Rightmost shape. Must contain the same number of elements as the current shape.
        /// \return An alias of the array with the new shape and stride.
        Array reshape(size4_t shape) const {
            size4_t new_stride;
            if (!indexing::reshape(m_shape, m_stride, shape, new_stride))
                NOA_THROW("An array of shape {} cannot be reshaped to an array of shape {}", m_shape, shape);
            return {m_ptr, shape, new_stride, options()};
        }

        /// Permutes the array.
        /// \param permutation  Rightmost permutation. Axes are numbered from 0 to 3, 3 being the innermost dimension.
        /// \param copy         Whether the permuted array should be copied into a contiguous array, completely
        ///                     independent from the original one.
        /// \return The permuted array. If \p copy is false, this new array is an alias from the original array.
        Array permute(uint4_t permutation, bool copy = false) const {
            const size4_t permuted_shape = indexing::reorder(m_shape, permutation);
            Array<T> out;
            if (copy) {
                if constexpr (!noa::traits::is_data_v<T>) {
                    NOA_THROW("This type ({}) is not supported by memory::transpose()", string::human<T>());
                } else {
                    out = Array<T>{permuted_shape, m_options};
                    details::arrayTranspose(*this, out, permutation);
                }
            } else {
                out = Array<T>{m_ptr, permuted_shape, indexing::reorder(m_stride, permutation), m_options};
            }
            return out;
        }

    public: // Operators
        Array& operator=(T value) {
            // fill;
        }

        Array& operator+=(T value);
        Array& operator-=(T value);
        Array& operator*=(T value);
        Array& operator/=(T value);

        Array& operator+=(Array& value);
        Array& operator-=(Array& value);
        Array& operator*=(Array& value);
        Array& operator/=(Array& value);



        // TODO operator=(), operator+(), ..., fancy indexing etc.
        // NOTE do not add unsafe accesses, like loop-like indexing.

    public: // Subregion
        template<typename A,
                 typename B = indexing::full_extent_t,
                 typename C = indexing::full_extent_t,
                 typename D = indexing::full_extent_t,
                 typename = std::enable_if_t<is_indexable_v<A> && is_indexable_v<B> &&
                                             is_indexable_v<C> && is_indexable_v<D>>>
        constexpr Array subregion(A&& i0, B&& i1 = {}, C&& i2 = {}, D&& i3 = {}) const {
            const indexing::Subregion<int64_t> indexer =
                    indexing::Subregion<int64_t>{long4_t{m_shape}, long4_t{m_stride}}(i0, i1, i2, i3);
            return {std::shared_ptr<T[]>{m_ptr, m_ptr.get() + indexer.offset},
                    size4_t{indexer.shape}, size4_t{indexer.stride}, m_options};
        }

        constexpr Array subregion(indexing::ellipsis_t) const {
            return *this;
        }

        template<typename A, typename = std::enable_if_t<is_indexable_v<A>>>
        constexpr Array subregion(indexing::ellipsis_t, A&& i3) const {
            return subregion(indexing::full_extent_t{}, indexing::full_extent_t{}, indexing::full_extent_t{}, i3);
        }

        template<typename A, typename B,
                typename = std::enable_if_t<is_indexable_v<A> && is_indexable_v<B>>>
        constexpr Array subregion(indexing::ellipsis_t, A&& i2, B&& i3) const {
            return subregion(indexing::full_extent_t{}, indexing::full_extent_t{}, i2, i3);
        }

        template<typename A, typename B, typename C,
                typename = std::enable_if_t<is_indexable_v<A> && is_indexable_v<B> && is_indexable_v<C>>>
        constexpr Array subregion(indexing::ellipsis_t, A&& i1, B&& i2, C&& i3) const {
            return subregion(indexing::full_extent_t{}, i1, i2, i3);
        }

    private:
        void alloc_() {
            NOA_PROFILE_FUNCTION();
            const size_t elements = m_shape.elements();
            const Device device = m_options.device();
            switch (m_options.allocator()) {
                case Allocator::NONE:
                    break;
                case Allocator::DEFAULT:
                    if (device.cpu()) {
                        m_ptr = cpu::memory::PtrHost<T>::alloc(elements);
                    } else {
                        #ifdef NOA_ENABLE_CUDA
                        DeviceGuard guard(device);
                        m_ptr = cuda::memory::PtrDevice<T>::alloc(elements);
                        #endif
                    }
                    break;
                case Allocator::DEFAULT_ASYNC:
                    if (device.cpu()) {
                        m_ptr = cpu::memory::PtrHost<T>::alloc(elements);
                    } else {
                        #ifdef NOA_ENABLE_CUDA
                        m_ptr = cuda::memory::PtrDevice<T>::alloc(elements, Stream::current(device).cuda());
                        #endif
                    }
                    break;
                case Allocator::PITCHED:
                    if (device.cpu()) {
                        m_ptr = cpu::memory::PtrHost<T>::alloc(elements);
                    } else {
                        #ifdef NOA_ENABLE_CUDA
                        DeviceGuard guard(device);
                        // PtrDevicePadded requires sizeof(T) <= 16 bytes.
                        // We could remove this restriction, but for now since it is only for
                        // static vectors and matrices, just switch to classic cudaMalloc.
                        if constexpr (noa::traits::is_data_v<T>) {
                            auto[ptr, pitch] = cuda::memory::PtrDevicePadded<T>::alloc(m_shape);
                            m_ptr = std::move(ptr);
                            m_stride = size4_t{m_shape[0], m_shape[1], m_shape[2], pitch}.stride();
                        } else {
                            m_ptr = cuda::memory::PtrDevice<T>::alloc(elements);
                        }
                        #endif
                    }
                    break;
                case Allocator::PINNED: {
                    if (device.cpu() && !Device::any(Device::GPU)) {
                        m_ptr = cpu::memory::PtrHost<T>::alloc(elements);
                    } else {
                        #ifdef NOA_ENABLE_CUDA
                        DeviceGuard guard(device.gpu() ? device : Device::current(Device::GPU));
                        m_ptr = cuda::memory::PtrPinned<T>::alloc(elements);
                        #endif
                    }
                    break;
                }
                case Allocator::MANAGED: {
                    if (device.cpu() && !Device::any(Device::GPU)) {
                        m_ptr = cpu::memory::PtrHost<T>::alloc(elements);
                    } else {
                        #ifdef NOA_ENABLE_CUDA
                        const Device gpu = device.gpu() ? device : Device::current(Device::GPU);
                        const DeviceGuard guard(gpu); // could be helpful when retrieving device
                        cuda::Stream& cuda_stream = Stream::current(gpu).cuda();
                        m_ptr = cuda::memory::PtrManaged<T>::alloc(elements, cuda_stream);
                        #endif
                    }
                    break;
                }
                case Allocator::MANAGED_GLOBAL: {
                    if (device.cpu() && !Device::any(Device::GPU)) {
                        m_ptr = cpu::memory::PtrHost<T>::alloc(elements);
                    } else {
                        #ifdef NOA_ENABLE_CUDA
                        DeviceGuard guard(device.gpu() ? device : Device::current(Device::GPU));
                        m_ptr = cuda::memory::PtrManaged<T>::alloc(elements);
                        #endif
                    }
                    break;
                }
            }
        }

        static void validate_(void* ptr, ArrayOption option) {
            const Allocator alloc = option.allocator();
            if (option.device().cpu() &&
                alloc != Allocator::DEFAULT &&
                alloc != Allocator::DEFAULT_ASYNC &&
                alloc != Allocator::PITCHED)
                return; // there's no simple way to validate the data

            #ifdef NOA_ENABLE_CUDA
            if (alloc == Allocator::PINNED ||
                alloc == Allocator::MANAGED ||
                alloc == Allocator::MANAGED_GLOBAL) {
                const cudaPointerAttributes attr = cuda::util::getAttributes(ptr);
                NOA_CHECK(option.device().cpu() || attr.device == option.device().id(),
                          "The entered GPU ID ({}) does not match the registered GPU ID for that pointer ({})",
                          attr.device, option.device().id());
                NOA_CHECK((option.allocator() == Allocator::PINNED && attr.type == cudaMemoryTypeHost) ||
                          (option.allocator() == Allocator::MANAGED && attr.type == cudaMemoryTypeManaged) ||
                          (option.allocator() == Allocator::MANAGED_GLOBAL && attr.type == cudaMemoryTypeManaged),
                          "The entered allocator ({}) does not match the registered allocator (cudaMemoryType:{}) "
                          "for that pointer", option.allocator(), attr.type);
            }
            #else
            (void) ptr;
            return;
            #endif
        }

    private:
        size4_t m_shape;
        size4_t m_stride;
        std::shared_ptr<T[]> m_ptr;
        ArrayOption m_options;
    };
}
