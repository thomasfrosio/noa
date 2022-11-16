#pragma once
#ifndef NOA_UNIFIED_ARRAY_
#error "Cannot include this private header"
#endif

namespace noa {
    template<typename Value>
    constexpr Array<Value>::Array(dim_t elements, ArrayOption option)
            : m_shape(dim4_t{1, 1, 1, elements}),
              m_strides(dim4_t{elements, elements, elements, 1}),
              m_options(option) { alloc_(); }

    template<typename Value>
    constexpr Array<Value>::Array(dim4_t shape, ArrayOption option)
            : m_shape(shape), m_strides(shape.strides()), m_options(option) { alloc_(); }

    template<typename Value>
    constexpr Array<Value>::Array(pointer_type data, dim_t elements, ArrayOption option)
            : m_shape(dim4_t{1, 1, 1, elements}),
              m_strides(dim4_t{elements, elements, elements, 1}),
              m_ptr(data, [](void*) {}),
              m_options(option) {
        validate_(data, option);
    }

    template<typename Value>
    constexpr Array<Value>::Array(pointer_type data, dim4_t shape, dim4_t strides, ArrayOption option)
            : m_shape(shape), m_strides(strides), m_ptr(data, [](void*) {}), m_options(option) {
        validate_(data, option);
    }

    template<typename Value>
    constexpr Array<Value>::Array(shared_type data, dim_t elements, ArrayOption option)
            : m_shape(dim4_t{1, 1, 1, elements}),
              m_strides(dim4_t{elements, elements, elements, 1}),
              m_ptr(std::move(data)),
              m_options(option) {
        validate_(m_ptr.get(), option);
    }

    template<typename Value>
    constexpr Array<Value>::Array(shared_type data, dim4_t shape, dim4_t strides, ArrayOption option)
            : m_shape(shape), m_strides(strides), m_ptr(std::move(data)), m_options(option) {
        validate_(m_ptr.get(), option);
    }

    template<typename Value>
    constexpr ArrayOption Array<Value>::options() const noexcept { return m_options; }

    template<typename Value>
    constexpr Device Array<Value>::device() const noexcept { return m_options.device(); }

    template<typename Value>
    constexpr Allocator Array<Value>::allocator() const noexcept { return m_options.allocator(); }

    template<typename Value>
    constexpr bool Array<Value>::dereferenceable() const noexcept { return m_options.dereferenceable(); }

    template<typename Value>
    bool Array<Value>::empty() const noexcept { return !m_ptr || any(m_shape == 0); }

    template<typename Value>
    const dim4_t& Array<Value>::shape() const noexcept { return m_shape; }

    template<typename Value>
    const dim4_t& Array<Value>::strides() const noexcept { return m_strides; }

    template<typename Value>
    dim_t Array<Value>::elements() const noexcept { return m_shape.elements(); }

    template<typename Value>
    dim_t Array<Value>::size() const noexcept { return elements(); }

    template<typename Value>
    template<char ORDER>
    bool Array<Value>::contiguous() const noexcept {
        return indexing::areContiguous<ORDER>(m_strides, m_shape);
    }

    template<typename Value>
    constexpr Value* Array<Value>::get() const noexcept { return m_ptr.get(); }

    template<typename Value>
    constexpr Value* Array<Value>::data() const noexcept { return m_ptr.get(); }

    template<typename Value>
    const Array<Value>& Array<Value>::eval() const {
        Stream::current(device()).synchronize();
        return *this;
    }

    template<typename Value>
    Array<Value> Array<Value>::release() noexcept {
        return std::exchange(*this, Array<Value>{});
    }

    template<typename Value>
    template<typename Int, typename>
    [[nodiscard]] NOA_HD constexpr Value& Array<Value>::operator[](Int offset) const noexcept {
        NOA_ASSERT(!empty() && dereferenceable() &&
                   static_cast<dim_t>(offset) <= indexing::at(m_shape - 1, m_strides));
        return m_ptr.get()[offset];
    }

    template<typename Value>
    template<typename Index0>
    [[nodiscard]] NOA_HD constexpr Value& Array<Value>::operator()(Index0 i0) const noexcept {
        NOA_ASSERT(dereferenceable() &&
                   static_cast<dim_t>(i0) < m_shape[0]);
        using accessor_t = typename View<Value, dim_t>::accessor_reference_type;
        return accessor_t(get(), m_strides.get())(i0);
    }

    template<typename Value>
    template<typename Index0, typename Index1>
    [[nodiscard]] NOA_HD constexpr Value& Array<Value>::operator()(Index0 i0, Index1 i1) const noexcept {
        NOA_ASSERT(dereferenceable() &&
                   static_cast<dim_t>(i0) < m_shape[0] &&
                   static_cast<dim_t>(i1) < m_shape[1]);
        using accessor_t = typename View<Value, dim_t>::accessor_reference_type;
        return accessor_t(get(), m_strides.get())(i0, i1);
    }

    template<typename Value>
    template<typename Index0, typename Index1, typename Index2>
    [[nodiscard]] NOA_HD constexpr Value& Array<Value>::operator()(Index0 i0, Index1 i1, Index2 i2) const noexcept {
        NOA_ASSERT(dereferenceable() &&
                   static_cast<dim_t>(i0) < m_shape[0] &&
                   static_cast<dim_t>(i1) < m_shape[1] &&
                   static_cast<dim_t>(i2) < m_shape[2]);
        using accessor_t = typename View<Value, dim_t>::accessor_reference_type;
        return accessor_t(get(), m_strides.get())(i0, i1, i2);
    }

    template<typename Value>
    template<typename Index0, typename Index1, typename Index2, typename Index3>
    [[nodiscard]] NOA_HD constexpr Value& Array<Value>::operator()(Index0 i0, Index1 i1, Index2 i2, Index3 i3) const noexcept {
        NOA_ASSERT(dereferenceable() &&
                   static_cast<dim_t>(i0) < m_shape[0] &&
                   static_cast<dim_t>(i1) < m_shape[1] &&
                   static_cast<dim_t>(i2) < m_shape[2] &&
                   static_cast<dim_t>(i3) < m_shape[3]);
        using accessor_t = typename View<Value, dim_t>::accessor_reference_type;
        return accessor_t(get(), m_strides.get())(i0, i1, i2, i3);
    }

    template<typename Value>
    constexpr const std::shared_ptr<Value[]>& Array<Value>::share() const noexcept { return m_ptr; }

    template<typename Value>
    template<typename Value0, typename Index>
    constexpr View<Value0, Index> Array<Value>::view() const noexcept {
        return View<Value, Index>(get(), Int4<Index>(m_shape), Int4<Index>(m_strides)).template as<Value0>();
    }

    template<typename Value>
    [[nodiscard]] constexpr auto Array<Value>::accessor() const noexcept {
        using output_t = Accessor<Value, 4, dim_t, AccessorTraits::DEFAULT>;
        return output_t(m_ptr, m_strides.get());
    }

    template<typename Value>
    template<typename Value0, int NDIM, typename Index, AccessorTraits TRAITS>
    [[nodiscard]] constexpr auto Array<Value>::accessor() const noexcept {
        return View<Value, Index>(get(), Int4<Index>(m_shape), Int4<Index>(m_strides))
                .template accessor<Value0, NDIM, Index, TRAITS>();
    }

    template<typename Value>
    void Array<Value>::to(const Array& output) const {
        NOA_CHECK(!this->empty() && !output.empty(), "Empty array detected");
        NOA_CHECK(!indexing::isOverlap(*this, output), "The input and output should not overlap");

        dim4_t input_strides = m_strides;
        if (!indexing::broadcast(m_shape, input_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      m_shape, output.shape());
        }

        const Device input_device = this->device();
        const Device output_device = output.device();

        if (input_device.cpu() && output_device.cpu()) {
            cpu::memory::copy(this->share(), input_strides,
                              output.share(), output.strides(),
                              output.shape(), Stream::current(input_device).cpu());
        } else if (output_device.cpu()) { // gpu -> cpu
            #ifdef NOA_ENABLE_CUDA
            Stream::current(output_device).synchronize();
            cuda::Stream& cuda_stream = Stream::current(input_device).cuda();
            cuda::memory::copy(this->share(), input_strides,
                               output.share(), output.strides(),
                               output.shape(), cuda_stream);
            cuda_stream.synchronize();
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        } else { // gpu -> gpu or cpu -> gpu
            #ifdef NOA_ENABLE_CUDA
            if (input_device != output_device)
                Stream::current(input_device).synchronize(); // wait for the input
            cuda::memory::copy(this->share(), input_strides,
                               output.share(), output.strides(),
                               output.shape(), Stream::current(output_device).cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename Value>
    Array<Value> Array<Value>::to(ArrayOption option) const {
        Array out(m_shape, option);
        to(out);
        return out;
    }

    template<typename Value>
    Array<Value> Array<Value>::copy() const {
        return to(m_options);
    }

    template<typename Value>
    template<typename Value0>
    Array<Value0> Array<Value>::as() const {
        const auto out = indexing::Reinterpret<Value, dim_t>(m_shape, m_strides, get()).template as<Value0>();
        return {std::shared_ptr<Value0[]>(m_ptr, out.ptr), out.shape, out.strides, options()};
    }

    template<typename Value>
    Array<Value> Array<Value>::as(Device::Type type) const {
        const Allocator alloc = m_options.allocator();
        if (type == Device::CPU && device().gpu()) { // see as CPU array
            NOA_CHECK(alloc == Allocator::PINNED ||
                      alloc == Allocator::MANAGED ||
                      alloc == Allocator::MANAGED_GLOBAL,
                      "GPU array with the allocator {} cannot be reinterpreted as a CPU array. "
                      "This is only supported for pinned and managed arrays", alloc);
            return {m_ptr, m_shape, m_strides, ArrayOption(m_options).device(Device(type))};

        } else if (type == Device::GPU && device().cpu()) { // see as GPU array
            NOA_CHECK(Device::any(Device::GPU), "No GPU detected");
            NOA_CHECK(alloc == Allocator::PINNED ||
                      alloc == Allocator::MANAGED ||
                      alloc == Allocator::MANAGED_GLOBAL,
                      "CPU array with the allocator {} cannot be reinterpreted as a GPU array. "
                      "This is only supported for pinned and managed arrays", alloc);
            Device gpu;
            #ifdef NOA_ENABLE_CUDA
            if (alloc == Allocator::PINNED || alloc == Allocator::MANAGED) {
                // NOTE: CUDA doesn't document what the attr.device is for managed memory.
                //       Hopefully this is the device against which the allocation was performed
                //       and not the current device.
                // NOTE: With "stream-attached" managed memory, it is up to the user to know what
                //       stream was used to perform the allocation.
                const cudaPointerAttributes attr = cuda::utils::getAttributes(m_ptr.get());
                gpu = Device(Device::GPU, attr.device, true);
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
            return {m_ptr, m_shape, m_strides, ArrayOption(m_options).device(gpu)};
        } else {
            return *this;
        }
    }

    template<typename Value>
    Array<Value> Array<Value>::reshape(dim4_t shape) const {
        dim4_t new_strides;
        if (!indexing::reshape(m_shape, m_strides, shape, new_strides))
            NOA_THROW("An array of shape {} cannot be reshaped to an array of shape {}", m_shape, shape);
        return {m_ptr, shape, new_strides, options()};
    }

    template<typename Value>
    Array<Value> Array<Value>::flat(int axis) const {
        dim4_t output_shape(1);
        output_shape[axis] = m_shape.elements();
        return reshape(output_shape);
    }

    template<typename Value>
    Array<Value> Array<Value>::permute(dim4_t permutation, bool copy) const {
        const dim4_t permuted_shape = indexing::reorder(m_shape, permutation);
        if (!copy)
            return Array<Value>(m_ptr, permuted_shape, indexing::reorder(m_strides, permutation), m_options);

        if constexpr (noa::traits::is_data_v<Value>) {
            Array<Value> out(permuted_shape, m_options);
            Stream& stream = Stream::current(device());
            if (device().cpu()) {
                cpu::memory::permute(m_ptr, m_strides, m_shape,
                                     out.share(), out.strides(),
                                     permutation, stream.cpu());
            } else {
                #ifdef NOA_ENABLE_CUDA
                cuda::memory::permute(m_ptr, m_strides, m_shape,
                                      out.share(), out.strides(),
                                      permutation, stream.cuda());
                #else
                NOA_THROW("No GPU backend detected");
                #endif
            }
            return out;
        } else {
            NOA_THROW("This type ({}) is not supported by memory::permute()", string::human<Value>());
        }
    }

    template<typename Value>
    Array<Value>& Array<Value>::operator=(std::nullptr_t) {
        *this = Array<Value>{};
        return *this;
    }

    template<typename Value>
    template<typename A, typename B, typename C, typename D, typename>
    constexpr Array<Value> Array<Value>::subregion(A&& i0, B&& i1, C&& i2, D&& i3) const {
        const indexing::Subregion indexer = indexing::Subregion(m_shape, m_strides)(i0, i1, i2, i3);
        return {shared_type(m_ptr, m_ptr.get() + indexer.offset()),
                indexer.shape(), indexer.strides(), m_options};
    }

    template<typename Value>
    constexpr Array<Value> Array<Value>::subregion(indexing::ellipsis_t) const {
        return *this;
    }

    template<typename Value>
    template<typename A, typename>
    constexpr Array<Value> Array<Value>::subregion(indexing::ellipsis_t, A&& i3) const {
        return subregion(indexing::full_extent_t{}, indexing::full_extent_t{}, indexing::full_extent_t{}, i3);
    }

    template<typename Value>
    template<typename A, typename B, typename>
    constexpr Array<Value> Array<Value>::subregion(indexing::ellipsis_t, A&& i2, B&& i3) const {
        return subregion(indexing::full_extent_t{}, indexing::full_extent_t{}, i2, i3);
    }

    template<typename Value>
    template<typename A, typename B, typename C, typename>
    constexpr Array<Value> Array<Value>::subregion(indexing::ellipsis_t, A&& i1, B&& i2, C&& i3) const {
        return subregion(indexing::full_extent_t{}, i1, i2, i3);
    }

    template<typename Value>
    void Array<Value>::alloc_() {
        const dim_t elements = m_shape.elements();
        if (!elements) {
            m_shape = 0;
            m_ptr = nullptr;
            return;
        }

        const Device device = m_options.device();
        switch (m_options.allocator()) {
            case Allocator::NONE:
                break;
            case Allocator::DEFAULT:
                if (device.cpu()) {
                    m_ptr = cpu::memory::PtrHost<value_type>::alloc(elements);
                } else {
                    #ifdef NOA_ENABLE_CUDA
                    DeviceGuard guard(device);
                    m_ptr = cuda::memory::PtrDevice<value_type>::alloc(elements);
                    #endif
                }
                break;
            case Allocator::DEFAULT_ASYNC:
                if (device.cpu()) {
                    m_ptr = cpu::memory::PtrHost<value_type>::alloc(elements);
                } else {
                    #ifdef NOA_ENABLE_CUDA
                    m_ptr = cuda::memory::PtrDevice<value_type>::alloc(elements, Stream::current(device).cuda());
                    #endif
                }
                break;
            case Allocator::PITCHED:
                if (device.cpu()) {
                    m_ptr = cpu::memory::PtrHost<value_type>::alloc(elements);
                } else {
                    #ifdef NOA_ENABLE_CUDA
                    DeviceGuard guard(device);
                    // PtrDevicePadded requires sizeof(T) <= 16 bytes.
                    // We could remove this restriction, but for now since it is only for
                    // static vectors and matrices, just switch to classic cudaMalloc.
                    if constexpr (noa::traits::is_data_v<value_type>) {
                        auto [ptr, pitch] = cuda::memory::PtrDevicePadded<value_type>::alloc(m_shape);
                        m_ptr = std::move(ptr);
                        m_strides = dim4_t{m_shape[0], m_shape[1], m_shape[2], pitch}.strides();
                    } else {
                        m_ptr = cuda::memory::PtrDevice<value_type>::alloc(elements);
                    }
                    #endif
                }
                break;
            case Allocator::PINNED: {
                if (device.cpu() && !Device::any(Device::GPU)) {
                    m_ptr = cpu::memory::PtrHost<value_type>::alloc(elements);
                } else {
                    #ifdef NOA_ENABLE_CUDA
                    DeviceGuard guard(device.gpu() ? device : Device::current(Device::GPU));
                    m_ptr = cuda::memory::PtrPinned<value_type>::alloc(elements);
                    #endif
                }
                break;
            }
            case Allocator::MANAGED: {
                if (device.cpu() && !Device::any(Device::GPU)) {
                    m_ptr = cpu::memory::PtrHost<value_type>::alloc(elements);
                } else {
                    #ifdef NOA_ENABLE_CUDA
                    const Device gpu = device.gpu() ? device : Device::current(Device::GPU);
                    const DeviceGuard guard(gpu); // could be helpful when retrieving device
                    cuda::Stream& cuda_stream = Stream::current(gpu).cuda();
                    m_ptr = cuda::memory::PtrManaged<value_type>::alloc(elements, cuda_stream);
                    #endif
                }
                break;
            }
            case Allocator::MANAGED_GLOBAL: {
                if (device.cpu() && !Device::any(Device::GPU)) {
                    m_ptr = cpu::memory::PtrHost<value_type>::alloc(elements);
                } else {
                    #ifdef NOA_ENABLE_CUDA
                    DeviceGuard guard(device.gpu() ? device : Device::current(Device::GPU));
                    m_ptr = cuda::memory::PtrManaged<value_type>::alloc(elements);
                    #endif
                }
                break;
            }
            case Allocator::CUDA_ARRAY:
                NOA_THROW("CUDA arrays are not supported by the Array allocator. See Texture instead");
            default:
                NOA_THROW("Allocator {} is not supported by the Array allocator", m_options.allocator());
        }
    }

    template<typename Value>
    void Array<Value>::validate_([[maybe_unused]] void* ptr, ArrayOption option) {
        const Allocator alloc = option.allocator();
        NOA_CHECK(alloc != Allocator::CUDA_ARRAY,
                  "CUDA arrays are not supported by the Array class. See Texture instead");
        NOA_CHECK(alloc != Allocator::NONE || ptr == nullptr, "Allocator {} is for nullptr only", Allocator::NONE);

        if (option.device().cpu()) {
            if (!Device::any(Device::GPU))
                return; // Everything is allocated using PtrHost
            #ifdef NOA_ENABLE_CUDA
            const cudaPointerAttributes attr = cuda::utils::getAttributes(ptr);
            switch (attr.type) {
                case cudaMemoryTypeUnregistered:
                    if (alloc != Allocator::DEFAULT &&
                        alloc != Allocator::DEFAULT_ASYNC &&
                        alloc != Allocator::PITCHED)
                        NOA_THROW("The entered allocator ({}) is not compatible the CPU-only (i.e. CUDA unregistered) "
                                  "pointer of this CPU array", alloc);
                    break;
                case cudaMemoryTypeHost:
                    if (alloc != Allocator::PINNED)
                        NOA_THROW("The entered allocator ({}) is not compatible with the underlying pinned pointer "
                                  "of this CPU array", alloc);
                    break;
                case cudaMemoryTypeDevice:
                    NOA_THROW("CPU array is pointing to a device pointer");
                case cudaMemoryTypeManaged:
                    if (alloc != Allocator::DEFAULT &&
                        alloc != Allocator::DEFAULT_ASYNC &&
                        alloc != Allocator::PITCHED &&
                        alloc != Allocator::MANAGED &&
                        alloc != Allocator::MANAGED_GLOBAL)
                        NOA_THROW("The entered allocator ({}) is not compatible with the (CUDA) managed pointer "
                                  "of this CPU array", alloc);
                    break;
            }
            #endif

        } else if (option.device().gpu()) {
            #ifdef NOA_ENABLE_CUDA
            const cudaPointerAttributes attr = cuda::utils::getAttributes(ptr);
            switch (attr.type) {
                case cudaMemoryTypeUnregistered:
                    NOA_THROW("GPU array is pointing to a host pointer");
                case cudaMemoryTypeHost:
                    if (alloc != Allocator::PINNED)
                        NOA_THROW("The entered allocator ({}) is not compatible with the underlying pinned pointer "
                                  "of this GPU array", alloc);
                    break;
                case cudaMemoryTypeDevice:
                    if (attr.device != option.device().id())
                        NOA_THROW("The device ID of this GPU array ({}) does not match the device of the underlying "
                                  "pointer ({})", option.device().id(), attr.device);
                    break;
                case cudaMemoryTypeManaged:
                    if (alloc != Allocator::DEFAULT &&
                        alloc != Allocator::DEFAULT_ASYNC &&
                        alloc != Allocator::PITCHED &&
                        alloc != Allocator::MANAGED &&
                        alloc != Allocator::MANAGED_GLOBAL)
                        NOA_THROW("The entered allocator ({}) is not compatible with the (CUDA) managed pointer "
                                  "of this GPU array", alloc);
                    break;
            }
            #endif
        }
    }
}

namespace noa::indexing {
    template<typename Value>
    Array<Value> broadcast(const Array<Value>& input, dim4_t shape) {
        dim4_t strides = input.strides();
        if (!broadcast(input.shape(), strides, shape))
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}", input.shape(), shape);
        return Array<Value>{input.share(), shape, strides, input.options()};
    }

    template<typename Value0, typename Value1>
    bool isOverlap(const Array<Value0>& lhs, const Array<Value1>& rhs) {
        if (lhs.empty() || rhs.empty())
            return false;

        return isOverlap(reinterpret_cast<uintptr_t>(lhs.get()),
                         reinterpret_cast<uintptr_t>(lhs.get() + at(lhs.shape() - 1, lhs.strides())),
                         reinterpret_cast<uintptr_t>(rhs.get()),
                         reinterpret_cast<uintptr_t>(rhs.get() + at(rhs.shape() - 1, rhs.strides())));
    }

    template<typename Int, typename Value, typename>
    constexpr dim4_t indexes(Int offset, const Array<Value>& array) {
        NOA_CHECK(!any(array.strides() == 0),
                  "Cannot retrieve the 4D index from a broadcast array. Got strides:{}",
                  array.strides());
        return indexing::indexes(offset, array.strides(), array.shape());
    }
}
