#pragma once
#ifndef NOA_UNIFIED_ARRAY_
#error "Cannot include this private header"
#endif

namespace noa::details {
    template<typename T, typename U, typename BinaryOp, typename>
    void arrayAssign(const Array<T>& lhs, U rhs, BinaryOp binary_op) {
        const Device device = lhs.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::ewise(lhs.share(), lhs.stride(), rhs,
                             lhs.share(), lhs.stride(), lhs.shape(),
                             binary_op, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::math::details::is_valid_ewise_binary_v<T, U, T, BinaryOp>) {
                cuda::math::ewise(lhs.share(), lhs.stride(), rhs,
                                  lhs.share(), lhs.stride(), lhs.shape(),
                                  binary_op, stream.cuda());
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See noa::cuda::math::ewise(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename U, typename BinaryOp>
    void arrayAssign(const Array<T>& lhs, const Array<U>& rhs, BinaryOp binary_op) {
        size4_t rhs_stride = rhs.stride();
        if (!indexing::broadcast(rhs.shape(), rhs_stride, lhs.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      rhs.shape(), lhs.shape());
        }

        const Device device = lhs.device();
        NOA_CHECK(device == rhs.device(),
                  "The input and output arrays must be on the same device, but got input:{} and output:{}",
                  rhs.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::ewise(lhs.share(), lhs.stride(),
                             rhs.share(), rhs_stride,
                             lhs.share(), lhs.stride(), lhs.shape(),
                             binary_op, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::math::details::is_valid_ewise_binary_v<T, U, T, BinaryOp>) {
                cuda::math::ewise(lhs.share(), lhs.stride(),
                                  rhs.share(), rhs_stride,
                                  lhs.share(), lhs.stride(), lhs.shape(),
                                  binary_op, stream.cuda());
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See noa::cuda::math::ewise(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}

namespace noa {
    template<typename T>
    constexpr Array<T>::Array(size_t elements, ArrayOption option)
            : m_shape(size4_t{1, 1, 1, elements}),
              m_stride(size4_t{elements, elements, elements, 1}),
              m_options(option) { alloc_(); }

    template<typename T>
    constexpr Array<T>::Array(size4_t shape, ArrayOption option)
            : m_shape(shape), m_stride(shape.stride()), m_options(option) { alloc_(); }

    template<typename T>
    constexpr Array<T>::Array(T* data, size_t elements, ArrayOption option)
            : m_shape(size4_t{1, 1, 1, elements}),
              m_stride(size4_t{elements, elements, elements, 1}),
              m_ptr(data, [](void*) {}),
              m_options(option) {
        validate_(data, option);
    }

    template<typename T>
    constexpr Array<T>::Array(T* data, size4_t shape, size4_t stride, ArrayOption option)
            : m_shape(shape), m_stride(stride), m_ptr(data, [](void*) {}), m_options(option) {
        validate_(data, option);
    }

    template<typename T>
    constexpr Array<T>::Array(shared_t<T[]> data, size_t elements, ArrayOption option)
            : m_shape(size4_t{1, 1, 1, elements}),
              m_stride(size4_t{elements, elements, elements, 1}),
              m_ptr(std::move(data)),
              m_options(option) {
        validate_(m_ptr.get(), option);
    }

    template<typename T>
    constexpr Array<T>::Array(shared_t<T[]> data, size4_t shape, size4_t stride, ArrayOption option)
            : m_shape(shape), m_stride(stride), m_ptr(std::move(data)), m_options(option) {
        validate_(m_ptr.get(), option);
    }

    template<typename T>
    constexpr ArrayOption Array<T>::options() const noexcept { return m_options; }

    template<typename T>
    constexpr Device Array<T>::device() const noexcept { return m_options.device(); }

    template<typename T>
    constexpr Allocator Array<T>::allocator() const noexcept { return m_options.allocator(); }

    template<typename T>
    constexpr bool Array<T>::dereferencable() const noexcept {
        return m_options.device().cpu() || allocator() == Allocator::PINNED ||
               allocator() == Allocator::MANAGED || allocator() == Allocator::MANAGED_GLOBAL;
    }

    template<typename T>
    bool Array<T>::empty() const noexcept { return !m_ptr || !m_shape.elements(); }

    template<typename T>
    const size4_t& Array<T>::shape() const noexcept { return m_shape; }

    template<typename T>
    const size4_t& Array<T>::stride() const noexcept { return m_stride; }

    template<typename T>
    bool4_t Array<T>::contiguous() const noexcept {
        return indexing::isContiguous(m_stride, m_shape);
    }

    template<typename T>
    constexpr T* Array<T>::get() noexcept { return m_ptr.get(); }

    template<typename T>
    constexpr const T* Array<T>::get() const noexcept { return m_ptr.get(); }

    template<typename T>
    constexpr T* Array<T>::data() noexcept { return m_ptr.get(); }

    template<typename T>
    constexpr const T* Array<T>::data() const noexcept { return m_ptr.get(); }

    template<typename T>
    constexpr const std::shared_ptr<T[]>& Array<T>::share() const noexcept { return m_ptr; }

    template<typename T>
    template<typename I>
    constexpr View<T, I> Array<T>::view() noexcept {
        return {get(), Int4<I>{m_shape}, Int4<I>{m_stride}};
    }

    template<typename T>
    template<typename I>
    constexpr View<const T, I> Array<T>::view() const noexcept {
        return {get(), Int4<I>{m_shape}, Int4<I>{m_stride}};
    }

    template<typename T>
    const Array<T>& Array<T>::eval() const {
        Stream::current(device()).synchronize();
        return *this;
    }

    template<typename T>
    void Array<T>::to(const Array& output) const {
        size4_t input_stride = m_stride;
        if (!indexing::broadcast(m_shape, input_stride, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      m_shape, output.shape());
        }

        const Device input_device = this->device();
        const Device output_device = output.device();

        if (input_device.cpu() && output_device.cpu()) {
            cpu::memory::copy(this->share(), input_stride,
                              output.share(), output.stride(),
                              output.shape(), Stream::current(input_device).cpu());
        } else if (output_device.cpu()) { // gpu -> cpu
            #ifdef NOA_ENABLE_CUDA
            Stream::current(output_device).synchronize();
            cuda::Stream& cuda_stream = Stream::current(input_device).cuda();
            cuda::memory::copy(this->share(), input_stride,
                               output.share(), output.stride(),
                               output.shape(), cuda_stream);
            cuda_stream.synchronize();
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        } else { // gpu -> gpu or cpu -> gpu
            #ifdef NOA_ENABLE_CUDA
            if (input_device != output_device)
                Stream::current(input_device).synchronize(); // wait for the input
            cuda::memory::copy(this->share(), input_stride,
                               output.share(), output.stride(),
                               output.shape(), Stream::current(output_device).cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T>
    Array<T> Array<T>::to(ArrayOption option) const {
        Array out{m_shape, option};
        to(out);
        return out;
    }

    template<typename T>
    template<typename U>
    Array<U> Array<T>::as() const {
        const auto out = indexing::Reinterpret<T, size_t>{m_shape, m_stride, get()}.template as<U>();
        return {std::shared_ptr<U[]>{m_ptr, out.ptr}, out.shape, out.stride, options()};
    }

    template<typename T>
    template<typename U>
    Array<U> Array<T>::as() {
        const auto out = indexing::Reinterpret<T, size_t>{m_shape, m_stride, get()}.template as<U>();
        return {std::shared_ptr<U[]>{m_ptr, out.ptr}, out.shape, out.stride, options()};
    }

    template<typename T>
    Array<T> Array<T>::as(Device::Type type) const {
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

    template<typename T>
    Array<T> Array<T>::reshape(size4_t shape) const {
        size4_t new_stride;
        if (!indexing::reshape(m_shape, m_stride, shape, new_stride))
            NOA_THROW("An array of shape {} cannot be reshaped to an array of shape {}", m_shape, shape);
        return {m_ptr, shape, new_stride, options()};
    }

    template<typename T>
    Array<T> Array<T>::permute(uint4_t permutation, bool copy) const {
        const size4_t permuted_shape = indexing::reorder(m_shape, permutation);
        if (!copy)
            return Array<T>{m_ptr, permuted_shape, indexing::reorder(m_stride, permutation), m_options};

        if constexpr (noa::traits::is_data_v<T>) {
            Array<T> out{permuted_shape, m_options};
            Stream& stream = Stream::current(device());
            if (device().cpu()) {
                cpu::memory::transpose(m_ptr, m_stride, m_shape,
                                       out.share(), out.stride(),
                                       permutation, stream.cpu());
            } else {
                #ifdef NOA_ENABLE_CUDA
                cuda::memory::transpose(m_ptr, m_stride, m_shape,
                                        out.share(), out.stride(),
                                        permutation, stream.cuda());
                #else
                NOA_THROW("No GPU backend detected");
                #endif
            }
            return out;
        } else {
            NOA_THROW("This type ({}) is not supported by memory::transpose()", string::human<T>());
        }
    }

    template<typename T>
    Array<T>& Array<T>::operator=(T value) {
        Stream& stream = Stream::current(device());
        if (device().cpu()) {
            cpu::memory::set(m_ptr, m_stride, m_shape, value, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::memory::set(m_ptr, m_stride, m_shape, value, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T>
    Array<T>& Array<T>::operator+=(T value) {
        details::arrayAssign(*this, value, math::plus_t{});
        return *this;
    }

    template<typename T>
    Array<T>& Array<T>::operator-=(T value) {
        details::arrayAssign(*this, value, math::minus_t{});
        return *this;
    }

    template<typename T>
    Array<T>& Array<T>::operator*=(T value) {
        details::arrayAssign(*this, value, math::multiply_t{});
        return *this;
    }

    template<typename T>
    Array<T>& Array<T>::operator/=(T value) {
        details::arrayAssign(*this, value, math::divide_t{});
        return *this;
    }

    template<typename T>
    template<typename U>
    Array<T>& Array<T>::operator+=(const Array<U>& array) {
        details::arrayAssign(*this, array, math::plus_t{});
        return *this;
    }

    template<typename T>
    template<typename U>
    Array<T>& Array<T>::operator-=(const Array<U>& array) {
        details::arrayAssign(*this, array, math::minus_t{});
        return *this;
    }

    template<typename T>
    template<typename U>
    Array<T>& Array<T>::operator*=(const Array<U>& array) {
        details::arrayAssign(*this, array, math::multiply_t{});
        return *this;
    }

    template<typename T>
    template<typename U>
    Array<T>& Array<T>::operator/=(const Array<U>& array) {
        details::arrayAssign(*this, array, math::divide_t{});
        return *this;
    }

    template<typename T>
    template<typename U, typename>
    Array<T>& Array<T>::operator+=(U value) {
        details::arrayAssign(*this, static_cast<traits::value_type_t<T>>(value), math::plus_t{});
        return *this;
    }

    template<typename T>
    template<typename U, typename>
    Array<T>& Array<T>::operator-=(U value) {
        details::arrayAssign(*this, static_cast<traits::value_type_t<T>>(value), math::minus_t{});
        return *this;
    }

    template<typename T>
    template<typename U, typename>
    Array<T>& Array<T>::operator*=(U value) {
        details::arrayAssign(*this, static_cast<traits::value_type_t<T>>(value), math::multiply_t{});
        return *this;
    }

    template<typename T>
    template<typename U, typename>
    Array<T>& Array<T>::operator/=(U value) {
        details::arrayAssign(*this, static_cast<traits::value_type_t<T>>(value), math::divide_t{});
        return *this;
    }

    template<typename T>
    template<typename A, typename B, typename C, typename D, typename>
    constexpr Array<T> Array<T>::subregion(A&& i0, B&& i1, C&& i2, D&& i3) const {
        const indexing::Subregion<int64_t> indexer =
                indexing::Subregion<int64_t>{long4_t{m_shape}, long4_t{m_stride}}(i0, i1, i2, i3);
        return {std::shared_ptr<T[]>{m_ptr, m_ptr.get() + indexer.offset},
                size4_t{indexer.shape}, size4_t{indexer.stride}, m_options};
    }

    template<typename T>
    constexpr Array<T> Array<T>::subregion(indexing::ellipsis_t) const {
        return *this;
    }

    template<typename T>
    template<typename A, typename>
    constexpr Array<T> Array<T>::subregion(indexing::ellipsis_t, A&& i3) const {
        return subregion(indexing::full_extent_t{}, indexing::full_extent_t{}, indexing::full_extent_t{}, i3);
    }

    template<typename T>
    template<typename A, typename B, typename>
    constexpr Array<T> Array<T>::subregion(indexing::ellipsis_t, A&& i2, B&& i3) const {
        return subregion(indexing::full_extent_t{}, indexing::full_extent_t{}, i2, i3);
    }

    template<typename T>
    template<typename A, typename B, typename C, typename>
    constexpr Array<T> Array<T>::subregion(indexing::ellipsis_t, A&& i1, B&& i2, C&& i3) const {
        return subregion(indexing::full_extent_t{}, i1, i2, i3);
    }

    template<typename T>
    void Array<T>::alloc_() {
        NOA_PROFILE_FUNCTION();
        const size_t elements = m_shape.elements();
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

    template<typename T>
    void Array<T>::validate_(void* ptr, ArrayOption option) {
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
}

namespace noa::indexing {
    template<typename T>
    Array<T> broadcast(const Array<T>& input, size4_t shape) {
        size4_t stride = input.stride();
        if (!broadcast(input.shape(), stride, shape))
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}", input.shape(), shape);
        return Array<T>{input.share(), shape, stride, input.options()};
    }
}
