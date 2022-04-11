#pragma once

#ifndef NOA_UNIFIED_ARRAY_
#error "This is a private header"
#endif

namespace noa::details {
    template<typename T>
    void arrayCopy(const Array<T>& src, const Array<T>& dst);

    template<typename T>
    void arrayTranspose(const Array<T>& src, const Array<T>& dst, uint4_t permutation);
}

namespace noa {
    template<typename T>
    template<typename U>
    NOA_IH Array<U> Array<T>::as() const {
        const indexing::Reinterpret<U, size_t> out =
                indexing::Reinterpret<T, size_t>{m_shape, m_stride, get()}.template as<U>();
        return {std::shared_ptr<U[]>{m_ptr, out.ptr}, out.shape, out.stride, options()};
    }

    template<typename T>
    NOA_IH Array<T> Array<T>::as(Device::Type type) const {
        if (type == Device::CPU && device().gpu()) { // see as GPU array
            NOA_CHECK(m_options.allocator() != Allocator::DEFAULT &&
                      m_options.allocator() != Allocator::DEFAULT_ASYNC &&
                      m_options.allocator() != Allocator::PITCHED,
                      "GPU arrays with the allocator {} cannot be reinterpreted as CPU arrays",
                      m_options.allocator());
            return {m_ptr, m_shape, m_stride, ArrayOption{m_options}.device(Device{Device::CPU})};

        } else if (type == Device::GPU && device().cpu()) { // see as GPU array
            NOA_CHECK(Device::any(Device::GPU), "No GPU detected");
            NOA_CHECK(m_options.allocator() != Allocator::DEFAULT &&
                      m_options.allocator() != Allocator::DEFAULT_ASYNC &&
                      m_options.allocator() != Allocator::PITCHED,
                      "CPU arrays with the allocator {} cannot be reinterpreted as GPU arrays",
                      m_options.allocator());
            Device gpu;
            #ifdef NOA_ENABLE_CUDA
            if (m_options.allocator() == Allocator::PINNED || m_options.allocator() == Allocator::MANAGED) {
                // NOTE: CUDA doesn't document what the attr.device is for managed memory.
                //       Hopefully this is the device against which the allocation was performed
                //       and not the current device.
                // NOTE: With "stream-attached" managed memory, it is up to the user to know what
                //       stream was used to perform the allocation.
                const cudaPointerAttributes attr = cuda::util::getAttributes(m_ptr.get());
                gpu = Device{Device::GPU, attr.device, true};
                NOA_ASSERT((m_options.allocator() == Allocator::PINNED && attr.type == cudaMemoryTypeHost) ||
                           (m_options.allocator() == Allocator::MANAGED && attr.type == cudaMemoryTypeManaged));
                // TODO Add cudaPrefetchAsync when it is added to cuda::PtrManaged.

            } else if (m_options.allocator() == Allocator::MANAGED_GLOBAL) {
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
    NOA_IH Array<T> Array<T>::reshape(size4_t shape) const {
        if (contiguous()) {
            NOA_CHECK(shape.elements() == m_shape.elements(),
                      "An array with a shape of {} cannot be reshaped to {}",
                      m_shape, shape);
            return {m_ptr, shape, shape.stride(), options()};
        } else {
            NOA_THROW("TODO: Reshaping non-contiguous arrays is currently not supported");
            // see https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/TensorUtils.cpp
        }
    }

    template<typename T>
    Array<T> Array<T>::to(ArrayOption option, bool as_contiguous) const {
        Array out{m_shape, as_contiguous ? m_shape.stride() : m_stride, option};
        if constexpr (noa::traits::is_data_v<T>) {
            details::arrayCopy(*this, out);
        } else {
            // TODO Update when nvrtc
            NOA_CHECK(option.device().cpu(),
                      "This type ({}) is not supported by copy() on the GPU backend", string::human<T>());
            cpu::Stream& stream = Stream::current(option.device()).cpu();
            cpu::memory::copy(this->share(), out.share(), stream);
        }
        return out;
    }

    template<typename T>
    template<typename I>
    Array<T> Array<T>::permute(Int4<I> permutation, bool copy) const {
        const size4_t permuted_shape = indexing::reorder(m_shape, permutation);
        Array<T> out;
        if (copy) {
            if constexpr (!noa::traits::is_data_v<T>) {
                NOA_THROW("This type ({}) is not supported by transpose()", string::human<T>());
            } else {
                out = Array<T>{permuted_shape, permuted_shape.stride(), m_options};
                details::arrayTranspose(*this, out, uint4_t{permutation});
            }
        } else {
            out = Array<T>{m_ptr, permuted_shape, indexing::reorder(m_stride, permutation), m_options};
        }
        return out;
    }

    template<typename T>
    NOA_IH void Array<T>::alloc_(bool strided) { // m_shape and m_options should be set
        NOA_PROFILE_FUNCTION();
        size_t elements;
        if (strided) {
            // In case the strides are not ordered (e.g. permuted array).
            size4_t shape = m_shape;
            size4_t stride = m_stride;
            const auto order = indexing::order(stride);
            indexing::reorder(shape, order);
            indexing::reorder(stride, order);
            elements = shape[0] * stride[0];
        } else {
            elements = m_shape.elements();
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
                NOA_CHECK(!strided, "Pitched memory is not supported with user-specified strides");
                if (device.cpu()) {
                    m_ptr = cpu::memory::PtrHost<T>::alloc(elements);
                } else {
                    #ifdef NOA_ENABLE_CUDA
                    DeviceGuard guard(device);
                    auto [ptr, pitch] = cuda::memory::PtrDevicePadded<T>::alloc(m_shape);
                    m_ptr = std::move(ptr);
                    m_stride = size4_t{m_shape[0], m_shape[1], m_shape[2], pitch}.stride();
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
    NOA_IH void Array<T>::validate_(void* ptr, ArrayOption option) {
        const Allocator alloc = option.allocator();
        if (option.device().cpu() &&
            alloc != Allocator::DEFAULT &&
            alloc != Allocator::DEFAULT_ASYNC &&
            alloc != Allocator::PITCHED)
            return;

        #ifdef NOA_ENABLE_CUDA
        if (alloc == Allocator::PINNED ||
            alloc == Allocator::MANAGED ||
            alloc == Allocator::MANAGED_GLOBAL) {
            const cudaPointerAttributes attr = cuda::util::getAttributes(ptr);
            NOA_CHECK(option.device().cpu() || attr.device == option.device().id(),
                      "The entered GPU ID ({}) does not match the registered GPU ID for that pointer ({})",
                      attr.device, option.device().id());
            NOA_CHECK((option.allocator() == Allocator::PINNED && attr.type == cudaMemoryTypeHost) ||
                      (option.allocator() == Allocator::MANAGED && attr.type == cudaMemoryTypeManaged),
                      "The entered allocator ({}) does not match the registered allocator (cudaMemoryType:{}) ",
                      "for that pointer", option.allocator(), cudaMemoryTypeHost);
        }
        #else
        (void) ptr;
            return;
        #endif
    }
}
