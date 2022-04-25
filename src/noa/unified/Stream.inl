#ifndef NOA_UNIFIED_STREAM_
#error "Implementation header"
#endif

namespace noa {
    template<typename T>
    NOA_IH void* Stream::StreamModel<T>::addr() noexcept {
        return &stream;
    }

    template<typename T>
    NOA_IH const void* Stream::StreamModel<T>::addr() const noexcept {
        return &stream;
    }

    template<typename T>
    NOA_IH void Stream::StreamModel<T>::synchronize() {
        stream.synchronize();
    }

    template<typename T>
    NOA_IH bool Stream::StreamModel<T>::busy() {
        return stream.busy();
    }
}

namespace noa {
    template<size_t N, size_t A>
    NOA_IH Stream::StreamConcept* Stream::StreamStorage<N, A>::stream() noexcept {
        return reinterpret_cast<StreamConcept*>(&storage);
    }

    template<size_t N, size_t A>
    NOA_IH const Stream::StreamConcept* Stream::StreamStorage<N, A>::stream() const noexcept {
        return reinterpret_cast<const StreamConcept*>(&storage);
    }

    template<size_t N, size_t A>
    NOA_IH Stream::StreamStorage<N, A>::StreamStorage(const Stream::StreamStorage<N, A>& src) {
        if (!src.is_allocated)
            return;

        // FIXME Is there a better way of doing this?
        const StreamConcept* tmp = src.stream();
        if (const auto* ptr = dynamic_cast<const StreamModel<cpu::Stream>*>(tmp)) {
            const cpu::Stream& src_stream = *reinterpret_cast<const cpu::Stream*>(ptr->addr());
            this->emplace<StreamModel<cpu::Stream>>(src_stream); // copy
            return;
        }
        if (const auto* ptr = dynamic_cast<const StreamModel<cuda::Stream>*>(tmp)) {
            const cuda::Stream& src_stream = *reinterpret_cast<const cuda::Stream*>(ptr->addr());
            this->emplace<StreamModel<cuda::Stream>>(src_stream); // copy
            return;
        }
    }

    template<size_t N, size_t A>
    NOA_IH Stream::StreamStorage<N, A>::StreamStorage(Stream::StreamStorage<N, A>&& src) noexcept {
        if (src.is_allocated) {
            std::copy(src.storage, src.storage + N, storage);
            src.is_allocated = false;
            is_allocated = true;
        }
    }

    template<size_t N, size_t A>
    NOA_IH Stream::StreamStorage<N, A>&
    Stream::StreamStorage<N, A>::operator=(const Stream::StreamStorage<N, A>& src) {
        if (this != &src)
            *this = StreamStorage(src); // move
        return *this;
    }

    template<size_t N, size_t A>
    NOA_IH Stream::StreamStorage<N, A>&
    Stream::StreamStorage<N, A>::operator=(Stream::StreamStorage<N, A>&& src) noexcept {
        if (this != &src) {
            clear();
            if (src.is_allocated) {
                std::copy(src.storage, src.storage + N, storage);
                src.is_allocated = false;
                is_allocated = true;
            }
        }
        return *this;
    }

    template<size_t N, size_t A>
    template<typename T, typename... Args>
    NOA_IH void Stream::StreamStorage<N, A>::emplace(Args&& ... args) {
        static_assert(sizeof(T) <= N);
        static_assert(alignof(T) <= A);
        clear();
        new(this->stream()) T(std::forward<Args>(args)...);
        is_allocated = true;
    }

    template<size_t N, size_t A>
    NOA_IH void Stream::StreamStorage<N, A>::clear() noexcept {
        if (is_allocated) {
            stream()->~StreamConcept();
            is_allocated = false;
        }
    }

    template<size_t N, size_t A>
    NOA_IH Stream::StreamStorage<N, A>::StreamStorage::~StreamStorage() {
        clear();
    }
}

namespace noa {
    NOA_IH Stream::Stream(Device device, Mode mode) : m_device(device) {
        if (m_device.cpu()) {
            const auto cpu_mode = mode == Stream::Mode::ASYNC ? cpu::Stream::ASYNC : cpu::Stream::DEFAULT;
            m_storage.emplace<StreamModel<cpu::Stream>>(cpu::Stream(cpu_mode));
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::Device cuda_device(m_device.id());
            const auto cuda_mode = mode == Stream::Mode::ASYNC ? cuda::Stream::ASYNC : cuda::Stream::DEFAULT;
            m_storage.emplace<StreamModel<cuda::Stream>>(cuda::Stream(cuda_device, cuda_mode));
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    NOA_IH Stream::Stream(T&& stream) {
        m_storage.emplace<StreamModel<T>>(std::forward<T>(stream));
    }

    NOA_IH void Stream::synchronize() {
        m_storage.stream()->synchronize();
    }

    NOA_IH bool Stream::busy() {
        return m_storage.stream()->busy();
    }

    NOA_IH Device Stream::device() const noexcept {
        return m_device;
    }

    NOA_IH cpu::Stream& Stream::cpu() {
        if (!m_device.cpu())
            NOA_THROW("The stream is not a CPU stream");

        StreamConcept* tmp = m_storage.stream();
        auto* d = static_cast<StreamModel<cpu::Stream>*>(tmp); // dynamic_cast?
        return *reinterpret_cast<cpu::Stream*>(d->addr());
    }

    NOA_IH gpu::Stream& Stream::gpu() {
        #ifdef NOA_ENABLE_CUDA
        return cuda();
        #else
        NOA_THROW("No GPU backend detected");
        #endif
    }

    NOA_IH cuda::Stream& Stream::cuda() {
        if (!m_device.gpu())
            NOA_THROW("The stream is not a GPU stream");

        #ifdef NOA_ENABLE_CUDA
        StreamConcept* tmp = m_storage.stream();
        auto* d = static_cast<StreamModel<cuda::Stream>*>(tmp); // dynamic_cast?
        return *reinterpret_cast<cuda::Stream*>(d->addr());
        #else
        NOA_THROW("No GPU backend detected");
        #endif
    }
}
