#ifndef NOA_UNIFIED_TEXTURE_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

namespace noa {
    template<typename Value>
    Texture<Value>::Texture(const Array<value_type>& array, Device device_target,
                            InterpMode interp_mode, BorderMode border_mode,
                            value_type cvalue, bool layered, bool prefilter)
            : m_shape(array.shape()), m_interp(interp_mode), m_border(border_mode) {

        if (prefilter &&
            (interp_mode == INTERP_CUBIC_BSPLINE ||
             interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
            geometry::bspline::prefilter(array, array);
        }

        if (device_target.cpu()) {
            NOA_CHECK(array.device() == device_target,
                      "CPU textures can only be constructed/updated from CPU arrays, but got array device {}",
                      array.device());
            m_texture = cpu::Texture<value_type>{array.strides(), array.share(), cvalue};
            m_options = array.options();

        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<value_type>) >= 8) {
                NOA_THROW("Double-precision textures are not supported by the CUDA backend");
            } else {
                NOA_CHECK(!array.empty(), "Empty array detected");

                using namespace noa::cuda::memory;
                const DeviceGuard guard(device_target);

                cuda::Texture<value_type> texture;
                texture.array = PtrArray<value_type>::alloc(array.shape(), layered ? cudaArrayLayered : cudaArrayDefault);
                texture.texture = PtrTexture::alloc(texture.array.get(), interp_mode, border_mode);

                Stream& stream = Stream::current(device_target);
                if (device_target != array.device())
                    array.eval();
                cuda::memory::copy(array.share(), array.strides(), texture.array, array.shape(), stream.cuda());

                m_texture = texture;
                m_options = ArrayOption{device_target, Allocator::CUDA_ARRAY};
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename Value>
    Texture<Value>::Texture(dim4_t shape, Device device_target,
                            InterpMode interp_mode, BorderMode border_mode,
                            value_type cvalue, bool layered)
            : m_shape(shape), m_interp(interp_mode), m_border(border_mode) {

        if (device_target.cpu()) {
            m_texture = cpu::Texture<value_type>{dim4_t{}, nullptr, cvalue};
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<value_type>) >= 8) {
                NOA_THROW("Double-precision textures are not supported by the CUDA backend");
            } else {

                using namespace noa::cuda::memory;
                const DeviceGuard guard(device_target);

                cuda::Texture<value_type> texture;
                texture.array = PtrArray<value_type>::alloc(shape, layered ? cudaArrayLayered : cudaArrayDefault);
                texture.texture = PtrTexture::alloc(texture.array.get(), interp_mode, border_mode);
                m_texture = texture;
                m_options = ArrayOption{device_target, Allocator::CUDA_ARRAY};
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename Value>
    constexpr ArrayOption Texture<Value>::options() const noexcept { return m_options; }

    template<typename Value>
    constexpr Device Texture<Value>::device() const noexcept { return m_options.device(); }

    template<typename Value>
    constexpr Allocator Texture<Value>::allocator() const noexcept { return m_options.allocator(); }

    template<typename Value>
    bool Texture<Value>::empty() const noexcept {
        return std::holds_alternative<std::monostate>(m_texture);
    }

    template<typename Value>
    const dim4_t& Texture<Value>::shape() const noexcept {
        return m_shape;
    }

    template<typename Value>
    dim4_t Texture<Value>::strides() const {
        if (device().cpu())
            return cpu().strides;
        else
            return m_shape.strides<'C'>();
    }

    template<typename Value>
    template<char ORDER>
    bool Texture<Value>::contiguous() const noexcept {
        if (device().cpu())
            return indexing::areContiguous<ORDER>(cpu().strides, m_shape);
        else
            return ORDER == 'C' || ORDER == 'c';
    }

    template<typename Value>
    const Texture<Value>& Texture<Value>::eval() const {
        Stream::current(device()).synchronize();
        return *this;
    }

    template<typename Value>
    cpu::Texture<Value>& Texture<Value>::cpu() {
        auto* ptr = std::get_if<cpu::Texture<value_type>>(&m_texture);
        if (!ptr)
            NOA_THROW("Texture is not initialized or trying to retrieve at CPU texture from a GPU texture");
        return *ptr;
    }

    template<typename Value>
    const cpu::Texture<Value>& Texture<Value>::cpu() const {
        auto* ptr = std::get_if<cpu::Texture<value_type>>(&m_texture);
        if (!ptr)
            NOA_THROW("Texture is not initialized or trying to retrieve at CPU texture from a GPU texture");
        return *ptr;
    }

    template<typename Value>
    gpu::Texture<Value>& Texture<Value>::gpu() {
        #ifdef NOA_ENABLE_CUDA
        return this->cuda();
        #else
        NOA_THROW("No GPU backend detected");
        #endif
    }

    template<typename Value>
    const gpu::Texture<Value>& Texture<Value>::gpu() const {
        #ifdef NOA_ENABLE_CUDA
        return this->cuda();
        #else
        NOA_THROW("No GPU backend detected");
        #endif
    }

    template<typename Value>
    cuda::Texture<Value>& Texture<Value>::cuda() {
        #ifdef NOA_ENABLE_CUDA
        auto* ptr = std::get_if<cuda::Texture<value_type>>(&m_texture);
        if (!ptr)
            NOA_THROW("Texture is not initialized or trying to retrieve at GPU texture from a CPU texture");
        return *ptr;
        #else
        NOA_THROW("No GPU backend detected");
        #endif
    }

    template<typename Value>
    const cuda::Texture<Value>& Texture<Value>::cuda() const {
        #ifdef NOA_ENABLE_CUDA
        auto* ptr = std::get_if<cuda::Texture<value_type>>(&m_texture);
        if (!ptr)
            NOA_THROW("Texture is not initialized or trying to retrieve at GPU texture from a CPU texture");
        return *ptr;
        #else
        NOA_THROW("No GPU backend detected");
        #endif
    }

    template<typename Value>
    InterpMode Texture<Value>::interp() const noexcept {
        return m_interp;
    }

    template<typename Value>
    BorderMode Texture<Value>::border() const noexcept {
        return m_border;
    }

    template<typename Value>
    bool Texture<Value>::layered() const {
        if (device().cpu()) {
            return true;
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::memory::PtrArray<value_type>::isLayered(this->cuda().array.get());
            #else
            return false;
            #endif
        }
    }

    template<typename Value>
    Texture<Value> Texture<Value>::release() noexcept {
        return std::exchange(*this, Texture<value_type>{});
    }

    template<typename Value>
    void Texture<Value>::update(const Array<value_type>& array, bool prefilter) {
        NOA_CHECK(!empty(), "Trying to update an empty texture is not allowed. Create the texture first.");
        NOA_CHECK(!array.empty(), "Empty array detected");
        NOA_CHECK(all(array.shape() == m_shape), // TODO Broadcast?
                  "The input array should have the same shape as the texture {}, but got {}", m_shape, array.shape());

        if (prefilter &&
            (m_interp == INTERP_CUBIC_BSPLINE ||
             m_interp == INTERP_CUBIC_BSPLINE_FAST)) {
            geometry::bspline::prefilter(array, array);
        }

        const Device device_target = device();
        if (device_target.cpu()) {
            NOA_CHECK(array.device() == device_target,
                      "CPU textures can only be constructed/updated from CPU arrays, but got array device {}",
                      array.device());
            cpu::Texture<value_type>& cpu_texture = this->cpu();
            cpu_texture.strides = array.strides();
            cpu_texture.ptr = array.share();
            m_options = array.options();

        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<value_type>) >= 8) {
                NOA_THROW("Double-precision textures are not supported by the CUDA backend");
            } else {
                if (device_target != array.device())
                    array.eval();

                Stream& stream = Stream::current(device_target);
                cuda::memory::copy(
                        array.share(), array.strides(),
                        this->cuda().array,
                        m_shape, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
