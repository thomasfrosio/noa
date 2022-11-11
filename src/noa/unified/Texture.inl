#ifndef NOA_UNIFIED_TEXTURE_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

namespace noa {
    template<typename value_t>
    Texture<value_t>::Texture(const Array<value_t>& array, Device device_target,
                              InterpMode interp_mode, BorderMode border_mode,
                              value_t cvalue, bool layered, bool prefilter)
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
            m_texture = cpu::Texture<value_t>{array.strides(), array.share(), cvalue};
            m_options = array.options();

        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<value_t>) >= 8) {
                NOA_THROW("Double-precision textures are not supported by the CUDA backend");
            } else {
                NOA_CHECK(!array.empty(), "Empty array detected");

                using namespace noa::cuda::memory;
                const DeviceGuard guard(device_target);

                cuda::Texture<value_t> texture{
                        PtrArray<value_t>::alloc(array.shape(), layered ? cudaArrayLayered : cudaArrayDefault),
                        PtrTexture::alloc(texture.array.get(), interp_mode, border_mode),
                        layered
                };

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

    template<typename value_t>
    Texture<value_t>::Texture(dim4_t shape, Device device_target,
                              InterpMode interp_mode, BorderMode border_mode,
                              value_t cvalue, bool layered)
            : m_shape(shape), m_interp(interp_mode), m_border(border_mode) {

        if (device_target.cpu()) {
            m_texture = cpu::Texture<value_t>{dim4_t{}, nullptr, cvalue};
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<value_t>) >= 8) {
                NOA_THROW("Double-precision textures are not supported by the CUDA backend");
            } else {

                using namespace noa::cuda::memory;
                const DeviceGuard guard(device_target);

                cuda::Texture<value_t> texture{
                        PtrArray<value_t>::alloc(shape, layered ? cudaArrayLayered : cudaArrayDefault),
                        PtrTexture::alloc(texture.array.get(), interp_mode, border_mode),
                        layered
                };
                m_texture = texture;
                m_options = ArrayOption{device_target, Allocator::CUDA_ARRAY};
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename value_t>
    constexpr ArrayOption Texture<value_t>::options() const noexcept { return m_options; }

    template<typename value_t>
    constexpr Device Texture<value_t>::device() const noexcept { return m_options.device(); }

    template<typename value_t>
    constexpr Allocator Texture<value_t>::allocator() const noexcept { return m_options.allocator(); }

    template<typename value_t>
    bool Texture<value_t>::empty() const noexcept {
        return std::holds_alternative<std::monostate>(m_texture);
    }

    template<typename value_t>
    const dim4_t& Texture<value_t>::shape() const noexcept {
        return m_shape;
    }

    template<typename value_t>
    const dim4_t Texture<value_t>::strides() const {
        if (device().cpu())
            return cpu().strides;
        else
            return m_shape.strides<'C'>();
    }

    template<typename value_t>
    template<char ORDER>
    bool Texture<value_t>::contiguous() const noexcept {
        if (device().cpu())
            return indexing::areContiguous<ORDER>(cpu().strides, m_shape);
        else
            return ORDER == 'C' || ORDER == 'c';
    }

    template<typename value_t>
    cpu::Texture<value_t>& Texture<value_t>::cpu() {
        auto* ptr = std::get_if<cpu::Texture<value_t>>(&m_texture);
        if (!ptr)
            NOA_THROW("Texture is not initialized or trying to retrieve at CPU texture from a GPU texture");
        return *ptr;
    }

    template<typename value_t>
    const cpu::Texture<value_t>& Texture<value_t>::cpu() const {
        auto* ptr = std::get_if<cpu::Texture<value_t>>(&m_texture);
        if (!ptr)
            NOA_THROW("Texture is not initialized or trying to retrieve at CPU texture from a GPU texture");
        return *ptr;
    }

    template<typename value_t>
    gpu::Texture<value_t>& Texture<value_t>::gpu() {
        #ifdef NOA_ENABLE_CUDA
        return this->cuda();
        #else
        NOA_THROW("No GPU backend detected");
        #endif
    }

    template<typename value_t>
    const gpu::Texture<value_t>& Texture<value_t>::gpu() const {
        #ifdef NOA_ENABLE_CUDA
        return this->cuda();
        #else
        NOA_THROW("No GPU backend detected");
        #endif
    }

    template<typename value_t>
    cuda::Texture<value_t>& Texture<value_t>::cuda() {
        #ifdef NOA_ENABLE_CUDA
        auto* ptr = std::get_if<cuda::Texture<value_t>>(&m_texture);
        if (!ptr)
            NOA_THROW("Texture is not initialized or trying to retrieve at GPU texture from a CPU texture");
        return *ptr;
        #else
        NOA_THROW("No GPU backend detected");
        #endif
    }

    template<typename value_t>
    const cuda::Texture<value_t>& Texture<value_t>::cuda() const {
        #ifdef NOA_ENABLE_CUDA
        auto* ptr = std::get_if<cuda::Texture<value_t>>(&m_texture);
        if (!ptr)
            NOA_THROW("Texture is not initialized or trying to retrieve at GPU texture from a CPU texture");
        return *ptr;
        #else
        NOA_THROW("No GPU backend detected");
        #endif
    }

    template<typename value_t>
    InterpMode Texture<value_t>::interp() const noexcept {
        return m_interp;
    }

    template<typename value_t>
    BorderMode Texture<value_t>::border() const noexcept {
        return m_border;
    }

    template<typename value_t>
    bool Texture<value_t>::layered() const {
        if (device().cpu())
            return true;
        else
            return this->cuda().layered;
    }

    template<typename value_t>
    Texture<value_t> Texture<value_t>::release() noexcept {
        return std::exchange(*this, Texture<value_t>{});
    }

    template<typename value_t>
    void Texture<value_t>::update(const Array<value_t>& array, bool prefilter) {
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
            cpu::Texture<value_t>& cpu_texture = this->cpu();
            cpu_texture.strides = array.strides();
            cpu_texture.ptr = array.share();
            m_options = array.options();

        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<value_t>) >= 8) {
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
