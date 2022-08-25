#ifndef NOA_UNIFIED_TEXTURE_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

namespace noa {
    template<typename T>
    Texture<T>::Texture(const Array<T>& array, Device device_target,
                        InterpMode interp_mode, BorderMode border_mode,
                        T cvalue, bool prefilter)
            : m_shape(array.shape()), m_interp(interp_mode), m_border(border_mode) {

        if (prefilter &&
            (interp_mode == INTERP_CUBIC_BSPLINE ||
             interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
            geometry::bspline::prefilter(array, array);
        }

        if (device_target.cpu()) {
            NOA_CHECK(array.device() == device_target,
                      "CPU textures can only be constructed from CPU arrays, but got input device {}", array.device());
            m_texture = cpu::Texture<T>{array.strides(), array.share(), cvalue};
            m_options = array.options();

        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<T>) >= 8) {
                NOA_THROW("Double-precision textures are not supported by the CUDA backend");
            } else {
                NOA_CHECK(array.shape().ndim() <= 3, "CUDA textures cannot be batched");
                NOA_CHECK(indexing::isRightmost(array.strides()) &&
                          indexing::isContiguous(array.strides(), array.shape())[1] && array.strides()[3] == 1,
                          "The depth and width dimension of the array should be contiguous, but got shape {} "
                          "and strides {}", array.shape(), array.strides());

                using namespace noa::cuda::memory;
                const size3_t shape_3d(array.shape().get(1));
                m_texture = cuda::Texture<T>{};
                m_texture.array = PtrArray<T>::alloc(shape_3d);
                m_texture.texture = PtrTexture::alloc(m_texture.array.get(), interp_mode, border_mode);
                m_options = ArrayOption{device_target, Allocator::CUDA_ARRAY};

                if (device_target != array.device())
                    array.eval();

                Stream& stream = Stream::current(device_target);
                cuda::memory::copy(array.share(), array.strides()[2], m_texture.array, shape_3d, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T>
    constexpr ArrayOption Texture<T>::options() const noexcept { return m_options; }

    template<typename T>
    constexpr Device Texture<T>::device() const noexcept { return m_options.device(); }

    template<typename T>
    constexpr Allocator Texture<T>::allocator() const noexcept { return m_options.allocator(); }

    template<typename T>
    bool Texture<T>::empty() const noexcept {
        return std::holds_alternative<std::monostate>(m_texture);
    }

    template<typename T>
    const size4_t& Texture<T>::shape() const noexcept {
        return m_shape;
    }

    template<typename T>
    const size4_t Texture<T>::strides() const {
        if (device().cpu())
            return cpu().strides;
        else
            return m_shape.strides<'C'>();
    }

    template<typename T>
    template<char ORDER>
    bool Texture<T>::contiguous() const noexcept {
        if (device().cpu())
            return indexing::areContiguous<ORDER>(cpu().strides, m_shape);
        else
            return ORDER == 'C' || ORDER == 'c';
    }

    template<typename T>
    cpu::Texture<T>& Texture<T>::cpu() {
        if (!device().cpu())
            NOA_THROW("");

        auto* ptr = std::get_if<cpu::Texture<T>>(m_texture);
        if (!ptr)
            NOA_THROW("empty");
        return *ptr;
    }

    template<typename T>
    gpu::Texture<T>& Texture<T>::gpu() {
    #ifdef NOA_ENABLE_CUDA
        return cuda();
    #else
        NOA_THROW("No GPU backend detected");
    #endif
    }

    template<typename T>
    cuda::Texture<T>& Texture<T>::cuda() {
        if (!device().gpu())
            NOA_THROW("");

        #ifdef NOA_ENABLE_CUDA
        auto* ptr = std::get_if<cuda::Texture<T>>(m_texture);
        if (!ptr)
            NOA_THROW("empty");
        return *ptr;
        #else
        NOA_THROW("No GPU backend detected");
        #endif
    }

    template<typename T>
    InterpMode Texture<T>::interp() const noexcept {
        return m_interp;
    }

    template<typename T>
    BorderMode Texture<T>::border() const noexcept {
        return m_border;
    }

    template<typename T>
    Texture<T> Texture<T>::release() noexcept {
        return std::exchange(*this, Texture<T>{});
    }

    template<typename T>
    void Texture<T>::update(const Array<T>& array, bool prefilter) {
        NOA_CHECK(!empty(), "The texture should already be created");
        NOA_CHECK(all(array.shape() == m_shape),
                  "The input array should have the same shape as the texture {}, but got {}", m_shape, array.shape());

        if (prefilter &&
            (m_interp == INTERP_CUBIC_BSPLINE ||
             m_interp == INTERP_CUBIC_BSPLINE_FAST)) {
            geometry::bspline::prefilter(array, array);
        }

        const Device device_target = device();
        if (device_target.cpu()) {
            NOA_CHECK(array.device() == device_target,
                      "CPU textures can only be constructed from CPU arrays, but got input device {}", array.device());
            m_texture = cpu::Texture<T>{array.strides(), array.share()};
            m_options = array.options();

        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<T>) >= 8) {
                NOA_THROW("Double-precision textures are not supported by the CUDA backend");
            } else {
                NOA_CHECK(indexing::isRightmost(array.strides()) &&
                          indexing::isContiguous(array.strides(), array.shape())[1] && array.strides()[3] == 1,
                          "The depth and width dimension of the array should be contiguous, but got shape {} "
                          "and strides {}", array.shape(), array.strides());

                if (device_target != array.device())
                    array.eval();

                Stream& stream = Stream::current(device_target);
                cuda::memory::copy(array.share(), array.strides()[2],
                                   m_texture.array, size3_t(m_shape.get(1)), stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
