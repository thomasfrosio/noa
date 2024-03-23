#pragma once

#include <variant>

#include "noa/unified/Array.hpp"
#include "noa/unified/ArrayOption.hpp"
#include "noa/unified/geometry/CubicBSplinePrefilter.hpp"

#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/AllocatorTexture.hpp"
#include "noa/gpu/cuda/Copy.cuh"
#endif

namespace noa::cpu {
    template<typename T>
    struct Texture {
        Strides4<i64> strides{};
        Array<T>::shared_type handle{};
        const T* pointer{};
        T cvalue{};
    };
}

namespace noa::gpu {
    #ifdef NOA_ENABLE_CUDA
    template<typename T>
    using Texture = noa::cuda::AllocatorTexture<T>::shared_type;
    #else
    template<typename T>
    using Texture = std::shared_ptr<void>;
    #endif
}

namespace noa {
    /// Unified texture.
    /// \details This template class constructs and encapsulates a texture. Textures are used for fast interpolation
    ///          and/or multidimensional caching. On the CPU, this simply points to an array. However, on the GPU,
    ///          it allocates and initializes a proper GPU texture. If multiple calls with the same input or even
    ///          the same input type and shape, it is more likely to be more efficient to create and reuse textures.
    /// \tparam Value f32, f64, c32, c64.
    template<typename T>
    class Texture {
    public:
        static_assert(nt::is_any_v<T, f32, f64, c32, c64>);

        using value_type = T;
        using mutable_value_type = T;
        using shape_type = Shape4<i64>;
        using strides_type = Strides4<i64>;
        using cpu_texture_type = noa::cpu::Texture<value_type>;
        using gpu_texture_type = noa::gpu::Texture<value_type>;
        using variant_type = std::variant<std::monostate, cpu_texture_type, gpu_texture_type>;

        struct Options {
            Border border{Border::ZERO};

            /// Constant value to use for out-of-bounds coordinates.
            /// Only used if border is Border::VALUE.
            value_type cvalue{};

            /// Whether the input array should be prefiltered.
            /// If true and if the interpolation is Interp::{CUBIC_BSPLINE|CUBIC_BSPLINE_FAST},
            /// the input is prefiltered in-place before creating/updating the texture.
            bool prefilter{true};
        };

    public:
        /// Creates an empty texture.
        constexpr Texture() = default;

        /// Creates a texture.
        /// \param[in,out] array    Array or mutable view to transform into the new texture.
        /// \param device_target    Device where the texture should be constructed.
        /// \param interp_mode      Interpolation mode.
        /// \param options          Texture options.
        ///
        /// \note If \p device_target is a GPU, a CUDA array is allocated with the same type and shape as \p array,
        ///       a texture is attached to the new array's memory and the new CUDA array is initialized with
        ///       the values from \p array. Limitations:\n
        ///         - Double precision is not supported.\n
        ///         - \p array should be in the rightmost order and its depth and width dimensions should be C-contiguous.
        ///           In other words, \p array should be C-contiguous or have a valid "pitch" as defined in CUDA.\n
        ///         - \p array can be on any device, including the CPU.\n
        ///         - \p options.border should be \c Border::{ZERO|CLAMP|PERIODIC|MIRROR}.\n
        ///         - \c Interp::{NEAREST|LINEAR_FAST} are the only modes supporting \c Border::{MIRROR|PERIODIC}.
        /// \note If \p device_target is a CPU, no computation is performed (other the the optional pre-filtering)
        ///       and the texture simply points to \p array. Limitations:\n
        ///         - \p array should be on the CPU.\n
        ///
        /// \warning For GPU textures, \p array can be on any device, effectively allowing to create a GPU texture
        ///          from a CPU array. Note however that while the API will make sure that stream ordering will be
        ///          respected (by possibly synchronizing the current stream of the \p array device), the caller
        ///          should not modify the underlying values of \p array until the texture is created. See eval().
        template<typename VArray> requires nt::is_varray_of_any_v<VArray, value_type>
        Texture(
                const VArray& array,
                Device device_target,
                Interp interp_mode,
                const Options& options = {}
        ) : m_shape(array.shape()),
            m_interp(interp_mode),
            m_border(options.border)
        {
            check(not array.is_empty(), "Empty array detected");

            if (options.prefilter and
                (interp_mode == Interp::CUBIC_BSPLINE or
                 interp_mode == Interp::CUBIC_BSPLINE_FAST)) {
                noa::geometry::cubic_bspline_prefilter(array, array);
            }

            if (device_target.is_cpu()) {
                check(array.device() == device_target,
                      "CPU textures can only be constructed/updated from CPU arrays, but got device {}",
                      array.device());
                if constexpr (nt::is_view_v<VArray>) {
                    m_texture = cpu_texture_type{
                            .strides=array.strides(),
                            .pointer=array.get(),
                            .cvalue=options.cvalue,
                    };
                } else {
                    m_texture = cpu_texture_type{
                            .strides=array.strides(),
                            .handle=array.share(),
                            .pointer=array.get(),
                            .cvalue=options.cvalue,
                    };
                }
                m_options = array.options();

            } else {
                #ifdef NOA_ENABLE_CUDA
                if constexpr (sizeof(nt::value_type_t<value_type>) >= 8) {
                    panic("Double-precision textures are not supported by the CUDA backend");
                } else {
                    const auto guard = DeviceGuard(device_target);
                    auto texture = noa::cuda::AllocatorTexture<value_type>::allocate(
                            array.shape(), interp_mode, options.border,
                            array.shape().ndim() == 2 ? cudaArrayLayered : cudaArrayDefault
                    );

                    // Copy input to CUDA array.
                    if (device_target != array.device())
                        array.eval();
                    auto& cuda_stream = Stream::current(device_target).cuda();
                    noa::cuda::copy(
                            array.get(), array.strides(),
                            texture->array, array.shape(),
                            cuda_stream);
                    cuda_stream.enqueue_attach(array, texture);

                    m_texture = std::move(texture);
                    m_options = ArrayOption{device_target, Allocator(MemoryResource::CUDA_ARRAY)};
                }
                #else
                panic("No GPU backend detected");
                #endif
            }
        }

        /// Creates a texture.
        /// \param shape            BDHW shape of the new texture.
        /// \param device_target    Device where the texture should be constructed.
        /// \param interp_mode      Interpolation mode.
        /// \param options          Texture options (prefilter is ignored).
        ///
        /// \note If \p device_target is a GPU, a CUDA array is allocated of \p T type and \p shape,
        ///       a texture is attached to the new array's memory. The new CUDA array is left
        ///       uninitialized (see update()). Limitations:\n
        ///         - Double precision is not supported.\n
        ///         - \p options.border should be \c Border::{ZERO|CLAMP|PERIODIC|MIRROR}.\n
        ///         - \c Interp::{NEAREST|LINEAR_FAST} are the only modes supporting \c Border::{MIRROR|PERIODIC}.
        ///
        /// \note If \p device_target is a CPU, no computation is performed. The texture is non-empty and valid,
        ///       but the underlying managed data (i.e. the cpu::Texture) points to a null pointer. Use update()
        ///       to set the texture to a valid memory region.
        Texture(shape_type shape, Device device_target, Interp interp_mode, const Options& options = {})
                : m_shape(shape), m_interp(interp_mode), m_border(options.border) {

            if (device_target.is_cpu()) {
                m_texture = cpu_texture_type{.cvalue=options.cvalue};
            } else {
                #ifdef NOA_ENABLE_CUDA
                if constexpr (sizeof(nt::value_type_t<value_type>) >= 8) {
                    panic("Double-precision textures are not supported by the CUDA backend");
                } else {
                    const auto guard = DeviceGuard(device_target);
                    m_texture = noa::cuda::AllocatorTexture<value_type>::allocate(
                            shape, interp_mode, options.border,
                            shape.ndim() == 2 ? cudaArrayLayered : cudaArrayDefault);
                    m_options = ArrayOption{device_target, Allocator(MemoryResource::CUDA_ARRAY)};
                }
                #else
                panic("No GPU backend detected");
                #endif
            }
        }

    public: // Copy
        /// Updates the texture values with \p array.
        /// \param[in,out] array    Array or mutable view to copy into the texture.
        /// \param prefilter        Whether the input \p array should be prefiltered first.
        ///                         If true and if the texture uses \c Interp::CUBIC_BSPLINE(_FAST),
        ///                         the input is prefiltered in-place.
        /// \note With GPU textures, \p array should have the same shape as the texture and
        ///       a deep copy is performed from \p array to the managed texture data.
        ///       Limitations:
        ///         - \p array should be in the rightmost order and its depth and width dimensions should be C-contiguous.
        ///           In other words, \p array should be C-contiguous or have a valid "pitch" as defined in CUDA.\n
        ///         - \p array can be on any device, including the CPU.\n
        /// \note With CPU textures, no computation is performed (other than the optional pre-filtering)
        ///       and the texture pointer is simply updated to point to \p array, which should be a CPU array.
        ///
        /// \warning For GPU textures, \p array can be on any device, effectively allowing to update a GPU texture
        ///          from a CPU array. Note however that while the API will make sure that stream ordering will be
        ///          respected (by possibly synchronizing the current stream of the \p array device), the caller
        ///          should not modify the underlying values of \p array until the texture is updated. See eval().
        template<typename VArray> requires nt::is_varray_of_any_v<VArray, value_type>
        void update(const VArray& array, bool prefilter = true) {
            check(not is_empty(), "Trying to update an empty texture is not allowed. Create a valid the texture first");
            check(not array.is_empty(), "Empty array detected");
            check(all(array.shape() == m_shape),
                  "The input array should have the same shape as the texture, but got texture:shape={} and array:shape={}",
                  m_shape, array.shape());

            if (prefilter and
                (m_interp == Interp::CUBIC_BSPLINE or
                 m_interp == Interp::CUBIC_BSPLINE_FAST)) {
                noa::geometry::cubic_bspline_prefilter(array, array);
            }

            const Device device_target = device();
            if (device_target.is_cpu()) {
                check(array.device() == device_target,
                      "CPU textures can only be constructed/updated from CPU arrays, but got array:device={}",
                      array.device());
                cpu_texture_type& cpu_texture = cpu_();
                cpu_texture.strides = array.strides();
                if constexpr (nt::is_view_v<VArray>)
                    cpu_texture.ptr = std::shared_ptr<T[]>(array.get(), [](void*) {});
                else
                    cpu_texture.ptr = array.share();
                m_options = array.options();

            } else {
                #ifdef NOA_ENABLE_CUDA
                if constexpr (sizeof(nt::value_type_t<value_type>) >= 8) {
                    panic("Double-precision textures are not supported by the CUDA backend");
                } else {
                    if (device_target != array.device())
                        array.eval();

                    gpu_texture_type& cuda_texture = cuda_();
                    auto& cuda_stream = Stream::current(device_target).cuda();
                    noa::cuda::copy(
                            array.get(), array.strides(),
                            cuda_texture->array,
                            m_shape, cuda_stream);
                    cuda_stream.enqueue_attach(array, cuda_texture->array);
                }
                #else
                panic("No GPU backend detected");
                #endif
            }
        }

    public: // Getters
        /// Returns the options used to create the array.
        [[nodiscard]] constexpr ArrayOption options() const noexcept { return m_options; }

        /// Returns the device used to create the array.
        [[nodiscard]] constexpr Device device() const noexcept { return m_options.device; }

        /// Returns the device used to create the array.
        [[nodiscard]] constexpr Allocator allocator() const noexcept { return m_options.allocator; }

        /// Whether the array is empty.
        [[nodiscard]] bool is_empty() const noexcept { return std::holds_alternative<std::monostate>(m_texture); }

        /// Returns the BDHW shape of the array.
        [[nodiscard]] const shape_type& shape() const noexcept { return m_shape; }

        /// Returns the BDHW strides of the array.
        [[nodiscard]] strides_type strides() const {
            if (device().is_cpu())
                return cpu().strides;
            else
                return m_shape.strides();
        }

        /// Whether the dimensions of the array are C or F contiguous.
        template<char ORDER = 'C'>
        [[nodiscard]] bool are_contiguous() const noexcept {
            if (device().is_cpu())
                return ni::are_contiguous<ORDER>(cpu().strides, m_shape);
            else
                return ORDER == 'C' or ORDER == 'c';
        }

        /// Synchronizes the current stream of the Texture's device.
        /// \details It guarantees safe access to the underlying data and indicates no operations are pending.
        const Texture& eval() const {
            Stream::current(device()).synchronize();
            return *this;
        }

        /// Releases the texture. The current instance is left empty.
        Texture release() noexcept {
            return std::exchange(*this, Texture<value_type>{});
        }

        /// Returns the underlying pointer of CPU array.
        /// This is used to provide an Array-like API.
        [[nodiscard]] constexpr const value_type* get() const noexcept {
            return cpu().pointer;
        }

        /// Returns the underlying CPU array as a View.
        /// This is used to provide an Array-like API.
        [[nodiscard]] constexpr View<const value_type> view() const noexcept {
            const auto& cpu_texture = cpu();
            return View<const value_type>(cpu_texture.pointer, shape(), cpu_texture.strides, options());
        }

        /// Returns a reference of the managed resource.
        /// This is used to provide an Array-like API.
        /// \warning Depending on the current stream of this array's device,
        ///          reading/writing to this pointer may be illegal or create a data race.
        [[nodiscard]] std::shared_ptr<void> share() const noexcept {
            return std::visit(m_texture,
                       [](const cpu_texture_type& texture ){ return texture.handle; },
                       [](const gpu_texture_type& texture ){ return texture; });
        }

        /// Gets the underlying texture, assuming it is a CPU texture (i.e. device is CPU).
        /// Otherwise, throws an exception.
        [[nodiscard]] const cpu_texture_type& cpu() const {
            auto* ptr = std::get_if<cpu_texture_type>(&m_texture);
            check(ptr, "Texture is not initialized or trying to retrieve at CPU texture from a GPU texture");
            return *ptr;
        }

        /// Gets the underlying texture, assuming it is a GPU texture (i.e. device is GPU).
        /// Otherwise, throws an exception.
        [[nodiscard]] const gpu_texture_type& gpu() const {
            #ifdef NOA_ENABLE_CUDA
            return this->cuda();
            #else
            panic("No GPU backend detected");
            #endif
        }

        /// Gets the underlying texture, assuming it is a CUDA texture (i.e. device is a CUDA-capable GPU).
        /// Otherwise, throws an exception.
        [[nodiscard]] const gpu_texture_type& cuda() const {
            #ifdef NOA_ENABLE_CUDA
            auto* ptr = std::get_if<gpu_texture_type>(&m_texture);
            check(ptr, "Texture is not initialized or trying to retrieve at GPU texture from a CPU texture");
            return *ptr;
            #else
            panic("No GPU backend detected");
            #endif
        }

        [[nodiscard]] Interp interp_mode() const noexcept { return m_interp; }
        [[nodiscard]] Border border_mode() const noexcept { return m_border; }

        [[nodiscard]] value_type cvalue() const noexcept {
            if (device().is_cpu())
                return cpu().cvalue;
            else
                return {}; // GPU textures do not support Border::VALUE
        }

    private: // For now, keep the right to modify the underlying textures to yourself
        [[nodiscard]] cpu_texture_type& cpu_() {
            auto* ptr = std::get_if<cpu_texture_type>(&m_texture);
            check(ptr, "Texture is not initialized or trying to retrieve at CPU texture from a GPU texture");
            return *ptr;
        }

        [[nodiscard]] gpu_texture_type& gpu_() {
            #ifdef NOA_ENABLE_CUDA
            return this->cuda();
            #else
            panic("No GPU backend detected");
            #endif
        }

        [[nodiscard]] gpu_texture_type& cuda_() {
            #ifdef NOA_ENABLE_CUDA
            auto* ptr = std::get_if<gpu_texture_type>(&m_texture);
            check(ptr, "Texture is not initialized or trying to retrieve at GPU texture from a CPU texture");
            return *ptr;
            #else
            panic("No GPU backend detected");
            #endif
        }

    private:
        variant_type m_texture;
        Shape4<i64> m_shape{};
        ArrayOption m_options;
        Interp m_interp{};
        Border m_border{};
    };
}

namespace noa::traits {
    template<typename T> struct proclaim_is_texture<Texture<T>> : std::true_type {};
}
