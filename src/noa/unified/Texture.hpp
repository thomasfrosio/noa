#pragma once

#include <variant>

#include "noa/unified/Array.hpp"
#include "noa/unified/ArrayOption.hpp"
#include "noa/unified/geometry/Prefilter.h"

#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/memory/PtrArray.hpp"
#include "noa/gpu/cuda/memory/PtrTexture.hpp"
#include "noa/gpu/cuda/memory/Copy.hpp"
#else
namespace noa::cuda {
    template<typename T>
    struct Texture {};
}
#endif

namespace noa::cpu {
    template<typename T>
    struct Texture {
        Strides4<i64> strides;
        Shared<T[]> ptr;
        T cvalue;
    };
}

namespace noa::gpu {
    template<typename T>
    using Texture = noa::cuda::Texture<T>;
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
        static_assert(noa::traits::is_any_v<T, f32, f64, c32, c64>);

        using value_type = T;
        using shape_type = Shape4<i64>;
        using strides_type = Strides4<i64>;
        using cpu_texture_type = cpu::Texture<value_type>;
        using gpu_texture_type = gpu::Texture<value_type>;
        using variant_type = std::variant<std::monostate, cpu_texture_type, gpu_texture_type>;

    public:
        /// Creates an empty texture.
        constexpr Texture() = default;

        /// Creates a texture.
        /// \param[in,out] array    Array or mutable view to transform into the new texture.
        /// \param device_target    Device where the texture should be constructed.
        /// \param interp_mode      Interpolation mode.
        /// \param border_mode      Border mode.
        /// \param cvalue           Constant value to use for out-of-bounds coordinates.
        ///                         Only used if \p border_mode is \c BorderMode::VALUE.
        /// \param layered          Whether the GPU texture should be a 2D layered texture.
        ///                         The number of layers is equal to the batch dimension of \p array.
        ///                         This is ignored for CPU textures, since they are always considered layered.
        /// \param prefilter        Whether the input \p array should be prefiltered first.
        ///                         If true and if \p interp_mode is \c InterpMode::{CUBIC_BSPLINE|CUBIC_BSPLINE_FAST},
        ///                         the input is prefiltered in-place.
        ///
        /// \note If \p device_target is a GPU, a CUDA array is allocated with the same type and shape as \p array,
        ///       a texture is attached to the new array's memory and the new CUDA array is initialized with
        ///       the values from \p array. Limitations:\n
        ///         - Double precision is not supported.\n
        ///         - \p array should be in the rightmost order and its depth and width dimensions should be C-contiguous.
        ///           In other words, \p array should be C-contiguous or have a valid "pitch" as defined in CUDA.\n
        ///         - \p array can be on any device, including the CPU.\n
        ///         - If \p layered is false or if \p array is a 3D array, \p array cannot be batched.\n
        ///         - \p border_mode should be \c BorderMode::{ZERO|CLAMP|PERIODIC|MIRROR}.\n
        ///         - \c InterpMode::{NEAREST|LINEAR_FAST} are the only modes supporting \c BorderMode::{MIRROR|PERIODIC}.
        /// \note If \p device_target is a CPU, no computation is performed (other the the optional pre-filtering)
        ///       and the texture simply points to \p array. Limitations:\n
        ///         - \p array should be on the CPU.\n
        ///
        /// \warning For GPU textures, \p array can be on any device, effectively allowing to create a GPU texture
        ///          from a CPU array. Note however that while the API will make sure that stream ordering will be
        ///          respected (by possibly synchronizing the current stream of the \p array device), the caller
        ///          should not modify the underlying values of \p array until the texture is created. See eval().
        template<typename ArrayOrView, typename = std::enable_if_t<
                 noa::traits::is_array_or_view_of_any_v<ArrayOrView, value_type>>>
        Texture(const ArrayOrView& array, Device device_target, InterpMode interp_mode, BorderMode border_mode,
                value_type cvalue = value_type{0}, bool layered = false, bool prefilter = true)
                : m_shape(array.shape()), m_interp(interp_mode), m_border(border_mode) {

            NOA_CHECK(!array.is_empty(), "Empty array detected");

            if (prefilter &&
                (interp_mode == InterpMode::CUBIC_BSPLINE ||
                 interp_mode == InterpMode::CUBIC_BSPLINE_FAST)) {
                noa::geometry::cubic_bspline_prefilter(array, array);
            }

            if (device_target.is_cpu()) {
                NOA_CHECK(array.device() == device_target,
                          "CPU textures can only be constructed/updated from CPU arrays, but got device {}",
                          array.device());
                if constexpr (noa::traits::is_view_v<ArrayOrView>)
                    m_texture = cpu_texture_type{array.strides(), Shared<T[]>(array.get(), [](void*) {}), cvalue};
                else
                    m_texture = cpu_texture_type{array.strides(), array.share(), cvalue};
                m_options = array.options();

            } else {
                #ifdef NOA_ENABLE_CUDA
                if constexpr (sizeof(noa::traits::value_type_t<value_type>) >= 8) {
                    NOA_THROW("Double-precision textures are not supported by the CUDA backend");
                } else {
                    const auto guard = DeviceGuard(device_target);

                    gpu_texture_type texture;
                    texture.array = noa::cuda::memory::PtrArray<value_type>::alloc(
                            array.shape(), layered ? cudaArrayLayered : cudaArrayDefault);
                    texture.texture = noa::cuda::memory::PtrTexture::alloc(
                            texture.array.get(), interp_mode, border_mode);

                    if (device_target != array.device())
                        array.eval();
                    noa::cuda::memory::copy(
                            array.share(), array.strides(),
                            texture.array, array.shape(),
                            Stream::current(device_target).cuda());

                    m_texture = texture;
                    m_options = ArrayOption{device_target, Allocator::CUDA_ARRAY};
                }
                #else
                NOA_THROW("No GPU backend detected");
                #endif
            }
        }

        /// Creates a texture.
        /// \param shape            BDHW shape of the new texture.
        /// \param device_target    Device where the texture should be constructed.
        /// \param interp_mode      Interpolation mode.
        /// \param border_mode      Border mode.
        /// \param cvalue           Constant value to use for out-of-bounds coordinates.
        ///                         Only used if \p border_mode is \c BorderMode::VALUE.
        /// \param layered          Whether the GPU texture should be a 2D layered texture.
        ///                         The number of layers is equal to the batch dimension of \p array.
        ///                         This is ignored for CPU textures, since they are always considered layered.
        ///
        /// \note If \p device_target is a GPU, a CUDA array is allocated of \p T type and \p shape,
        ///       a texture is attached to the new array's memory. The new CUDA array is left
        ///       uninitialized (see update()). Limitations:\n
        ///         - Double precision is not supported.\n
        ///         - If \p layered is false or if \p shape describes a 3D array, \p shape cannot be batched.\n
        ///         - \p border_mode should be \c BorderMode::{ZERO|CLAMP|PERIODIC|MIRROR}.\n
        ///         - \c InterpMode::{NEAREST|LINEAR_FAST} are the only modes supporting \c BorderMode::{MIRROR|PERIODIC}.
        ///
        /// \note If \p device_target is a CPU, no computation is performed. The texture is non-empty and valid,
        ///       but the underlying managed data (i.e. the cpu::Texture) points to a null pointer. Use update()
        ///       to set the texture to a valid memory region.
        Texture(shape_type shape, Device device_target, InterpMode interp_mode, BorderMode border_mode,
                value_type cvalue = value_type{0}, bool layered = false)
                : m_shape(shape), m_interp(interp_mode), m_border(border_mode) {

            if (device_target.is_cpu()) {
                m_texture = cpu_texture_type{{}, nullptr, cvalue};
            } else {
                #ifdef NOA_ENABLE_CUDA
                if constexpr (sizeof(traits::value_type_t<value_type>) >= 8) {
                    NOA_THROW("Double-precision textures are not supported by the CUDA backend");
                } else {
                    const auto guard = DeviceGuard(device_target);

                    gpu_texture_type texture;
                    texture.array = noa::cuda::memory::PtrArray<value_type>::alloc(
                            shape, layered ? cudaArrayLayered : cudaArrayDefault);
                    texture.texture = noa::cuda::memory::PtrTexture::alloc(
                            texture.array.get(), interp_mode, border_mode);
                    m_texture = texture;
                    m_options = ArrayOption{device_target, Allocator::CUDA_ARRAY};
                }
                #else
                NOA_THROW("No GPU backend detected");
                #endif
            }
        }

    public: // Copy
        /// Updates the texture values with \p array.
        /// \param[in,out] array    Array or mutable view to copy into the texture.
        /// \param prefilter        Whether the input \p array should be prefiltered first.
        ///                         If true and if the texture uses \c InterpMode::CUBIC_BSPLINE(_FAST),
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
        template<typename ArrayOrView, typename = std::enable_if_t<
                 noa::traits::is_array_or_view_of_any_v<ArrayOrView, value_type>>>
        void update(const ArrayOrView& array, bool prefilter = true) {
            NOA_CHECK(!is_empty(), "Trying to update an empty texture is not allowed. Create a valid the texture first");
            NOA_CHECK(!array.is_empty(), "Empty array detected");
            NOA_CHECK(noa::all(array.shape() == m_shape), // TODO Broadcast?
                      "The input array should have the same shape as the texture {}, but got {}",
                      m_shape, array.shape());

            if (prefilter &&
                (m_interp == InterpMode::CUBIC_BSPLINE ||
                 m_interp == InterpMode::CUBIC_BSPLINE_FAST)) {
                noa::geometry::cubic_bspline_prefilter(array, array);
            }

            const Device device_target = device();
            if (device_target.is_cpu()) {
                NOA_CHECK(array.device() == device_target,
                          "CPU textures can only be constructed/updated from CPU arrays, but got device {}",
                          array.device());
                cpu_texture_type& cpu_texture = this->cpu();
                cpu_texture.strides = array.strides();
                if constexpr (noa::traits::is_view_v<ArrayOrView>)
                    cpu_texture.ptr = Shared<T[]>(array.get(), [](void*) {});
                else
                    cpu_texture.ptr = array.share();
                m_options = array.options();

            } else {
                #ifdef NOA_ENABLE_CUDA
                if constexpr (sizeof(traits::value_type_t<value_type>) >= 8) {
                    NOA_THROW("Double-precision textures are not supported by the CUDA backend");
                } else {
                    if (device_target != array.device())
                        array.eval();
                    cuda::memory::copy(
                            array.share(), array.strides(),
                            this->cuda().array,
                            m_shape, Stream::current(device_target).cuda());
                }
                #else
                NOA_THROW("No GPU backend detected");
                #endif
            }
        }

    public: // Getters
        /// Returns the options used to create the array.
        [[nodiscard]] constexpr ArrayOption options() const noexcept { return m_options; }

        /// Returns the device used to create the array.
        [[nodiscard]] constexpr Device device() const noexcept { return m_options.device(); }

        /// Returns the device used to create the array.
        [[nodiscard]] constexpr Allocator allocator() const noexcept { return m_options.allocator(); }

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
        [[nodiscard]] bool is_contiguous() const noexcept {
            if (device().is_cpu())
                return noa::indexing::are_contiguous<ORDER>(cpu().strides, m_shape);
            else
                return ORDER == 'C' || ORDER == 'c';
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

        /// Gets the underlying texture, assuming it is a CPU texture (i.e. device is CPU).
        /// Otherwise, throws an exception.
        [[nodiscard]] const cpu::Texture<value_type>& cpu() const {
            auto* ptr = std::get_if<cpu_texture_type>(&m_texture);
            if (!ptr)
                NOA_THROW("Texture is not initialized or trying to retrieve at CPU texture from a GPU texture");
            return *ptr;
        }

        /// Gets the underlying texture, assuming it is a GPU texture (i.e. device is GPU).
        /// Otherwise, throws an exception.
        [[nodiscard]] const gpu_texture_type& gpu() const {
            #ifdef NOA_ENABLE_CUDA
            return this->cuda();
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }

        /// Gets the underlying texture, assuming it is a CUDA texture (i.e. device is a CUDA-capable GPU).
        /// Otherwise, throws an exception.
        [[nodiscard]] const gpu_texture_type& cuda() const {
            #ifdef NOA_ENABLE_CUDA
            auto* ptr = std::get_if<gpu_texture_type>(&m_texture);
            if (!ptr)
                NOA_THROW("Texture is not initialized or trying to retrieve at GPU texture from a CPU texture");
            return *ptr;
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }

        [[nodiscard]] InterpMode interp_mode() const noexcept { return m_interp; }
        [[nodiscard]] BorderMode border_mode() const noexcept { return m_border; }

        [[nodiscard]] bool is_layered() const {
            if (device().is_cpu()) {
                return true;
            } else {
                #ifdef NOA_ENABLE_CUDA
                return noa::cuda::memory::PtrArray<value_type>::is_layered(this->cuda().array.get());
                #else
                return false;
                #endif
            }
        }

    private:
        variant_type m_texture;
        Shape4<i64> m_shape;
        ArrayOption m_options;
        InterpMode m_interp{};
        BorderMode m_border{};
    };
}
