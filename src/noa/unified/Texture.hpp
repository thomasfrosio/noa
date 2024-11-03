#pragma once

#include "noa/core/Config.hpp"

#ifdef NOA_IS_OFFLINE
#include <variant>

#include "noa/unified/Array.hpp"
#include "noa/unified/ArrayOption.hpp"
#include "noa/unified/geometry/CubicBSplinePrefilter.hpp"

#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/IncludeGuard.cuh"
#include "noa/gpu/cuda/Allocators.hpp"
#include "noa/gpu/cuda/Copy.cuh"
namespace noa::gpu {
    template<typename T>
    using TextureResource = AllocatorTexture::shared_type;
}
#else
namespace noa::gpu {
    template<typename T>
    using TextureResource = std::shared_ptr<void>;
}
#endif

namespace noa::cpu {
    /// A simple handle to an array or view.
    template<typename T>
    struct TextureResource {
        Strides4<i64> strides{};
        std::shared_ptr<T[]> handle{};
        const T* pointer{};
    };
}

namespace noa::inline types {
    /// Unified texture.
    /// \details This template class constructs and encapsulates a texture. Textures are used for fast interpolation
    ///          and/or multidimensional caching. On the CPU, this simply points to an array. However, on the GPU,
    ///          it allocates and initializes a proper GPU texture.
    ///
    /// \note CUDA textures have limitations on the addressing/Border, but the Interpolator hides them from the user by
    ///       adding software support for the modes that CUDA does not natively support. As such, if a GPU texture
    ///       is created with Border::{VALUE|REFLECT|NOTHING} (addressing modes that are not supported by CUDA),
    ///       hardware interpolation is turned off and the Interpolator falls back using software interpolation and
    ///       addressing. Note that the texture is still used to read values, so the texture cache can still speed up
    ///       the interpolation.
    ///
    /// \note For spectrum interpolation, hardware interpolation and addressing is only supported if the input
    ///       spectrum is centered, i.e. REMAP.xc2xx() == true. Otherwise, the InterpolatorSpectrum falls back on
    ///       software interpolation and addressing.
    ///
    /// \see Interpolator and InterpolatorSpectrum for more details.
    template<typename T>
    class Texture {
    public:
        static_assert(nt::any_of<T, f32, f64, c32, c64>);

        using value_type = T;
        using mutable_value_type = T;
        using shape_type = Shape4<i64>;
        using strides_type = Strides4<i64>;
        using cpu_texture_type = noa::cpu::TextureResource<value_type>;
        using gpu_texture_type = noa::gpu::TextureResource<value_type>;
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

        /// Creates and initializes a texture.
        /// \param[in,out] array    Array or mutable view to transform into the new texture.
        /// \param device_target    Device where the texture should be constructed.
        /// \param interp           Interpolation mode.
        /// \param options          Texture options.
        ///
        /// \note If device_target is a CUDA-capable GPU, a CUDA array is allocated with the same type and shape
        ///       as array, a texture is attached to this new array's memory, and the new CUDA array is initialized
        ///       with the values from array. Limitations:
        ///         - double precision is not supported.
        ///         - the array should be in the rightmost order, and its depth and width dimensions should be
        ///           C-contiguous. In other words, it should be C-contiguous or have a valid "pitch".
        ///         - the array can be on any device, including the CPU.
        ///
        /// \note If device_target is a CPU, no computation is performed (other than the optional pre-filtering)
        ///       and the texture simply points to array. Limitations:
        ///         - array should be on the CPU.
        ///
        /// \warning For GPU textures, the array can be on any device, effectively allowing to create a GPU texture
        ///          from a CPU array. Note however that while the API will make sure that stream ordering will be
        ///          respected (by possibly synchronizing the current stream of the array's device), the caller
        ///          should not modify the underlying values of the array until the texture is created. This can
        ///          be guaranteed by calling Texture::eval() after creating the texture, for example.
        template<nt::varray_decay_of_any<value_type> VArray>
        Texture(
            VArray&& array,
            Device device_target,
            Interp interp,
            const Options& options = {}
        ) :
            m_shape(array.shape()),
            m_interp(interp),
            m_border(options.border),
            m_cvalue(options.cvalue)
        {
            check(not array.is_empty(), "Empty array detected");

            if (options.prefilter and interp.is_almost_any(Interp::CUBIC_BSPLINE))
                noa::geometry::cubic_bspline_prefilter(array, array);

            if (device_target.is_cpu()) {
                check(array.device() == device_target,
                      "CPU textures can only be constructed/updated from other CPU arrays, but got array:device={}",
                      array.device());
                m_options = array.options();
                m_texture = cpu_texture_type{
                    .strides = array.strides(),
                    .pointer = array.get(),
                };
                if constexpr (nt::view_decay<VArray>)
                    m_texture.handle = std::forward<VArray>(array).share();

            } else {
                #ifdef NOA_ENABLE_CUDA
                if constexpr (sizeof(nt::value_type_t<value_type>) >= 8) {
                    panic("Double-precision textures are not supported by the CUDA backend");
                } else {
                    const auto guard = DeviceGuard(device_target);
                    auto texture = noa::cuda::AllocatorTexture::allocate<value_type>(
                        array.shape(), interp, options.border);

                    // Copy input to CUDA array.
                    if (device_target != array.device())
                        array.eval();
                    auto& cuda_stream = Stream::current(device_target).cuda();
                    noa::cuda::copy(
                        array.get(), array.strides(),
                        texture->array, array.shape(),
                        cuda_stream);
                    cuda_stream.enqueue_attach(std::forward<VArray>(array), texture);

                    m_texture = std::move(texture);
                    m_options = ArrayOption{device_target, Allocator::CUDA_ARRAY};
                }
                #else
                panic();
                #endif
            }
        }

        /// Creates a texture.
        /// \param shape            BDHW shape of the new texture.
        /// \param device_target    Device where the texture should be constructed.
        /// \param interp           Interpolation mode.
        /// \param options          Texture options (prefilter is ignored).
        ///
        /// \note If device_target is a GPU, a CUDA array of T type and shape is allocated,
        ///       and a texture is attached to this new array's memory. The new CUDA array is left
        ///       uninitialized (see update()). Limitations:
        ///         - double precision is not supported.
        ///
        /// \note If device_target is a CPU, no computation is performed. The texture is valid, but the underlying
        ///       managed data points to a null pointer. Use update() to set the texture to a valid memory region.
        Texture(const shape_type& shape, Device device_target, Interp interp, const Options& options = {}) :
            m_shape(shape),
            m_interp(interp),
            m_border(options.border),
            m_cvalue(options.cvalue)
        {
            if (device_target.is_cpu()) {
                m_texture = cpu_texture_type{};
            } else {
                #ifdef NOA_ENABLE_CUDA
                if constexpr (sizeof(nt::value_type_t<value_type>) >= 8) {
                    panic("Double-precision textures are not supported by the CUDA backend");
                } else {
                    const auto guard = DeviceGuard(device_target);
                    m_texture = noa::cuda::AllocatorTexture::allocate<value_type>(shape, interp, options.border);
                    m_options = ArrayOption{device_target, Allocator::CUDA_ARRAY};
                }
                #else
                panic();
                #endif
            }
        }

    public: // Copy
        /// Updates the texture values.
        /// \param[in,out] array    Array or mutable view to copy into the texture.
        /// \param prefilter        Whether the input array should be prefiltered first.
        ///                         If true and if the texture uses Interp::CUBIC_BSPLINE(_FAST),
        ///                         the input is prefiltered in-place.
        ///
        /// \note With GPU textures, the array should have the same shape as the texture and a deep copy is performed
        ///       from the array to the managed texture data. Limitations:
        ///         - the array should be in the rightmost order and its depth and width dimensions should be
        ///           C-contiguous. In other words, it should be C-contiguous or have a valid "pitch".
        ///         - the array can be on any device, including the CPU.
        /// \note With CPU textures, no computation is performed (other than the optional pre-filtering) and the
        ///       texture pointer is simply updated to point to this array instead (which should be a CPU array).
        ///
        /// \warning For GPU textures, the array can be on any device, effectively allowing to create a GPU texture
        ///          from a CPU array. Note however that while the API will make sure that stream ordering will be
        ///          respected (by possibly synchronizing the current stream of the array's device), the caller
        ///          should not modify the underlying values of the array until the texture is created. This can
        ///          be guaranteed by calling Texture::eval() after creating the texture, for example.
        template<nt::varray_decay_of_any<value_type> VArray>
        void update(VArray&& array, bool prefilter = true) {
            check(not is_empty(), "Trying to update an empty texture is not allowed. Create a valid the texture first");
            check(not array.is_empty(), "Empty array detected");
            check(all(array.shape() == m_shape),
                  "The input array should have the same shape as the texture, "
                  "but got texture:shape={} and array:shape={}",
                  m_shape, array.shape());

            if (prefilter and m_interp.is_almost_any(Interp::CUBIC_BSPLINE))
                noa::geometry::cubic_bspline_prefilter(array, array);

            const Device device_target = device();
            if (device_target.is_cpu()) {
                check(array.device() == device_target,
                      "CPU textures can only be constructed/updated from CPU arrays, but got array:device={}",
                      array.device());

                // Reset the underlying array to this new one.
                m_options = array.options();
                cpu_texture_type& cpu_texture = cpu_();
                cpu_texture.strides = array.strides();
                cpu_texture.pointer = array.get();
                if constexpr (nt::array_decay<VArray>)
                    cpu_texture.handle = array.share();

            } else {
                #ifdef NOA_ENABLE_CUDA
                if constexpr (sizeof(nt::value_type_t<value_type>) >= 8) {
                    panic("Double-precision textures are not supported by the CUDA backend");
                } else {
                    if (device_target != array.device())
                        array.eval();

                    // Update the CUDA array with the new values.
                    gpu_texture_type& cuda_texture = cuda_();
                    auto& cuda_stream = Stream::current(device_target).cuda();
                    noa::cuda::copy(
                        array.get(), array.strides(),
                        cuda_texture->array,
                        m_shape, cuda_stream);
                    cuda_stream.enqueue_attach(std::forward<VArray>(array), cuda_texture);
                }
                #else
                panic();
                #endif
            }
        }

    public: // Getters
        [[nodiscard]] constexpr auto options() const noexcept -> ArrayOption { return m_options; }
        [[nodiscard]] constexpr auto device() const noexcept -> Device { return m_options.device; }
        [[nodiscard]] constexpr auto allocator() const noexcept -> Allocator { return m_options.allocator; }
        [[nodiscard]] constexpr auto is_empty() const noexcept -> bool { return std::holds_alternative<std::monostate>(m_texture); }
        [[nodiscard]] constexpr auto interp() const noexcept -> Interp { return m_interp; }
        [[nodiscard]] constexpr auto border() const noexcept -> Border { return m_border; }
        [[nodiscard]] constexpr auto cvalue() const noexcept -> value_type { return m_cvalue; }

        [[nodiscard]] constexpr auto shape() const noexcept -> const shape_type& { return m_shape; }
        [[nodiscard]] constexpr auto strides() const noexcept -> strides_type {
            if (device().is_cpu())
                return cpu().strides;
            return m_shape.strides();
        }
        [[nodiscard]] constexpr auto strides_full() const noexcept -> strides_type { return strides(); }

        /// Whether the dimensions of the array are C or F contiguous.
        template<char ORDER = 'C'>
        [[nodiscard]] auto are_contiguous() const noexcept -> bool {
            if (device().is_cpu())
                return ni::are_contiguous<ORDER>(cpu().strides, m_shape);
            return ORDER == 'C' or ORDER == 'c';
        }

        /// Synchronizes the current stream of the Texture's device.
        /// \details It guarantees safe access to the underlying data and indicates no operations are pending.
        auto& eval() const {
            Stream::current(device()).synchronize();
            return *this;
        }

        /// Releases the texture. The current instance is left empty.
        auto release() noexcept -> Texture {
            return std::exchange(*this, Texture{});
        }

        /// Returns the underlying pointer of CPU array.
        /// This is used to provide an Array-like API.
        [[nodiscard]] constexpr auto get() const noexcept -> const value_type* {
            return cpu().pointer;
        }

        /// Returns the underlying CPU array as a View.
        /// This is used to provide an Array-like API.
        [[nodiscard]] constexpr auto view() const noexcept -> View<const value_type> {
            const auto& cpu_texture = cpu();
            return View<const value_type>(cpu_texture.pointer, shape(), cpu_texture.strides, options());
        }

        /// Returns a reference of the managed resource.
        /// This is used to provide an Array-like API.
        /// \warning Depending on the current stream of this array's device,
        ///          reading/writing to this pointer may be illegal or create a data race.
        [[nodiscard]] auto share() const noexcept {
            return std::visit([]<typename U>(const U& t) -> std::shared_ptr<void> {
                if constexpr (std::is_same_v<U, cpu_texture_type>)
                    return t.handle;
                else if constexpr (std::is_same_v<U, gpu_texture_type>)
                    return t;
                else // std::monostate
                    return {};
            }, m_texture);
        }

        /// Gets the underlying texture, assuming it is a CPU texture (i.e. device is CPU).
        /// Otherwise, throw an exception.
        [[nodiscard]] auto cpu() const -> const cpu_texture_type& {
            auto* ptr = std::get_if<cpu_texture_type>(&m_texture);
            check(ptr, "Texture is not initialized or trying to retrieve at CPU texture from a GPU texture");
            return *ptr;
        }

        /// Gets the underlying texture, assuming it is a GPU texture (i.e. device is GPU).
        /// Otherwise, throw an exception.
        [[nodiscard]] auto gpu() const -> const gpu_texture_type& {
            #ifdef NOA_ENABLE_CUDA
            return this->cuda();
            #else
            panic();
            #endif
        }

#ifdef NOA_ENABLE_CUDA
        /// Gets the underlying texture, assuming it is a CUDA texture (i.e. device is a CUDA-capable GPU).
        /// Otherwise, throw an exception.
        [[nodiscard]] auto cuda() const -> const gpu_texture_type& {
            auto* ptr = std::get_if<gpu_texture_type>(&m_texture);
            check(ptr, "Texture is not initialized or trying to retrieve at GPU texture from a CPU texture");
            return *ptr;
        }
#endif

    private: // For now, keep the right to modify the underlying textures to yourself - TODO C++23 deducing this
        [[nodiscard]] auto cpu_() -> cpu_texture_type& {
            auto* ptr = std::get_if<cpu_texture_type>(&m_texture);
            check(ptr, "Texture is not initialized or trying to retrieve at CPU texture from a GPU texture");
            return *ptr;
        }

        [[nodiscard]] auto gpu_() -> gpu_texture_type& {
            #ifdef NOA_ENABLE_CUDA
            return this->cuda();
            #else
            panic();
            #endif
        }

#ifdef NOA_ENABLE_CUDA
        [[nodiscard]] auto cuda_() -> gpu_texture_type& {
            auto* ptr = std::get_if<gpu_texture_type>(&m_texture);
            check(ptr, "Texture is not initialized or trying to retrieve at GPU texture from a CPU texture");
            return *ptr;
        }
#endif

    private:
        variant_type m_texture{};
        Shape4<i64> m_shape{};
        ArrayOption m_options{};
        Interp m_interp{};
        Border m_border{};
        value_type m_cvalue{};
    };
}

namespace noa::traits {
    template<typename T> struct proclaim_is_texture<Texture<T>> : std::true_type {};
}
#endif
