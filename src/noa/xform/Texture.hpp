#pragma once

#include <variant>

#include "noa/runtime/Array.hpp"
#include "noa/runtime/ArrayOption.hpp"
#include "noa/xform/CubicBSplinePrefilter.hpp"
#include "noa/xform/core/Interp.hpp"
#include "noa/xform/Traits.hpp"

#ifdef NOA_ENABLE_CUDA
#include "noa/runtime/cuda/IncludeGuard.cuh"
#include "noa/runtime/cuda/Allocators.hpp"
#include "noa/runtime/cuda/Copy.cuh"
#include "noa/xform/cuda/Texture.cuh"
namespace noa::xform::gpu {
    template<typename>
    struct TextureResource {
        struct Managed {
            noa::cuda::AllocatorArray::allocate_type array;
            noa::xform::cuda::AllocatorTexture::allocate_type texture;
        };
        std::shared_ptr<Managed> handle{};
    };
}
#else
namespace noa::xform::gpu {
    template<typename T, usize N, ArrayOwnership>
    struct TextureGPUResource {
        template<char>
        constexpr auto is_contiguous() const noexcept -> bool { return true; }
        constexpr auto get() const noexcept -> T* { return nullptr; }
        constexpr auto shape() const noexcept -> auto { return Shape<isize, N>{}; }
        constexpr auto strides() const noexcept { return Strides<isize, N>{}; }
        constexpr auto options() const noexcept { return ArrayOption{}; }
        auto view() const noexcept -> Array<T, N, ArrayOwnership::VIEW> { return {}; }
        auto share() const noexcept -> std::shared_ptr<void> { return nullptr; }
    };
}
#endif

namespace noa::xform {
    /// Texture for fast 2D or 3D interpolation or caching.
    /// \details
    ///     This class template constructs and encapsulates a texture, and is meant for device agnostic use of textures.
    ///   - On the CPU, it is a simple array wrapper. On the GPU, it allocates and initializes a proper GPU texture.
    ///     For a N-D texture, the shape and strides are N + 1 (+1 to encode for the batch/layer dimension).
    ///
    /// \note
    ///     CUDA textures have limitations on the addressing/Border, but the Interpolator class hides it away by
    ///     adding software support for the modes that CUDA does not natively support. As such, if a GPU texture
    ///     is created with Border::{VALUE|REFLECT|NOTHING} (addressing modes that are not supported by CUDA),
    ///     hardware interpolation is turned off and the Interpolator falls back to using software interpolation and
    ///     addressing. Note that the texture is still used to read values, so the texture cache can still speed up
    ///     the interpolation. For spectrum interpolation, hardware interpolation and addressing are only supported
    ///     if the input spectrum is centered. Otherwise, the InterpolatorSpectrum falls back on software interpolation
    ///     and addressing. Interpolator and InterpolatorSpectrum for more details.
    template<typename T, usize N, ArrayOwnership O = ArrayOwnership::RC>
    class Texture {
    public:
        static_assert(nt::almost_any_of<T, f32, f64, c32, c64>);
        static_assert(N == 2 or N == 3);
        static constexpr usize SIZE = N;
        static constexpr isize SSIZE = N;
        static constexpr bool IS_VIEW = O == ArrayOwnership::VIEW;

        using value_type = T;
        using mutable_value_type = std::remove_const<T>;
        using const_value_type = std::add_const_t<mutable_value_type>;
        using shape_type = Shape<isize, N + 1>;
        using strides_type = Strides<isize, N + 1>;
        using cpu_texture_type = Array<value_type, N + 1, O>;
        using gpu_texture_type = gpu::TextureGPUResource<value_type, N + 1, O>;
        using variant_type = std::variant<cpu_texture_type, gpu_texture_type>;

        struct CTorOptions {
            /// Addressing for out-of-bounds coordinates.
            Border border{Border::ZERO};

            /// Constant value to use for out-of-bounds coordinates.
            /// Only used if border is Border::VALUE.
            value_type cvalue{};

            /// Whether the input array should be prefiltered, in-place.
            /// If true and if the interpolation is Interp::{CUBIC_BSPLINE|CUBIC_BSPLINE_FAST},
            /// the input is prefiltered in-place before creating/updating the texture.
            bool prefilter{true};
        };

        struct UpdateOptions {
            /// Whether the input array should be prefiltered, in-place.
            /// If true and if the interpolation is Interp::{CUBIC_BSPLINE|CUBIC_BSPLINE_FAST},
            /// the input is prefiltered in-place before creating/updating the texture.
            bool prefilter{true};
        };

    public:
        /// Creates an empty texture.
        constexpr Texture() = default;

        /// Creates a const array from an existing non-const array.
        template<nt::mutable_of<value_type> U> requires std::is_const_v<value_type>
        constexpr /*implicit*/ Texture(Texture<U, N, O> texture) noexcept :
            m_interp{texture.m_interp},
            m_border{texture.m_border},
            m_cvalue{texture.m_cvalue}
        {
            if (texture.device().is_cpu()) {
                m_variant = cpu_texture_type(std::move(texture).cpu());
            } else {
                #ifdef NOA_ENABLE_CUDA
                panic(); // TODO
                #endif
            }
        }

        /// Creates a view of an owning array.
        template<nt::almost_same_as<value_type> U> requires IS_VIEW
        constexpr /*implicit*/ Texture(const Texture<U, N, ArrayOwnership::RC>& texture) noexcept :
            m_interp{texture.m_interp},
            m_border{texture.m_border},
            m_cvalue{texture.m_cvalue}
        {
            if (texture.device().is_cpu()) {
                m_variant = cpu_texture_type(texture.cpu());
            } else {
                #ifdef NOA_ENABLE_CUDA
                panic(); // TODO
                #endif
            }
        }

        /// Creates and initializes a texture.
        /// \param[in,out] array:
        ///     Mutable array to transform into the new texture.
        ///     2D: ((b...,)h,w), 3D: ((b...,)d,h,w).
        ///     Batch dimensions (if any) should be collapsable.
        /// \param target_device:
        ///     Device where the texture should be constructed.
        ///   - If CPU, no computation is performed (other than the optional pre-filtering) and the texture simply
        ///     points to the input array, which should be on the CPU too. If the array is a view, the user should
        ///     make sure its lifetime exceeds the lifetime of the texture.
        ///   - If CUDA-capable GPU, a CUDA array is allocated with the same type and shape as the input array,
        ///     a texture is attached to this new array's memory, and the new CUDA array is initialized with the values
        ///     from the array. In this case, 1) double precision is not supported, 2) the array should be in the
        ///     rightmost order, and its depth and width dimensions should be C-contiguous. In other words, it should
        ///     be C-contiguous or have a valid "pitch" aka padded layout and 3) the array can be on any device,
        ///     including the CPU, effectively allowing to create a GPU texture from a CPU array. Note however that
        ///     while the API will make sure that stream ordering is respected (by possibly synchronizing the current
        ///     stream of the array's device), the caller should not modify the underlying values of the array until
        ///     the texture is created because the stream of the target device is not synchronized. Use Texture::eval()
        ///     after creating the texture, for example, to make sure the texture is ready.
        /// \param interp:
        ///     Interpolation mode.
        /// \param options:
        ///     Texture options.
        ///     An error is thrown if a const-valued array is passed with
        ///     options.prefilter=true and interp == CUBIC_BSPLINE.
        template<nt::array_decay_of_any<mutable_value_type> Input>
            requires (not std::is_const_v<value_type> and nt::array_size_v<Input> >= N)
        Texture(
            Input&& array,
            Device target_device,
            Interp interp,
            const CTorOptions& options = {}
        ) :
            m_interp(interp),
            m_border(options.border),
            m_cvalue(options.cvalue)
        {
            check(not array.is_empty(), "Empty array detected");

            if (options.prefilter and interp.is_almost_any(Interp::CUBIC_BSPLINE))
                nx::cubic_bspline_prefilter(array, array);

            if (target_device.is_cpu()) {
                check(array.device() == target_device,
                      "CPU textures can only be constructed/updated from other CPU arrays, but got array:device={}",
                      array.device());
                // Save the array (shallow copy) as collapsed.
                m_variant = std::forward<Input>(array).template as<value_type, N + 1>();
            } else {
                #ifdef NOA_ENABLE_CUDA
                if constexpr (sizeof(nt::value_type_t<value_type>) >= 8) {
                    panic("Double-precision textures are not supported by the CUDA backend");
                } else {
                    auto texture = gpu_texture_type{};
                    texture.handle = std::make_shared<typename gpu_texture_type::Managed>();

                    // Allocate the CUDA array.
                    auto& cuda_array = texture.handle->array;
                    auto cuda_device = noa::cuda::Device(target_device.id(), Unchecked{});
                    cuda_array = noa::cuda::AllocatorArray::allocate<value_type>(array.shape(), cuda_device);

                    // Copy input into CUDA array.
                    if (target_device != array.device())
                        array.eval();
                    auto& cuda_stream = Stream::current(target_device).cuda();
                    noa::cuda::copy(array.get(), array.strides(), cuda_array.get(), array.shape(), cuda_stream);
                    cuda_stream.enqueue_attach(std::forward<VArray>(array), texture.handle);

                    // Create the texture.
                    texture.handle->texture = noa::xform::cuda::AllocatorTexture::allocate(
                        cuda_array.get(), interp, options.border);

                    m_variant = std::move(texture);
                    m_options = ArrayOption{target_device, Allocator::CUDA_ARRAY};
                }
                #else
                panic();
                #endif
            }
        }

        /// Creates a texture, postponing texel initialization.
        /// \param shape:
        ///     Shape of the new texture.
        ///     2D: ((b...,)h,w), 3D: ((b...,)d,h,w).
        /// \param target_device:
        ///     Device where the texture should be constructed.
        ///   - If CPU, no computation is performed, and the shape and device are not saved.
        ///     The underlying managed array is empty and one should use update() to set the texture.
        ///   - If CUDA-capable GPU, a CUDA array of type T and shape is allocated, and a texture is attached to the new
        ///     array's memory. The new CUDA array is left uninitialized and one should use update() to set the texture.
        ///     In this case, double precision is not supported.
        /// \param interp:
        ///     Interpolation mode.
        /// \param options:
        ///     Texture options (prefilter is ignored).
        Texture(
            const shape_type& shape,
            Device target_device,
            Interp interp,
            const CTorOptions& options = {}
        ) requires (not std::is_const_v<value_type>) :
            m_interp(interp),
            m_border(options.border),
            m_cvalue(options.cvalue)
        {
            if (target_device.is_cpu()) {
                m_variant = cpu_texture_type{};
                (void) shape;
            } else {
                #ifdef NOA_ENABLE_CUDA
                if constexpr (sizeof(nt::value_type_t<value_type>) >= 8) {
                    panic("Double-precision textures are not supported by the CUDA backend");
                } else {
                    auto texture = gpu_texture_type{};
                    texture.handle = std::make_shared<typename gpu_texture_type::Managed>();
                    texture.handle->array = noa::cuda::AllocatorArray::allocate<value_type>(
                        shape, noa::cuda::Device(target_device.id(), Unchecked{}));
                    texture.handle->texture = noa::xform::cuda::AllocatorTexture::allocate(
                       texture.handle->array.get(), interp, options.border);
                    m_variant = std::move(texture);
                    m_options = ArrayOption{target_device, Allocator::CUDA_ARRAY};
                }
                #else
                panic();
                #endif
            }
        }

    public: // Copy
        /// Updates the texture values.
        /// \param[in,out] array:
        ///     Mutable array to wrap/copy into the texture.
        ///     2D: ((b...,)h,w), 3D: ((b...,)d,h,w).
        ///     Batch dimensions (if any) should be collapsable.
        ///   - With CPU textures, no computation is performed (other than the optional pre-filtering) and the
        ///     texture array is simply updated to point to this array instead (which should be a CPU array).
        ///     If it is a view, the caller should make sure its lifetime exceeds the lifetime of the texture.
        ///   - With GPU textures, the array should have the same collapsed shape as the texture and a deep copy is
        ///     performed from the array to the managed texture data. In this case, 1) the array should be in the
        ///     rightmost order and its depth and width dimensions should be C-contiguous. In other words, it should
        ///     be C-contiguous or have a valid "pitch" aka padded layout. 2) the array can be on any device, including
        ///     the CPU, effectively allowing to create a GPU texture from a CPU array. Note however that while the
        ///     API will make sure that stream ordering is respected (by possibly synchronizing the current stream of
        ///     the array's device), the caller should not modify the underlying values of the array until the texture
        ///     is created because the stream of the target device is not synchronized. Use Texture::eval() after
        ///     creating the texture, for example, to make sure the texture is ready.
        /// \param options:
        ///     Update options.
        template<usize N0> requires (not std::is_const_v<value_type>)
        void update(Array<value_type, N0, O> array, UpdateOptions options = {}) {
            check(not is_empty(), "Trying to update an empty texture is not allowed. Create a valid the texture first");
            check(not array.is_empty(), "Empty array detected");

            if (options.prefilter and m_interp.is_almost_any(Interp::CUBIC_BSPLINE))
                nx::cubic_bspline_prefilter(array, array);

            const Device device_target = device();
            if (device_target.is_cpu()) {
                check(array.device() == device_target,
                      "CPU textures can only be constructed/updated from CPU arrays, but got array:device={}",
                      array.device());
                cpu() = std::move(array).template as<value_type, N + 1>();
            } else {
                #ifdef NOA_ENABLE_CUDA
                check(array.shape() == m_shape,
                  "The input array should have the same shape as the texture, "
                  "but got texture:shape={} and array:shape={}",
                  m_shape, array.shape());
                if constexpr (sizeof(nt::value_type_t<value_type>) >= 8) {
                    panic("Double-precision textures are not supported by the CUDA backend");
                } else {
                    if (device_target != array.device())
                        array.eval();

                    // Update the CUDA array with the new values.
                    auto& handle = cuda_().handle;
                    auto& cuda_stream = Stream::current(device_target).cuda();
                    noa::cuda::copy(array.get(), array.strides(), handle->array.get(), m_shape, cuda_stream);
                    cuda_stream.enqueue_attach(std::forward<VArray>(array), handle);
                }
                #else
                panic();
                #endif
            }
        }

    public: // Getters
        [[nodiscard]] constexpr auto interp() const noexcept -> Interp { return m_interp; }
        [[nodiscard]] constexpr auto border() const noexcept -> Border { return m_border; }
        [[nodiscard]] constexpr auto cvalue() const noexcept -> value_type { return m_cvalue; }

        [[nodiscard]] constexpr auto options() const noexcept -> ArrayOption {
            return std::visit([](auto&& v) { return v.options(); }, m_variant);
        }
        [[nodiscard]] constexpr auto device() const noexcept -> Device { return options().device; }

        [[nodiscard]] constexpr auto allocator() const noexcept -> Allocator { return options().allocator; }

        [[nodiscard]] constexpr auto is_empty() const noexcept -> bool {
            return std::visit([](auto&& v) { return v.is_empty(); }, m_variant);
        }

        [[nodiscard]] constexpr auto shape() const noexcept -> const shape_type& {
            return std::visit([](auto&& v) { return v.shape(); }, m_variant);
        }
        [[nodiscard]] constexpr auto strides() const noexcept -> strides_type {
            return std::visit([](auto&& v) { return v.strides(); }, m_variant);
        }
        [[nodiscard]] constexpr auto strides_full() const noexcept -> strides_type { return strides(); }

        template<char ORDER = 'C'>
        [[nodiscard]] auto is_contiguous() const noexcept -> bool {
            return std::visit([](auto&& v) { return v.template is_contiguous<ORDER>(); }, m_variant);
        }

        /// Synchronizes the current stream of the Texture's device.
        /// \details It guarantees safe access to the underlying data and indicates no operations are pending.
        auto eval() const& -> const Texture& {
            Stream::current(device()).synchronize();
            return *this;
        }
        auto eval() & -> Texture& {
            Stream::current(device()).synchronize();
            return *this;
        }
        auto eval() && -> Texture&& {
            Stream::current(device()).synchronize();
            return std::move(*this);
        }

        /// Drops the resource of this texture into the returned texture.
        auto drop() & -> Texture {
            Texture out = *this;
            *this = Texture{};
            return out;
        }
        auto drop() && -> Texture {
            Texture out = std::move(*this);
            *this = Texture{};
            return out;
        }

        /// Returns the underlying pointer of the texture.
        /// GPU textures returns false.
        [[nodiscard]] constexpr auto get() const noexcept -> value_type* {
            return std::visit([](auto&& v) { return v.get(); }, m_variant);
        }

        /// Returns a view of the underlying CPU array.
        /// GPU textures returns an empty view.
        [[nodiscard]] constexpr auto view() const noexcept -> Array<value_type, N + 1, ArrayOwnership::VIEW> {
            return std::visit([](auto&& v) { return v.view(); }, m_variant);
        }

        /// Returns a reference of the managed resource.
        [[nodiscard]] auto share() const& noexcept {
            return std::visit([]<typename U>(auto const& v) -> std::shared_ptr<void> {
                return v.share();
            }, m_variant);
        }
        [[nodiscard]] auto share() && noexcept {
            return std::visit([]<typename U>(auto&& v) -> std::shared_ptr<void> {
                return std::move(v).share();
            }, std::move(m_variant));
        }

        /// Gets the underlying CPU texture.
        /// Throws if the texture is a GPU texture.
        [[nodiscard]] auto cpu() const& -> const cpu_texture_type& {
            check(std::holds_alternative<cpu_texture_type>(m_variant),
                  "Texture is not initialized or trying to retrieve at CPU texture from a GPU texture");
            return std::get<cpu_texture_type>(m_variant);
        }
        [[nodiscard]] auto cpu() & -> cpu_texture_type& {
            check(std::holds_alternative<cpu_texture_type>(m_variant),
                  "Texture is not initialized or trying to retrieve at CPU texture from a GPU texture");
            return std::get<cpu_texture_type>(m_variant);
        }
        [[nodiscard]] auto cpu() && -> cpu_texture_type&& {
            check(std::holds_alternative<cpu_texture_type>(m_variant),
                  "Texture is not initialized or trying to retrieve at CPU texture from a GPU texture");
            return std::get<cpu_texture_type>(std::move(m_variant));
        }

        /// Gets the underlying GPU texture.
        /// Throws if the texture is a CPU texture.
        [[nodiscard]] auto gpu() const -> const gpu_texture_type& {
            #ifdef NOA_ENABLE_CUDA
            return this->cuda();
            #else
            panic("The texture is a CPU texture (no GPU backend detected)");
            #endif
        }

#ifdef NOA_ENABLE_CUDA
        [[nodiscard]] auto cuda() const -> const gpu_texture_type& {
            auto* ptr = std::get_if<gpu_texture_type>(&m_variant);
            check(ptr, "Texture is not initialized or trying to retrieve at GPU texture from a CPU texture");
            return *ptr;
        }
#endif

    private:
        variant_type m_variant{};
        Interp m_interp{};
        Border m_border{};
        value_type m_cvalue{};
    };
}

namespace noa::traits {
    template<typename T, usize N, ArrayOwnership O> struct proclaim_is_texture<nx::Texture<T, N, O>> : std::true_type {};
}

namespace noa::details {
    template<typename Int, typename T, typename I, usize N>
    requires (nt::texture<T> and N == nt::array_size_v<T> and nt::same_as<I, isize>)
    [[nodiscard]] constexpr bool is_accessor_access_safe(const T& input, const Shape<I, N>& shape) {
        return is_accessor_access_safe<Int>(input.strides_full(), shape);
    }
}
