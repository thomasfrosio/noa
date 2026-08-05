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
#include "noa/xform/cuda/Allocators.hpp"
#include "noa/xform/cuda/Texture.cuh"
namespace noa::xform::details {
    template<usize N, ArrayOwnership O>
    struct TextureGPUResource {
        using object_type = noa::xform::cuda::AllocatorTexture::allocate_type::element_type;
        using share_type = std::conditional_t<O == ArrayOwnership::RC, std::shared_ptr<object_type>, const object_type*>;

        Shape<isize, N> m_shape{};
        Strides<isize, N> m_strides{};
        ArrayOption m_options{};
        share_type m_share{};

        static auto from_rc(const TextureGPUResource<N, ArrayOwnership::RC>& input) noexcept -> TextureGPUResource<N, ArrayOwnership::VIEW> {
            TextureGPUResource<N, ArrayOwnership::VIEW> output;
            output.m_shape = input.m_shape;
            output.m_strides = input.m_strides;
            output.m_options = input.m_options;
            output.m_share = input.m_share.get();
            return output;
        }

        template<char>
        constexpr auto is_contiguous() const noexcept -> bool { return false; }
        constexpr auto is_empty() const noexcept -> bool { return m_shape.is_empty(); }
        constexpr auto shape() const noexcept -> auto { return m_shape; }
        constexpr auto strides() const noexcept { return m_strides; }
        constexpr auto options() const noexcept { return m_options; }
        auto share() const& noexcept -> const share_type& { return m_share; }
        auto share() && noexcept -> share_type&& { return std::move(m_share); }
        auto view() const noexcept -> TextureGPUResource<N, ArrayOwnership::VIEW> {
            if constexpr (O == ArrayOwnership::RC) {
                return {m_shape, m_options, m_share.get()};
            } else {
                return *this;
            }
        }
    };
}
#else
namespace noa::xform::details {
    template<usize N, ArrayOwnership>
    struct TextureGPUResource {
        static auto from_rc(const TextureGPUResource<N, ArrayOwnership::RC>&) noexcept -> TextureGPUResource<N, ArrayOwnership::VIEW> { return {}; }

        template<char>
        constexpr auto is_contiguous() const noexcept -> bool { return true; }
        constexpr auto is_empty() const noexcept -> bool { return true; }
        constexpr auto shape() const noexcept -> auto { return Shape<isize, N>{}; }
        constexpr auto strides() const noexcept { return Strides<isize, N>{}; }
        constexpr auto options() const noexcept { return ArrayOption{}; }
        auto view() const noexcept -> TextureGPUResource<N, ArrayOwnership::VIEW> { return {}; }
        auto share() const noexcept -> std::shared_ptr<void> { return nullptr; }
    };
}
#endif

namespace noa::xform {
    template<typename T>
    struct TextureCTorOptions {
        /// Addressing for out-of-bounds coordinates.
        Border border{Border::ZERO};

        /// Constant value to use for out-of-bounds coordinates.
        /// Only used if border is Border::VALUE.
        T cvalue{};

        /// Whether the input array should be prefiltered, in-place.
        /// If true and if the interpolation is Interp::{CUBIC_BSPLINE|CUBIC_BSPLINE_FAST},
        /// the input is prefiltered in-place before creating/updating the texture.
        bool prefilter{true};
    };

    struct TextureUpdateOptions {
        /// Whether the input array should be prefiltered, in-place.
        /// If true and if the interpolation is Interp::{CUBIC_BSPLINE|CUBIC_BSPLINE_FAST},
        /// the input is prefiltered in-place before creating/updating the texture.
        bool prefilter{true};
    };

    /// Texture for fast 2D or 3D interpolation or caching.
    /// \details
    ///     This class template constructs and encapsulates a texture, and is meant for device agnostic use of textures.
    ///   - On the CPU, it is a simple array wrapper. On the GPU, it allocates and initializes a proper GPU texture.
    ///     3D GPU textures currently do no support batch axes, as opposed to 2D textures which support an arbitrary
    ///     number of non-empty batch axes (RANK <= N).
    ///
    /// \note
    ///     CUDA textures have limitations on the addressing mode (see Border), but the Interpolator class hides it by
    ///     adding software support for the modes that CUDA does not natively support. As such, if a GPU texture
    ///     is created with Border::{VALUE|REFLECT|NOTHING} (addressing modes that are not supported by CUDA),
    ///     hardware interpolation is turned off and the Interpolator falls back to using software interpolation and
    ///     addressing. Note that the texture is still used to read values, so the texture cache can still speed up
    ///     the interpolation. For spectrum interpolation, hardware interpolation and addressing are only supported
    ///     if the input spectrum is centered (see fftshift). Otherwise, the InterpolatorSpectrum falls back on
    ///     software interpolation and addressing. See Interpolator and InterpolatorSpectrum for more details.
    template<usize RANK, typename T, usize N, ArrayOwnership O = ArrayOwnership::RC>
    class Texture {
    public:
        static_assert(nt::almost_any_of<T, f32, f64, c32, c64>);
        static_assert(RANK == 2 or RANK == 3);
        static_assert(RANK <= N);
        static constexpr usize SIZE = N;
        static constexpr isize SSIZE = N;
        static constexpr bool IS_VIEW = O == ArrayOwnership::VIEW;
        static constexpr bool IS_CONST = std::is_const_v<T>;

        using value_type = T;
        using mutable_value_type = std::remove_const_t<T>;
        using const_value_type = std::add_const_t<mutable_value_type>;
        using shape_type = Shape<isize, N>;
        using strides_type = Strides<isize, N>;
        using cpu_texture_type = Array<value_type, N, O>;
        using gpu_texture_type = details::TextureGPUResource<N, O>;
        using variant_type = std::variant<cpu_texture_type, gpu_texture_type>;

    public:
        /// Creates an empty texture.
        constexpr Texture() = default;

        /// Creates and initializes a texture from an array.
        /// \param[in,out] array:
        ///     Mutable array to transform into the new texture.
        ///     2D: ((B...,)H,W), 3D: ((B...,)D,H,W).
        /// \param target_device:
        ///     Device where the texture should be constructed.
        ///   - If CPU, no computation is performed (other than the optional pre-filtering) and the texture simply
        ///     points to the input array, which should be on the CPU too. If the array is a view, the user should
        ///     make sure its lifetime exceeds the lifetime of the texture.
        ///   - If CUDA-capable GPU, a CUDA array is allocated with the same type and shape as the input array,
        ///     a texture is attached to this new array's memory, and the new CUDA array is initialized with the values
        ///     from the array. In this case: 1) Double precision is not supported. 2) The array should be in the
        ///     rightmost order, and its batch axes and width should be C-contiguous. In other words, it should
        ///     be C-contiguous or have a valid "pitch" aka padded layout. 3) The array can be on any device,
        ///     including the CPU, effectively allowing to create a GPU texture from a CPU array. Note however that
        ///     while the API will make sure that stream ordering is respected (by possibly synchronizing the current
        ///     stream of the array's device), the caller should not modify the underlying values of the array until
        ///     the texture is created because the stream of the target device is not synchronized. Use Texture::eval()
        ///     after creating the texture, for example, to make sure the texture is ready.
        /// \param interp:
        ///     Interpolation mode.
        /// \param options:
        ///     Texture options.
        template<ArrayOwnership O1> requires (not IS_VIEW)
        Texture(
            Array<value_type, N, O1> array,
            Device target_device,
            Interp interp,
            const TextureCTorOptions<value_type>& options = {}
        ) :
            m_interp(interp),
            m_border(options.border),
            m_cvalue(options.cvalue)
        {
            check(not array.is_empty(), "Empty array detected");

            if (options.prefilter and interp.is_almost_any(Interp::CUBIC_BSPLINE)) {
                if constexpr (IS_CONST)
                    panic("Cannot inplace prefilter a const-array. Initialize a mutable value-typed texture first, then make it const.");
                else
                    nx::cubic_bspline_prefilter<RANK>(array, array);
            }

            if (target_device.is_cpu()) {
                check(array.device() == target_device,
                      "CPU textures can only be constructed or updated from other CPU arrays, but got array:device={}",
                      array.device());
                cpu() = std::move(array);
            } else {
                #ifdef NOA_ENABLE_CUDA
                if constexpr (sizeof(nt::value_type_t<value_type>) >= 8) {
                    panic("Double-precision textures are not supported by the CUDA backend");
                } else if constexpr (IS_VIEW) {
                    panic("Cannot create a non-owning GPU texture from an array. Initialize an owning GPU texture first, then make it a view.");
                } else if constexpr (IS_CONST) {
                    panic("Cannot create a const value-typed GPU texture from an array. Initialize a mutable value-typed texture first, then make it const.");
                } else {
                    const auto [shape_3d, strides_layers] =
                        noa::xform::cuda::AllocatorTexture::array_shape_and_strides(array.shape(), RANK);

                    auto texture = gpu_texture_type{};
                    texture.m_shape = array.shape();
                    texture.m_strides = strides_layers;
                    texture.m_options = ArrayOption{.device = target_device, .allocator = Allocator::CUDA_ARRAY};

                    // Allocate the CUDA array and create the texture.
                    texture.m_share = noa::xform::cuda::AllocatorTexture::allocate<value_type>(
                        shape_3d, RANK, noa::cuda::Device(target_device.id(), Unchecked{}), interp, options.border);

                    // Copy input into CUDA array.
                    if (array.device().is_cpu())
                        array.eval();
                    auto& cuda_stream = Stream::current(target_device).cuda();
                    Strides3 strides_3d;
                    check(noa::reshape(array.shape(), array.strides(), shape_3d, strides_3d),
                          "The input array should have its batch axes collapsible, but got array:shape={} and array:strides={}, rank={}, target_shape={}",
                          array.shape(), array.strides(), RANK, shape_3d);
                    noa::cuda::copy(array.get(), strides_3d, texture.m_share->array, shape_3d, cuda_stream);
                    cuda_stream.enqueue_attach(std::move(array), texture.m_share);

                    m_variant = std::move(texture);
                }
                #else
                panic();
                #endif
            }
        }

        /// Creates a texture, postponing texel initialization.
        /// \param shape:
        ///     Shape of the new texture.
        ///     2D: ((B...,)H,W), 3D: ((B...,)D,H,W).
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
            const TextureCTorOptions<value_type>& options = {}
        ) requires (not std::is_const_v<value_type>) :
            m_interp(interp),
            m_border(options.border),
            m_cvalue(options.cvalue)
        {
            check(not shape.is_empty(), "Empty shape detected");

            if (target_device.is_cpu()) {
                m_variant = cpu_texture_type(shape, {.device = target_device, .allocator = Allocator::NONE});
            } else {
                #ifdef NOA_ENABLE_CUDA
                if constexpr (sizeof(nt::value_type_t<value_type>) >= 8) {
                    panic("Double-precision textures are not supported by the CUDA backend");
                } else if constexpr (IS_VIEW) {
                    panic("Cannot create a non-owning GPU texture from an array. Initialize an owning GPU texture first, then make it a view.");
                } else if constexpr (IS_CONST) {
                    panic("Cannot create a const value-typed GPU texture from an array. Initialize a mutable value-typed texture first, then make it const.");
                } else {
                    const auto [shape_3d, strides_layers] =
                        noa::xform::cuda::AllocatorTexture::array_shape_and_strides(shape, RANK);

                    auto texture = gpu_texture_type{};
                    texture.m_shape = shape;
                    texture.m_strides = strides_layers;
                    texture.m_options = ArrayOption{.device = target_device, .allocator = Allocator::CUDA_ARRAY};
                    texture.m_share = noa::xform::cuda::AllocatorTexture::allocate<value_type>(
                        shape_3d, RANK, noa::cuda::Device(target_device.id(), Unchecked{}), interp, options.border);
                    m_variant = std::move(texture);
                }
                #else
                panic();
                #endif
            }
        }

        /// Creates a const texture from an existing mutable texture.
        template<nt::mutable_of<value_type> U> requires std::is_const_v<value_type>
        constexpr /*implicit*/ Texture(Texture<RANK, U, N, O> texture) noexcept :
            m_interp{texture.interp()},
            m_border{texture.border()},
            m_cvalue{texture.cvalue()}
        {
            if (texture.device().is_cpu())
                m_variant = cpu_texture_type(std::move(texture).cpu());
            else
                m_variant = gpu_texture_type(std::move(texture).gpu());
        }

        /// Creates a view of an owning texture.
        template<nt::almost_same_as<value_type> U> requires IS_VIEW
        constexpr explicit Texture(const Texture<RANK, U, N>& texture) noexcept :
            m_interp{texture.interp()},
            m_border{texture.border()},
            m_cvalue{texture.cvalue()}
        {
            if (texture.device().is_cpu()) {
                m_variant = cpu_texture_type(texture.cpu());
            } else {
                m_variant = gpu_texture_type::from_rc(texture.gpu());
            }
        }

    public: // Copy
        /// Updates the texture values.
        /// \param[in,out] array:
        ///     Mutable array to wrap/copy into the texture.
        ///     2D: ((B...,)H,W), 3D: ((B...,)D,H,W).
        ///   - With CPU textures, no computation is performed (other than the optional pre-filtering) and the
        ///     texture array is simply updated to point to this array instead (which should be a CPU array).
        ///     If it is a view, the caller should make sure its lifetime exceeds the lifetime of the texture.
        ///   - With GPU textures, the array should have the same collapsed shape as the texture and a deep copy is
        ///     performed from the array to the managed texture data. In this case: 1) The array should be in the
        ///     rightmost order and its batch axes and width should be C-contiguous. In other words, it should
        ///     be C-contiguous or have a valid "pitch" aka padded layout. 2) The array can be on any device, including
        ///     the CPU, effectively allowing to create a GPU texture from a CPU array. Note however that while the
        ///     API will make sure that stream ordering is respected (by possibly synchronizing the current stream of
        ///     the array's device), the caller should not modify the underlying values of the array until the texture
        ///     is created because the stream of the target device is not synchronized. Use Texture::eval() after
        ///     creating the texture, for example, to make sure the texture is ready.
        /// \param options:
        ///     Update options.
        void update(Array<value_type, N, O> array, TextureUpdateOptions options = {}) {
            check(not array.is_empty(), "Empty array detected");
            check(not shape().is_empty(), "Trying to update an empty texture is not allowed. Create a valid the texture first");
            check(array.shape() == shape(),
                  "The input array should have the same shape as the texture, but got texture:shape={} and array:shape={}",
                  shape(), array.shape());

            if (options.prefilter and m_interp.is_almost_any(Interp::CUBIC_BSPLINE)) {
                if constexpr (IS_CONST)
                    panic("Cannot inplace prefilter a const-array. Initialize a mutable value-typed texture first, then make it const.");
                else
                    nx::cubic_bspline_prefilter<RANK>(array, array);
            }

            const Device device_target = device();
            if (device_target.is_cpu()) {
                check(array.device() == device_target,
                      "CPU textures can only be constructed or updated from CPU arrays, but got array:device={}",
                      array.device());
                cpu() = std::move(array);
            } else {
                #ifdef NOA_ENABLE_CUDA
                if constexpr (sizeof(nt::value_type_t<value_type>) >= 8) {
                    panic("Double-precision textures are not supported by the CUDA backend");
                } else if constexpr (IS_CONST) {
                    panic("Cannot update a const value-typed GPU texture from an array. Initialize a mutable value-typed texture first, then make it const.");
                } else {
                    if (array.device().is_cpu())
                        array.eval();

                    // Update the CUDA array with the new values.
                    auto& cuda_stream = Stream::current(device_target).cuda();
                    const auto shape_3d = noa::xform::cuda::AllocatorTexture::array_shape(array.shape(), RANK);
                    Strides3 strides_3d;
                    check(noa::reshape(array.shape(), array.strides(), shape_3d, strides_3d),
                          "The input array should have its batch axes collapsible, but got array:shape={} and array:strides={}, rank={}, target_shape={}",
                          array.shape(), array.strides(), RANK, shape_3d);
                    auto& gpu_resource = gpu();
                    noa::cuda::copy(array.get(), strides_3d, gpu_resource.share()->array, shape_3d, cuda_stream);
                    cuda_stream.enqueue_attach(std::move(array), gpu_resource.share());
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

        [[nodiscard]] constexpr auto shape() const noexcept -> shape_type {
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

        /// Returns a view of the texture.
        [[nodiscard]] constexpr auto view() const noexcept -> Texture<RANK, T, N, ArrayOwnership::VIEW> {
            return Texture<RANK, T, N, ArrayOwnership::VIEW>(*this);
        }

        /// Returns a reference of the managed resource.
        [[nodiscard]] auto share() const& noexcept {
            return std::visit([](const auto& v) -> std::shared_ptr<void> {
                return v.share();
            }, m_variant);
        }
        [[nodiscard]] auto share() && noexcept {
            return std::visit([](auto&& v) -> std::shared_ptr<void> {
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
        [[nodiscard]] auto gpu() const& -> const gpu_texture_type& {
            check(std::holds_alternative<gpu_texture_type>(m_variant),
                  "Texture is not initialized or trying to retrieve at GPU texture from a CPU texture");
            return std::get<gpu_texture_type>(m_variant);
        }
        [[nodiscard]] auto gpu() & -> gpu_texture_type& {
            check(std::holds_alternative<gpu_texture_type>(m_variant),
                  "Texture is not initialized or trying to retrieve at GPU texture from a CPU texture");
            return std::get<gpu_texture_type>(m_variant);
        }
        [[nodiscard]] auto gpu() && -> gpu_texture_type&& {
            check(std::holds_alternative<gpu_texture_type>(m_variant),
                  "Texture is not initialized or trying to retrieve at GPU texture from a CPU texture");
            return std::get<gpu_texture_type>(std::move(m_variant));
        }

    private:
        variant_type m_variant{};
        Interp m_interp{};
        Border m_border{};
        value_type m_cvalue{};
    };

    template<typename T, usize N, ArrayOwnership O = ArrayOwnership::RC>
    using Texture2D = Texture<2, T, N, O>;

    template<typename T, usize N, ArrayOwnership O = ArrayOwnership::RC>
    using Texture3D = Texture<3, T, N, O>;
}

namespace noa::traits {
    template<usize R, typename T, usize N, ArrayOwnership O>
    struct proclaim_is_texture<noa::xform::Texture<R, T, N, O>> : std::true_type {};
}

namespace noa::details {
    template<typename Int, typename T, typename I, usize N>
    requires (nt::texture<T> and N == nt::array_size_v<T> and nt::same_as<I, isize>) // FIXME gpu strides?
    [[nodiscard]] constexpr bool is_accessor_access_safe(const T& input, const Shape<I, N>& shape) {
        return is_accessor_access_safe<Int>(input.strides_full(), shape);
    }
}
