#include "noa/common/Assert.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/PtrArray.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/PtrTexture.h"
#include "noa/gpu/cuda/geometry/Interpolator.h"
#include "noa/gpu/cuda/geometry/fft/Transform.h"

namespace {
    using namespace ::noa;

    constexpr dim3 THREADS(32, 8);

    template<bool IS_DST_CENTERED>
    __forceinline__ __device__ int32_t index2frequency_(int32_t idx, int32_t dim) {
        if constexpr(IS_DST_CENTERED)
            return idx - dim / 2;
        else
            return idx < (dim + 1) / 2 ? idx : idx - dim;
    }

    __forceinline__ __device__ cfloat_t phaseShift_(float3_t shift, float3_t freq) {
        const float factor = -math::dot(shift, freq);
        cfloat_t phase_shift;
        math::sincos(factor, &phase_shift.imag, &phase_shift.real);
        return phase_shift;
    }

    // Interpolates the (complex) value at the normalized frequency "freq".
    template<typename interpolator_t>
    inline __device__ auto interpolateFFT_(const interpolator_t& interpolator, float3_t freq, float3_t norm_shape) {
        using data_t = typename interpolator_t::data_t;
        using real_t = typename interpolator_t::real_t;

        [[maybe_unused]] real_t conj = 1;
        if (freq[2] < 0.f) {
            freq = -freq;
            if constexpr (traits::is_complex_v<data_t>)
                conj = -1;
        }

        freq[0] += 0.5f;
        freq[1] += 0.5f;
        freq *= norm_shape;
        data_t value = interpolator(freq);
        if constexpr (traits::is_complex_v<data_t>)
            value.imag *= conj;
        else
            (void) conj;
        return value;
    }

    template<bool IS_DST_CENTERED, bool APPLY_SHIFT, bool IS_IDENTITY, typename data_t, typename interpolator_t>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void transform3D_(interpolator_t interpolator,
                      Accessor<data_t, 4, uint32_t> output,
                      int3_t shape, float3_t norm_shape,
                      float33_t matrix, const float33_t* sym_matrices, uint32_t sym_count,
                      float3_t shift, float scaling, float cutoff_sqd, uint32_t blocks_x) {

        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const uint4_t gid{blockIdx.z,
                          blockIdx.y,
                          index[0] * THREADS.y + threadIdx.y,
                          index[1] * THREADS.x + threadIdx.x};
        if (gid[2] >= shape[0] || gid[3] >= shape[1] / 2 + 1)
            return;

        const int32_t w = index2frequency_<IS_DST_CENTERED>(gid[1], shape[0]);
        const int32_t v = index2frequency_<IS_DST_CENTERED>(gid[2], shape[1]);
        float3_t freq{w, v, gid[3]};
        freq /= norm_shape; // [-0.5, 0.5]
        if (math::dot(freq, freq) > cutoff_sqd) {
            output(gid) = 0;
            return;
        }

        if constexpr (!IS_IDENTITY)
            freq = matrix * freq;
        else
            (void) matrix;

        data_t value = interpolateFFT_(interpolator, freq, norm_shape);
        for (uint32_t i = 0; i < sym_count; ++i) {
            const float3_t i_freq(sym_matrices[i] * freq);
            value += interpolateFFT_(interpolator, i_freq, norm_shape);
        }

        value *= scaling;
        if constexpr (traits::is_complex_v<data_t> && APPLY_SHIFT)
            value *= phaseShift_(shift, float3_t{w, v, gid[3]});
        else
            (void) shift;

        output(gid) = value;
    }

    template<fft::Remap REMAP, typename T = void>
    constexpr bool parseRemap_() noexcept {
        using Layout = ::noa::fft::Layout;
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (!IS_SRC_CENTERED || REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL)
            static_assert(traits::always_false_v<T>);

        return IS_DST_CENTERED;
    }

    template<bool IS_DST_CENTERED, bool APPLY_SHIFT, typename data_t>
    void launch_(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                 data_t* output, dim4_t output_strides, dim4_t output_shape,
                 float33_t matrix, const geometry::Symmetry& symmetry, float3_t shift,
                 float cutoff, bool normalize, cuda::Stream& stream) {
        const auto o_strides = safe_cast<uint4_t>(output_strides);
        const auto s_shape = safe_cast<int3_t>(dim3_t(output_shape.get(1)));
        const float3_t f_shape(s_shape / 2 * 2 + int3_t(s_shape == 1)); // if odd, n-1

        cutoff = noa::math::clamp(cutoff, 0.f, 0.5f);
        cutoff *= cutoff;

        // TODO Move symmetry matrices to constant memory?
        const dim_t count = symmetry.count();
        const float33_t* symmetry_matrices = symmetry.get();
        using unique_ptr_t = cuda::memory::PtrDevice<float33_t>::alloc_unique_t;
        unique_ptr_t d_matrices = cuda::memory::PtrDevice<float33_t>::alloc(count, stream);
        cuda::memory::copy(symmetry_matrices, d_matrices.get(), count, stream);
        const float scaling = normalize ? 1 / static_cast<float>(count + 1) : 1;

        if constexpr (APPLY_SHIFT)
            shift *= math::Constants<float>::PI2 / float3_t(s_shape);

        const uint32_t blocks_x = math::divideUp(s_shape[2] / 2 + 1, static_cast<int32_t>(THREADS.x));
        const uint32_t blocks_y = math::divideUp(s_shape[1], static_cast<int32_t>(THREADS.y));
        const dim3 blocks(blocks_x * blocks_y, output_shape[1], output_shape[0]);
        const cuda::LaunchConfig config{blocks, THREADS};
        const Accessor<data_t, 4, uint32_t> output_accessor(output, o_strides);

        const bool is_identity = matrix == float33_t{};
        switch (texture_interp_mode) {
            case INTERP_NEAREST: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_NEAREST, data_t>;
                return stream.enqueue(
                        "geometry::fft::transform3D",
                        is_identity ?
                        transform3D_<IS_DST_CENTERED, APPLY_SHIFT, true, data_t, interpolator_t> :
                        transform3D_<IS_DST_CENTERED, APPLY_SHIFT, false, data_t, interpolator_t>,
                        config, interpolator_t(texture), output_accessor, s_shape, f_shape,
                        matrix, d_matrices.get(), count, shift, scaling, cutoff, blocks_x);
            }
            case INTERP_LINEAR: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_LINEAR, data_t>;
                return stream.enqueue(
                        "geometry::fft::transform3D",
                        is_identity ?
                        transform3D_<IS_DST_CENTERED, APPLY_SHIFT, true, data_t, interpolator_t> :
                        transform3D_<IS_DST_CENTERED, APPLY_SHIFT, false, data_t, interpolator_t>,
                        config, interpolator_t(texture), output_accessor, s_shape, f_shape,
                        matrix, d_matrices.get(), count, shift, scaling, cutoff, blocks_x);
            }
            case INTERP_COSINE: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_COSINE, data_t>;
                return stream.enqueue(
                        "geometry::fft::transform3D",
                        is_identity ?
                        transform3D_<IS_DST_CENTERED, APPLY_SHIFT, true, data_t, interpolator_t> :
                        transform3D_<IS_DST_CENTERED, APPLY_SHIFT, false, data_t, interpolator_t>,
                        config, interpolator_t(texture), output_accessor, s_shape, f_shape,
                        matrix, d_matrices.get(), count, shift, scaling, cutoff, blocks_x);
            }
            case INTERP_LINEAR_FAST: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_LINEAR_FAST, data_t>;
                return stream.enqueue(
                        "geometry::fft::transform3D",
                        is_identity ?
                        transform3D_<IS_DST_CENTERED, APPLY_SHIFT, true, data_t, interpolator_t> :
                        transform3D_<IS_DST_CENTERED, APPLY_SHIFT, false, data_t, interpolator_t>,
                        config, interpolator_t(texture), output_accessor, s_shape, f_shape,
                        matrix, d_matrices.get(), count, shift, scaling, cutoff, blocks_x);
            }
            case INTERP_COSINE_FAST: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_COSINE_FAST, data_t>;
                return stream.enqueue(
                        "geometry::fft::transform3D",
                        is_identity ?
                        transform3D_<IS_DST_CENTERED, APPLY_SHIFT, true, data_t, interpolator_t> :
                        transform3D_<IS_DST_CENTERED, APPLY_SHIFT, false, data_t, interpolator_t>,
                        config, interpolator_t(texture), output_accessor, s_shape, f_shape,
                        matrix, d_matrices.get(), count, shift, scaling, cutoff, blocks_x);
            }
            default:
                NOA_THROW_FUNC("transform3D", "{} is not supported", texture_interp_mode);
        }
    }

    template<fft::Remap REMAP, typename data_t>
    void launchTexture3D_(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                          data_t* output, dim4_t output_strides, dim4_t output_shape,
                          float33_t matrix, const geometry::Symmetry& symmetry, float3_t shift,
                          float cutoff, bool normalize, cuda::Stream& stream) {
        constexpr bool IS_DST_CENTERED = parseRemap_<REMAP>();
        if (any(shift != 0.f)) {
            launch_<IS_DST_CENTERED, true>(
                    texture, texture_interp_mode, output, output_strides, output_shape,
                    matrix, symmetry, shift, cutoff, normalize, stream);
        } else {
            launch_<IS_DST_CENTERED, false>(
                    texture, texture_interp_mode, output, output_strides, output_shape,
                    matrix, symmetry, {}, cutoff, normalize, stream);
        }
    }
}

namespace noa::cuda::geometry::fft {
    template<Remap REMAP, typename T, typename>
    void transform3D(const shared_t<cudaArray>& array,
                     const shared_t<cudaTextureObject_t>& texture, InterpMode texture_interp_mode,
                     const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                     float33_t matrix, const Symmetry& symmetry, float3_t shift,
                     float cutoff, bool normalize, Stream& stream) {
        NOA_ASSERT(array && texture && all(output_shape > 0));
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        launchTexture3D_<REMAP>(*texture, texture_interp_mode, output.get(), output_strides,
                                output_shape, matrix, symmetry, shift, cutoff, normalize, stream);
        stream.attach(array, texture, output, symmetry.share());
    }

    template<Remap REMAP, typename T, typename>
    void transform3D(const shared_t<T[]>& input, dim4_t input_strides,
                     const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                     float33_t matrix, const Symmetry& symmetry, float3_t shift,
                     float cutoff, InterpMode interp_mode, bool normalize, Stream& stream) {
        if (!symmetry.count())
            return transform3D<REMAP>(input, input_strides, output, output_strides, shape,
                                      matrix, shift, cutoff, interp_mode, stream);

        NOA_ASSERT(input && all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        NOA_ASSERT(indexing::isRightmost(dim3_t(input_strides.get(1))) &&
                   indexing::isContiguous(input_strides, shape.fft())[3] &&
                   indexing::isContiguous(input_strides, shape.fft())[1]);

        const size3_t shape_3d{shape[1], shape[2], shape[3] / 2 + 1};
        memory::PtrArray<T> array(shape_3d);
        memory::PtrTexture texture(array.get(), interp_mode, BORDER_ZERO);

        dim_t iter;
        dim4_t o_shape;
        if (input_strides[0] == 0) {
            iter = 1;
            o_shape = shape;
        } else {
            iter = shape[0];
            o_shape = {1, shape[1], shape[2], shape[3]};
        }
        for (dim_t i = 0; i < iter; ++i) {
            cuda::memory::copy(input.get() + i * input_strides[0], input_strides[2], array.get(), shape_3d, stream);
            launchTexture3D_<REMAP>(texture.get(), interp_mode, output.get() + i * output_strides[0], output_strides,
                                    o_shape, matrix, symmetry, shift, cutoff, normalize, stream);
        }
        stream.attach(input, output, symmetry.share(), array.share(), texture.share());
    }

    #define NOA_INSTANTIATE_TRANSFORM_3D_(T)                                                                                                                                                                                            \
    template void transform3D<Remap::HC2HC, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float33_t, const Symmetry&, float3_t, float, InterpMode, bool, Stream&);                                       \
    template void transform3D<Remap::HC2H, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float33_t, const Symmetry&, float3_t, float, InterpMode, bool, Stream&);                                        \
    template void transform3D<Remap::HC2HC, T, void>(const shared_t<cudaArray>&, const shared_t<cudaTextureObject_t>&, InterpMode, const shared_t<T[]>&, dim4_t, dim4_t, float33_t, const Symmetry&, float3_t, float, bool, Stream&);   \
    template void transform3D<Remap::HC2H, T, void>(const shared_t<cudaArray>&, const shared_t<cudaTextureObject_t>&, InterpMode, const shared_t<T[]>&, dim4_t, dim4_t, float33_t, const Symmetry&, float3_t, float, bool, Stream&)

    NOA_INSTANTIATE_TRANSFORM_3D_(float);
    NOA_INSTANTIATE_TRANSFORM_3D_(cfloat_t);
}
