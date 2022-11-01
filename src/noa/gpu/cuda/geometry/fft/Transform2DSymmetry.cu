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
        return 0;
    }

    __forceinline__ __device__ cfloat_t phaseShift_(float2_t shift, float2_t freq) {
        const float factor = -math::dot(shift, freq);
        cfloat_t phase_shift;
        math::sincos(factor, &phase_shift.imag, &phase_shift.real);
        return phase_shift;
    }

    // Interpolates the (complex) value at the normalized frequency "freq".
    template<typename interpolator_t>
    inline __device__ auto interpolateFFT_(const interpolator_t& interpolator, float2_t freq, float2_t norm_shape) {
        using data_t = typename interpolator_t::data_t;
        using real_t = typename interpolator_t::real_t;

        real_t conj = 1;
        if (freq[1] < 0.f) {
            freq = -freq;
            if constexpr (traits::is_complex_v<data_t>)
                conj = -1;
        }

        freq[0] += 0.5f;
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
    void transform2D_(interpolator_t interpolator,
                      Accessor<data_t, 3, uint32_t> output,
                      int2_t shape, float2_t norm_shape,
                      float22_t matrix, const float33_t* sym_matrices, uint32_t sym_count,
                      float2_t shift, float scaling, float cutoff_sqd) {
        const int3_t gid{blockIdx.z,
                         blockIdx.y * THREADS.y + threadIdx.y,
                         blockIdx.x * THREADS.x + threadIdx.x};
        if (gid[1] >= shape[0] || gid[2] >= shape[1] / 2 + 1)
            return;

        const int32_t v = index2frequency_<IS_DST_CENTERED>(gid[1], shape[0]);
        float2_t freq{v, gid[2]};
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
            const float33_t& m = sym_matrices[i];
            const float22_t sym_matrix{m[1][1], m[1][2],
                                       m[2][1], m[2][2]};
            const float2_t i_freq = sym_matrix * freq;
            value += interpolateFFT_(interpolator, i_freq, norm_shape);
        }

        value *= scaling;
        if constexpr (traits::is_complex_v<data_t> && APPLY_SHIFT)
            value *= phaseShift_(shift, float2_t{v, gid[2]});
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

    template<bool IS_DST_CENTERED, bool APPLY_SHIFT, typename T>
    void launch_(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                 T* output, dim4_t output_strides, dim4_t output_shape,
                 float22_t matrix, const geometry::Symmetry& symmetry, float2_t shift,
                 float cutoff, bool normalize, cuda::Stream& stream) {
        NOA_ASSERT(output_shape[1] == 1);
        const auto o_strides = safe_cast<uint3_t>(dim3_t{output_strides[0], output_strides[2], output_strides[3]});
        const auto s_shape = safe_cast<int2_t>(dim2_t(output_shape.get(2)));
        const float2_t f_shape(s_shape / 2 * 2 + int2_t(s_shape == 1)); // if odd, n-1

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
            shift *= math::Constants<float>::PI2 / float2_t(s_shape);

        const dim3 blocks(math::divideUp(s_shape[1] / 2 + 1, static_cast<int>(THREADS.x)),
                          math::divideUp(s_shape[0], static_cast<int>(THREADS.y)),
                          output_shape[0]);
        const cuda::LaunchConfig config{blocks, THREADS};
        const Accessor<T, 3, uint32_t> output_accessor(output, o_strides);

        const bool is_identity = matrix == float22_t{};
        switch (texture_interp_mode) {
            case INTERP_NEAREST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_NEAREST, T>;
                return stream.enqueue(
                        "geometry::fft::transform2D",
                        is_identity ?
                        transform2D_<IS_DST_CENTERED, APPLY_SHIFT, true, T, interpolator_t> :
                        transform2D_<IS_DST_CENTERED, APPLY_SHIFT, false, T, interpolator_t>,
                        config, interpolator_t(texture), output_accessor, s_shape, f_shape,
                        matrix, d_matrices.get(), count, shift, scaling, cutoff);
            }
            case INTERP_LINEAR: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR, T>;
                return stream.enqueue(
                        "geometry::fft::transform2D",
                        is_identity ?
                        transform2D_<IS_DST_CENTERED, APPLY_SHIFT, true, T, interpolator_t> :
                        transform2D_<IS_DST_CENTERED, APPLY_SHIFT, false, T, interpolator_t>,
                        config, interpolator_t(texture), output_accessor, s_shape, f_shape,
                        matrix, d_matrices.get(), count, shift, scaling, cutoff);
            }
            case INTERP_COSINE: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_COSINE, T>;
                return stream.enqueue(
                        "geometry::fft::transform2D",
                        is_identity ?
                        transform2D_<IS_DST_CENTERED, APPLY_SHIFT, true, T, interpolator_t> :
                        transform2D_<IS_DST_CENTERED, APPLY_SHIFT, false, T, interpolator_t>,
                        config, interpolator_t(texture), output_accessor, s_shape, f_shape,
                        matrix, d_matrices.get(), count, shift, scaling, cutoff);
            }
            case INTERP_LINEAR_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR_FAST, T>;
                return stream.enqueue(
                        "geometry::fft::transform2D",
                        is_identity ?
                        transform2D_<IS_DST_CENTERED, APPLY_SHIFT, true, T, interpolator_t> :
                        transform2D_<IS_DST_CENTERED, APPLY_SHIFT, false, T, interpolator_t>,
                        config, interpolator_t(texture), output_accessor, s_shape, f_shape,
                        matrix, d_matrices.get(), count, shift, scaling, cutoff);
            }
            case INTERP_COSINE_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_COSINE_FAST, T>;
                return stream.enqueue(
                        "geometry::fft::transform2D",
                        is_identity ?
                        transform2D_<IS_DST_CENTERED, APPLY_SHIFT, true, T, interpolator_t> :
                        transform2D_<IS_DST_CENTERED, APPLY_SHIFT, false, T, interpolator_t>,
                        config, interpolator_t(texture), output_accessor, s_shape, f_shape,
                        matrix, d_matrices.get(), count, shift, scaling, cutoff);
            }
            default:
                NOA_THROW_FUNC("transform2D", "{} is not supported", texture_interp_mode);
        }
    }

    template<fft::Remap REMAP, typename T>
    void launchTexture2D_(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                          T* output, dim4_t output_strides, dim4_t output_shape,
                          float22_t matrix, const geometry::Symmetry& symmetry, float2_t shift,
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
    void transform2D(const shared_t<cudaArray>& array,
                     const shared_t<cudaTextureObject_t>& texture, InterpMode texture_interp_mode,
                     const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                     float22_t matrix, const Symmetry& symmetry, float2_t shift,
                     float cutoff, bool normalize, Stream& stream) {
        NOA_ASSERT(array && texture && all(output_shape > 0));
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        launchTexture2D_<REMAP>(*texture, texture_interp_mode, output.get(), output_strides,
                                output_shape, matrix, symmetry, shift, cutoff, normalize, stream);
        stream.attach(array, texture, output, symmetry.share());
    }

    template<Remap REMAP, typename T, typename>
    void transform2D(const shared_t<T[]>& input, dim4_t input_strides,
                     const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                     float22_t matrix, const Symmetry& symmetry, float2_t shift,
                     float cutoff, InterpMode interp_mode, bool normalize, Stream& stream) {
        if (!symmetry.count())
            return transform2D<REMAP>(input, input_strides, output, output_strides, shape,
                                      matrix, shift, cutoff, interp_mode, stream);

        NOA_ASSERT(input && all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        NOA_ASSERT(indexing::isRightmost(dim2_t(input_strides.get(2))) &&
                   indexing::isContiguous(input_strides, shape.fft())[3]);
        NOA_ASSERT(shape[1] == 1);

        const dim3_t shape_3d{1, shape[2], shape[3] / 2 + 1};
        memory::PtrArray<T> array(shape_3d);
        memory::PtrTexture texture(array.get(), interp_mode, BORDER_ZERO);

        dim_t iter;
        dim4_t o_shape;
        if (input_strides[0] == 0) {
            iter = 1;
            o_shape = {shape[0], 1, shape[2], shape[3]};
        } else {
            iter = shape[0];
            o_shape = {1, 1, shape[2], shape[3]};
        }
        for (dim_t i = 0; i < iter; ++i) {
            memory::copy(input.get() + i * input_strides[0], input_strides[2], array.get(), shape_3d, stream);
            launchTexture2D_<REMAP>(texture.get(), interp_mode, output.get() + i * output_strides[0], output_strides,
                                    o_shape, matrix, symmetry, shift, cutoff, normalize, stream);
        }
        stream.attach(input, output, symmetry.share(), array.share(), texture.share());
    }

    #define NOA_INSTANTIATE_TRANSFORM_2D_(T)                                                                                                                                                                                            \
    template void transform2D<Remap::HC2HC, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float22_t, const Symmetry&, float2_t, float, InterpMode, bool, Stream&);                                       \
    template void transform2D<Remap::HC2H, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float22_t, const Symmetry&, float2_t, float, InterpMode, bool, Stream&);                                        \
    template void transform2D<Remap::HC2HC, T, void>(const shared_t<cudaArray>&, const shared_t<cudaTextureObject_t>&, InterpMode, const shared_t<T[]>&, dim4_t, dim4_t, float22_t, const Symmetry&, float2_t, float, bool, Stream&);   \
    template void transform2D<Remap::HC2H, T, void>(const shared_t<cudaArray>&, const shared_t<cudaTextureObject_t>&, InterpMode, const shared_t<T[]>&, dim4_t, dim4_t, float22_t, const Symmetry&, float2_t, float, bool, Stream&)

    NOA_INSTANTIATE_TRANSFORM_2D_(float);
    NOA_INSTANTIATE_TRANSFORM_2D_(cfloat_t);
}
