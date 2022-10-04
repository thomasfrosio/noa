#include "noa/common/Assert.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/util/Pointers.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/PtrArray.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/PtrTexture.h"
#include "noa/gpu/cuda/geometry/Interpolate.h"
#include "noa/gpu/cuda/geometry/fft/Transform.h"

namespace {
    using namespace ::noa;
    constexpr dim3 THREADS(32, 8);

    template<bool IS_DST_CENTERED>
    __forceinline__ __device__ int32_t getFrequency_(int32_t idx, int32_t dim) {
        if constexpr(IS_DST_CENTERED)
            return idx - dim / 2;
        else
            return idx < (dim + 1) / 2 ? idx : idx - dim;
        return 0;
    }

    __forceinline__ __device__ cfloat_t getPhaseShift_(float2_t shift, float2_t freq) {
        const float factor = -math::dot(shift, freq);
        cfloat_t phase_shift;
        math::sincos(factor, &phase_shift.imag, &phase_shift.real);
        return phase_shift;
    }

    template<bool IS_DST_CENTERED, bool APPLY_SHIFT, InterpMode INTERP,
             typename T, typename MAT, typename VEC>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void transform2D_(cudaTextureObject_t tex, Accessor<T, 3, uint32_t> output,
                      int2_t shape, float2_t norm_shape,
                      MAT matrices, // const float22_t* or float22_t
                      [[maybe_unused]] VEC shifts, // const float2_t* or float2_t
                      float cutoff_sqd) {

        const int3_t gid{blockIdx.z,
                         blockIdx.y * THREADS.y + threadIdx.y,
                         blockIdx.x * THREADS.x + threadIdx.x};
        if (gid[1] >= shape[0] || gid[2] >= shape[1] / 2 + 1)
            return;

        // Compute the frequency corresponding to the gid and inverse transform.
        const int32_t v = getFrequency_<IS_DST_CENTERED>(gid[1], shape[0]);
        float2_t freq{v, gid[2]};
        freq /= norm_shape; // [-0.5, 0.5]
        if (math::dot(freq, freq) > cutoff_sqd) {
            output(gid) = 0;
            return;
        }

        if constexpr (std::is_pointer_v<MAT>)
            freq = matrices[gid[0]] * freq;
        else
            freq = matrices * freq;

        // Non-redundant transform, so flip to the valid Hermitian pair, if necessary.
        using real_t = traits::value_type_t<T>;
        [[maybe_unused]] real_t conj = 1;
        if (freq[1] < 0.f) {
            freq = -freq;
            if constexpr (traits::is_complex_v<T>)
                conj = -1;
        }

        // Convert back to index and fetch value from input texture.
        freq[0] += 0.5f; // [0, 1]
        freq *= norm_shape; // [0, N-1]
        T value = cuda::geometry::tex2D<T, INTERP>(tex, freq + 0.5f);
        if constexpr (traits::is_complex_v<T>)
            value.imag *= conj;
        else
            (void) conj;

        // Phase shift the interpolated value.
        if constexpr (traits::is_complex_v<T> && APPLY_SHIFT) {
            if constexpr (std::is_pointer_v<VEC>) {
                const float2_t shift = shifts[gid[0]] * math::Constants<float>::PI2 / float2_t(shape);
                value *= getPhaseShift_(shift, float2_t{v, gid[2]});
            } else {
                value *= getPhaseShift_(shifts, float2_t{v, gid[2]});
            }
        } else {
            (void) shifts;
        }

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

    template<bool IS_DST_CENTERED, bool APPLY_SHIFT,
             typename T, typename MAT, typename VEC>
    void launch_(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                 T* output, dim4_t output_strides, dim4_t output_shape,
                 MAT matrices, VEC shifts,
                 float cutoff, cuda::Stream& stream) {
        NOA_ASSERT(output_shape[1] == 1);
        const auto s_shape = safe_cast<int2_t>(dim2_t(output_shape.get(2)));
        const auto o_strides = safe_cast<uint3_t>(dim3_t{output_strides[0], output_strides[2], output_strides[3]});
        const float2_t f_shape(s_shape / 2 * 2 + int2_t(s_shape == 1)); // if odd, n-1

        cutoff = noa::math::clamp(cutoff, 0.f, 0.5f);
        cutoff *= cutoff;

        if constexpr (!std::is_pointer_v<VEC>)
            shifts *= math::Constants<float>::PI2 / float2_t(s_shape);

        const dim3 blocks(math::divideUp(s_shape[1] / 2 + 1, static_cast<int>(THREADS.x)),
                          math::divideUp(s_shape[0], static_cast<int>(THREADS.y)),
                          output_shape[0]);
        const cuda::LaunchConfig config{blocks, THREADS};
        const Accessor<T, 3, uint32_t> output_accessor(output, o_strides);

        NOA_ASSERT(!cuda::memory::PtrTexture::hasNormalizedCoordinates(texture));
        switch (texture_interp_mode) {
            case INTERP_NEAREST:
                return stream.enqueue(
                        "geometry::fft::transform2D",
                        transform2D_<IS_DST_CENTERED, APPLY_SHIFT, INTERP_NEAREST, T, MAT, VEC>, config,
                        texture, output_accessor, s_shape, f_shape, matrices, shifts, cutoff);
            case INTERP_LINEAR:
                return stream.enqueue(
                        "geometry::fft::transform2D",
                        transform2D_<IS_DST_CENTERED, APPLY_SHIFT, INTERP_LINEAR, T, MAT, VEC>, config,
                        texture, output_accessor, s_shape, f_shape, matrices, shifts, cutoff);
            case INTERP_COSINE:
                return stream.enqueue(
                        "geometry::fft::transform2D",
                        transform2D_<IS_DST_CENTERED, APPLY_SHIFT, INTERP_COSINE, T, MAT, VEC>, config,
                        texture, output_accessor, s_shape, f_shape, matrices, shifts, cutoff);
            case INTERP_LINEAR_FAST:
                return stream.enqueue(
                        "geometry::fft::transform2D",
                        transform2D_<IS_DST_CENTERED, APPLY_SHIFT, INTERP_LINEAR_FAST, T, MAT, VEC>, config,
                        texture, output_accessor, s_shape, f_shape, matrices, shifts, cutoff);
            case INTERP_COSINE_FAST:
                return stream.enqueue(
                        "geometry::fft::transform2D",
                        transform2D_<IS_DST_CENTERED, APPLY_SHIFT, INTERP_COSINE_FAST, T, MAT, VEC>, config,
                        texture, output_accessor, s_shape, f_shape, matrices, shifts, cutoff);
            default:
                NOA_THROW_FUNC("transform2D", "{} is not supported", texture_interp_mode);
        }
    }

    template<fft::Remap REMAP, typename T, typename M, typename S>
    void launchTexture2D_(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                          T* output, dim4_t output_strides, dim4_t shape,
                          M matrices, S shifts, float cutoff,
                          cuda::Stream& stream) {
        constexpr bool IS_DST_CENTERED = parseRemap_<REMAP>();

        bool do_shift;
        if constexpr (!traits::is_floatX_v<S>) {
            do_shift = shifts != nullptr;
        } else {
            do_shift = any(shifts != 0.f);
        }

        if (do_shift) {
            launch_<IS_DST_CENTERED, true>(
                    texture, texture_interp_mode, output, output_strides, shape,
                    matrices, shifts, cutoff, stream);
        } else {
            launch_<IS_DST_CENTERED, false>(
                    texture, texture_interp_mode, output, output_strides, shape,
                    matrices, shifts, cutoff, stream);
        }
    }
}

namespace noa::cuda::geometry::fft {
    template<Remap REMAP, typename T, typename M, typename S, typename>
    void transform2D(const shared_t<T[]>& input, dim4_t input_strides,
                     const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                     const M& matrices, const S& shifts,
                     float cutoff, InterpMode interp_mode, Stream& stream) {
        NOA_ASSERT(indexing::isRightmost(input_strides) &&
                   indexing::isContiguous(input_strides, shape.fft())[3]);
        NOA_ASSERT(shape[1] == 1);

        constexpr bool SINGLE_MATRIX = traits::is_floatXX_v<M>;
        using matrix_t = std::conditional_t<SINGLE_MATRIX, float22_t, const float22_t*>;
        matrix_t matrices_;
        cuda::memory::PtrDevice<float22_t> m_buffer;
        if constexpr (!SINGLE_MATRIX)
            matrices_ = cuda::util::ensureDeviceAccess(matrices.get(), stream, m_buffer, shape[0]);
        else
            matrices_ = matrices;

        constexpr bool SINGLE_SHIFT = traits::is_floatX_v<S>;
        using shift_t = std::conditional_t<SINGLE_SHIFT, float2_t, const float2_t*>;
        shift_t shifts_;
        cuda::memory::PtrDevice<float2_t> s_buffer;
        if constexpr (!SINGLE_SHIFT)
            shifts_ = shifts.get() ? cuda::util::ensureDeviceAccess(shifts.get(), stream, s_buffer, shape[0]) : nullptr;
        else
            shifts_ = shifts;

        const size3_t shape_3d{1, shape[2], shape[3] / 2 + 1};
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
            cuda::memory::copy(input.get() + i * input_strides[0], input_strides[2], array.get(), shape_3d, stream);
            launchTexture2D_<REMAP>(texture.get(), interp_mode,
                                    output.get() + i * output_strides[0], output_strides,
                                    o_shape, matrices_, shifts_, cutoff, stream);

            if constexpr (!SINGLE_MATRIX)
                ++matrices_;
            if constexpr (!SINGLE_SHIFT)
                ++shifts_;
        }
        stream.attach(input, output, array.share(), texture.share());
        if constexpr (!SINGLE_MATRIX)
            stream.attach(matrices);
        if constexpr (!SINGLE_SHIFT)
            stream.attach(shifts);
    }

    template<Remap REMAP, typename T, typename M, typename S, typename>
    void transform2D(const shared_t<cudaArray>& array,
                     const shared_t<cudaTextureObject_t>& texture, InterpMode texture_interp_mode,
                     const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                     const M& matrices, const S& shifts, float cutoff, Stream& stream) {
        constexpr bool SINGLE_MATRIX = traits::is_floatXX_v<M>;
        using matrix_t = std::conditional_t<SINGLE_MATRIX, float22_t, const float22_t*>;
        matrix_t matrices_;
        cuda::memory::PtrDevice<float22_t> m_buffer;
        if constexpr (!SINGLE_MATRIX)
            matrices_ = cuda::util::ensureDeviceAccess(matrices.get(), stream, m_buffer, output_shape[0]);
        else
            matrices_ = matrices;

        constexpr bool SINGLE_SHIFT = traits::is_floatX_v<S>;
        using shift_t = std::conditional_t<SINGLE_SHIFT, float2_t, const float2_t*>;
        shift_t shifts_;
        cuda::memory::PtrDevice<float2_t> s_buffer;
        if constexpr (!SINGLE_SHIFT) {
            shifts_ = shifts.get() ?
                      cuda::util::ensureDeviceAccess(shifts.get(), stream, s_buffer, output_shape[0]) :
                      nullptr;
        } else {
            shifts_ = shifts;
        }

        launchTexture2D_<REMAP>(*texture, texture_interp_mode,
                                output.get(), output_strides,
                                output_shape, matrices_, shifts_, cutoff, stream);

        stream.attach(array, texture, output);
        if constexpr (!SINGLE_MATRIX)
            stream.attach(matrices);
        if constexpr (!SINGLE_SHIFT)
            stream.attach(shifts);
    }

    #define NOA_INSTANTIATE_TRANSFORM2D_(T, M, S)                                                                                                                                                                   \
    template void transform2D<Remap::HC2H,  T, M, S, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const M&, const S&, float, InterpMode, Stream&);                                     \
    template void transform2D<Remap::HC2HC, T, M, S, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const M&, const S&, float, InterpMode, Stream&);                                     \
    template void transform2D<Remap::HC2H,  T, M, S, void>(const shared_t<cudaArray>&, const shared_t<cudaTextureObject_t>&, InterpMode, const shared_t<T[]>&, dim4_t, dim4_t, const M&, const S&, float, Stream&); \
    template void transform2D<Remap::HC2HC, T, M, S, void>(const shared_t<cudaArray>&, const shared_t<cudaTextureObject_t>&, InterpMode, const shared_t<T[]>&, dim4_t, dim4_t, const M&, const S&, float, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM2D_ALL_(T)                                     \
    NOA_INSTANTIATE_TRANSFORM2D_(T, shared_t<float22_t[]>, shared_t<float2_t[]>);   \
    NOA_INSTANTIATE_TRANSFORM2D_(T, shared_t<float22_t[]>, float2_t);               \
    NOA_INSTANTIATE_TRANSFORM2D_(T, float22_t, shared_t<float2_t[]>);               \
    NOA_INSTANTIATE_TRANSFORM2D_(T, float22_t, float2_t)

    NOA_INSTANTIATE_TRANSFORM2D_ALL_(float);
    NOA_INSTANTIATE_TRANSFORM2D_ALL_(cfloat_t);
}
