#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/PtrArray.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/PtrTexture.h"
#include "noa/gpu/cuda/util/Atomic.cuh"
#include "noa/gpu/cuda/util/Pointers.h"
#include "noa/gpu/cuda/geometry/Interpolate.h"
#include "noa/gpu/cuda/geometry/fft/Project.h"

// This implementation is almost identical to the CPU backend's. See implementation details/comments there.
// TODO The 3D grid seems to be a good use case for CUDA surfaces. Might worth a try.

namespace {
    using namespace ::noa;
    constexpr dim3 THREADS(32, 8);

    template<bool IS_CENTERED>
    __device__ __forceinline__ int getIndex_(int frequency, int volume_dim) {
        if constexpr (IS_CENTERED) {
            return frequency + volume_dim / 2;
        } else {
            return frequency < 0 ? frequency + volume_dim : frequency;
        }
        return 0; // unreachable - remove false warning
    }

    template<bool IS_CENTERED>
    __device__ __forceinline__ int getFrequency_(int index, int shape) {
        if constexpr (IS_CENTERED)
            return index - shape / 2;
        else
            return index < (shape + 1) / 2 ? index : index - shape;
    }

    __device__ inline void setGriddingWeights_(int3_t base0, float3_t freq, float o_weights[2][2][2]) {
        float3_t fraction[2];
        fraction[1] = freq - float3_t{base0};
        fraction[0] = 1.f - fraction[1];
        for (size_t w = 0; w < 2; ++w)
            for (size_t v = 0; v < 2; ++v)
                for (size_t u = 0; u < 2; ++u)
                    o_weights[w][v][u] = fraction[w][0] * fraction[v][1] * fraction[u][2];
    }

    __device__ inline void setBoundary_(int3_t base0, int3_t shape, bool2_t o_bound[3]) {
        const int3_t base1{base0 + 1};
        const int3_t idx_max = (shape - 1) / 2;

        o_bound[0][0] = base0[0] < -idx_max[0] || base0[0] > idx_max[0];
        o_bound[0][1] = base1[0] < -idx_max[0] || base1[0] > idx_max[0];

        o_bound[1][0] = base0[1] < -idx_max[1] || base0[1] > idx_max[1];
        o_bound[1][1] = base1[1] < -idx_max[1] || base1[1] > idx_max[1];

        o_bound[2][0] = base0[2] > idx_max[2];
        o_bound[2][1] = base1[2] > idx_max[2];
    }

    template<bool IS_CENTERED, typename T>
    __device__ void addByGridding_(T* grid, uint3_t grid_stride, int3_t grid_shape, T data, float3_t frequency) {
        using real_t = traits::value_type_t<T>;
        namespace atomic = noa::cuda::util::atomic;

        const int3_t base0{math::floor(frequency)};

        float kernel[2][2][2];
        setGriddingWeights_(base0, frequency, kernel);

        bool2_t is_valid[3];
        setBoundary_(base0, grid_shape, is_valid);

        for (int w = 0; w < 2; ++w) {
            for (int v = 0; v < 2; ++v) {
                for (int u = 0; u < 2; ++u) {
                    if (is_valid[0][w] && is_valid[1][v] && is_valid[2][u]) {
                        const int idx_w = getIndex_<IS_CENTERED>(base0[0] + w, grid_shape[0]);
                        const int idx_v = getIndex_<IS_CENTERED>(base0[1] + v, grid_shape[1]);
                        const int idx_u = base0[2] + u;
                        const auto fraction = static_cast<real_t>(kernel[w][v][u]);
                        atomic::add(grid + indexing::at(idx_w, idx_v, idx_u, grid_stride), data * fraction);
                    }
                }
            }
        }

        if (base0[2] == 0 && (base0[1] != 0 || base0[0] != 0)) {
            if constexpr (traits::is_complex_v<T>)
                data.imag = -data.imag;
            for (int w = 0; w < 2; ++w) {
                for (int v = 0; v < 2; ++v) {
                    if (is_valid[0][w] && is_valid[1][v]) {
                        const int idx_w = getIndex_<IS_CENTERED>(-(base0[0] + w), grid_shape[0]);
                        const int idx_v = getIndex_<IS_CENTERED>(-(base0[1] + v), grid_shape[1]);
                        const auto fraction = static_cast<real_t>(kernel[w][v][0]);
                        atomic::add(grid + indexing::at(idx_w, idx_v, grid_stride), data * fraction);
                    }
                }
            }
        }
    }

    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED, typename T>
    __global__ void __launch_bounds__(THREADS.x * THREADS.y)
    fourierInsert_(const T* slice, uint3_t slice_stride, int2_t slice_shape, float2_t f_slice_shape,
                   T* grid, uint3_t grid_stride, int3_t grid_shape, float3_t f_grid_shape,
                   const float22_t* inv_scaling_factors, const float33_t* rotations,
                   float cutoff_sqd, float2_t ews_diam_inv) {
        using real_t = traits::value_type_t<T>;
        const int3_t gid{blockIdx.z,
                         blockIdx.y * THREADS.y + threadIdx.y,
                         blockIdx.x * THREADS.x + threadIdx.x};
        if (gid[1] >= slice_shape[0] || gid[2] >= slice_shape[1])
            return;

        const int v = getFrequency_<IS_SRC_CENTERED>(gid[1], slice_shape[0]);
        const float2_t orig_freq{v, gid[2]};
        float2_t freq_2d = orig_freq / f_slice_shape;

        if (inv_scaling_factors)
            freq_2d = inv_scaling_factors[gid[0]] * freq_2d;

        const float z = math::sum(ews_diam_inv * freq_2d * freq_2d);
        float3_t freq_3d{z, freq_2d[0], freq_2d[1]};
        freq_3d = rotations[gid[0]] * freq_3d;

        if (math::dot(freq_3d, freq_3d) > cutoff_sqd)
            return;

        real_t conj = 1;
        if (freq_3d[2] < 0) {
            freq_3d = -freq_3d;
            if constexpr(traits::is_complex_v<T>)
                conj = -1;
        }
        freq_3d *= f_grid_shape;

        T value = slice[indexing::at(gid, slice_stride)];
        if constexpr(traits::is_complex_v<T>)
            value.imag *= conj;
        else
            (void) conj;

        addByGridding_<IS_DST_CENTERED>(grid, grid_stride, grid_shape, value, freq_3d);
    }

    template<bool IS_DST_CENTERED, typename T>
    __global__ void __launch_bounds__(THREADS.x * THREADS.y)
    fourierExtract_(cudaTextureObject_t grid, float3_t f_grid_shape,
                    T* slice, uint3_t slice_stride, int2_t slice_shape, float2_t f_slice_shape,
                    const float22_t* inv_scaling_factors, const float33_t* rotations,
                    float cutoff_sqd, float2_t ews_diam_inv) {
        using real_t = traits::value_type_t<T>;
        const int3_t gid{blockIdx.z,
                         blockIdx.y * THREADS.y + threadIdx.y,
                         blockIdx.x * THREADS.x + threadIdx.x};
        if (gid[1] >= slice_shape[0] || gid[2] >= slice_shape[1])
            return;

        // ---- Same as fourierInsert_ ---- //
        const int v = getFrequency_<IS_DST_CENTERED>(gid[1], slice_shape[0]);
        const float2_t orig_freq{v, gid[2]};
        float2_t freq_2d = orig_freq / f_slice_shape;

        if (inv_scaling_factors)
            freq_2d = inv_scaling_factors[gid[0]] * freq_2d;

        const float z = math::sum(ews_diam_inv * freq_2d * freq_2d);
        float3_t freq_3d{z, freq_2d[0], freq_2d[1]};
        freq_3d = rotations[gid[0]] * freq_3d;

        if (math::dot(freq_3d, freq_3d) > cutoff_sqd)
            return;

        real_t conj = 1;
        if (freq_3d[2] < 0) {
            freq_3d = -freq_3d;
            if constexpr(traits::is_complex_v<T>)
                conj = -1;
        }
        freq_3d *= f_grid_shape;
        // -------------------------------- //

        T value = cuda::geometry::tex3D<T, INTERP_LINEAR>(grid, freq_3d);
        if constexpr(traits::is_complex_v<T>)
            value.imag *= conj;
        else
            (void) conj;

        slice[indexing::at(gid, slice_stride)] = value;
    }

    template<bool POST_CORRECTION, typename T>
    __global__ void __launch_bounds__(THREADS.x * THREADS.y)
    correctGriddingSinc2_(const T* input, uint4_t input_stride,
                          T* output, uint4_t output_stride,
                          uint2_t shape, float3_t f_shape, float3_t half, uint blocks_x) {
        constexpr float PI = math::Constants<float>::PI;

        const uint2_t indexes = indexing::indexes(blockIdx.x, blocks_x);
        const uint4_t gid{blockIdx.z,
                          blockIdx.y,
                          indexes[0] * THREADS.y + threadIdx.y,
                          indexes[1] * THREADS.x + threadIdx.x};
        if (gid[2] >= shape[0] || gid[3] >= shape[1])
            return;

        float3_t dist{gid[1], gid[2], gid[3]};
        dist -= half;
        dist /= f_shape;

        const float radius = math::sqrt(math::dot(dist, dist));
        const float sinc = math::sinc(PI * radius);
        const T sinc2 = static_cast<T>(sinc * sinc); // > 0.05

        const uint offset = indexing::at(gid, input_stride);
        output[indexing::at(gid, output_stride)] =
                POST_CORRECTION ? input[offset] / sinc2 : input[offset] * sinc2;
    }
}

namespace noa::cuda::geometry::fft {
    template<Remap REMAP, typename T, typename>
    void insert3D(const shared_t<T[]>& slice, size4_t slice_stride, size4_t slice_shape,
                  const shared_t<T[]>& grid, size4_t grid_stride, size4_t grid_shape,
                  const shared_t<float22_t[]>& scaling_factors,
                  const shared_t<float33_t[]>& rotations,
                  float cutoff, float2_t ews_radius, Stream& stream) {
        using Layout = ::noa::fft::Layout;
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL)
            static_assert(traits::always_false_v<T>);

        NOA_ASSERT(slice_shape[1] == 1);
        NOA_ASSERT(grid_shape[0] == 1);

        // Dimensions:
        const size_t count = slice_shape[0];
        const int2_t slice_shape_{slice_shape[2], slice_shape[3]};
        const int3_t grid_shape_{grid_shape[1], grid_shape[2], grid_shape[3]};
        const uint3_t slice_stride_{slice_stride[0], slice_stride[2], slice_stride[3]};
        const uint3_t grid_stride_{grid_stride[1], grid_stride[2], grid_stride[3]};
        const float2_t f_slice_shape{slice_shape_ / 2 * 2 + int2_t{slice_shape_ == 1}};
        const float3_t f_grid_shape{grid_shape_ / 2 * 2 + int3_t{grid_shape_ == 1}};

        // Launch config:
        const uint2_t tmp{slice_shape_};
        const dim3 blocks(math::divideUp(tmp[1] / 2 + 1, THREADS.x),
                          math::divideUp(tmp[0], THREADS.y),
                          count);
        const LaunchConfig config{blocks, THREADS};

        // Some preprocessing:
        const float2_t ews_diam_inv = any(ews_radius != 0) ? 1 / (2 * ews_radius) : float2_t{};
        cutoff = math::clamp(cutoff, 0.f, 0.5f);
        cutoff *= cutoff;

        // Ensure transformation parameters are accessible to the GPU:
        memory::PtrDevice<float22_t> b0;
        memory::PtrDevice<float33_t> b1;
        using namespace util;
        const auto* ptr0 = scaling_factors ? ensureDeviceAccess(scaling_factors.get(), stream, b0, count) : nullptr;
        const auto* ptr1 = ensureDeviceAccess(rotations.get(), stream, b1, count);

        stream.enqueue("geometry::fft::insert3D", fourierInsert_<IS_SRC_CENTERED, IS_DST_CENTERED, T>, config,
                       slice.get(), slice_stride_, slice_shape_, f_slice_shape,
                       grid.get(), grid_stride_, grid_shape_, f_grid_shape,
                       ptr0, ptr1, cutoff, ews_diam_inv);
        stream.attach(slice, grid, scaling_factors, rotations);
    }

    template<Remap REMAP, typename T, typename>
    void extract3D(const shared_t<T[]>& grid, size4_t grid_stride, size4_t grid_shape,
                   const shared_t<T[]>& slice, size4_t slice_stride, size4_t slice_shape,
                   const shared_t<float22_t[]>& scaling_factors,
                   const shared_t<float33_t[]>& rotations,
                   float cutoff, float2_t ews_radius, Stream& stream) {
        NOA_ASSERT(slice_shape[1] == 1);
        NOA_ASSERT(grid_shape[0] == 1);
        NOA_ASSERT(grid_stride[1] == 1 && indexing::isContiguous(grid_stride, grid_shape)[1]);

        memory::PtrArray<T> array{size3_t{grid_shape[1], grid_shape[2], grid_shape[3] / 2 + 1}};
        memory::PtrTexture texture{array.get(), INTERP_LINEAR, BORDER_ZERO}; // todo INTERP_LINEAR_FAST ?
        memory::copy(grid, grid_shape[2], array.share(), array.shape(), stream);

        extract3D<REMAP>(texture.get(), int3_t{grid_shape.get() + 1}, slice.get(), slice_stride, slice_shape,
                         scaling_factors.get(), rotations.get(), cutoff, ews_radius, stream);
        stream.attach(array.share(), texture.share(), slice, scaling_factors, rotations);
    }

    template<Remap REMAP, typename T, typename>
    void extract3D(cudaTextureObject_t grid, int3_t grid_shape,
                   T* slice, size4_t slice_stride, size4_t slice_shape,
                   const float22_t* scaling_factors, const float33_t* rotations,
                   float cutoff, float2_t ews_radius, Stream& stream) {
        using Layout = ::noa::fft::Layout;
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_NON_CENTERED ||
                      REMAP_ & Layout::SRC_FULL ||
                      REMAP_ & Layout::DST_FULL)
            static_assert(traits::always_false_v<T>);

        // Dimensions:
        const size_t count = slice_shape[0];
        const int2_t slice_shape_{slice_shape[2], slice_shape[3]};
        const uint3_t slice_stride_{slice_stride[0], slice_stride[2], slice_stride[3]};
        const float2_t f_slice_shape{slice_shape_ / 2 * 2 + int2_t{slice_shape_ == 1}};
        const float3_t f_grid_shape{grid_shape / 2 * 2 + int3_t{grid_shape == 1}};

        // Launch config:
        const uint2_t tmp{slice_shape.get() + 2};
        const dim3 blocks(math::divideUp(tmp[1] / 2 + 1, THREADS.x),
                          math::divideUp(tmp[0], THREADS.y),
                          count);
        const LaunchConfig config{blocks, THREADS};

        // Some preprocessing:
        const float2_t ews_diam_inv = any(ews_radius != 0) ? 1 / (2 * ews_radius) : float2_t{};
        cutoff = math::clamp(cutoff, 0.f, 0.5f);
        cutoff *= cutoff;

        // Ensure transformation parameters are accessible to the GPU:
        memory::PtrDevice<float22_t> b0;
        memory::PtrDevice<float33_t> b1;
        using namespace util;
        const auto* ptr0 = scaling_factors ? ensureDeviceAccess(scaling_factors, stream, b0, count) : nullptr;
        const auto* ptr1 = ensureDeviceAccess(rotations, stream, b1, count);

        stream.enqueue("geometry::fft::extract3D", fourierExtract_<IS_DST_CENTERED, T>, config,
                       grid, f_grid_shape, slice, slice_stride_, slice_shape_, f_slice_shape,
                       ptr0, ptr1, cutoff, ews_diam_inv);
    }

    template<typename T, typename>
    void griddingCorrection(const shared_t<T[]>& input, size4_t input_stride,
                            const shared_t<T[]>& output, size4_t output_stride,
                            size4_t shape, bool post_correction, Stream& stream) {
        const uint2_t shape_{shape.get() + 2};
        const uint blocks_x = math::divideUp(shape_[1], THREADS.x);
        const uint blocks_y = math::divideUp(shape_[0], THREADS.y);
        const dim3 blocks(blocks_x * blocks_y,
                          shape[1],
                          shape[0]);
        const LaunchConfig config{blocks, THREADS};

        const int3_t l_shape{shape.get() + 1};
        const float3_t f_shape{l_shape};
        const float3_t half{f_shape / 2 * float3_t{l_shape != 1}}; // if size == 1, half should be 0

        stream.enqueue("geometry::fft::griddingCorrection",
                       post_correction ? correctGriddingSinc2_<true, T> : correctGriddingSinc2_<false, T>, config,
                       input.get(), uint4_t{input_stride}, output.get(), uint4_t{output_stride},
                       shape_, f_shape, half, blocks_x);
        stream.attach(input, output);
    }

    #define NOA_INSTANTIATE_INSERT_(T, R)                                                                               \
    template void insert3D<R, T, void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t,  \
                                       const shared_t<float22_t[]>&, const shared_t<float33_t[]>&, float, float2_t, Stream&)

    #define NOA_INSTANTIATE_PROJECT_(T)         \
    NOA_INSTANTIATE_INSERT_(T, Remap::H2H);     \
    NOA_INSTANTIATE_INSERT_(T, Remap::H2HC);    \
    NOA_INSTANTIATE_INSERT_(T, Remap::HC2H);    \
    NOA_INSTANTIATE_INSERT_(T, Remap::HC2HC);   \
    template void griddingCorrection<T, void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, bool, Stream&)

    NOA_INSTANTIATE_PROJECT_(float);
    NOA_INSTANTIATE_PROJECT_(double);

    #define NOA_INSTANTIATE_EXTRACT_(T, R)                                                                              \
    template void extract3D<R, T, void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, \
                                        const shared_t<float22_t[]>&, const shared_t<float33_t[]>&, float, float2_t, Stream&)

    NOA_INSTANTIATE_EXTRACT_(float, Remap::HC2HC);
    NOA_INSTANTIATE_EXTRACT_(float, Remap::HC2H);
}
