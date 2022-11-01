#include "noa/common/geometry/Interpolator.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/PtrArray.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/PtrTexture.h"
#include "noa/gpu/cuda/util/Atomic.cuh"
#include "noa/gpu/cuda/util/Pointers.h"
#include "noa/gpu/cuda/geometry/Interpolator.h"
#include "noa/gpu/cuda/geometry/fft/Project.h"

// This implementation is almost identical to the CPU backend's.
// See implementation details/comments there.
namespace {
    using namespace ::noa;
    constexpr dim3 THREADS(32, 8);

    template<typename T>
    struct GridNoTexture {
        const T* __restrict__ ptr;
        uint3_t strides;
        int3_t shape;
    };

    template<bool IS_CENTERED>
    [[nodiscard]] __device__ __forceinline__ int32_t getIndex_(int32_t frequency, int32_t volume_dim) {
        if constexpr (IS_CENTERED) {
            return frequency + volume_dim / 2;
        } else {
            return frequency < 0 ? frequency + volume_dim : frequency;
        }
        return 0; // unreachable - remove false warning
    }

    template<bool IS_CENTERED>
    [[nodiscard]] __device__ __forceinline__ int32_t getFrequency_(int32_t index, int32_t shape) {
        if constexpr (IS_CENTERED)
            return index - shape / 2;
        else
            return index < (shape + 1) / 2 ? index : index - shape;
        return 0; // unreachable
    }

    __device__ inline void setGriddingWeights_(int3_t base0, float3_t freq, float o_weights[2][2][2]) {
        float3_t fraction[2];
        fraction[1] = freq - float3_t(base0);
        fraction[0] = 1.f - fraction[1];
        for (int64_t w = 0; w < 2; ++w)
            for (int64_t v = 0; v < 2; ++v)
                for (int64_t u = 0; u < 2; ++u)
                    o_weights[w][v][u] = fraction[w][0] * fraction[v][1] * fraction[u][2];
    }

    __device__ inline void setBoundary_(int3_t base0, int3_t shape, bool2_t o_bound[3]) {
        const int3_t base1(base0 + 1);
        const int3_t idx_max = (shape - 1) / 2;

        o_bound[0][0] = base0[0] >= -idx_max[0] && base0[0] <= idx_max[0];
        o_bound[0][1] = base1[0] >= -idx_max[0] && base1[0] <= idx_max[0];

        o_bound[1][0] = base0[1] >= -idx_max[1] && base0[1] <= idx_max[1];
        o_bound[1][1] = base1[1] >= -idx_max[1] && base1[1] <= idx_max[1];

        o_bound[2][0] = base0[2] <= idx_max[2];
        o_bound[2][1] = base1[2] <= idx_max[2];
    }

    template<bool IS_CENTERED, typename T>
    __device__ void addByGridding_(T* grid, uint3_t grid_strides, int3_t grid_shape, T data, float3_t frequency) {
        using real_t = traits::value_type_t<T>;
        namespace atomic = noa::cuda::util::atomic;

        const int3_t base0(math::floor(frequency));

        float kernel[2][2][2];
        setGriddingWeights_(base0, frequency, kernel);

        bool2_t is_valid[3];
        setBoundary_(base0, grid_shape, is_valid);

        for (int32_t w = 0; w < 2; ++w) {
            for (int32_t v = 0; v < 2; ++v) {
                for (int32_t u = 0; u < 2; ++u) {
                    if (is_valid[0][w] && is_valid[1][v] && is_valid[2][u]) {
                        const int32_t idx_w = getIndex_<IS_CENTERED>(base0[0] + w, grid_shape[0]);
                        const int32_t idx_v = getIndex_<IS_CENTERED>(base0[1] + v, grid_shape[1]);
                        const int32_t idx_u = base0[2] + u;
                        const auto fraction = static_cast<real_t>(kernel[w][v][u]);
                        atomic::add(grid + indexing::at(idx_w, idx_v, idx_u, grid_strides), data * fraction);
                    }
                }
            }
        }

        if (base0[2] == 0 && (base0[1] != 0 || base0[0] != 0)) {
            if constexpr (traits::is_complex_v<T>)
                data.imag = -data.imag;
            for (int32_t w = 0; w < 2; ++w) {
                for (int32_t v = 0; v < 2; ++v) {
                    if (is_valid[0][w] && is_valid[1][v]) {
                        const int32_t idx_w = getIndex_<IS_CENTERED>(-(base0[0] + w), grid_shape[0]);
                        const int32_t idx_v = getIndex_<IS_CENTERED>(-(base0[1] + v), grid_shape[1]);
                        const auto fraction = static_cast<real_t>(kernel[w][v][0]);
                        atomic::add(grid + indexing::at(idx_w, idx_v, grid_strides), data * fraction);
                    }
                }
            }
        }
    }

    template<typename T>
    [[nodiscard]] __device__ T linear3D_(const T* __restrict__ grid, uint3_t strides, int3_t shape, float3_t frequency) {
        int3_t idx[2];
        idx[0] = int3_t(noa::math::floor(frequency));
        idx[1] = idx[0] + 1;

        const bool cond_z[2] = {idx[0][0] >= 0 && idx[0][0] < shape[0], idx[1][0] >= 0 && idx[1][0] < shape[0]};
        const bool cond_y[2] = {idx[0][1] >= 0 && idx[0][1] < shape[1], idx[1][1] >= 0 && idx[1][1] < shape[1]};
        const bool cond_x[2] = {idx[0][2] >= 0 && idx[0][2] < shape[2], idx[1][2] >= 0 && idx[1][2] < shape[2]};

        const uint32_t off_z[2] = {idx[0][0] * strides[0], idx[1][0] * strides[0]};
        const uint32_t off_y[2] = {idx[0][1] * strides[1], idx[1][1] * strides[1]};
        const uint32_t off_x[2] = {idx[0][2] * strides[2], idx[1][2] * strides[2]};

        const float ry = frequency[1] - static_cast<float>(idx[0][1]);
        const float rx = frequency[2] - static_cast<float>(idx[0][2]);

        const T v000 = cond_z[0] && cond_y[0] && cond_x[0] ? grid[off_z[0] + off_y[0] + off_x[0]] : T{0};
        const T v001 = cond_z[0] && cond_y[0] && cond_x[1] ? grid[off_z[0] + off_y[0] + off_x[1]] : T{0};
        const T v010 = cond_z[0] && cond_y[1] && cond_x[0] ? grid[off_z[0] + off_y[1] + off_x[0]] : T{0};
        const T v011 = cond_z[0] && cond_y[1] && cond_x[1] ? grid[off_z[0] + off_y[1] + off_x[1]] : T{0};
        T tmp1 = geometry::interpolate::lerp2D(v000, v001, v010, v011, rx, ry);

        const T v100 = cond_z[1] && cond_y[0] && cond_x[0] ? grid[off_z[1] + off_y[0] + off_x[0]] : T{0};
        const T v101 = cond_z[1] && cond_y[0] && cond_x[1] ? grid[off_z[1] + off_y[0] + off_x[1]] : T{0};
        const T v110 = cond_z[1] && cond_y[1] && cond_x[0] ? grid[off_z[1] + off_y[1] + off_x[0]] : T{0};
        const T v111 = cond_z[1] && cond_y[1] && cond_x[1] ? grid[off_z[1] + off_y[1] + off_x[1]] : T{0};
        T tmp2 = geometry::interpolate::lerp2D(v100, v101, v110, v111, rx, ry);

        const float rz = frequency[0] - static_cast<float>(idx[0][0]);
        return geometry::interpolate::lerp1D(tmp1, tmp2, rz);
    }

    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED, typename T, typename S, typename R>
    __global__ void __launch_bounds__(THREADS.x * THREADS.y)
    fourierInsert_(const T* __restrict__ slice, uint3_t slice_strides, int2_t slice_shape_fft, float2_t f_slice_shape,
                   T* __restrict__ grid, uint3_t grid_strides, int3_t grid_shape,
                   S inv_scaling_factors, R rotations,
                   float cutoff_sqd, float3_t f_target_shape, float2_t ews_diam_inv) {
        using real_t = traits::value_type_t<T>;
        const int3_t gid{blockIdx.z,
                         blockIdx.y * THREADS.y + threadIdx.y,
                         blockIdx.x * THREADS.x + threadIdx.x};
        if (gid[1] >= slice_shape_fft[0] || gid[2] >= slice_shape_fft[1])
            return;

        const int32_t v = getFrequency_<IS_SRC_CENTERED>(gid[1], slice_shape_fft[0]);
        const float2_t orig_freq{v, gid[2]};
        float2_t freq_2d = orig_freq / f_slice_shape;

        if constexpr (std::is_pointer_v<S>) {
            if (inv_scaling_factors)
                freq_2d = inv_scaling_factors[gid[0]] * freq_2d;
        } else {
            freq_2d = inv_scaling_factors * freq_2d;
        }

        const float z = math::sum(ews_diam_inv * freq_2d * freq_2d);
        float3_t freq_3d{z, freq_2d[0], freq_2d[1]};
        if constexpr (std::is_pointer_v<R>)
            freq_3d = rotations[gid[0]] * freq_3d;
        else
            freq_3d = rotations * freq_3d;

        if (math::dot(freq_3d, freq_3d) > cutoff_sqd)
            return;

        real_t conj = 1;
        if (freq_3d[2] < 0) {
            freq_3d = -freq_3d;
            if constexpr(traits::is_complex_v<T>)
                conj = -1;
        }
        freq_3d *= f_target_shape;

        T value = slice[indexing::at(gid, slice_strides)];
        if constexpr(traits::is_complex_v<T>)
            value.imag *= conj;
        else
            (void) conj;

        addByGridding_<IS_DST_CENTERED>(grid, grid_strides, grid_shape, value, freq_3d);
    }

    template<bool IS_DST_CENTERED, typename data_t, typename interpolator_t, typename scale_t, typename rotate_t>
    __global__ void __launch_bounds__(THREADS.x * THREADS.y)
    fourierExtract_(interpolator_t grid,
                    data_t* __restrict__ slice, uint3_t slice_strides, int2_t slice_shape_fft, float2_t f_slice_shape,
                    scale_t inv_scaling_factors, rotate_t rotations,
                    float cutoff_sqd, float3_t f_target_shape, float2_t f_offset, float2_t ews_diam_inv) {
        using real_t = traits::value_type_t<data_t>;
        const int3_t gid{blockIdx.z,
                         blockIdx.y * THREADS.y + threadIdx.y,
                         blockIdx.x * THREADS.x + threadIdx.x};
        if (gid[1] >= slice_shape_fft[0] || gid[2] >= slice_shape_fft[1])
            return;

        // -------------------------------- //
        const int32_t v = getFrequency_<IS_DST_CENTERED>(gid[1], slice_shape_fft[0]);
        const float2_t orig_freq{v, gid[2]};
        float2_t freq_2d = orig_freq / f_slice_shape;
        if constexpr (std::is_pointer_v<scale_t>) {
            if (inv_scaling_factors)
                freq_2d = inv_scaling_factors[gid[0]] * freq_2d;
        } else {
            freq_2d = inv_scaling_factors * freq_2d;
        }

        const float z = math::sum(ews_diam_inv * freq_2d * freq_2d);
        float3_t freq_3d{z, freq_2d[0], freq_2d[1]};
        if constexpr (std::is_pointer_v<rotate_t>)
            freq_3d = rotations[gid[0]] * freq_3d;
        else
            freq_3d = rotations * freq_3d;

        if (math::dot(freq_3d, freq_3d) > cutoff_sqd) {
            slice[indexing::at(gid, slice_strides)] = data_t{0};
            return;
        }

        real_t conj = 1;
        if (freq_3d[2] < 0) {
            freq_3d = -freq_3d;
            if constexpr(traits::is_complex_v<data_t>)
                conj = -1;
        }
        freq_3d *= f_target_shape; // [-0.5 <- 0 -> 0.5] to [-N//2 <- 0 -> N//2]
        freq_3d[0] += f_offset[0]; // offset to the grid center index
        freq_3d[1] += f_offset[1];
        // -------------------------------- //

        data_t value = grid(freq_3d);
        if constexpr(traits::is_complex_v<data_t>)
            value.imag *= conj;
        else
            (void) conj;

        slice[indexing::at(gid, slice_strides)] = value;
    }

    template<bool POST_CORRECTION, typename T>
    __global__ void __launch_bounds__(THREADS.x * THREADS.y)
    correctGriddingSinc2_(Accessor<const T, 4, uint32_t> input,
                          Accessor<T, 4, uint32_t> output,
                          uint2_t shape, float3_t f_shape, float3_t half, uint32_t blocks_x) {
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
        output(gid) = POST_CORRECTION ? input(gid) / sinc2 : input(gid) * sinc2;
    }

    template<typename T>
    using matrixOrRawConstPtr_t = std::conditional_t<
            traits::is_floatXX_v<T>,
            traits::remove_ref_cv_t<T>,
            const typename traits::element_type_t<T>*>;

    template<typename T>
    auto matrixOrRawConstPtr(T v) {
        using output_t = matrixOrRawConstPtr_t<T>;
        if constexpr (traits::is_floatXX_v<T>) {
            return output_t(v);
        } else {
            return static_cast<output_t>(v.get());
        }
    }

    template<fft::Remap REMAP, typename data_t, typename interpolator_t, typename scale_t, typename rotate_t>
    void launchExtract3D_(interpolator_t grid, int3_t grid_shape,
                          data_t* slice, dim4_t slice_strides, dim4_t slice_shape,
                          scale_t scaling_factors, rotate_t rotations,
                          float cutoff, dim4_t target_shape, float2_t ews_radius, cuda::Stream& stream) {
        using Layout = ::noa::fft::Layout;
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_NON_CENTERED ||
                      REMAP_ & Layout::SRC_FULL ||
                      REMAP_ & Layout::DST_FULL)
            static_assert(traits::always_false_v<data_t>);

        // Dimensions:
        const dim_t count = slice_shape[0];
        const auto slice_shape_ = safe_cast<int2_t>(dim2_t{slice_shape[2], slice_shape[3]});
        const auto slice_strides_ = safe_cast<uint3_t>(dim3_t{slice_strides[0], slice_strides[2], slice_strides[3]});
        const float2_t f_slice_shape(slice_shape_ / 2 * 2 + int2_t(slice_shape_ == 1));

        // Use the grid shape as backup.
        const int3_t target_shape_ = any(target_shape == 0) ? grid_shape : int3_t(target_shape.get(1));
        const float3_t f_target_shape(target_shape_ / 2 * 2 + int3_t(target_shape_ == 1));
        const float2_t f_offset(grid_shape[0] / 2, grid_shape[1] / 2); // grid ZY center

        // Launch config:
        const uint2_t tmp(slice_shape.get(2));
        const dim3 blocks(math::divideUp(tmp[1] / 2 + 1, THREADS.x),
                          math::divideUp(tmp[0], THREADS.y),
                          count);
        const cuda::LaunchConfig config{blocks, THREADS};

        // Some preprocessing:
        const float2_t ews_diam_inv = any(ews_radius != 0) ? 1 / (2 * ews_radius) : float2_t{};
        cutoff = math::clamp(cutoff, 0.f, 0.5f);
        cutoff *= cutoff;

        // Ensure transformation parameters are accessible to the GPU:
        cuda::memory::PtrDevice<float22_t> b0;
        using scaling_t = matrixOrRawConstPtr_t<scale_t>;
        scaling_t ptr0 = matrixOrRawConstPtr(scaling_factors);
        if constexpr (std::is_pointer_v<scaling_t>)
            ptr0 = scaling_factors ? cuda::util::ensureDeviceAccess(ptr0, stream, b0, count) : nullptr;

        cuda::memory::PtrDevice<float33_t> b1;
        using rotation_t = matrixOrRawConstPtr_t<rotate_t>;
        rotation_t ptr1 = matrixOrRawConstPtr(rotations);
        if constexpr (std::is_pointer_v<rotation_t>) {
            NOA_ASSERT(ptr1 != nullptr);
            ptr1 = cuda::util::ensureDeviceAccess(ptr1, stream, b1, count);
        }

        stream.enqueue("geometry::fft::extract3D",
                       fourierExtract_<IS_DST_CENTERED, data_t, interpolator_t, scaling_t, rotation_t>, config,
                       grid, slice, slice_strides_, slice_shape_.fft(), f_slice_shape,
                       ptr0, ptr1, cutoff, f_target_shape, f_offset, ews_diam_inv);
    }
}

namespace noa::cuda::geometry::fft {
    template<Remap REMAP, typename T, typename S, typename R, typename>
    void insert3D(const shared_t<T[]>& slice, dim4_t slice_strides, dim4_t slice_shape,
                  const shared_t<T[]>& grid, dim4_t grid_strides, dim4_t grid_shape,
                  const S& scaling_factors, const R& rotations,
                  float cutoff, dim4_t target_shape, float2_t ews_radius, Stream& stream) {
        using Layout = ::noa::fft::Layout;
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL)
            static_assert(traits::always_false_v<T>);

        NOA_ASSERT(slice.get() != grid.get() && all(slice_shape > 0) && all(grid_shape > 0));
        NOA_ASSERT_DEVICE_PTR(slice.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(grid.get(), stream.device());
        NOA_ASSERT(slice_shape[1] == 1);
        NOA_ASSERT(grid_shape[0] == 1);

        // Dimensions:
        const dim_t count = slice_shape[0];
        const auto slice_shape_ = safe_cast<int2_t>(dim2_t{slice_shape[2], slice_shape[3]});
        const auto grid_shape_ = safe_cast<int3_t>(dim3_t(grid_shape.get(1)));
        const auto slice_strides_ = safe_cast<uint3_t>(dim3_t{slice_strides[0], slice_strides[2], slice_strides[3]});
        const auto grid_strides_ = safe_cast<uint3_t>(dim3_t(grid_strides.get(1)));
        const float2_t f_slice_shape(slice_shape_ / 2 * 2 + int2_t(slice_shape_ == 1));

        // Use the grid shape as backup.
        const int3_t target_shape_ = any(target_shape == 0) ? grid_shape_ : int3_t(target_shape.get(1));
        const float3_t f_target_shape(target_shape_ / 2 * 2 + int3_t(target_shape_ == 1));

        // Launch config:
        const uint2_t tmp(slice_shape_);
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
        using scaling_t = matrixOrRawConstPtr_t<S>;
        scaling_t ptr0 = matrixOrRawConstPtr(scaling_factors);
        if constexpr (std::is_pointer_v<scaling_t>)
            ptr0 = scaling_factors ? util::ensureDeviceAccess(ptr0, stream, b0, count) : nullptr;

        memory::PtrDevice<float33_t> b1;
        using rotation_t = matrixOrRawConstPtr_t<R>;
        rotation_t ptr1 = matrixOrRawConstPtr(rotations);
        if constexpr (std::is_pointer_v<rotation_t>) {
            NOA_ASSERT(ptr1 != nullptr);
            ptr1 = util::ensureDeviceAccess(ptr1, stream, b1, count);
        }

        stream.enqueue("geometry::fft::insert3D",
                       fourierInsert_<IS_SRC_CENTERED, IS_DST_CENTERED, T, scaling_t, rotation_t>, config,
                       slice.get(), slice_strides_, slice_shape_.fft(), f_slice_shape,
                       grid.get(), grid_strides_, grid_shape_,
                       ptr0, ptr1, cutoff, f_target_shape, ews_diam_inv);
        stream.attach(slice, grid);
        if constexpr (std::is_pointer_v<scaling_t>)
            stream.attach(scaling_factors);
        if constexpr (std::is_pointer_v<rotation_t>)
            stream.attach(rotations);
    }

    template<Remap REMAP, typename T, typename S, typename R, typename>
    void extract3D(const shared_t<T[]>& grid, dim4_t grid_strides, dim4_t grid_shape,
                   const shared_t<T[]>& slice, dim4_t slice_strides, dim4_t slice_shape,
                   const S& scaling_factors, const R& rotations,
                   float cutoff, dim4_t target_shape, float2_t ews_radius,
                   bool use_texture, Stream& stream) {
        NOA_ASSERT(all(slice_shape > 0) && all(grid_shape > 0));
        NOA_ASSERT_DEVICE_PTR(slice.get(), stream.device());
        NOA_ASSERT(slice_shape[1] == 1);
        NOA_ASSERT(grid_shape[0] == 1);
        const auto int3_grid_shape = safe_cast<int3_t>(dim3_t(grid_shape.get(1)));

        if (!use_texture) {
            NOA_ASSERT(slice.get() != grid.get());
            NOA_ASSERT_DEVICE_PTR(grid.get(), stream.device());

            const auto grid_strides_3d = safe_cast<uint3_t>(dim3_t(grid_strides.get(1)));
            const Accessor<const T, 3, uint32_t> grid_accessor(grid.get(), grid_strides_3d);
            const auto interpolator = noa::geometry::interpolator3D<BORDER_ZERO, INTERP_LINEAR>(
                    grid_accessor, int3_grid_shape.fft(), T{0});

            launchExtract3D_<REMAP>(interpolator, int3_grid_shape,
                                    slice.get(), slice_strides, slice_shape,
                                    scaling_factors, rotations,
                                    cutoff, target_shape, ews_radius, stream);
            stream.attach(grid, slice);
        } else {
            if constexpr (traits::is_any_v<T, double, cdouble_t>) {
                NOA_THROW("Double precision is not supported in this mode. Use no_texture=true instead");
            } else {
                NOA_ASSERT(grid_strides[3] == 1 && indexing::isContiguous(grid_strides, grid_shape.fft())[1]);

                memory::PtrArray<T> array(dim3_t(int3_grid_shape.fft()));
                memory::PtrTexture texture(array.get(), INTERP_LINEAR, BORDER_ZERO);
                memory::copy(grid, grid_strides[2], array.share(), array.shape(), stream);
                const noa::cuda::geometry::Interpolator3D<INTERP_LINEAR, T> interpolator(texture.get());

                launchExtract3D_<REMAP>(interpolator, int3_grid_shape,
                                        slice.get(), slice_strides, slice_shape,
                                        scaling_factors, rotations,
                                        cutoff, target_shape, ews_radius, stream);
                stream.attach(array.share(), texture.share(), slice);
            }
        }
    }

    template<Remap REMAP, typename T, typename S, typename R, typename>
    void extract3D(const shared_t<cudaArray>& array,
                   const shared_t<cudaTextureObject_t>& grid, int3_t grid_shape,
                   const shared_t<T[]>& slice, dim4_t slice_strides, dim4_t slice_shape,
                   const S& scaling_factors, const R& rotations,
                   float cutoff, dim4_t target_shape, float2_t ews_radius, Stream& stream) {
        NOA_ASSERT(array && grid && all(slice_shape > 0) && all(grid_shape > 0));
        NOA_ASSERT_DEVICE_PTR(slice.get(), stream.device());

        const noa::cuda::geometry::Interpolator3D<INTERP_LINEAR, T> interpolator(*grid); // FIXME
        launchExtract3D_<REMAP>(interpolator, grid_shape,
                                slice.get(), slice_strides, slice_shape,
                                scaling_factors, rotations,
                                cutoff, target_shape, ews_radius, stream);
        stream.attach(array, grid, slice);
    }

    template<typename T, typename>
    void griddingCorrection(const shared_t<T[]>& input, dim4_t input_strides,
                            const shared_t<T[]>& output, dim4_t output_strides,
                            dim4_t shape, bool post_correction, Stream& stream) {
        NOA_ASSERT(all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());

        const uint2_t shape_(shape.get(2));
        const uint32_t blocks_x = math::divideUp(shape_[1], THREADS.x);
        const uint32_t blocks_y = math::divideUp(shape_[0], THREADS.y);
        const dim3 blocks(blocks_x * blocks_y,
                          shape[1],
                          shape[0]);
        const LaunchConfig config{blocks, THREADS};

        const auto i_shape = safe_cast<int3_t>(dim3_t(shape.get(1)));
        const float3_t f_shape(i_shape);
        const float3_t half(f_shape / 2 * float3_t(i_shape != 1)); // if size == 1, half should be 0
        const Accessor<const T, 4, uint32_t> input_accessor(input.get(), safe_cast<uint4_t>(input_strides));
        const Accessor<T, 4, uint32_t> output_accessor(output.get(), safe_cast<uint4_t>(output_strides));

        stream.enqueue("geometry::fft::griddingCorrection",
                       post_correction ? correctGriddingSinc2_<true, T> : correctGriddingSinc2_<false, T>, config,
                       input_accessor, output_accessor, shape_, f_shape, half, blocks_x);
        stream.attach(input, output);
    }

    #define NOA_INSTANTIATE_INSERT_(T, REMAP, S, R) \
    template void insert3D<REMAP, T, S, R, void>(   \
        const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const S&, const R&, float, dim4_t, float2_t, Stream&)

    #define NOA_INSTANTIATE_INSERT_ALL_REMAP(T, S, R)   \
    NOA_INSTANTIATE_INSERT_(T, Remap::H2H, S, R);       \
    NOA_INSTANTIATE_INSERT_(T, Remap::H2HC, S, R);      \
    NOA_INSTANTIATE_INSERT_(T, Remap::HC2H, S, R);      \
    NOA_INSTANTIATE_INSERT_(T, Remap::HC2HC, S, R)

    #define NOA_INSTANTIATE_INSERT_ALL_(T)                                             \
    NOA_INSTANTIATE_INSERT_ALL_REMAP(T, float22_t, float33_t);                         \
    NOA_INSTANTIATE_INSERT_ALL_REMAP(T, shared_t<float22_t[]>, float33_t);             \
    NOA_INSTANTIATE_INSERT_ALL_REMAP(T, float22_t, shared_t<float33_t[]>);             \
    NOA_INSTANTIATE_INSERT_ALL_REMAP(T, shared_t<float22_t[]>, shared_t<float33_t[]>); \
    template void griddingCorrection<T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, bool, Stream&)

    NOA_INSTANTIATE_INSERT_ALL_(float);
    NOA_INSTANTIATE_INSERT_ALL_(double);
    NOA_INSTANTIATE_INSERT_ALL_(cfloat_t);
    NOA_INSTANTIATE_INSERT_ALL_(cdouble_t);

    #define NOA_INSTANTIATE_EXTRACT_(T, REMAP, S, R)                                \
    template void extract3D<REMAP, T, S, R, void>(                                  \
        const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, \
        const S&, const R&, float, dim4_t, float2_t, bool, Stream&);                \
    template void extract3D<REMAP, T, S, R, void>(                                  \
        const shared_t<cudaArray>&, const shared_t<cudaTextureObject_t>&, int3_t,   \
        const shared_t<T[]>&, dim4_t, dim4_t,                                       \
        const S&, const R&, float, dim4_t, float2_t, Stream&)

    #define NOA_INSTANTIATE_EXTRACT_ALL_REMAP(T, S, R)  \
    NOA_INSTANTIATE_EXTRACT_(T, Remap::HC2HC, S, R);    \
    NOA_INSTANTIATE_EXTRACT_(T, Remap::HC2H, S, R)

    #define NOA_INSTANTIATE_EXTRACT_ALL_(T)                                             \
    NOA_INSTANTIATE_EXTRACT_ALL_REMAP(T, float22_t, float33_t);                         \
    NOA_INSTANTIATE_EXTRACT_ALL_REMAP(T, shared_t<float22_t[]>, float33_t);             \
    NOA_INSTANTIATE_EXTRACT_ALL_REMAP(T, float22_t, shared_t<float33_t[]>);             \
    NOA_INSTANTIATE_EXTRACT_ALL_REMAP(T, shared_t<float22_t[]>, shared_t<float33_t[]>); \

    NOA_INSTANTIATE_EXTRACT_ALL_(float);
    NOA_INSTANTIATE_EXTRACT_ALL_(cfloat_t);
}
