#include "noa/common/transform/Geometry.h"

#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/memory/PtrPinned.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/util/Atomic.cuh"
#include "noa/gpu/cuda/reconstruct/ProjectBackward.h"

// This implementation is very similar to the CPU backend's. See implementation details there.

// TODO The implementation takes a complex projection/volume and read/write from global memory. I think it might be
//      more efficient to take the interleaved arrays for proper global memory coalescing. Add the implementation and
//      benchmark to see if there's a difference.

namespace {
    using namespace ::noa;
    constexpr dim3 THREADS(32, 8);

    __device__ inline cfloat_t getPhaseShift_(float2_t shift, float shift_scaling, float2_t freq) {
        shift *= shift_scaling;
        float factor = -math::sum(shift * freq);
        cfloat_t phase_shift;
        math::sincos(factor, &phase_shift.imag, &phase_shift.real);
        return phase_shift;
    }

    template<bool IS_VOLUME_CENTERED>
    __device__ __forceinline__ int toIndex_(int frequency, int volume_dim) {
        if constexpr(IS_VOLUME_CENTERED) {
            return frequency + volume_dim / 2;
        } else {
            return frequency < 0 ? frequency + volume_dim : frequency;
        }
        return 0; // unreachable - remove false warning
    }

    // Linear interpolation on the 2x2x2 cube.
    template<bool IS_VOLUME_CENTERED, typename T>
    __device__ void addByInterpolation_(Complex<T>* v1, Complex<T> a1, T* v2, T a2,
                                        float3_t freq, int volume_dim, int volume_pitch) {

        int3_t base0(math::floor(freq));
        float3_t fractions[2]; // TODO(TF) register pressure seems OK; store in half precision is still likely to help
        fractions[1] = freq - float3_t(base0);
        fractions[0] = 1.f - fractions[1];

        bool oob_z[2], oob_y[2], oob_x[2]; // TODO(TF) uint8_t bitfield?
        {
            int3_t base1(base0 + 1);
            const int idx_max = (volume_dim - 1) / 2;
            oob_z[0] = base0.z < -idx_max || base0.z > idx_max;
            oob_z[1] = base1.z < -idx_max || base1.z > idx_max;
            oob_y[0] = base0.y < -idx_max || base0.y > idx_max;
            oob_y[1] = base1.y < -idx_max || base1.y > idx_max;
            oob_x[0] = base0.x > idx_max;
            oob_x[1] = base1.x > idx_max;
        }

        T fraction;
        using namespace cuda;
        for (int z = 0; z < 2; ++z) {
            int64_t idx_z = toIndex_<IS_VOLUME_CENTERED>(base0.z + z, volume_dim);
            for (int y = 0; y < 2; ++y) {
                int64_t idx_y = toIndex_<IS_VOLUME_CENTERED>(base0.y + y, volume_dim);
                for (int x = 0; x < 2; ++x) {
                    if (oob_z[z] || oob_y[y] || oob_x[x])
                        continue;
                    fraction = static_cast<T>(fractions[z].z * fractions[y].y * fractions[x].x);
                    if (v1)
                        atomic::add(v1 + (idx_z * volume_dim + idx_y) * volume_pitch + base0.x + x, a1 * fraction);
                    if (v2)
                        atomic::add(v2 + (idx_z * volume_dim + idx_y) * volume_pitch + base0.x + x, a2 * fraction);
                }
            }
        }

        if (base0.x == 0 && (base0.y != 0 || base0.z != 0)) {
            a1.imag = -a1.imag;
            for (int z = 0; z < 2; ++z) {
                int64_t idx_z = toIndex_<IS_VOLUME_CENTERED>(-(base0.z + z), volume_dim);
                for (int y = 0; y < 2; ++y) {
                    int64_t idx_y = toIndex_<IS_VOLUME_CENTERED>(-(base0.y + y), volume_dim);
                    if (oob_z[z] || oob_y[y])
                        continue;
                    fraction = static_cast<T>(fractions[z].z * fractions[y].y * fractions[0].x);
                    if (v1)
                        atomic::add(v1 + (idx_z * volume_dim + idx_y) * volume_pitch, a1 * fraction);
                    if (v2)
                        atomic::add(v2 + (idx_z * volume_dim + idx_y) * volume_pitch, a2 * fraction);
                }
            }
        }
    }

    // Launch config: 2D XY block mapping the projection, with projection ID (batch) in Z. One thread per projected element.
    // If scaling is a pointer, it expects to have one 2x2 matrix per projection. Otherwise, the same matrix is used.
    // In debug, this is about 30-32 registers for float, and 40 for double (+ double atomic is 16 if ARCH < 6).
    template<bool IS_PROJ_CENTERED, bool IS_VOLUME_CENTERED, bool APPLY_SHIFT, bool APPLY_MAG, typename T, typename U>
    __global__ void __launch_bounds__(THREADS.x * THREADS.y)
    fourierInsert_(const Complex<T>* proj, const T* proj_weights, int proj_pitch, int proj_dim,
                   Complex<T>* volume, T* volume_weights, int volume_pitch, int volume_dim,
                   [[maybe_unused]] const float2_t* shifts, [[maybe_unused]] float shift_scaling,
                   [[maybe_unused]] U scaling, // const float22_t* or float22_t
                   const float33_t* rotation_oversampling, float freq_max_sqd, float ewald_sphere_diam_inv) {

        // gid = current index in the projection. z is the projection ID.
        const int3_t gid(blockIdx.x * THREADS.x + threadIdx.x,
                         blockIdx.y * THREADS.y + threadIdx.y,
                         blockIdx.z);
        // frequencies u and v:
        //  u is equal to gid.x because proj is non-redundant
        //  v depends on the projection layout.
        const int frequency_v = IS_PROJ_CENTERED ?
                                gid.y - proj_dim / 2 :
                                gid.y < (proj_dim + 1) / 2 ? gid.y : gid.y - proj_dim;
        float2_t freq_2d(gid.x, frequency_v);

        if constexpr(APPLY_MAG) {
            if constexpr(std::is_pointer_v<U>)
                freq_2d = scaling[gid.z] * freq_2d;
            else
                freq_2d = scaling * freq_2d;
        } else {
            (void) scaling;
        }

        float3_t freq_3d(freq_2d.x, freq_2d.y, ewald_sphere_diam_inv * math::dot(freq_2d, freq_2d));
        if (math::dot(freq_3d, freq_3d) > freq_max_sqd)
            return;

        freq_3d = rotation_oversampling[gid.z] * freq_3d;
        T conj = 1;
        if (freq_3d.x < 0) {
            freq_3d = -freq_3d;
            conj = -1;
        }

        Complex<T> proj_value;
        if (volume) { // should be predicted and shouldn't diverge
            proj += gid.z * proj_dim * proj_pitch;
            proj_value = proj[gid.y * proj_pitch + gid.x];
            proj_value.imag *= conj;
            if constexpr (APPLY_SHIFT) {
                float2_t orig_freq(gid.x, frequency_v); // gid.x == frequency_u
                proj_value *= Complex<T>(getPhaseShift_(shifts[gid.z], shift_scaling, orig_freq));
            } else {
                (void) shifts;
                (void) shift_scaling;
            }
        }
        T proj_weight;
        if (volume_weights) {
            proj_weights += gid.z * proj_dim * proj_pitch;
            proj_weight = proj_weights[gid.y * proj_pitch + gid.x];
        }

        addByInterpolation_<IS_VOLUME_CENTERED>(volume, proj_value, volume_weights, proj_weight,
                                                freq_3d, volume_dim, volume_pitch);
    }

    inline float prepareEWS_(size_t proj_dim, float ewald_sphere_radius) {
        return 1.f / (static_cast<float>(2 * proj_dim) * ewald_sphere_radius);
    }
}

namespace noa::cuda::reconstruct {
    // Since we want to batch all projections in one kernel launch, we need to allocate and copy to the device
    // the rotation and scaling matrices, which ultimately requires a stream synchronization.
    template<bool IS_PROJ_CENTERED, bool IS_VOLUME_CENTERED, typename T>
    void projectBackward(const Complex<T>* proj, const T* proj_weights, size_t proj_pitch, size_t proj_dim,
                         Complex<T>* volume, T* volume_weights, size_t volume_pitch, size_t volume_dim,
                         const float2_t* proj_shifts, const float3_t* proj_scaling_factors,
                         const float33_t* proj_rotations, uint proj_count,
                         float freq_max, float ewald_sphere_radius, Stream& stream) {

        // Oversampling & max frequency.
        const auto samples = static_cast<float>(proj_dim);
        const auto oversampling = static_cast<float>(volume_dim) / samples;
        const auto sample_max = static_cast<uint>(math::clamp(freq_max, 0.f, 0.5f) * samples);
        const auto sample_max_sqd = static_cast<float>(sample_max * sample_max);

        // Prepare for the EWS curvature.
        if (ewald_sphere_radius != 0.f)
            ewald_sphere_radius = prepareEWS_(proj_dim, ewald_sphere_radius); // returns 1/diameter

        // Launch config.
        const auto tmp = static_cast<uint>(proj_dim);
        const dim3 blocks(math::divideUp(tmp / 2 + 1, THREADS.x), // non-redundant
                          math::divideUp(tmp, THREADS.y),
                          proj_count);

        // Prepare rotation matrices.
        cuda::memory::PtrDevice<float33_t> d_rotations(proj_count);
        cuda::memory::PtrPinned<float33_t> h_rotations(proj_count);
        for (uint idx = 0; idx < proj_count; ++idx)
            h_rotations[idx] = proj_rotations[idx] * float33_t(oversampling);
        memory::copy(h_rotations.get(), d_rotations.get(), proj_count, stream);

        const int p_pitch = static_cast<int>(proj_pitch);
        const int p_dim = static_cast<int>(proj_dim);
        const int v_pitch = static_cast<int>(volume_pitch);
        const int v_dim = static_cast<int>(volume_dim);
        const float shift_scaling = math::Constants<float>::PI2 / static_cast<float>(p_dim);

        // Prepare magnifications (which are optional)
        const bool apply_shift = proj_shifts != nullptr;
        if (proj_scaling_factors) {
            // Scaling is inverted in Fourier space.
            cuda::memory::PtrPinned<float22_t> h_distortion(proj_count);
            cuda::memory::PtrDevice<float22_t> d_distortion(proj_count);
            for (uint idx = 0; idx < proj_count; ++idx) {
                const float3_t& i_scale = proj_scaling_factors[idx];
                h_distortion[idx] = transform::rotate(-i_scale.z) *
                                    transform::scale(1.f / float2_t(i_scale.x, i_scale.y)) *
                                    transform::rotate(i_scale.z);
            }
            memory::copy(h_distortion.get(), d_distortion.get(), proj_count, stream);

            if (apply_shift) {
                fourierInsert_<IS_PROJ_CENTERED, IS_VOLUME_CENTERED, true, true, T, const float22_t*>
                <<<blocks, THREADS, 0, stream.get()>>>(
                        proj, proj_weights, p_pitch, p_dim, volume, volume_weights, v_pitch, v_dim,
                        proj_shifts, shift_scaling, d_distortion.get(), d_rotations.get(),
                        sample_max_sqd, ewald_sphere_radius);
            } else {
                fourierInsert_<IS_PROJ_CENTERED, IS_VOLUME_CENTERED, false, true, T, const float22_t*>
                <<<blocks, THREADS, 0, stream.get()>>>(
                        proj, proj_weights, p_pitch, p_dim, volume, volume_weights, v_pitch, v_dim,
                        nullptr, 0.f, d_distortion.get(), d_rotations.get(), sample_max_sqd, ewald_sphere_radius);
            }
        } else { // without mag
            if (apply_shift) {
                fourierInsert_<IS_PROJ_CENTERED, IS_VOLUME_CENTERED, true, false, T, const float22_t*>
                <<<blocks, THREADS, 0, stream.get()>>>(
                        proj, proj_weights, p_pitch, p_dim, volume, volume_weights, v_pitch, v_dim,
                        proj_shifts, shift_scaling, nullptr, d_rotations.get(),
                        sample_max_sqd, ewald_sphere_radius);
            } else {
                fourierInsert_<IS_PROJ_CENTERED, IS_VOLUME_CENTERED, false, false, T, const float22_t*>
                <<<blocks, THREADS, 0, stream.get()>>>(
                        proj, proj_weights, p_pitch, p_dim, volume, volume_weights, v_pitch, v_dim,
                        nullptr, 0.f, nullptr, d_rotations.get(), sample_max_sqd, ewald_sphere_radius);
            }
        }
        NOA_THROW_IF(cudaGetLastError());
        stream.synchronize();
    }

    // Same mag for all projections. Otherwise, it's the same as above.
    template<bool IS_PROJ_CENTERED, bool IS_VOLUME_CENTERED, typename T>
    void projectBackward(const Complex<T>* proj, const T* proj_weights, size_t proj_pitch, size_t proj_dim,
                         Complex<T>* volume, T* volume_weights, size_t volume_pitch, size_t volume_dim,
                         const float2_t* proj_shifts, float3_t proj_scaling_factor,
                         const float33_t* proj_rotations, uint proj_count,
                         float freq_max, float ewald_sphere_radius, Stream& stream) {

        const auto samples = static_cast<float>(proj_dim);
        const auto oversampling = static_cast<float>(volume_dim) / samples;
        const auto sample_max = static_cast<uint>(math::clamp(freq_max, 0.f, 0.5f) * samples);
        const auto sample_max_sqd = static_cast<float>(sample_max * sample_max);

        if (ewald_sphere_radius != 0.f)
            ewald_sphere_radius = prepareEWS_(volume_dim, ewald_sphere_radius);

        cuda::memory::PtrDevice<float33_t> d_rotations(proj_count);
        cuda::memory::PtrPinned<float33_t> h_rotations(proj_count);
        for (uint idx = 0; idx < proj_count; ++idx)
            h_rotations[idx] = proj_rotations[idx] * float33_t(oversampling);
        memory::copy(h_rotations.get(), d_rotations.get(), proj_count, stream);

        const bool apply_shift = proj_shifts != nullptr;
        float22_t distortion(transform::rotate(-proj_scaling_factor.z) *
                             transform::scale(1.f / float2_t(proj_scaling_factor.x, proj_scaling_factor.y)) *
                             transform::rotate(proj_scaling_factor.z));

        const int p_pitch = static_cast<int>(proj_pitch);
        const int p_dim = static_cast<int>(proj_dim);
        const int v_pitch = static_cast<int>(volume_pitch);
        const int v_dim = static_cast<int>(volume_dim);
        const float shift_scaling = math::Constants<float>::PI2 / static_cast<float>(p_dim);

        const auto tmp = static_cast<uint>(proj_dim);
        const dim3 blocks(math::divideUp(tmp / 2 + 1, THREADS.x),
                          math::divideUp(tmp, THREADS.y),
                          proj_count);

        if (apply_shift) {
            fourierInsert_<IS_PROJ_CENTERED, IS_VOLUME_CENTERED, true, true, T, float22_t>
            <<<blocks, THREADS, 0, stream.get()>>>(
                    proj, proj_weights, p_pitch, p_dim, volume, volume_weights, v_pitch, v_dim,
                    proj_shifts, shift_scaling, distortion, d_rotations.get(),
                    sample_max_sqd, ewald_sphere_radius);
        } else {
            fourierInsert_<IS_PROJ_CENTERED, IS_VOLUME_CENTERED, false, true, T, float22_t>
            <<<blocks, THREADS, 0, stream.get()>>>(
                    proj, proj_weights, p_pitch, p_dim, volume, volume_weights, v_pitch, v_dim,
                    nullptr, 0.f, distortion, d_rotations.get(), sample_max_sqd, ewald_sphere_radius);
        }
        NOA_THROW_IF(cudaGetLastError());
        stream.synchronize();
    }

    #define NOA_INSTANTIATE_BACKWARD_(T)                                                                                            \
    template void projectBackward<true, true, T>(const Complex<T>*, const T*, size_t, size_t, Complex<T>*, T*, size_t, size_t,      \
                                                 const float2_t*, const float3_t*, const float33_t*, uint, float, float, Stream&);  \
    template void projectBackward<true, false, T>(const Complex<T>*, const T*, size_t, size_t, Complex<T>*, T*, size_t, size_t,     \
                                                  const float2_t*, const float3_t*, const float33_t*, uint, float, float, Stream&); \
    template void projectBackward<false, true, T>(const Complex<T>*, const T*, size_t, size_t, Complex<T>*, T*, size_t, size_t,     \
                                                  const float2_t*, const float3_t*, const float33_t*, uint, float, float, Stream&); \
    template void projectBackward<false, false, T>(const Complex<T>*, const T*, size_t, size_t, Complex<T>*, T*, size_t, size_t,    \
                                                   const float2_t*, const float3_t*, const float33_t*, uint, float, float, Stream&);\
    template void projectBackward<true, true, T>(const Complex<T>*, const T*, size_t, size_t, Complex<T>*, T*, size_t, size_t,      \
                                                 const float2_t*, float3_t, const float33_t*, uint, float, float, Stream&);         \
    template void projectBackward<true, false, T>(const Complex<T>*, const T*, size_t, size_t, Complex<T>*, T*, size_t, size_t,     \
                                                  const float2_t*, float3_t, const float33_t*, uint, float, float, Stream&);        \
    template void projectBackward<false, true, T>(const Complex<T>*, const T*, size_t, size_t, Complex<T>*, T*, size_t, size_t,     \
                                                  const float2_t*, float3_t, const float33_t*, uint, float, float, Stream&);        \
    template void projectBackward<false, false, T>(const Complex<T>*, const T*, size_t, size_t, Complex<T>*, T*, size_t, size_t,    \
                                                   const float2_t*, float3_t, const float33_t*, uint, float, float, Stream&)
    NOA_INSTANTIATE_BACKWARD_(float);
    NOA_INSTANTIATE_BACKWARD_(double);
}
