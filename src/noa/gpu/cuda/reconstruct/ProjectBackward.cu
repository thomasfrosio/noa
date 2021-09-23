#include "noa/common/transform/Euler.h"
#include "noa/common/transform/Geometry.h"
#include "noa/cpu/memory/PtrHost.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/util/Atomic.h"
#include "noa/gpu/cuda/reconstruct/ProjectBackward.h"

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

    // Go from the DC-centered frequency to a valid index in the output volume.
    // This operation of course depends on the volume FFT layout.
    // The input frequency should be in-bound, i.e. -n/2 <= frequency <= (n-1)/2
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
    // v1 and v2 can be nullptr, freq is the frequency centered on DC (i.e. freq == 0 is the DC),
    //  with negative frequencies on the left and the base is the floor(freq).
    template<bool IS_VOLUME_CENTERED, typename T>
    __device__ void addByInterpolation_(Complex<T>* v1, Complex<T> a1, T* v2, T a2,
                                        float3_t freq, int3_t base, int volume_dim, int volume_pitch) {
        using namespace cuda;

        // So if the coordinate is centered on the left point (the base), i.e. its decimal is 0,
        // the corresponding fraction for this element should be 1.
        float3_t fractions[2]; // TODO(TF) check register pressure; store in half precision likely to help
        fractions[1] = freq - float3_t(base);
        fractions[0] = 1.f - fractions[1];

        T fraction;
        int idx_y, idx_z; // idx_x == base.x (+1)
        for (int z = 0; z < 2; ++z) {
            idx_z = toIndex_<IS_VOLUME_CENTERED>(base.z + z, volume_dim);
            for (int y = 0; y < 2; ++y) {
                idx_y = toIndex_<IS_VOLUME_CENTERED>(base.y + y, volume_dim);
                #pragma unroll
                for (int x = 0; x < 2; ++x) {
                    fraction = static_cast<T>(fractions[z].z * fractions[y].y * fractions[x].x);
                    if (v1) atomic::add(v1 + (idx_z * volume_dim + idx_y) * volume_pitch + base.x + x, a1 * fraction);
                    if (v2) atomic::add(v2 + (idx_z * volume_dim + idx_y) * volume_pitch + base.x + x, a2 * fraction);
                }
            }
        }

        // Redundant plane: the x==0 plane (excluding the DC) is redundant, so to preserve the hermitian symmetry,
        // we simply need to add the conjugate at (x=0, -y, -z) with the same fraction.
        if (base.x == 0 && (base.y != 0 || base.z != 0)) {
            a1.imag = -a1.imag;
            for (int z = 0; z < 2; ++z) {
                idx_z = toIndex_<IS_VOLUME_CENTERED>(-(base.z + z), volume_dim);
                for (int y = 0; y < 2; ++y) {
                    idx_y = toIndex_<IS_VOLUME_CENTERED>(-(base.y + y), volume_dim);

                    fraction = static_cast<T>(fractions[z].z * fractions[y].y * fractions[0].x);
                    if (v1) atomic::add(v1 + (idx_z * volume_dim + idx_y) * volume_pitch, a1 * fraction);
                    if (v2) atomic::add(v2 + (idx_z * volume_dim + idx_y) * volume_pitch, a2 * fraction);
                }
            }
        }
    }

    // Launch config: 2D XY block mapping the projection, with projection ID (batch) in Z. One thread per projected element.
    // If magnification is a pointer, it expects to have one 2x2 matrix per projection. Otherwise, the same matrix is used.
    // In debug, this is about 30-32 registers for float, and 40 for double (+ double atomic is 16 if ARCH < 6).
    template<bool IS_PROJ_CENTERED, bool IS_VOLUME_CENTERED, bool APPLY_MAG, typename T, typename U>
    __global__ void __launch_bounds__(THREADS.x * THREADS.y)
    fourierInsert_(const Complex<T>* proj, const T* proj_weights, int proj_pitch, int proj_dim,
                   Complex<T>* volume, T* volume_weights, int volume_pitch, int volume_dim,
                   const float2_t* shifts, float shift_scaling,
                   U magnifications, // const float22_t* or float22_t
                   const float33_t* rotation_oversampling, float freq_max_sqd, float ewald_sphere_diam_inv) {

        // gid = current index in the projection. z is the projection ID.
        const int3_t gid(blockIdx.x * THREADS.x,
                         blockIdx.y * THREADS.y,
                         blockIdx.z);
        // frequencies u and v: u is equal to gid.x because proj is non-redundant, v depends on the projection layout.
        const int frequency_v = IS_PROJ_CENTERED ?
                                gid.y - proj_dim / 2 :
                                gid.y < (proj_dim + 1) / 2 ? gid.y : gid.y - proj_dim;

        // Correct for the (anisotropic) magnification.
        float2_t freq_2d(gid.x, frequency_v);
        if constexpr(APPLY_MAG) {
            if constexpr(std::is_pointer_v<U>)
                freq_2d = magnifications[gid.z] * freq_2d;
            else
                freq_2d = magnifications * freq_2d;
        }

        // Curve the projection to match the EWS.
        float3_t freq_3d(freq_2d.x, freq_2d.y, ewald_sphere_diam_inv * math::innerProduct(freq_2d, freq_2d));

        // If the frequency is larger than the max authorized, stop here.
        // This is enough to remove most OOB conditions. However, since the interpolation window is
        // 2 elements wide, there can still be rare cases of OOB after that point.
        if (math::innerProduct(freq_3d, freq_3d) > freq_max_sqd)
            return;

        // Rotate and scale for the oversampling.
        // The projection is non-redundant and the transformation can result in a negative v frequency.
        // In this case, take the conjugate.
        freq_3d = rotation_oversampling[gid.z] * freq_3d;
        T conj = 1;
        if (freq_3d.x < 0) {
            freq_3d = -freq_3d;
            conj = -1;
        }

        // Extract the base. The base is the left element of the interpolation window: [base, base+1].
        // n = 10: -5 -4 -3 -2 -1 0 1 2 3 4     -    base = -5 ... to ... 3
        // n = 9:     -4 -3 -2 -1 0 1 2 3 4     -    base = -4 ... to ... 3
        int3_t base(math::floor(freq_3d));
        NOA_ASSERT(base.x >= 0); // it shouldn't be possible for x to be negative since freq_3d.x >= 0
        const int volume_half = volume_dim / 2;
        if (base.x >= volume_half ||
            base.y < -volume_half || base.y >= (volume_dim - 1) / 2 ||
            base.z < -volume_half || base.z >= (volume_dim - 1) / 2)
            return;

        // At this point, we know for sure we are going to use the projection value and weight.
        Complex<T> proj_value;
        if (volume) { // should be predicted and shouldn't diverge
            proj += gid.z * proj_dim * proj_pitch;
            proj_value = proj[gid.y * proj_pitch + gid.x];
            proj_value.imag *= conj;

            // Phase shift value to apply the desired real space shift.
            // Use the untransformed frequencies.
            float2_t orig_freq(gid.x, frequency_v); // gid.x == frequency_u
            proj_value *= getPhaseShift_(shifts[gid.z], shift_scaling, orig_freq);
        }
        T proj_weight;
        if (volume_weights) {
            proj_weight += gid.z * proj_dim * proj_pitch;
            proj_weight = proj_weights[gid.y * proj_pitch + gid.x];
        }

        addByInterpolation_<IS_VOLUME_CENTERED>(volume, proj_value, volume_weights, proj_weight,
                                                freq_3d, base, volume_dim, volume_pitch);
    }

    // Same as above but without phase shifts.
    template<bool IS_PROJ_CENTERED, bool IS_VOLUME_CENTERED, bool APPLY_MAG, typename T, typename U>
    __global__ void __launch_bounds__(THREADS.x * THREADS.y)
    fourierInsert_(const Complex<T>* proj, const T* proj_weights, int proj_pitch, int proj_dim,
                   Complex<T>* volume, T* volume_weights, int volume_pitch, int volume_dim,
                   U magnifications, // const float22_t* or float22_t
                   const float33_t* rotation_oversampling, float freq_max_sqd, float ewald_sphere_diam_inv) {

        const int3_t gid(blockIdx.x * THREADS.x,
                         blockIdx.y * THREADS.y,
                         blockIdx.z);
        const int frequency_v = IS_PROJ_CENTERED ?
                                gid.y - proj_dim / 2 :
                                gid.y < (proj_dim + 1) / 2 ? gid.y : gid.y - proj_dim;

        float2_t freq_2d(gid.x, frequency_v);
        if constexpr(APPLY_MAG) {
            if constexpr(std::is_pointer_v<U>)
                freq_2d = magnifications[gid.z] * freq_2d;
            else
                freq_2d = magnifications * freq_2d;
        }

        float3_t freq_3d(freq_2d.x, freq_2d.y, ewald_sphere_diam_inv * math::innerProduct(freq_2d, freq_2d));
        if (math::innerProduct(freq_3d, freq_3d) > freq_max_sqd)
            return;

        freq_3d = rotation_oversampling[gid.z] * freq_3d;
        T conj = 1;
        if (freq_3d.x < 0) {
            freq_3d = -freq_3d;
            conj = -1;
        }

        int3_t base(math::floor(freq_3d));
        NOA_ASSERT(base.x >= 0);
        const int volume_half = volume_dim / 2;
        if (base.x >= volume_half ||
            base.y < -volume_half || base.y >= (volume_dim - 1) / 2 ||
            base.z < -volume_half || base.z >= (volume_dim - 1) / 2)
            return;

        Complex<T> proj_value;
        if (volume) {
            proj += gid.z * proj_dim * proj_pitch;
            proj_value = proj[gid.y * proj_pitch + gid.x];
            proj_value.imag *= conj;
        }
        T proj_weight;
        if (volume_weights) {
            proj_weight += gid.z * proj_dim * proj_pitch;
            proj_weight = proj_weights[gid.y * proj_pitch + gid.x];
        }

        addByInterpolation_<IS_VOLUME_CENTERED>(volume, proj_value, volume_weights, proj_weight,
                                                freq_3d, base, volume_dim, volume_pitch);
    }

    // The EWS radius should be 1/wavelength in SI, so in m^-1.
    // To curve the slice and match the EWS, we need to compute the Z.
    // Using the small-angle approximation, Z = wavelength / 2 * (X^2 + Y^2).
    // See doi:10.1016/S0304-3991(99)00120-5 for a derivation.
    inline float prepareEWS_(size_t volume_dim, float ewald_sphere_radius) {
        ewald_sphere_radius *= static_cast<float>(volume_dim); // in pixel/meter
        return 1.f / (2.f * ewald_sphere_radius); // wavelength / 2
    }
}

namespace noa::cuda::reconstruct {
    // Since we want to batch all projections in one kernel launch, we need to allocate and copy to the device
    // the rotation and scaling matrices, which ultimately requires a stream synchronization.
    template<bool IS_PROJ_CENTERED, bool IS_VOLUME_CENTERED, typename T>
    void projectBackward(const Complex<T>* proj, const T* proj_weights, size_t proj_pitch, size_t proj_dim,
                         const float2_t* proj_shifts, const float3_t* proj_magnifications,
                         const float3_t* proj_angles, uint proj_count,
                         Complex<T>* volume, T* volume_weights, size_t volume_pitch, size_t volume_dim,
                         float freq_max, float ewald_sphere_radius, Stream& stream) {

        // Oversampling & max frequency.
        const auto samples = static_cast<float>(proj_dim);
        const auto oversampling = static_cast<float>(volume_dim) / samples;
        const auto sample_max = static_cast<uint>(math::clamp(freq_max, 0.f, 0.5f) * samples);
        const auto sample_max_sqd = static_cast<float>(sample_max * sample_max);

        // Prepare for the EWS curvature.
        if (ewald_sphere_radius != 0.f)
            ewald_sphere_radius = prepareEWS_(volume_dim, ewald_sphere_radius);

        // Launch config.
        const auto tmp = static_cast<uint>(proj_dim);
        const dim3 blocks(math::divideUp(tmp, THREADS.x),
                          math::divideUp(tmp, THREADS.y),
                          proj_count);

        // Prepare rotation matrices.
        cuda::memory::PtrDevice<float33_t> d_rotations(proj_count);
        cpu::memory::PtrHost<float33_t> h_rotations(proj_count); // PtrPinned?
        for (uint idx = 0; idx < proj_count; ++idx) {
            h_rotations[idx] = (transform::toMatrix<true>(proj_angles[idx])); // invert rotation
            h_rotations[idx] *= float33_t(oversampling); // scale for sampling during rotation
        }
        memory::copy(h_rotations.get(), d_rotations.get(), proj_count, stream);

        const int p_pitch = static_cast<int>(proj_pitch);
        const int p_dim = static_cast<int>(proj_dim);
        const int v_pitch = static_cast<int>(volume_pitch);
        const int v_dim = static_cast<int>(volume_dim);
        const float shift_scaling = math::Constants<float>::PI2 / static_cast<float>(p_dim); // phase shift constant

        // Prepare magnifications (which are optional)
        const bool apply_shift = proj_shifts != nullptr;
        if (proj_magnifications) {
            cuda::memory::PtrDevice<float22_t> d_mag_corrections(proj_count);
            cpu::memory::PtrHost<float22_t> h_mag_corrections(proj_count);
            for (uint idx = 0; idx < proj_count; ++idx) {
                const float3_t& i_mag = proj_magnifications[idx]; // z is angle
                h_mag_corrections[idx] = transform::rotate(-i_mag.z) *
                                         transform::scale(float2_t(i_mag.x, i_mag.y)) *
                                         transform::rotate(i_mag.z);
            }
            memory::copy(h_mag_corrections.get(), d_mag_corrections.get(), proj_count, stream);

            if (apply_shift) {
                fourierInsert_<IS_PROJ_CENTERED, IS_VOLUME_CENTERED, true, T, const float22_t*>
                <<<blocks, THREADS, 0, stream.get()>>>(
                        proj, proj_weights, p_pitch, p_dim, volume, volume_weights, v_pitch, v_dim,
                        proj_shifts, shift_scaling, d_mag_corrections.get(), d_rotations.get(),
                        sample_max_sqd, ewald_sphere_radius);
            } else {
                fourierInsert_<IS_PROJ_CENTERED, IS_VOLUME_CENTERED, true, T, const float22_t*>
                <<<blocks, THREADS, 0, stream.get()>>>(
                        proj, proj_weights, p_pitch, p_dim, volume, volume_weights, v_pitch, v_dim,
                        d_mag_corrections.get(), d_rotations.get(), sample_max_sqd, ewald_sphere_radius);
            }
        } else { // without mag
            if (apply_shift) {
                fourierInsert_<IS_PROJ_CENTERED, IS_VOLUME_CENTERED, false, T, const float22_t*>
                <<<blocks, THREADS, 0, stream.get()>>>(
                        proj, proj_weights, p_pitch, p_dim, volume, volume_weights, v_pitch, v_dim,
                        proj_shifts, shift_scaling, nullptr, d_rotations.get(),
                        sample_max_sqd, ewald_sphere_radius);
            } else {
                fourierInsert_<IS_PROJ_CENTERED, IS_VOLUME_CENTERED, false, T, const float22_t*>
                <<<blocks, THREADS, 0, stream.get()>>>(
                        proj, proj_weights, p_pitch, p_dim, volume, volume_weights, v_pitch, v_dim,
                        nullptr, d_rotations.get(), sample_max_sqd, ewald_sphere_radius);
            }
        }
        NOA_THROW_IF(cudaPeekAtLastError());
        stream.synchronize();
    }

    // Same mag for all projections. Otherwise, it's the same as above.
    template<bool IS_PROJ_CENTERED, bool IS_VOLUME_CENTERED, typename T>
    void projectBackward(const Complex<T>* proj, const T* proj_weights, size_t proj_pitch, size_t proj_dim,
                         const float2_t* proj_shifts, float3_t proj_magnification,
                         const float3_t* proj_angles, uint proj_count,
                         Complex<T>* volume, T* volume_weights, size_t volume_pitch, size_t volume_dim,
                         float freq_max, float ewald_sphere_radius, Stream& stream) {

        const auto samples = static_cast<float>(proj_dim);
        const auto oversampling = static_cast<float>(volume_dim) / samples;
        const auto sample_max = static_cast<uint>(math::clamp(freq_max, 0.f, 0.5f) * samples);
        const auto sample_max_sqd = static_cast<float>(sample_max * sample_max);

        if (ewald_sphere_radius != 0.f)
            ewald_sphere_radius = prepareEWS_(volume_dim, ewald_sphere_radius);

        cuda::memory::PtrDevice<float33_t> d_rotations(proj_count);
        cpu::memory::PtrHost<float33_t> h_rotations(proj_count);
        for (uint idx = 0; idx < proj_count; ++idx) {
            h_rotations[idx] = (transform::toMatrix<true>(proj_angles[idx]));
            h_rotations[idx] *= float33_t(oversampling);
        }
        memory::copy(h_rotations.get(), d_rotations.get(), proj_count, stream);

        const bool apply_shift = proj_shifts != nullptr;
        float22_t distortion(transform::rotate(-proj_magnification.z) * // z is angle
                             transform::scale(float2_t(proj_magnification.x, proj_magnification.y)) *
                             transform::rotate(proj_magnification.z));

        const int p_pitch = static_cast<int>(proj_pitch);
        const int p_dim = static_cast<int>(proj_dim);
        const int v_pitch = static_cast<int>(volume_pitch);
        const int v_dim = static_cast<int>(volume_dim);
        const float shift_scaling = math::Constants<float>::PI2 / static_cast<float>(p_dim);

        const auto tmp = static_cast<uint>(proj_dim);
        const dim3 blocks(math::divideUp(tmp, THREADS.x),
                          math::divideUp(tmp, THREADS.y),
                          proj_count);

        if (apply_shift) {
            fourierInsert_<IS_PROJ_CENTERED, IS_VOLUME_CENTERED, true, T, float22_t>
            <<<blocks, THREADS, 0, stream.get()>>>(
                    proj, proj_weights, p_pitch, p_dim, volume, volume_weights, v_pitch, v_dim,
                    proj_shifts, shift_scaling, distortion, d_rotations.get(),
                    sample_max_sqd, ewald_sphere_radius);
        } else {
            fourierInsert_<IS_PROJ_CENTERED, IS_VOLUME_CENTERED, true, T, float22_t>
            <<<blocks, THREADS, 0, stream.get()>>>(
                    proj, proj_weights, p_pitch, p_dim, volume, volume_weights, v_pitch, v_dim,
                    distortion, d_rotations.get(), sample_max_sqd, ewald_sphere_radius);
        }
        NOA_THROW_IF(cudaPeekAtLastError());
        stream.synchronize();
    }

    #define NOA_INSTANTIATE_BACKWARD_(T)                                                                    \
    template void projectBackward<true, true, T>(const Complex<T>*, const T*, size_t, size_t,               \
                                                 const float2_t*, const float3_t*, const float3_t*, uint,   \
                                                 Complex<T>*, T*, size_t, size_t, float, float, Stream&);   \
    template void projectBackward<true, false, T>(const Complex<T>*, const T*, size_t, size_t,              \
                                                  const float2_t*, const float3_t*, const float3_t*, uint,  \
                                                  Complex<T>*, T*, size_t, size_t, float, float, Stream&);  \
    template void projectBackward<false, true, T>(const Complex<T>*, const T*, size_t, size_t,              \
                                                  const float2_t*, const float3_t*, const float3_t*, uint,  \
                                                  Complex<T>*, T*, size_t, size_t, float, float, Stream&);  \
    template void projectBackward<false, false, T>(const Complex<T>*, const T*, size_t, size_t,             \
                                                  const float2_t*, const float3_t*, const float3_t*, uint,  \
                                                  Complex<T>*, T*, size_t, size_t, float, float, Stream&);  \
    template void projectBackward<true, true, T>(const Complex<T>*, const T*, size_t, size_t,               \
                                                  const float2_t*, float3_t, const float3_t*, uint,         \
                                                  Complex<T>*, T*, size_t, size_t, float, float, Stream&);  \
    template void projectBackward<true, false, T>(const Complex<T>*, const T*, size_t, size_t,              \
                                                  const float2_t*, float3_t, const float3_t*, uint,         \
                                                  Complex<T>*, T*, size_t, size_t, float, float, Stream&);  \
    template void projectBackward<false, true, T>(const Complex<T>*, const T*, size_t, size_t,              \
                                                  const float2_t*, float3_t, const float3_t*, uint,         \
                                                  Complex<T>*, T*, size_t, size_t, float, float, Stream&);  \
    template void projectBackward<false, false, T>(const Complex<T>*, const T*, size_t, size_t,             \
                                                  const float2_t*, float3_t, const float3_t*, uint,         \
                                                  Complex<T>*, T*, size_t, size_t, float, float, Stream&)
    NOA_INSTANTIATE_BACKWARD_(float);
    NOA_INSTANTIATE_BACKWARD_(double);
}
