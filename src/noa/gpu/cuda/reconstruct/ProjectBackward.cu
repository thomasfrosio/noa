#include "noa/common/transform/Euler.h"
#include "noa/common/transform/Geometry.h"
#include "noa/cpu/memory/PtrHost.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/util/Atomic.h"
#include "noa/gpu/cuda/reconstruct/ProjectBackward.h"

// TODO The implementation takes a complex projection/volume and read/write from global memory. I think it might be more efficient
//      to take the interleaved arrays for proper global memory coalescing. Add the implementation and benchmark to see if there's
//      a difference and how important the difference is.

namespace {
    using namespace ::noa;
    constexpr dim3 THREADS(32, 8);

    // Launch config: 2D XY block mapping the projection, with projection ID (batch) in Z. One thread per projected element.
    // If magnification is a pointer, it expects to have one 2x2 matrix per projection. Otherwise, the same matrix is used.
    template<bool PROJ_IS_CENTERED, bool VOLUME_IS_CENTERED, bool APPLY_SHIFT, bool APPLY_MAG, typename T, typename U>
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
        int proj_dim_half = proj_dim / 2;
        const int frequency_v = PROJ_IS_CENTERED ?
                                gid.y - proj_dim_half :
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
        float z = ewald_sphere_diam_inv * math::innerProduct(freq_2d, freq_2d);
        float3_t freq_3d(freq_2d.x, freq_2d.y, z);

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

        // Get weights/fractions. weight0 is the weight assigned to the first point on the left.
        // So if the coordinate is centered on the left point, i.e. decimal is 0, the weight for
        // this element should be 1.
        float3_t base(math::floor(freq_3d));
        float3_t weight1(freq_3d - base);
        float3_t weight0(1.f - weight1);

        // Compute the centered indexes corresponding to the current centered coordinates.
        // If the output volume is centered, these indexes match the physical indexes, otherwise
        // they will need to the decentered, i.e. ifftshift.
        int3_t idx0(base);
        int volume_dim_half = volume_dim / 2;
        idx0.y += volume_dim_half;
        idx0.z += volume_dim_half;
        int3_t idx1(idx0 + 1);

        // In most (if not all cases) this is fine...
        if (idx0.x < 0 || idx1.x >= volume_dim_half + 1 ||
            idx0.y < 0 || idx1.y >= volume_dim ||
            idx0.z < 0 || idx1.z >= volume_dim)
            return;

        // If the volume is not centered, decenter the coordinates.
        if constexpr(!VOLUME_IS_CENTERED) {
            idx0.y = math::iFFTShift(idx0.y, volume_dim);
            idx0.z = math::iFFTShift(idx0.z, volume_dim);
            idx1.y = math::iFFTShift(idx1.y, volume_dim);
            idx1.z = math::iFFTShift(idx1.z, volume_dim);
        }

        // At this point, we can add the current (x,y) element from the projection into the volume,
        // at the transformed coordinate which is located within idx0 and idx1.
        Complex<T> proj_value = proj[gid.y * proj_pitch + gid.x];
        T proj_weight = proj_weights[gid.y * proj_pitch + gid.x];
        proj_value.imag *= conj;

        // Phase shift value to apply the desired real space shift.
        // Use the untransformed frequencies.
        if constexpr(APPLY_SHIFT) {
            float2_t shift(shifts[gid.z]);
            shift *= shift_scaling;
            float factor = -math::sum(shift * float2_t(gid.x, frequency_v)); // gid.x == frequency_u
            cfloat_t phase_shift;
            math::sincos(factor, &phase_shift.imag, &phase_shift.real);
            proj_value *= phase_shift;
        }

        // Add to 2x2x2 cube to the cartesian grid.
        T tmp0 = static_cast<T>(weight0.z * weight0.y * weight0.x);
        T tmp1 = static_cast<T>(weight0.z * weight0.y * weight1.x);
        T tmp2 = static_cast<T>(weight0.z * weight1.y * weight0.x);
        T tmp3 = static_cast<T>(weight0.z * weight1.y * weight1.x);
        int offset_z = idx0.z * volume_dim * volume_pitch;
        cuda::atomic::add(volume + offset_z + idx0.y * volume_pitch + idx0.x, proj_value * tmp0);
        cuda::atomic::add(volume + offset_z + idx0.y * volume_pitch + idx1.x, proj_value * tmp1);
        cuda::atomic::add(volume + offset_z + idx1.y * volume_pitch + idx0.x, proj_value * tmp2);
        cuda::atomic::add(volume + offset_z + idx1.y * volume_pitch + idx1.x, proj_value * tmp3);
        cuda::atomic::add(volume_weights + offset_z + idx0.y * volume_pitch + idx0.x, proj_weight * tmp0);
        cuda::atomic::add(volume_weights + offset_z + idx0.y * volume_pitch + idx1.x, proj_weight * tmp1);
        cuda::atomic::add(volume_weights + offset_z + idx1.y * volume_pitch + idx0.x, proj_weight * tmp2);
        cuda::atomic::add(volume_weights + offset_z + idx1.y * volume_pitch + idx1.x, proj_weight * tmp3);

        tmp0 = static_cast<T>(weight1.z * weight0.y * weight0.x);
        tmp1 = static_cast<T>(weight1.z * weight0.y * weight1.x);
        tmp2 = static_cast<T>(weight1.z * weight1.y * weight0.x);
        tmp3 = static_cast<T>(weight1.z * weight1.y * weight1.x);
        offset_z = idx1.z * volume_dim * volume_pitch;
        cuda::atomic::add(volume + offset_z + idx0.y * volume_pitch + idx0.x, proj_value * tmp0);
        cuda::atomic::add(volume + offset_z + idx0.y * volume_pitch + idx1.x, proj_value * tmp1);
        cuda::atomic::add(volume + offset_z + idx1.y * volume_pitch + idx0.x, proj_value * tmp2);
        cuda::atomic::add(volume + offset_z + idx1.y * volume_pitch + idx1.x, proj_value * tmp3);
        cuda::atomic::add(volume_weights + offset_z + idx0.y * volume_pitch + idx0.x, proj_weight * tmp0);
        cuda::atomic::add(volume_weights + offset_z + idx0.y * volume_pitch + idx1.x, proj_weight * tmp1);
        cuda::atomic::add(volume_weights + offset_z + idx1.y * volume_pitch + idx0.x, proj_weight * tmp2);
        cuda::atomic::add(volume_weights + offset_z + idx1.y * volume_pitch + idx1.x, proj_weight * tmp3);
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
        auto samples = static_cast<float>(proj_dim);
        auto oversampling = static_cast<float>(volume_dim) / samples;
        auto sample_max = static_cast<uint>(math::clamp(freq_max, 0.f, 0.5f) * samples);
        auto sample_max_sqd = static_cast<float>(sample_max * sample_max);

        // Prepare for the EWS curvature.
        if (ewald_sphere_radius != 0.f)
            ewald_sphere_radius = prepareEWS_(volume_dim, ewald_sphere_radius);

        // Launch config.
        auto tmp = static_cast<uint>(proj_dim);
        const dim3 blocks(math::divideUp(tmp, THREADS.x),
                          math::divideUp(tmp, THREADS.x),
                          proj_count);

        // Prepare rotation matrices.
        cuda::memory::PtrDevice<float33_t> d_rotations(proj_count);
        cpu::memory::PtrHost<float33_t> h_rotations(proj_count); // PtrPinned?
        for (uint idx = 0; idx < proj_count; ++idx) {
            h_rotations[idx] = (transform::toMatrix<true>(proj_angles[idx])); // invert rotation
            h_rotations[idx] *= float33_t(oversampling); // scale for sampling during rotation
        }
        memory::copy(h_rotations.get(), d_rotations.get(), proj_count, stream);

        int p_pitch = static_cast<int>(proj_pitch);
        int p_dim = static_cast<int>(proj_dim);
        int v_pitch = static_cast<int>(volume_pitch);
        int v_dim = static_cast<int>(volume_dim);
        float shift_scaling = math::Constants<float>::PI2 / static_cast<float>(p_dim); // phase shift constant

        // Prepare magnifications (which are optional)
        bool apply_shift = proj_shifts != nullptr;
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
                fourierInsert_<IS_PROJ_CENTERED, IS_VOLUME_CENTERED, true, true, T, const float22_t*>
                <<<blocks, THREADS, 0, stream.get()>>>(
                        proj, proj_weights, p_pitch, p_dim, volume, volume_weights, v_pitch, v_dim,
                        proj_shifts, shift_scaling, d_mag_corrections.get(), d_rotations.get(),
                        sample_max_sqd, ewald_sphere_radius);
            } else {
                fourierInsert_<IS_PROJ_CENTERED, IS_VOLUME_CENTERED, false, true, T, const float22_t*>
                <<<blocks, THREADS, 0, stream.get()>>>(
                        proj, proj_weights, p_pitch, p_dim, volume, volume_weights, v_pitch, v_dim,
                        nullptr, 0.f, d_mag_corrections.get(), d_rotations.get(),
                        sample_max_sqd, ewald_sphere_radius);
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
                        nullptr, 0.f, nullptr, d_rotations.get(),
                        sample_max_sqd, ewald_sphere_radius);
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

        auto samples = static_cast<float>(proj_dim);
        auto oversampling = static_cast<float>(volume_dim) / samples;
        auto sample_max = static_cast<uint>(math::clamp(freq_max, 0.f, 0.5f) * samples);
        auto sample_max_sqd = static_cast<float>(sample_max * sample_max);

        if (ewald_sphere_radius != 0.f)
            ewald_sphere_radius = prepareEWS_(volume_dim, ewald_sphere_radius);

        cuda::memory::PtrDevice<float33_t> d_rotations(proj_count);
        cpu::memory::PtrHost<float33_t> h_rotations(proj_count);
        for (uint idx = 0; idx < proj_count; ++idx) {
            h_rotations[idx] = (transform::toMatrix<true>(proj_angles[idx]));
            h_rotations[idx] *= float33_t(oversampling);
        }
        memory::copy(h_rotations.get(), d_rotations.get(), proj_count, stream);

        bool apply_shift = proj_shifts != nullptr;
        float22_t distortion(1);
        distortion = transform::rotate(-proj_magnification.z) * // z is angle
                     transform::scale(float2_t(proj_magnification.x, proj_magnification.y)) *
                     transform::rotate(proj_magnification.z);

        int p_pitch = static_cast<int>(proj_pitch);
        int p_dim = static_cast<int>(proj_dim);
        int v_pitch = static_cast<int>(volume_pitch);
        int v_dim = static_cast<int>(volume_dim);
        float shift_scaling = math::Constants<float>::PI2 / static_cast<float>(p_dim);

        auto tmp = static_cast<uint>(proj_dim);
        const dim3 blocks(math::divideUp(tmp, THREADS.x),
                          math::divideUp(tmp, THREADS.x),
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
                    nullptr, 0.f, distortion, d_rotations.get(),
                    sample_max_sqd, ewald_sphere_radius);
        }
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
