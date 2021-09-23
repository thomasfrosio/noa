#include "noa/common/transform/Geometry.h"
#include "noa/cpu/memory/PtrHost.h"

#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/transform/Interpolate.h"
#include "noa/gpu/cuda/reconstruct/ProjectForward.h"

namespace {
    using namespace ::noa;
    constexpr dim3 THREADS(32, 8);

    template<InterpMode INTERP, bool IS_PROJ_CENTERED, bool APPLY_SHIFT, bool APPLY_MAG, typename T, typename U>
    __global__ void __launch_bounds__(THREADS.x * THREADS.y)
    fourierExtractCentered_(cudaTextureObject_t volume, T* proj, int proj_pitch, int proj_dim,
                            const float33_t* rotation_oversampling,
                            U magnifications, // const float22_t* or float22_t
                            const float2_t* shifts, float shift_scaling,
                            float freq_max_sqd, float ewald_sphere_diam_inv) {

        // gid = current index in the projection. z is the projection ID.
        const int3_t gid(blockIdx.x * THREADS.x,
                         blockIdx.y * THREADS.y,
                         blockIdx.z);
        // frequencies u and v: u is equal to gid.x because proj is non-redundant, v depends on the projection layout.
        const int proj_dim_half = proj_dim / 2;
        const int frequency_v = IS_PROJ_CENTERED ?
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
        if (math::innerProduct(freq_3d, freq_3d) > freq_max_sqd)
            return;

        // Rotate and scale for the oversampling.
        // The projection is non-redundant and the transformation can result in a negative v frequency.
        freq_3d = rotation_oversampling[gid.z] * freq_3d;
        using real_t = traits::value_type_t<T>;
        real_t conj = 1;
        if (freq_3d.x < 0) {
            freq_3d = -freq_3d;
            conj = -1;
        }

        // At this point, we can extract (i.e. interpolate) the value of the
        // current (u,v) frequency from the volume.
        T proj_value = cuda::transform::tex3D<T, INTERP>(volume, freq_3d.x, freq_3d.y, freq_3d.z);
        if constexpr (traits::is_complex_v<T>)
            proj_value.imag *= conj;

        // Phase shift value to apply the desired real space shift.
        // Use the untransformed frequencies.
        if constexpr(traits::is_complex_v<T> && APPLY_SHIFT) {
            float2_t shift(shifts[gid.z]);
            shift *= shift_scaling;
            float factor = -math::sum(shift * float2_t(gid.x, frequency_v)); // gid.x == frequency_u
            cfloat_t phase_shift;
            math::sincos(factor, &phase_shift.imag, &phase_shift.real);
            proj_value *= phase_shift;
        }
        proj[gid.y * proj_pitch + gid.x] = proj_value;
    }

    inline float prepareEWS_(size_t volume_dim, float ewald_sphere_radius) {
        ewald_sphere_radius *= static_cast<float>(volume_dim); // in pixel/meter
        return 1.f / (2.f * ewald_sphere_radius); // wavelength / 2
    }

    template<bool IS_PROJ_CENTERED, bool APPLY_SHIFT, bool APPLY_MAG, typename T, typename U>
    void launch_(cudaTextureObject_t volume, InterpMode volume_interp_mode,
                 T* proj, int proj_pitch, int proj_dim, uint proj_count,
                 const float33_t* rotation_oversampling,
                 U magnifications, // const float22_t* or float22_t
                 const float2_t* shifts, float shift_scaling,
                 float freq_max_sqd, float ewald_sphere_diam_inv, cuda::Stream& stream) {
        // Launch config.
        auto tmp = static_cast<uint>(proj_dim);
        const dim3 blocks(math::divideUp(tmp, THREADS.x),
                          math::divideUp(tmp, THREADS.y),
                          proj_count);

        switch (volume_interp_mode) {
            case INTERP_LINEAR:
                fourierExtractCentered_<INTERP_LINEAR, IS_PROJ_CENTERED, APPLY_SHIFT, APPLY_MAG>
                <<<blocks, THREADS, 0, stream.get()>>>(volume, proj, proj_pitch, proj_dim,
                                                       rotation_oversampling, magnifications, shifts, shift_scaling,
                                                       freq_max_sqd, ewald_sphere_diam_inv);
            case INTERP_LINEAR_FAST:
                fourierExtractCentered_<INTERP_LINEAR_FAST, IS_PROJ_CENTERED, APPLY_SHIFT, APPLY_MAG>
                <<<blocks, THREADS, 0, stream.get()>>>(volume, proj, proj_pitch, proj_dim,
                                                       rotation_oversampling, magnifications, shifts, shift_scaling,
                                                       freq_max_sqd, ewald_sphere_diam_inv);
            default:
                NOA_THROW_FUNC("projectForward", "Interpolation/filter mode should be {} or {}, got {}.",
                               INTERP_LINEAR, INTERP_LINEAR_FAST, volume_interp_mode);
        }
    }
}

namespace noa::cuda::reconstruct {
    template<bool IS_PROJ_CENTERED, bool IS_VOLUME_CENTERED, typename T>
    void projectForward(cudaTextureObject_t volume, InterpMode volume_interp_mode,
                        size_t volume_pitch, size_t volume_dim,
                        T* proj, size_t proj_pitch, size_t proj_dim, const float33_t* proj_rotations,
                        const float3_t* proj_magnifications, const float2_t* proj_shifts, uint proj_count,
                        float freq_max, float ewald_sphere_radius, Stream& stream) {
        // Oversampling & max frequency.
        const auto samples = static_cast<float>(proj_dim);
        const auto oversampling = static_cast<float>(volume_dim) / samples;
        const auto sample_max = static_cast<uint>(math::clamp(freq_max, 0.f, 0.5f) * samples);
        const auto sample_max_sqd = static_cast<float>(sample_max * sample_max);

        // Prepare for the EWS curvature.
        if (ewald_sphere_radius != 0.f)
            ewald_sphere_radius = prepareEWS_(volume_dim, ewald_sphere_radius);

        // Prepare rotation matrices.
        cuda::memory::PtrDevice<float33_t> d_rotations(proj_count);
        cpu::memory::PtrHost<float33_t> h_rotations(proj_count); // PtrPinned?
        for (uint idx = 0; idx < proj_count; ++idx) {
            h_rotations[idx] *= proj_rotations[idx] * float33_t(oversampling); // scale for sampling during rotation
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
                h_mag_corrections[idx] = noa::transform::rotate(-i_mag.z) *
                                         noa::transform::scale(1.f / float2_t(i_mag.x, i_mag.y)) * // invert
                                         noa::transform::rotate(i_mag.z);
            }
            memory::copy(h_mag_corrections.get(), d_mag_corrections.get(), proj_count, stream);

            if (apply_shift) {
                launch_<IS_PROJ_CENTERED, true, true, T, const float22_t*>(
                        volume, volume_interp_mode, proj, p_pitch, p_dim, proj_count, d_rotations.get(),
                        d_mag_corrections.get(), proj_shifts, shift_scaling,
                        sample_max_sqd, ewald_sphere_radius, stream);
            } else {
                launch_<IS_PROJ_CENTERED, false, true, T, const float22_t*>(
                        volume, volume_interp_mode, proj, p_pitch, p_dim, proj_count, d_rotations.get(),
                        d_mag_corrections.get(), nullptr, 0.f, sample_max_sqd, ewald_sphere_radius, stream);
            }
        } else { // without mag
            if (apply_shift) {
                launch_<IS_PROJ_CENTERED, true, false, T, const float22_t*>(
                        volume, volume_interp_mode, proj, p_pitch, p_dim, proj_count, d_rotations.get(),
                        nullptr, proj_shifts, shift_scaling, sample_max_sqd, ewald_sphere_radius, stream);
            } else {
                launch_<IS_PROJ_CENTERED, false, false, T, const float22_t*>(
                        volume, volume_interp_mode, proj, p_pitch, p_dim, proj_count, d_rotations.get(),
                        nullptr, nullptr, 0.f, sample_max_sqd, ewald_sphere_radius, stream);
            }
        }
        NOA_THROW_IF(cudaPeekAtLastError());
        stream.synchronize();
    }

    template<bool IS_PROJ_CENTERED, bool IS_VOLUME_CENTERED, typename T>
    void projectForward(cudaTextureObject_t volume, InterpMode volume_interp_mode,
                        size_t volume_pitch, size_t volume_dim,
                        T* proj, size_t proj_pitch, size_t proj_dim, const float33_t* proj_rotations,
                        float3_t proj_magnification, const float2_t* proj_shifts, uint proj_count,
                        float freq_max, float ewald_sphere_radius, Stream& stream) {
        // Oversampling & max frequency.
        const auto samples = static_cast<float>(proj_dim);
        const auto oversampling = static_cast<float>(volume_dim) / samples;
        const auto sample_max = static_cast<uint>(math::clamp(freq_max, 0.f, 0.5f) * samples);
        const auto sample_max_sqd = static_cast<float>(sample_max * sample_max);

        // Prepare for the EWS curvature.
        if (ewald_sphere_radius != 0.f)
            ewald_sphere_radius = prepareEWS_(volume_dim, ewald_sphere_radius);

        // Prepare rotation matrices.
        cuda::memory::PtrDevice<float33_t> d_rotations(proj_count);
        cpu::memory::PtrHost<float33_t> h_rotations(proj_count); // PtrPinned?
        for (uint idx = 0; idx < proj_count; ++idx)
            h_rotations[idx] *= proj_rotations[idx] * float33_t(oversampling); // scale for sampling during rotation
        memory::copy(h_rotations.get(), d_rotations.get(), proj_count, stream);

        float22_t distortion(noa::transform::rotate(-proj_magnification.z) * // z is angle
                             noa::transform::scale(1.f / float2_t(proj_magnification.x, proj_magnification.y)) *
                             noa::transform::rotate(proj_magnification.z));

        const int p_pitch = static_cast<int>(proj_pitch);
        const int p_dim = static_cast<int>(proj_dim);
        const int v_pitch = static_cast<int>(volume_pitch);
        const int v_dim = static_cast<int>(volume_dim);
        const float shift_scaling = math::Constants<float>::PI2 / static_cast<float>(p_dim); // phase shift constant

        if (proj_shifts != nullptr) {
            launch_<IS_PROJ_CENTERED, true, true, T, float22_t>(
                    volume, volume_interp_mode, proj, p_pitch, p_dim, proj_count, d_rotations.get(),
                    distortion, proj_shifts, shift_scaling, sample_max_sqd, ewald_sphere_radius, stream);
        } else {
            launch_<IS_PROJ_CENTERED, false, true, T, float22_t>(
                    volume, volume_interp_mode, proj, p_pitch, p_dim, proj_count, d_rotations.get(),
                    distortion, nullptr, 0.f, sample_max_sqd, ewald_sphere_radius, stream);
        }
        NOA_THROW_IF(cudaPeekAtLastError());
        stream.synchronize();
    }

    #define NOA_INSTANTIATE_FORWARD_(T)                                                                                             \
    template void projectForward<false, true, T>(cudaTextureObject_t, InterpMode, size_t, size_t, T*, size_t, size_t,               \
                                                 const float33_t*, const float3_t*, const float2_t*, uint, float, float, Stream&);  \
    template void projectForward<true, true, T>(cudaTextureObject_t, InterpMode, size_t, size_t, T*, size_t, size_t,                \
                                                const float33_t*, const float3_t*, const float2_t*, uint, float, float, Stream&);   \
    template void projectForward<false, true, T>(cudaTextureObject_t, InterpMode, size_t, size_t, T*, size_t, size_t,               \
                                                const float33_t*, float3_t, const float2_t*, uint, float, float, Stream&);          \
    template void projectForward<true, true, T>(cudaTextureObject_t, InterpMode, size_t, size_t, T*, size_t, size_t,                \
                                                const float33_t*, float3_t, const float2_t*, uint, float, float, Stream&)

    NOA_INSTANTIATE_FORWARD_(float);
    NOA_INSTANTIATE_FORWARD_(double);
    NOA_INSTANTIATE_FORWARD_(cfloat_t);
    NOA_INSTANTIATE_FORWARD_(cdouble_t);
}
