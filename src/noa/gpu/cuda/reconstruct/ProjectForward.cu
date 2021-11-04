#include <memory>

#include "noa/common/transform/Geometry.h"
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
                            U distortion, // const float22_t* or float22_t
                            const float33_t* rotation_oversampling,
                            const float2_t* shifts, float shift_scaling,
                            float freq_max_sqd, float ewald_sphere_diam_inv) {

        // gid = current index in the projection. z is the projection ID.
        const int3_t gid(blockIdx.x * THREADS.x,
                         blockIdx.y * THREADS.y,
                         blockIdx.z);
        // frequencies u and v:
        //  u is equal to gid.x because proj is non-redundant
        //  v depends on the projection layout.
        const int proj_dim_half = proj_dim / 2;
        const int frequency_v = IS_PROJ_CENTERED ?
                                gid.y - proj_dim_half :
                                gid.y < (proj_dim + 1) / 2 ? gid.y : gid.y - proj_dim;
        float2_t freq_2d(gid.x, frequency_v);

        if constexpr(APPLY_MAG) {
            if constexpr (std::is_pointer_v<U>)
                freq_2d = distortion[gid.z] * freq_2d;
            else
                freq_2d = distortion * freq_2d;
        }

        float z = ewald_sphere_diam_inv * math::innerProduct(freq_2d, freq_2d);
        float3_t freq_3d(freq_2d.x, freq_2d.y, z);

        if (math::innerProduct(freq_3d, freq_3d) > freq_max_sqd)
            return;

        freq_3d = rotation_oversampling[gid.z] * freq_3d;
        T proj_value;
        if constexpr (traits::is_complex_v<T>) {
            using real_t = traits::value_type_t<T>;
            real_t conj = 1;
            if (freq_3d.x < 0) {
                freq_3d = -freq_3d;
                conj = -1;
            }
            proj_value = cuda::transform::tex3D<T, INTERP>(volume, freq_3d);
            proj_value.imag *= conj;
        } else {
            proj_value = cuda::transform::tex3D<T, INTERP>(volume, freq_3d.x < 0 ? -freq_3d: freq_3d);
        }

        if constexpr (traits::is_complex_v<T> && APPLY_SHIFT) {
            float2_t shift(shifts[gid.z]);
            shift *= shift_scaling;
            float factor = -math::sum(shift * float2_t(gid.x, frequency_v)); // gid.x == frequency_u
            cfloat_t phase_shift;
            math::sincos(factor, &phase_shift.imag, &phase_shift.real);
            proj_value *= phase_shift;
        }
        proj[gid.y * proj_pitch + gid.x] = proj_value;
    }

    template<bool IS_PROJ_CENTERED, bool APPLY_SHIFT, bool APPLY_MAG, typename T, typename U>
    void launch_(cudaTextureObject_t volume, InterpMode volume_interp_mode,
                 T* proj, int proj_pitch, int proj_dim, uint proj_count,
                 U distortions, // const float22_t* or float22_t
                 const float33_t* rotation_oversampling,
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
                                                       distortions, rotation_oversampling, shifts, shift_scaling,
                                                       freq_max_sqd, ewald_sphere_diam_inv);
            case INTERP_LINEAR_FAST:
                fourierExtractCentered_<INTERP_LINEAR_FAST, IS_PROJ_CENTERED, APPLY_SHIFT, APPLY_MAG>
                <<<blocks, THREADS, 0, stream.get()>>>(volume, proj, proj_pitch, proj_dim,
                                                       distortions, rotation_oversampling, shifts, shift_scaling,
                                                       freq_max_sqd, ewald_sphere_diam_inv);
            default:
                NOA_THROW_FUNC("projectForward", "Interpolation/filter mode should be {} or {}, got {}.",
                               INTERP_LINEAR, INTERP_LINEAR_FAST, volume_interp_mode);
        }
    }

    inline float prepareEWS_(size_t proj_dim, float ewald_sphere_radius) {
        return 1.f / (static_cast<float>(2 * proj_dim) * ewald_sphere_radius);
    }
}

namespace noa::cuda::reconstruct {
    // IS_VOLUME_CENTERED=false is not instantiated
    template<bool IS_PROJ_CENTERED, bool IS_VOLUME_CENTERED, typename T>
    void projectForward(cudaTextureObject_t volume, InterpMode volume_interp_mode, size_t volume_dim,
                        T* proj, size_t proj_pitch, size_t proj_dim,
                        const float3_t* proj_scaling_factors, const float33_t* proj_rotations,
                        const float2_t* proj_shifts, uint proj_count,
                        float freq_max, float ewald_sphere_radius, Stream& stream) {
        // Oversampling & max frequency.
        const auto samples = static_cast<float>(proj_dim);
        const auto oversampling = static_cast<float>(volume_dim) / samples;
        const auto sample_max = static_cast<uint>(math::clamp(freq_max, 0.f, 0.5f) * samples);
        const auto sample_max_sqd = static_cast<float>(sample_max * sample_max);

        // Prepare for the EWS curvature.
        if (ewald_sphere_radius != 0.f)
            ewald_sphere_radius = prepareEWS_(proj_dim, ewald_sphere_radius);

        // Prepare rotation matrices.
        cuda::memory::PtrDevice<float33_t> d_rotations(proj_count);
        std::unique_ptr<float33_t[]> h_rotations = std::make_unique<float33_t[]>(proj_count); // PtrPinned?
        for (uint idx = 0; idx < proj_count; ++idx)
            h_rotations[idx] *= proj_rotations[idx] * float33_t(oversampling);
        memory::copy(h_rotations.get(), d_rotations.get(), proj_count, stream);

        const int p_pitch = static_cast<int>(proj_pitch);
        const int p_dim = static_cast<int>(proj_dim);
        const float shift_scaling = math::Constants<float>::PI2 / static_cast<float>(p_dim);

        // Prepare magnifications (which are optional)
        const bool apply_shift = traits::is_complex_v<T> && proj_shifts != nullptr;
        if (proj_scaling_factors) {
            cuda::memory::PtrDevice<float22_t> d_distortions(proj_count);
            std::unique_ptr<float22_t[]> h_distortions = std::make_unique<float22_t[]>(proj_count);
            for (uint idx = 0; idx < proj_count; ++idx) {
                const float3_t& i_scale = proj_scaling_factors[idx];
                h_distortions[idx] = noa::transform::rotate(-i_scale.z) *
                                     noa::transform::scale(1.f / float2_t(i_scale.x, i_scale.y)) *
                                     noa::transform::rotate(i_scale.z);
            }
            memory::copy(h_distortions.get(), d_distortions.get(), proj_count, stream);

            if (apply_shift) {
                launch_<IS_PROJ_CENTERED, true, true, T, const float22_t*>(
                        volume, volume_interp_mode, proj, p_pitch, p_dim, proj_count,
                        d_distortions.get(), d_rotations.get(), proj_shifts, shift_scaling,
                        sample_max_sqd, ewald_sphere_radius, stream);
            } else {
                launch_<IS_PROJ_CENTERED, false, true, T, const float22_t*>(
                        volume, volume_interp_mode, proj, p_pitch, p_dim, proj_count,
                        d_distortions.get(), d_rotations.get(), nullptr, 0.f,
                        sample_max_sqd, ewald_sphere_radius, stream);
            }
        } else { // without mag
            if (apply_shift) {
                launch_<IS_PROJ_CENTERED, true, false, T, const float22_t*>(
                        volume, volume_interp_mode, proj, p_pitch, p_dim, proj_count,
                        nullptr, d_rotations.get(), proj_shifts, shift_scaling,
                        sample_max_sqd, ewald_sphere_radius, stream);
            } else {
                launch_<IS_PROJ_CENTERED, false, false, T, const float22_t*>(
                        volume, volume_interp_mode, proj, p_pitch, p_dim, proj_count,
                        nullptr, d_rotations.get(), nullptr, 0.f,
                        sample_max_sqd, ewald_sphere_radius, stream);
            }
        }
        NOA_THROW_IF(cudaGetLastError());
        stream.synchronize();
    }

    template<bool IS_PROJ_CENTERED, bool IS_VOLUME_CENTERED, typename T>
    void projectForward(cudaTextureObject_t volume, InterpMode volume_interp_mode, size_t volume_dim,
                        T* proj, size_t proj_pitch, size_t proj_dim,
                        float3_t proj_scaling_factor, const float33_t* proj_rotations, const float2_t* proj_shifts,
                        uint proj_count, float freq_max, float ewald_sphere_radius, Stream& stream) {
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
        std::unique_ptr<float33_t[]> h_rotations = std::make_unique<float33_t[]>(proj_count); // PtrPinned?
        for (uint idx = 0; idx < proj_count; ++idx)
            h_rotations[idx] *= proj_rotations[idx] * float33_t(oversampling);
        memory::copy(h_rotations.get(), d_rotations.get(), proj_count, stream);

        float22_t distortion(noa::transform::rotate(-proj_scaling_factor.z) *
                             noa::transform::scale(1.f / float2_t(proj_scaling_factor.x, proj_scaling_factor.y)) *
                             noa::transform::rotate(proj_scaling_factor.z));

        const int p_pitch = static_cast<int>(proj_pitch);
        const int p_dim = static_cast<int>(proj_dim);
        const float shift_scaling = math::Constants<float>::PI2 / static_cast<float>(p_dim);

        if (proj_shifts != nullptr) {
            launch_<IS_PROJ_CENTERED, true, true, T, float22_t>(
                    volume, volume_interp_mode, proj, p_pitch, p_dim, proj_count,
                    distortion, d_rotations.get(), proj_shifts, shift_scaling,
                    sample_max_sqd, ewald_sphere_radius, stream);
        } else {
            launch_<IS_PROJ_CENTERED, false, true, T, float22_t>(
                    volume, volume_interp_mode, proj, p_pitch, p_dim, proj_count,
                    distortion, d_rotations.get(), nullptr, 0.f,
                    sample_max_sqd, ewald_sphere_radius, stream);
        }
        NOA_THROW_IF(cudaGetLastError());
        stream.synchronize();
    }

    #define NOA_INSTANTIATE_FORWARD_(T)                                                                                             \
    template void projectForward<false, true, T>(cudaTextureObject_t, InterpMode, size_t, T*, size_t, size_t,                       \
                                                 const float3_t*, const float33_t*, const float2_t*, uint, float, float, Stream&);  \
    template void projectForward<true, true, T>(cudaTextureObject_t, InterpMode, size_t, T*, size_t, size_t,                        \
                                                const float3_t*, const float33_t*, const float2_t*, uint, float, float, Stream&);   \
    template void projectForward<false, true, T>(cudaTextureObject_t, InterpMode, size_t, T*, size_t, size_t,                       \
                                                float3_t, const float33_t*, const float2_t*, uint, float, float, Stream&);          \
    template void projectForward<true, true, T>(cudaTextureObject_t, InterpMode, size_t, T*, size_t, size_t,                        \
                                                float3_t, const float33_t*, const float2_t*, uint, float, float, Stream&)

    NOA_INSTANTIATE_FORWARD_(float);
    NOA_INSTANTIATE_FORWARD_(cfloat_t);
}
