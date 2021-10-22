#include "noa/common/transform/Geometry.h"
#include "noa/cpu/transform/Interpolator.h"
#include "noa/cpu/reconstruct/ProjectForward.h"

namespace {
    using namespace ::noa;

    template<bool PROJ_IS_CENTERED, bool PHASE_SHIFT, typename T>
    void fourierExtractCentered_(const T* volume, int volume_dim, T* proj, int proj_dim,
                                 float33_t rotation_oversampling, float22_t magnification, float2_t shift,
                                 float freq_max_sqd, float ewald_sphere_diam_inv) {
        int proj_dim_half = proj_dim / 2;
        int proj_pitch = proj_dim_half + 1;
        int volume_pitch = volume_dim / 2 + 1;

        // The volume is non-redundant. Pass the physical size to the interpolator, not the logical.
        cpu::transform::Interpolator3D<T> interp(volume, size3_t(volume_pitch, volume_dim, volume_dim),
                                                 static_cast<size_t>(volume_pitch));

        if constexpr(traits::is_complex_v<T> && PHASE_SHIFT)
            shift *= math::Constants<float>::PI2 / static_cast<float>(proj_dim); // prepare phase-shift

        // The DC should be at index 0 to rotate around it, so Y needs to be
        // adjusted depending on its layout. The array is non-redundant, so X == U.
        int u, v;
        for (int y = 0; y < proj_dim; ++y) {
            if constexpr (PROJ_IS_CENTERED) // what's the frequency v saved at index y
                v = y - proj_dim_half;
            else
                v = y < (proj_dim + 1) / 2 ? y : y - proj_dim;
            for (u = 0; u < proj_dim_half + 1; ++u) { // x == u

                // Correct for the (anisotropic) magnification.
                float2_t freq_2d(u, v);
                freq_2d = magnification * freq_2d;

                // Curve the projection to match the EWS.
                float z = ewald_sphere_diam_inv * math::innerProduct(freq_2d, freq_2d);
                float3_t freq_3d(freq_2d.x, freq_2d.y, z);

                // If the frequency is larger than the max authorized, stop here.
                if (math::innerProduct(freq_3d, freq_3d) > freq_max_sqd)
                    continue;

                // Rotate and scale for the oversampling.
                // The projection is non-redundant and the transformation can result in a negative v frequency.
                // Then extract (i.e. interpolate) the value of the current (u,v) frequency from the volume.
                freq_3d = rotation_oversampling * freq_3d;
                T proj_value;
                if constexpr (traits::is_complex_v<T>) {
                    using real_t = traits::value_type_t<T>;
                    real_t conj = 1;
                    if (freq_3d.x < 0) {
                        freq_3d = -freq_3d;
                        conj = -1;
                    }
                    proj_value = interp.template get<INTERP_LINEAR, BORDER_ZERO>(freq_3d);
                    proj_value.imag *= conj;
                } else {
                    proj_value = interp.template get<INTERP_LINEAR, BORDER_ZERO>(freq_3d.x < 0 ? -freq_3d : freq_3d);
                }

                // Phase shift value to apply the desired real space shift.
                // Use the untransformed frequencies.
                if constexpr (traits::is_complex_v<T> && PHASE_SHIFT) {
                    float factor = -(shift.x * static_cast<float>(u) + shift.y * static_cast<float>(v));
                    cfloat_t phase_shift;
                    math::sincos(factor, &phase_shift.imag, &phase_shift.real);
                    proj_value *= phase_shift;
                }
                proj[y * proj_pitch + u] = proj_value; // x == u
            }
        }
    }

    inline float prepareEWS_(size_t proj_dim, float ewald_sphere_radius) {
        return 1.f / (static_cast<float>(2 * proj_dim) * ewald_sphere_radius);
    }
}

namespace noa::cpu::reconstruct {
    template<bool IS_PROJ_CENTERED, bool IS_VOLUME_CENTERED, typename T>
    void projectForward(const T* volume, size_t volume_dim, T* proj, size_t proj_dim,
                        const float3_t* proj_scaling_factors, const float33_t* proj_rotations,
                        const float2_t* proj_shifts, uint proj_count,
                        float freq_max, float ewald_sphere_radius) {
        // Oversampling & max frequency.
        auto samples = static_cast<float>(proj_dim);
        auto oversampling = static_cast<float>(volume_dim) / samples;
        auto sample_max = static_cast<uint>(math::clamp(freq_max, 0.f, 0.5f) * samples);
        auto sample_max_sqd = static_cast<float>(sample_max * sample_max);

        // Prepare for the EWS curvature.
        if (ewald_sphere_radius != 0.f)
            ewald_sphere_radius = prepareEWS_(proj_dim, ewald_sphere_radius);

        const bool apply_shift = traits::is_complex_v<T> && proj_shifts != nullptr;
        const bool apply_mag = proj_scaling_factors != nullptr;

        // Direct Fourier insertion, one projection at a time.
        size_t proj_elements = (proj_dim / 2 + 1) * proj_dim;
        for (uint idx = 0; idx < proj_count; ++idx) {
            T* i_proj = proj + proj_elements * idx;

            float33_t rotation(proj_rotations[idx] * float33_t(oversampling));
            float22_t distortion;
            if (apply_mag) {
                const float3_t& i_scale = proj_scaling_factors[idx];
                distortion = noa::transform::rotate(-i_scale.z) *
                             noa::transform::scale(1.f / float2_t(i_scale.x, i_scale.y)) *
                             noa::transform::rotate(i_scale.z);
            }

            if (apply_shift) {
                fourierExtractCentered_<IS_PROJ_CENTERED, true>(
                        volume, static_cast<int>(volume_dim), i_proj, static_cast<int>(proj_dim),
                        rotation, distortion, proj_shifts[idx], sample_max_sqd, ewald_sphere_radius);
            } else {
                fourierExtractCentered_<IS_PROJ_CENTERED, false>(
                        volume, static_cast<int>(volume_dim), i_proj, static_cast<int>(proj_dim),
                        rotation, distortion, float2_t{}, sample_max_sqd, ewald_sphere_radius);
            }
        }
    }

    template<bool IS_PROJ_CENTERED, bool IS_VOLUME_CENTERED, typename T>
    void projectForward(const T* volume, size_t volume_dim, T* proj, size_t proj_dim,
                        float3_t proj_scaling_factor, const float33_t* proj_rotations,
                        const float2_t* proj_shifts, uint proj_count,
                        float freq_max, float ewald_sphere_radius) {
        auto samples = static_cast<float>(proj_dim);
        auto oversampling = static_cast<float>(volume_dim) / samples;
        auto sample_max = static_cast<uint>(math::clamp(freq_max, 0.f, 0.5f) * samples);
        auto sample_max_sqd = static_cast<float>(sample_max * sample_max);

        if (ewald_sphere_radius != 0.f)
            ewald_sphere_radius = prepareEWS_(volume_dim, ewald_sphere_radius);

        float22_t distortion(noa::transform::rotate(-proj_scaling_factor.z) *
                             noa::transform::scale(1.f / float2_t(proj_scaling_factor.x, proj_scaling_factor.y)) *
                             noa::transform::rotate(proj_scaling_factor.z));

        const bool apply_shift = proj_shifts != nullptr;
        size_t proj_elements = (proj_dim / 2 + 1) * proj_dim;
        for (uint idx = 0; idx < proj_count; ++idx) {
            T* i_proj = proj + proj_elements * idx;

            float33_t rotation(proj_rotations[idx] * float33_t(oversampling));

            if (apply_shift) {
                fourierExtractCentered_<IS_PROJ_CENTERED, true>(
                        volume, static_cast<int>(volume_dim), i_proj, static_cast<int>(proj_dim),
                        rotation, distortion, proj_shifts[idx], sample_max_sqd, ewald_sphere_radius);
            } else {
                fourierExtractCentered_<IS_PROJ_CENTERED, false>(
                        volume, static_cast<int>(volume_dim), i_proj, static_cast<int>(proj_dim),
                        rotation, distortion, float2_t{}, sample_max_sqd, ewald_sphere_radius);
            }
        }
    }

    #define NOA_INSTANTIATE_FORWARD_(T) \
    template void projectForward<false, true, T>(const T*, size_t, T*, size_t, const float3_t*, const float33_t*, const float2_t*, uint, float, float); \
    template void projectForward<true, true, T>(const T*, size_t, T*, size_t, const float3_t*, const float33_t*, const float2_t*, uint, float, float);  \
    template void projectForward<false, true, T>(const T*, size_t, T*, size_t, float3_t, const float33_t*, const float2_t*, uint, float, float);        \
    template void projectForward<true, true, T>(const T*, size_t, T*, size_t, float3_t, const float33_t*, const float2_t*, uint, float, float)

    NOA_INSTANTIATE_FORWARD_(float);
    NOA_INSTANTIATE_FORWARD_(double);
    NOA_INSTANTIATE_FORWARD_(cfloat_t);
    NOA_INSTANTIATE_FORWARD_(cdouble_t);
}
