#include "noa/common/transform/Euler.h"
#include "noa/common/transform/Geometry.h"
#include "noa/cpu/reconstruct/ProjectBackward.h"

namespace {
    using namespace ::noa;

    // Data-driven interpolation: add the (curved) slice to the (oversampled) cartesian grid.
    // The volume weights keep track of what has been added to the volume.
    template<bool PROJ_IS_CENTERED, bool VOLUME_IS_CENTERED, bool PHASE_SHIFT, typename T>
    void fourierInsert(const Complex<T>* proj, const T* proj_weights, int proj_dim,
                       Complex<T>* volume, T* volume_weights, int volume_dim,
                       float2_t shift, float22_t magnification, float33_t rotation_oversampling,
                       float freq_max_sqd, float ewald_sphere_diam_inv) {
        int proj_dim_half = proj_dim / 2;
        int volume_dim_half = volume_dim / 2;
        int proj_pitch = proj_dim_half + 1;
        int volume_pitch = volume_dim_half + 1;

        if constexpr(PHASE_SHIFT)
            shift *= math::Constants<float>::PI2 / static_cast<float>(proj_dim); // prepare shifts

        // The DC should be at index 0 to rotate around it, so Y needs to be
        // adjusted depending on its layout. The array is non-redundant, so X is fine.
        int u, v;
        for (int y = 0; y < proj_dim; ++y) {
            if constexpr(PROJ_IS_CENTERED) // what's the frequency v saved at index y
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
                // This is enough to remove most OOB conditions. However, since the interpolation window is
                // 2 elements wide, there can still be rare cases of OOB after that point.
                if (math::innerProduct(freq_3d, freq_3d) > freq_max_sqd)
                    continue;

                // Rotate and scale for the oversampling.
                // The projection is non-redundant and the transformation can result in a negative v frequency.
                // In this case, take the conjugate.
                freq_3d = rotation_oversampling * freq_3d;
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
                idx0.y += volume_dim_half;
                idx0.z += volume_dim_half;
                int3_t idx1(idx0 + 1);

                // In most (if not all cases) this is fine...
                // idx0 can be -1, and idx1 can be volume_dim
                // volume_pitch is equal to the logical dimension, so that's fine here.
                // TODO debug and see if this is necessary
                if (idx0.x < 0 || idx1.x >= volume_pitch ||
                    idx0.y < 0 || idx1.y >= volume_dim ||
                    idx0.z < 0 || idx1.z >= volume_dim)
                    continue;

                // If the volume is not centered, decenter the coordinates.
                if constexpr(!VOLUME_IS_CENTERED) {
                    idx0.y = math::iFFTShift(idx0.y, volume_dim);
                    idx0.z = math::iFFTShift(idx0.z, volume_dim);
                    idx1.y = math::iFFTShift(idx1.y, volume_dim);
                    idx1.z = math::iFFTShift(idx1.z, volume_dim);
                }

                // At this point, we can add the current (x,y) element from the projection into the volume,
                // at the transformed coordinate which is located within idx0 and idx1.
                Complex<T> proj_value = proj[y * proj_pitch + u]; // x == u
                proj_value.imag *= conj;
                T proj_weight = proj_weights[y * proj_pitch + u];

                // Phase shift value to apply the desired real space shift.
                // Use the untransformed frequencies.
                if constexpr(PHASE_SHIFT) {
                    float factor = -(shift.x * static_cast<float>(u) + shift.y * static_cast<float>(v));
                    cfloat_t phase_shift;
                    math::sincos(factor, &phase_shift.imag, &phase_shift.real);
                    proj_value *= phase_shift;
                }

                // Add to 2x2x2 cube to the cartesian grid.
                T weights[8] = {static_cast<T>(weight0.z * weight0.y * weight0.x),
                                static_cast<T>(weight0.z * weight0.y * weight1.x),
                                static_cast<T>(weight0.z * weight1.y * weight0.x),
                                static_cast<T>(weight0.z * weight1.y * weight1.x),
                                static_cast<T>(weight1.z * weight0.y * weight0.x),
                                static_cast<T>(weight1.z * weight0.y * weight1.x),
                                static_cast<T>(weight1.z * weight1.y * weight0.x),
                                static_cast<T>(weight1.z * weight1.y * weight1.x)};

                volume[(idx0.z * volume_dim + idx0.y) * volume_pitch + idx0.x] += proj_value * weights[0];
                volume[(idx0.z * volume_dim + idx0.y) * volume_pitch + idx1.x] += proj_value * weights[1];
                volume[(idx0.z * volume_dim + idx1.y) * volume_pitch + idx0.x] += proj_value * weights[2];
                volume[(idx0.z * volume_dim + idx1.y) * volume_pitch + idx1.x] += proj_value * weights[3];
                volume[(idx1.z * volume_dim + idx0.y) * volume_pitch + idx0.x] += proj_value * weights[4];
                volume[(idx1.z * volume_dim + idx0.y) * volume_pitch + idx1.x] += proj_value * weights[5];
                volume[(idx1.z * volume_dim + idx1.y) * volume_pitch + idx0.x] += proj_value * weights[6];
                volume[(idx1.z * volume_dim + idx1.y) * volume_pitch + idx1.x] += proj_value * weights[7];

                volume_weights[(idx0.z * volume_dim + idx0.y) * volume_pitch + idx0.x] += proj_weight * weights[0];
                volume_weights[(idx0.z * volume_dim + idx0.y) * volume_pitch + idx1.x] += proj_weight * weights[1];
                volume_weights[(idx0.z * volume_dim + idx1.y) * volume_pitch + idx0.x] += proj_weight * weights[2];
                volume_weights[(idx0.z * volume_dim + idx1.y) * volume_pitch + idx1.x] += proj_weight * weights[3];
                volume_weights[(idx1.z * volume_dim + idx0.y) * volume_pitch + idx0.x] += proj_weight * weights[4];
                volume_weights[(idx1.z * volume_dim + idx0.y) * volume_pitch + idx1.x] += proj_weight * weights[5];
                volume_weights[(idx1.z * volume_dim + idx1.y) * volume_pitch + idx0.x] += proj_weight * weights[6];
                volume_weights[(idx1.z * volume_dim + idx1.y) * volume_pitch + idx1.x] += proj_weight * weights[7];
            }
        }
    }

    // The EWS radius should be 1/wavelength in SI, so in m^-1.
    // To curve the slice and match the EWS, we need to compute the Z.
    // Using the small-angle approximation, Z = wavelength / 2 * (X^2 + Y^2).
    // See doi:10.1016/S0304-3991(99)00120-5 for a derivation.
    inline float prepareEWS(size_t volume_dim, float ewald_sphere_radius) {
        ewald_sphere_radius *= static_cast<float>(volume_dim); // in pixel/meter
        return 1.f / (2.f * ewald_sphere_radius); // wavelength / 2
    }
}

namespace noa::cpu::reconstruct {
    template<bool IS_PROJ_CENTERED, bool IS_VOLUME_CENTERED, typename T>
    void projectBackward(const Complex<T>* proj, const T* proj_weights, size_t proj_dim,
                         const float2_t* proj_shifts, const float3_t* proj_magnifications,
                         const float3_t* proj_angles, uint proj_count,
                         Complex<T>* volume, T* volume_weights, size_t volume_dim,
                         float freq_max, float ewald_sphere_radius) {

        // Oversampling & max frequency.
        auto samples = static_cast<float>(proj_dim);
        auto oversampling = static_cast<float>(volume_dim) / samples;
        auto sample_max = static_cast<uint>(math::clamp(freq_max, 0.f, 0.5f) * samples);
        auto sample_max_sqd = static_cast<float>(sample_max * sample_max);

        // Prepare for the EWS curvature.
        if (ewald_sphere_radius != 0.f)
            ewald_sphere_radius = prepareEWS(volume_dim, ewald_sphere_radius);

        bool apply_shift = proj_shifts != nullptr;
        bool apply_mag = proj_magnifications != nullptr;

        // Direct Fourier insertion, one projection at a time.
        size_t proj_elements = (proj_dim / 2 + 1) * proj_dim;
        for (uint idx = 0; idx < proj_count; ++idx) {
            const Complex<T>* i_proj = proj + proj_elements * idx;
            const T* i_proj_weights = proj_weights + proj_elements * idx;

            float33_t rotation(transform::toMatrix<true>(proj_angles[idx])); // invert rotation
            rotation *= float33_t(oversampling); // scale for sampling during rotation

            // TODO Add CTF magnification struct somewhere and add corresponding overload for transform::scale.
            float22_t distortion(1);
            if (apply_mag) {
                const float3_t& i_mag = proj_magnifications[idx]; // z is angle
                distortion = transform::rotate(-i_mag.z) *
                             transform::scale(float2_t(i_mag.x, i_mag.y)) *
                             transform::rotate(i_mag.z);
            }

            if (apply_shift) {
                fourierInsert<IS_PROJ_CENTERED, IS_VOLUME_CENTERED, true>(
                        i_proj, i_proj_weights, static_cast<int>(proj_dim),
                        volume, volume_weights, static_cast<int>(volume_dim),
                        proj_shifts[idx], distortion, rotation, sample_max_sqd, ewald_sphere_radius);
            } else {
                fourierInsert<IS_PROJ_CENTERED, IS_VOLUME_CENTERED, false>(
                        i_proj, i_proj_weights, static_cast<int>(proj_dim),
                        volume, volume_weights, static_cast<int>(volume_dim),
                        float2_t(0.f), distortion, rotation, sample_max_sqd, ewald_sphere_radius);
            }
        }
    }

    template<bool IS_PROJ_CENTERED, bool IS_VOLUME_CENTERED, typename T>
    void projectBackward(const Complex<T>* proj, const T* proj_weights, size_t proj_dim,
                         const float2_t* proj_shifts, float3_t proj_magnification,
                         const float3_t* proj_angles, uint proj_count,
                         Complex<T>* volume, T* volume_weights, size_t volume_dim,
                         float freq_max, float ewald_sphere_radius) {
        // Oversampling & max frequency.
        auto samples = static_cast<float>(proj_dim);
        auto oversampling = static_cast<float>(volume_dim) / samples;
        auto sample_max = static_cast<uint>(math::clamp(freq_max, 0.f, 0.5f) * samples);
        auto sample_max_sqd = static_cast<float>(sample_max * sample_max);

        // Prepare for the EWS curvature.
        if (ewald_sphere_radius != 0.f)
            ewald_sphere_radius = prepareEWS(volume_dim, ewald_sphere_radius);

        // Prepare the magnification.
        float22_t distortion(transform::rotate(-proj_magnification.z) *
                             transform::scale(float2_t(proj_magnification.x, proj_magnification.y)) *
                             transform::rotate(proj_magnification.z));

        // Direct Fourier insertion, one projection at a time.
        bool apply_shift = proj_shifts != nullptr;
        size_t proj_elements = (proj_dim / 2 + 1) * proj_dim;
        for (uint idx = 0; idx < proj_count; ++idx) {
            const Complex<T>* i_proj = proj + proj_elements * idx;
            const T* i_proj_weights = proj_weights + proj_elements * idx;

            float33_t rotation(transform::toMatrix<true>(proj_angles[idx])); // invert rotation
            rotation *= float33_t(oversampling); // scale for sampling during rotation

            if (apply_shift) {
                fourierInsert<IS_PROJ_CENTERED, IS_VOLUME_CENTERED, true>(
                        i_proj, i_proj_weights, static_cast<int>(proj_dim),
                        volume, volume_weights, static_cast<int>(volume_dim),
                        proj_shifts[idx], distortion, rotation, sample_max_sqd, ewald_sphere_radius);
            } else {
                fourierInsert<IS_PROJ_CENTERED, IS_VOLUME_CENTERED, false>(
                        i_proj, i_proj_weights, static_cast<int>(proj_dim),
                        volume, volume_weights, static_cast<int>(volume_dim),
                        float2_t(0.f), distortion, rotation, sample_max_sqd, ewald_sphere_radius);
            }
        }
    }

    #define NOA_INSTANTIATE_BACKWARD_(T)                                                                                    \
    template void projectBackward<true, true, T>(const Complex<T>*, const T*, size_t, const float2_t*, const float3_t*,     \
                                                 const float3_t*, uint, Complex<T>*, T*, size_t, float, float);             \
    template void projectBackward<true, false, T>(const Complex<T>*, const T*, size_t, const float2_t*, const float3_t*,    \
                                                  const float3_t*, uint, Complex<T>*, T*, size_t, float, float);            \
    template void projectBackward<false, true, T>(const Complex<T>*, const T*, size_t, const float2_t*, const float3_t*,    \
                                                  const float3_t*, uint, Complex<T>*, T*, size_t, float, float);            \
    template void projectBackward<false, false, T>(const Complex<T>*, const T*, size_t, const float2_t*, const float3_t*,   \
                                                   const float3_t*, uint, Complex<T>*, T*, size_t, float, float);           \
    template void projectBackward<true, true, T>(const Complex<T>*, const T*, size_t, const float2_t*, float3_t,            \
                                                 const float3_t*, uint, Complex<T>*, T*, size_t, float, float);             \
    template void projectBackward<true, false, T>(const Complex<T>*, const T*, size_t, const float2_t*, float3_t,           \
                                                  const float3_t*, uint, Complex<T>*, T*, size_t, float, float);            \
    template void projectBackward<false, true, T>(const Complex<T>*, const T*, size_t, const float2_t*, float3_t,           \
                                                  const float3_t*, uint, Complex<T>*, T*, size_t, float, float);            \
    template void projectBackward<false, false, T>(const Complex<T>*, const T*, size_t, const float2_t*, float3_t,          \
                                                   const float3_t*, uint, Complex<T>*, T*, size_t, float, float)
    NOA_INSTANTIATE_BACKWARD_(float);
    NOA_INSTANTIATE_BACKWARD_(double);
}
