#include "noa/common/Types.h"
#include "noa/common/transform/Geometry.h"
#include "noa/cpu/reconstruct/ProjectBackward.h"

namespace {
    using namespace ::noa;

    inline cfloat_t getPhaseShift_(float2_t shift, float2_t freq) {
        float factor = -math::sum(shift * freq);
        cfloat_t phase_shift;
        math::sincos(factor, &phase_shift.imag, &phase_shift.real);
        return phase_shift;
    }

    // Go from the DC-centered frequency to a valid index in the output volume.
    // This operation of course depends on the volume's FFT layout.
    // The input frequency should be in-bound, i.e. -n/2 <= frequency <= (n-1)/2
    template<bool IS_VOLUME_CENTERED>
    inline int toIndex_(int frequency, int volume_dim) {
        if constexpr (IS_VOLUME_CENTERED) {
            return frequency + volume_dim / 2;
        } else {
            return frequency < 0 ? frequency + volume_dim : frequency;
        }
    }

    // Linear interpolation on the 2x2x2 cube.
    // v1 and v2 can be nullptr, freq is the frequency centered on DC (i.e. freq == 0 is the DC),
    // with negative frequencies on the left and the base is floor(freq).
    template<bool IS_VOLUME_CENTERED, typename T>
    void addByInterpolation_(Complex<T>* v1, Complex<T> a1, T* v2, T a2,
                             float3_t freq, int volume_dim, int volume_pitch) {

        // So if the coordinate is centered in the bottom left corner of the cube (base0),
        // i.e. its decimal is 0, the corresponding fraction for this element should be 1.
        int3_t base0(math::floor(freq));
        float weights[2][2][2];
        {
            float3_t fraction1(freq - float3_t(base0));
            float3_t fraction0(1.f - fraction1);
            weights[0][0][0] = fraction0.z * fraction0.y * fraction0.x;
            weights[0][0][1] = fraction0.z * fraction0.y * fraction1.x;
            weights[0][1][0] = fraction0.z * fraction1.y * fraction0.x;
            weights[0][1][1] = fraction0.z * fraction1.y * fraction1.x;
            weights[1][0][0] = fraction1.z * fraction0.y * fraction0.x;
            weights[1][0][1] = fraction1.z * fraction0.y * fraction1.x;
            weights[1][1][0] = fraction1.z * fraction1.y * fraction0.x;
            weights[1][1][1] = fraction1.z * fraction1.y * fraction1.x;
        }

        bool oob_z[2], oob_y[2], oob_x[2];
        {
            // For each voxel in the 2x2x2 cube, compute whether it is out of bound (OOB).
            // Note that for even dimensions, the negative (real) nyquist is considered OOB:
            // n = 10: -5 -4 -3 -2 -1 0 1 2 3 4    ->    allowed range: [-4, ..., 4]
            // n = 9:     -4 -3 -2 -1 0 1 2 3 4    ->    allowed range: [-4, ..., 4]
            int3_t base1(base0 + 1);
            const int idx_max = (volume_dim - 1) / 2;
            oob_z[0] = base0.z < -idx_max || base0.z > idx_max;
            oob_z[1] = base1.z < -idx_max || base1.z > idx_max;
            oob_y[0] = base0.y < -idx_max || base0.y > idx_max;
            oob_y[1] = base1.y < -idx_max || base1.y > idx_max;
            oob_x[0] = base0.x > idx_max;
            oob_x[1] = base1.x > idx_max;
        }

        for (int z = 0; z < 2; ++z) {
            if (oob_z[z])
                continue;
            int64_t idx_z = toIndex_<IS_VOLUME_CENTERED>(base0.z + z, volume_dim);
            for (int y = 0; y < 2; ++y) {
                if (oob_y[y])
                    continue;
                int64_t idx_y = toIndex_<IS_VOLUME_CENTERED>(base0.y + y, volume_dim);
                for (int x = 0; x < 2; ++x) {
                    if (oob_x[x])
                        continue;

                    T fraction = static_cast<T>(weights[z][y][x]);
                    int64_t offset = (idx_z * volume_dim + idx_y) * volume_pitch + base0.x + x;
                    if (v1)
                        v1[offset] += a1 * fraction;
                    if (v2)
                        v2[offset] += a2 * fraction;
                }
            }
        }

        // Redundant plane: the x==0 plane (excluding the DC) is redundant, so to preserve the hermitian symmetry,
        // we must add the conjugate at (x=0, -y, -z) with the same fraction. Fortunately, if frequency X is in-bound,
        // we know for sure that frequency -X is in-bound too.
        if (base0.x == 0 && (base0.y != 0 || base0.z != 0)) {
            a1.imag = -a1.imag;
            for (int z = 0; z < 2; ++z) {
                if (oob_z[z])
                    continue;
                int64_t idx_z = toIndex_<IS_VOLUME_CENTERED>(-(base0.z + z), volume_dim);
                for (int y = 0; y < 2; ++y) {
                    if (oob_y[y])
                        continue;
                    int64_t idx_y = toIndex_<IS_VOLUME_CENTERED>(-(base0.y + y), volume_dim);

                    T fraction = static_cast<T>(weights[z][y][0]);
                    int64_t offset = (idx_z * volume_dim + idx_y) * volume_pitch;
                    if (v1)
                        v1[offset] += a1 * fraction;
                    if (v2)
                        v2[offset] += a2 * fraction;
                }
            }
        }
    }

    // Data-driven interpolation: add the (curved) slice to the (oversampled) cartesian grid.
    // The volume weights keep track of what has been added to the volume.
    template<bool IS_PROJ_CENTERED, bool IS_VOLUME_CENTERED, bool PHASE_SHIFT, typename T>
    void fourierInsert_(const Complex<T>* proj, const T* proj_weights, int proj_dim,
                        Complex<T>* volume, T* volume_weights, int volume_dim,
                        [[maybe_unused]] float2_t shift, float22_t scaling, float33_t rotation_oversampling,
                        float freq_max_sqd, float ewald_sphere_diam_inv) {
        int proj_dim_half = proj_dim / 2;
        int volume_dim_half = volume_dim / 2;
        int proj_pitch = proj_dim_half + 1;
        int volume_pitch = volume_dim_half + 1;

        if constexpr ( PHASE_SHIFT)
            shift *= math::Constants<float>::PI2 / static_cast<float>(proj_dim); // prepare shifts

        // The DC should be at index 0 to rotate around it, so Y needs to be
        // adjusted depending on its layout. The array is non-redundant, so X is fine.
        // The transformation goes from the projection to the volume: forward transformation.
        int u, v;
        for (int y = 0; y < proj_dim; ++y) {
            if constexpr (IS_PROJ_CENTERED) // what's the frequency v saved at index y
                v = y - proj_dim_half;
            else
                v = y < (proj_dim + 1) / 2 ? y : y - proj_dim;
            for (u = 0; u < proj_dim_half + 1; ++u) { // x == u

                // Applies the (anisotropic) scaling to the projection.
                float2_t freq_2d(u, v);
                freq_2d = scaling * freq_2d;

                // Curve the projection to match the EWS.
                float z = ewald_sphere_diam_inv * math::dot(freq_2d, freq_2d);
                float3_t freq_3d(freq_2d.x, freq_2d.y, z);

                // If the frequency is larger than Nyquist, stop here.
                // This is enough to remove most OOB conditions. However, since the interpolation window is
                // 2 elements wide, there can still be rare cases of OOB after that point.
                if (math::dot(freq_3d, freq_3d) > freq_max_sqd)
                    continue;

                // Rotate and oversample.
                // The projection is non-redundant and the transformation can result in a negative v frequency.
                // In this case, take the conjugate.
                freq_3d = rotation_oversampling * freq_3d;
                T conj = 1;
                if (freq_3d.x < 0) {
                    freq_3d = -freq_3d;
                    conj = -1;
                }

                // At this point, we know for sure we are going to use the projection value and weight.
                Complex<T> proj_value;
                if (volume) { // should be predicted
                    proj_value = proj[y * proj_pitch + u];
                    proj_value.imag *= conj;

                    if constexpr (PHASE_SHIFT) {
                        // Phase shift value to apply the desired real space shift.
                        // Use the untransformed frequencies.
                        float2_t orig_freq(u, v);
                        proj_value *= getPhaseShift_(shift, orig_freq);
                    }
                }
                T proj_weight = volume_weights ? proj_weights[y * proj_pitch + u] : 0;

                addByInterpolation_<IS_VOLUME_CENTERED>(volume, proj_value, volume_weights, proj_weight,
                                                        freq_3d, volume_dim, volume_pitch);
            }
        }
    }

    // The EWS radius should be per pixels: pixel_size[A/pix]/lambda[A].
    // To curve the slice and match the EWS, we need to compute the Z.
    // Using the small-angle approximation, Z = wavelength / 2 * (X^2 + Y^2).
    // See doi:10.1016/S0304-3991(99)00120-5 for a derivation.
    inline float prepareEWS_(size_t proj_dim, float ewald_sphere_radius) {
        // scale to projection and return 1/diameter.
        return 1.f / (static_cast<float>(2 * proj_dim) * ewald_sphere_radius); // wavelength / 2, in 1/pixel
    }
}

namespace noa::cpu::reconstruct {
    template<bool IS_PROJ_CENTERED, bool IS_VOLUME_CENTERED, typename T>
    void projectBackward(const Complex<T>* proj, const T* proj_weights, size_t proj_dim,
                         Complex<T>* volume, T* volume_weights, size_t volume_dim,
                         const float2_t* proj_shifts, const float3_t* proj_scaling_factors,
                         const float33_t* proj_rotations, uint proj_count,
                         float freq_max, float ewald_sphere_radius) {

        // Oversampling & max frequency.
        auto samples = static_cast<float>(proj_dim);
        auto oversampling = static_cast<float>(volume_dim) / samples;
        auto sample_max = static_cast<uint>(math::clamp(freq_max, 0.f, 0.5f) * samples); // FIXME n/2*2 to fix odd sizes
        auto sample_max_sqd = static_cast<float>(sample_max * sample_max);

        // Prepare for the EWS curvature.
        if (ewald_sphere_radius != 0.f)
            ewald_sphere_radius = prepareEWS_(proj_dim, ewald_sphere_radius); // wavelength/2

        bool apply_shift = proj_shifts != nullptr;
        bool apply_mag = proj_scaling_factors != nullptr;

        // Direct Fourier insertion, one projection at a time.
        size_t proj_elements = (proj_dim / 2 + 1) * proj_dim;
        for (uint idx = 0; idx < proj_count; ++idx) {
            const Complex<T>* i_proj = proj + proj_elements * idx;
            const T* i_proj_weights = proj_weights + proj_elements * idx;

            // Oversample during the rotation.
            float33_t rotation(proj_rotations[idx] * float33_t(oversampling));

            float22_t distortion;
            if (apply_mag) {
                const float3_t& i_scale = proj_scaling_factors[idx];
                distortion = transform::rotate(-i_scale.z) *
                             transform::scale(1.f / float2_t(i_scale.x, i_scale.y)) *
                             transform::rotate(i_scale.z);
            }

            if (apply_shift) {
                fourierInsert_<IS_PROJ_CENTERED, IS_VOLUME_CENTERED, true>(
                        i_proj, i_proj_weights, static_cast<int>(proj_dim),
                        volume, volume_weights, static_cast<int>(volume_dim),
                        proj_shifts[idx], distortion, rotation, sample_max_sqd, ewald_sphere_radius);
            } else {
                fourierInsert_<IS_PROJ_CENTERED, IS_VOLUME_CENTERED, false>(
                        i_proj, i_proj_weights, static_cast<int>(proj_dim),
                        volume, volume_weights, static_cast<int>(volume_dim),
                        float2_t(0.f), distortion, rotation, sample_max_sqd, ewald_sphere_radius);
            }
        }
    }

    template<bool IS_PROJ_CENTERED, bool IS_VOLUME_CENTERED, typename T>
    void projectBackward(const Complex<T>* proj, const T* proj_weights, size_t proj_dim,
                         Complex<T>* volume, T* volume_weights, size_t volume_dim,
                         const float2_t* proj_shifts, float3_t proj_scaling_factor,
                         const float33_t* proj_rotations, uint proj_count,
                         float freq_max, float ewald_sphere_radius) {
        // Oversampling & max frequency.
        auto samples = static_cast<float>(proj_dim);
        auto oversampling = static_cast<float>(volume_dim) / samples;
        auto sample_max = static_cast<uint>(math::clamp(freq_max, 0.f, 0.5f) * samples);
        auto sample_max_sqd = static_cast<float>(sample_max * sample_max);

        // Prepare for the EWS curvature.
        if (ewald_sphere_radius != 0.f)
            ewald_sphere_radius = prepareEWS_(volume_dim, ewald_sphere_radius);

        // Prepare the magnification.
        float22_t distortion(transform::rotate(-proj_scaling_factor.z) *
                             transform::scale(1.f / float2_t(proj_scaling_factor.x, proj_scaling_factor.y)) *
                             transform::rotate(proj_scaling_factor.z));

        // Direct Fourier insertion, one projection at a time.
        bool apply_shift = proj_shifts != nullptr;
        size_t proj_elements = (proj_dim / 2 + 1) * proj_dim;
        for (uint idx = 0; idx < proj_count; ++idx) {
            const Complex<T>* i_proj = proj + proj_elements * idx;
            const T* i_proj_weights = proj_weights + proj_elements * idx;

            float33_t rotation(proj_rotations[idx] * float33_t(oversampling));

            if (apply_shift) {
                fourierInsert_<IS_PROJ_CENTERED, IS_VOLUME_CENTERED, true>(
                        i_proj, i_proj_weights, static_cast<int>(proj_dim),
                        volume, volume_weights, static_cast<int>(volume_dim),
                        proj_shifts[idx], distortion, rotation, sample_max_sqd, ewald_sphere_radius);
            } else {
                fourierInsert_<IS_PROJ_CENTERED, IS_VOLUME_CENTERED, false>(
                        i_proj, i_proj_weights, static_cast<int>(proj_dim),
                        volume, volume_weights, static_cast<int>(volume_dim),
                        float2_t(0.f), distortion, rotation, sample_max_sqd, ewald_sphere_radius);
            }
        }
    }

    #define NOA_INSTANTIATE_BACKWARD_(T)                                                                                    \
    template void projectBackward<true, true, T>(const Complex<T>*, const T*, size_t, Complex<T>*, T*, size_t,              \
                                                 const float2_t*, const float3_t*, const float33_t*, uint, float, float);   \
    template void projectBackward<true, false, T>(const Complex<T>*, const T*, size_t, Complex<T>*, T*, size_t,             \
                                                  const float2_t*, const float3_t*, const float33_t*, uint, float, float);  \
    template void projectBackward<false, true, T>(const Complex<T>*, const T*, size_t, Complex<T>*, T*, size_t,             \
                                                  const float2_t*, const float3_t*, const float33_t*, uint, float, float);  \
    template void projectBackward<false, false, T>(const Complex<T>*, const T*, size_t, Complex<T>*, T*, size_t,            \
                                                   const float2_t*, const float3_t*, const float33_t*, uint, float, float); \
    template void projectBackward<true, true, T>(const Complex<T>*, const T*, size_t, Complex<T>*, T*, size_t,              \
                                                 const float2_t*, float3_t, const float33_t*, uint, float, float);          \
    template void projectBackward<true, false, T>(const Complex<T>*, const T*, size_t, Complex<T>*, T*, size_t,             \
                                                  const float2_t*, float3_t,  const float33_t*, uint, float, float);        \
    template void projectBackward<false, true, T>(const Complex<T>*, const T*, size_t, Complex<T>*, T*, size_t,             \
                                                  const float2_t*, float3_t,  const float33_t*, uint, float, float);        \
    template void projectBackward<false, false, T>(const Complex<T>*, const T*, size_t, Complex<T>*, T*, size_t,            \
                                                   const float2_t*, float3_t, const float33_t*, uint, float, float)
    NOA_INSTANTIATE_BACKWARD_(float);
    NOA_INSTANTIATE_BACKWARD_(double);
}
