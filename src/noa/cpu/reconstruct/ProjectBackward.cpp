#include "noa/common/transform/Euler.h"
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

    template<bool IS_VOLUME_CENTERED>
    inline int toIndex_(int frequency, int volume_dim) {
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
    void addByInterpolation_(Complex<T>* v1, Complex<T> a1, T* v2, T a2,
                             float3_t freq, int3_t base, int volume_dim, int volume_pitch) {

        // So if the coordinate is centered on the left point (the base), i.e. its decimal is 0,
        // the corresponding fraction for this element should be 1.
        float weights[2][2][2];
        {
            float3_t fraction1(freq - float3_t(base));
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

        for (int z = 0; z < 2; ++z) {
            int idx_z = toIndex_<IS_VOLUME_CENTERED>(base.z + z, volume_dim);
            for (int y = 0; y < 2; ++y) {
                int idx_y = toIndex_<IS_VOLUME_CENTERED>(base.y + y, volume_dim);
                for (int x = 0; x < 2; ++x) {
                    T fraction = static_cast<T>(weights[z][y][x]);
                    int offset = (idx_z * volume_dim + idx_y) * volume_pitch + base.x + x;
                    if (v1)
                        v1[offset] += a1 * fraction;
                    if (v2)
                        v2[offset] += a2 * fraction;
                }
            }
        }

        // Redundant plane: the x==0 plane (excluding the DC) is redundant, so to preserve the hermitian symmetry,
        // we simply need to add the conjugate at (x=0, -y, -z) with the same fraction.
        if (base.x == 0 && (base.y != 0 || base.z != 0)) {
            a1.imag = -a1.imag;
            for (int z = 0; z < 2; ++z) {
                int idx_z = toIndex_<IS_VOLUME_CENTERED>(-(base.z + z), volume_dim);
                for (int y = 0; y < 2; ++y) {
                    int idx_y = toIndex_<IS_VOLUME_CENTERED>(-(base.y + y), volume_dim);

                    T fraction = static_cast<T>(weights[z][y][0]);
                    int offset = (idx_z * volume_dim + idx_y) * volume_pitch;
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
            if constexpr(IS_PROJ_CENTERED) // what's the frequency v saved at index y
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
                    proj_value = proj[y * proj_pitch + u];
                    proj_value.imag *= conj;

                    if constexpr(PHASE_SHIFT) {
                        // Phase shift value to apply the desired real space shift.
                        // Use the untransformed frequencies.
                        float2_t orig_freq(u, v);
                        proj_value *= getPhaseShift_(shift, orig_freq);
                    }
                }
                T proj_weight = volume_weights ? proj_weights[y * proj_pitch + u] : 0;

                addByInterpolation_<IS_VOLUME_CENTERED>(volume, proj_value, volume_weights, proj_weight,
                                                        freq_3d, base, volume_dim, volume_pitch);
            }
        }
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

namespace noa::cpu::reconstruct {
    template<bool IS_PROJ_CENTERED, bool IS_VOLUME_CENTERED, typename T>
    void projectBackward(const Complex<T>* proj, const T* proj_weights, size_t proj_dim,
                         Complex<T>* volume, T* volume_weights, size_t volume_dim,
                         const float2_t* proj_shifts, const float3_t* proj_magnifications,
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

        bool apply_shift = proj_shifts != nullptr;
        bool apply_mag = proj_magnifications != nullptr;

        // Direct Fourier insertion, one projection at a time.
        size_t proj_elements = (proj_dim / 2 + 1) * proj_dim;
        for (uint idx = 0; idx < proj_count; ++idx) {
            const Complex<T>* i_proj = proj + proj_elements * idx;
            const T* i_proj_weights = proj_weights + proj_elements * idx;

            // TODO Why do we need the invert matrices? We go from the projection to the volume...
            float33_t rotation(proj_rotations[idx] * float33_t(oversampling)); // scale for sampling during rotation

            // TODO Add CTF magnification struct somewhere and add corresponding overload for transform::scale.
            float22_t distortion(1);
            if (apply_mag) {
                const float3_t& i_mag = proj_magnifications[idx]; // z is angle
                distortion = transform::rotate(-i_mag.z) *
                             transform::scale(float2_t(i_mag.x, i_mag.y)) *
                             transform::rotate(i_mag.z);
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
                         const float2_t* proj_shifts, float3_t proj_magnification,
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
        float22_t distortion(transform::rotate(-proj_magnification.z) *
                             transform::scale(float2_t(proj_magnification.x, proj_magnification.y)) *
                             transform::rotate(proj_magnification.z));

        // Direct Fourier insertion, one projection at a time.
        bool apply_shift = proj_shifts != nullptr;
        size_t proj_elements = (proj_dim / 2 + 1) * proj_dim;
        for (uint idx = 0; idx < proj_count; ++idx) {
            const Complex<T>* i_proj = proj + proj_elements * idx;
            const T* i_proj_weights = proj_weights + proj_elements * idx;

            float33_t rotation(proj_rotations[idx] * float33_t(oversampling)); // scale for sampling during rotation

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
