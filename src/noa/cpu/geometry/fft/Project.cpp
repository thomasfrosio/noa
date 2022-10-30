#include "noa/common/Assert.h"
#include "noa/common/Types.h"
#include "noa/cpu/geometry/Interpolator.h"
#include "noa/cpu/geometry/fft/Project.h"

namespace {
    using namespace ::noa;

    // The input frequency should be in-bound, i.e. -size/2 <= frequency <= (size-1)/2
    template<bool IS_CENTERED>
    inline int64_t frequency2index_(int64_t frequency, int64_t size) {
        if constexpr (IS_CENTERED) {
            return frequency + size / 2;
        } else {
            return frequency < 0 ? frequency + size : frequency;
        }
    }

    template<bool IS_CENTERED>
    inline int64_t index2frequency_(int64_t index, int64_t size) {
        if constexpr (IS_CENTERED)
            return index - size / 2;
        else
            return index < (size + 1) / 2 ? index : index - size;
    }

    template<typename T>
    inline void atomicAdd_(T& grid, T value) {
        if constexpr (traits::is_complex_v<T>) {
            #pragma omp atomic
            grid[0] += value[0];
            #pragma omp atomic
            grid[1] += value[1];
        } else {
            #pragma omp atomic
            grid += value;
        }
    }

    // The gridding kernel is a tri-linear pulse. The total weight within the 2x2x2 cube is 1.
    inline void setGriddingWeights_(long3_t base0, float3_t freq, float o_weights[2][2][2]) {
        // So if the coordinate is centered in the bottom left corner of the cube (base0),
        // i.e. its decimal is 0, the corresponding fraction for this element should be 1.
        float3_t fraction[2];
        fraction[1] = freq - float3_t(base0);
        fraction[0] = 1.f - fraction[1];
        for (dim_t w = 0; w < 2; ++w)
            for (dim_t v = 0; v < 2; ++v)
                for (dim_t u = 0; u < 2; ++u)
                    o_weights[w][v][u] = fraction[w][0] * fraction[v][1] * fraction[u][2];
    }

    // Spread the value within a 2x2x2 centered on a particular frequency using linear interpolation.
    // "frequency" is the frequency, in samples, centered on DC, with negative frequencies on the left.
    template<bool IS_DST_CENTERED, typename T>
    void addByGridding_(const AccessorRestrict<T, 3, dim_t>& grid, long3_t grid_shape, T data, float3_t frequency) {
        using real_t = traits::value_type_t<T>;
        const auto base0 = long3_t(math::floor(frequency));

        float kernel[2][2][2]; // 2x2x2 tri-linear weights
        setGriddingWeights_(base0, frequency, kernel);

        for (int64_t w = 0; w < 2; ++w) {
            for (int64_t v = 0; v < 2; ++v) {
                for (int64_t u = 0; u < 2; ++u) {
                    const int64_t idx_w = frequency2index_<IS_DST_CENTERED>(base0[0] + w, grid_shape[0]);
                    const int64_t idx_v = frequency2index_<IS_DST_CENTERED>(base0[1] + v, grid_shape[1]);
                    const int64_t idx_u = base0[2] + u;

                    if (idx_w >= 0 && idx_w < grid_shape[0] &&
                        idx_v >= 0 && idx_v < grid_shape[1] &&
                        idx_u >= 0 && idx_u < grid_shape[2]) {
                        const auto fraction = static_cast<real_t>(kernel[w][v][u]);
                        atomicAdd_(grid(idx_w, idx_v, idx_u), data * fraction);
                    }
                }
            }
        }

        // The gridding doesn't preserve the hermitian symmetry, so enforce it on the redundant X==0 ZY plane.
        // So if a side of this plane was modified, add the conjugate at (x=0, -y, -z) with the same fraction.
        if (base0[2] == 0) {
            if constexpr (traits::is_complex_v<T>)
                data.imag = -data.imag;
            for (int64_t w = 0; w < 2; ++w) {
                for (int64_t v = 0; v < 2; ++v) {
                    const int64_t idx_w = frequency2index_<IS_DST_CENTERED>(-(base0[0] + w), grid_shape[0]);
                    const int64_t idx_v = frequency2index_<IS_DST_CENTERED>(-(base0[1] + v), grid_shape[1]);

                    if (idx_w >= 0 && idx_w < grid_shape[0] &&
                        idx_v >= 0 && idx_v < grid_shape[1]) {
                        const auto fraction = static_cast<real_t>(kernel[w][v][0]);
                        atomicAdd_(grid(idx_w, idx_v), data * fraction);
                    }
                }
            }
        }
    }

    // Interpolates the value at a given frequency, in cycle per pixel.
    template<typename interpolator_t>
    inline auto interpolateSliceValue_(float2_t frequency, float2_t slice_shape, float center_y,
                                       const interpolator_t& interp, size_t slice_offset) {
        using value_t = typename interpolator_t::mutable_value_t;
        using real_t = traits::value_type_t<value_t>;
        constexpr bool IS_COMPLEX = traits::is_complex_v<value_t>;

        // From the normalized frequency to the multidimensional index.
        [[maybe_unused]] real_t conj = 1;
        if (frequency[1] < 0) {
            frequency = -frequency;
            if constexpr (IS_COMPLEX)
                conj = -1;
        }
        frequency *= slice_shape;
        frequency[0] += center_y;

        value_t value = interp.template get<INTERP_LINEAR, BORDER_ZERO>(frequency, slice_offset);
        if constexpr (IS_COMPLEX)
            value.imag *= conj;
        return value;
    }

    // Interpolates the value at a given frequency, in cycle per pixel.
    template<typename interpolator_t>
    inline auto interpolateGridValue_(float3_t frequency, float3_t target_shape, float2_t grid_center_zy,
                                      const interpolator_t& interp) {
        using value_t = typename interpolator_t::mutable_value_t;
        using real_t = traits::value_type_t<value_t>;
        constexpr bool IS_COMPLEX = traits::is_complex_v<value_t>;

        [[maybe_unused]] real_t conj = 1;
        if (frequency[2] < 0) {
            frequency = -frequency;
            if constexpr(IS_COMPLEX)
                conj = -1;
        }
        frequency *= target_shape;
        frequency[0] += grid_center_zy[0];
        frequency[1] += grid_center_zy[1];

        value_t value = interp.template get<INTERP_LINEAR, BORDER_ZERO>(frequency);
        if constexpr(IS_COMPLEX)
            value.imag *= conj;
        return value;
    }

    // Transforms a 2D normalized frequency representing the slice to a 3D normalized frequency representing the grid.
    // This is a forward transformation of the frequency, but because it is in Fourier-space, the real-space scaling
    // is inverted.
    template<bool APPLY_EWS, bool APPLY_SCALE, typename scale_t, typename rotate_t, typename index_t>
    constexpr inline float3_t
    transformSliceToGrid_(float2_t frequency,
                          const scale_t& inv_scaling_matrices,
                          const rotate_t& fwd_rotation_matrices,
                          [[maybe_unused]] index_t index,
                          [[maybe_unused]] float2_t inv_ews_diameter) {
        if constexpr (APPLY_SCALE) {
            // If we apply the EWS curvature, the scaling factors should be corrected before applying the curvature,
            // and therefore before applying the rotation. That way, we use the correct frequencies to compute the
            // EWS, e.g. resulting in a spherical EWS even under anisotropic magnification.
            if constexpr (std::is_pointer_v<scale_t>) {
                frequency = inv_scaling_matrices[index] * frequency;
            } else {
                frequency = inv_scaling_matrices * frequency;
            }
        }

        // We use the Small Angle Approximation to compute the EWS curvature, so the frequency (u,v) is unchanged.
        // TODO Look at the cisTEM implementation to remove this approximation? RELION shows that even
        //      for low voltages and large boxes, it is probably not worth it though.
        const float3_t freq_3d{APPLY_EWS ? math::sum(inv_ews_diameter * frequency * frequency) : 0,
                               frequency[0],
                               frequency[1]};

        if constexpr (std::is_pointer_v<rotate_t>)
            return fwd_rotation_matrices[index] * freq_3d;
        else
            return fwd_rotation_matrices * freq_3d;
    }

    // Same as above, but in the other direction.
    template<bool APPLY_EWS, bool APPLY_SCALE, typename scale_t, typename rotate_t, typename index_t>
    inline Pair<float, float2_t>
    transformGridToSlice_(float3_t frequency,
                          const scale_t& fwd_scaling_matrices,
                          const rotate_t& inv_rotation_matrices,
                          [[maybe_unused]] index_t index,
                          [[maybe_unused]] float2_t inv_ews_diameter) {
        if constexpr (std::is_pointer_v<rotate_t>)
            frequency = inv_rotation_matrices[index] * frequency;
        else
            frequency = inv_rotation_matrices * frequency;

        float2_t freq_2d{frequency[1], frequency[2]};
        float freq_z = frequency[0];
        if constexpr (APPLY_EWS)
            freq_z -= math::sum(inv_ews_diameter * freq_2d * freq_2d);

        if constexpr (APPLY_SCALE) {
            // Same reason as for the forward transformation. Here the grid is correct, so rotate the EWS,
            // then compute the curvature and only then we can scale the slice.
            if constexpr (std::is_pointer_v<scale_t>) {
                freq_2d = fwd_scaling_matrices[index] * freq_2d;
            } else {
                freq_2d = fwd_scaling_matrices * freq_2d;
            }
        }
        return {freq_z, freq_2d};
    }

    // Z weighting for slice thickness.
    // The total weight along a Z slab of a slice is not normalized to 1,
    // because the projected data is supposed to be normalized by the project weights.
    template<int i = 2>
    float sliceZWeight(float freq_z, float freq_z_radius) {
        // https://www.desmos.com/calculator/ulcxogyr72
        freq_z = math::abs(freq_z) / freq_z_radius;
        if constexpr (i == 1) {
            return cpu::geometry::linear1D<float>(1, 0, freq_z);
        } else if constexpr (i == 2) {
            constexpr float PI = math::Constants<float>::PI;
            return math::sinc(PI * freq_z);
        } else if constexpr (i == 3) {
            constexpr float PI_HALF = math::Constants<float>::PI / 2;
            return math::cos(PI_HALF * freq_z);
        }
    }
}

namespace {
    // Direct Fourier insertion, using data-driven interpolation:
    // It is important to call this function twice, one for the data and one for the weights.
    // The data should not be interpreted before normalizing with the weights.
    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED, bool APPLY_EWS, bool APPLY_SCALE,
             typename T, typename S, typename R>
    void fourierInsert_(AccessorRestrict<const T, 3, dim_t> slice, long3_t slice_shape,
                        AccessorRestrict<T, 3, dim_t> grid, long3_t grid_shape,
                        S inv_scaling_matrices, R fwd_rotation_matrices,
                        float cutoff, long4_t target_shape, float2_t ews_radius, dim_t threads) {
        using real_t = traits::value_type_t<T>;
        const auto l_slice_shape = long2_t(slice_shape.get(1));
        const auto f_slice_shape = float2_t(l_slice_shape / 2 * 2 + long2_t(l_slice_shape == 1));

        // Use the grid shape as backup.
        const auto l_target_shape = any(target_shape == 0) ? grid_shape : long3_t(target_shape.get(1));
        const auto f_target_shape = float3_t(l_target_shape / 2 * 2 + long3_t(l_target_shape == 1));

        // Using the small-angle approximation, Z = wavelength / 2 * (X^2 + Y^2).
        // See doi:10.1016/S0304-3991(99)00120-5 for a derivation.
        const float2_t ews_diam_inv = any(ews_radius != 0) ? 1 / (2 * ews_radius) : float2_t{};

        cutoff = math::clamp(cutoff, 0.f, 0.5f);
        cutoff *= cutoff;

        #pragma omp parallel for collapse(3) default(none) num_threads(threads)                     \
        shared(slice, slice_shape, grid, grid_shape, inv_scaling_matrices, fwd_rotation_matrices,   \
               cutoff, ews_diam_inv, f_slice_shape, f_target_shape)

        for (int64_t i = 0; i < slice_shape[0]; ++i) {
            for (int64_t y = 0; y < slice_shape[1]; ++y) {
                for (int64_t u = 0; u < slice_shape[2] / 2 + 1; ++u) { // x == u

                    // We compute the forward transformation and use normalized frequencies.
                    // The oversampling is implicitly handled when scaling back to the target shape.
                    const int64_t v = index2frequency_<IS_SRC_CENTERED>(y, slice_shape[1]);
                    const float2_t freq_2d = float2_t{v, u} / f_slice_shape;
                    float3_t freq_3d = transformSliceToGrid_<APPLY_EWS, APPLY_SCALE>(
                            freq_2d, inv_scaling_matrices, fwd_rotation_matrices, i, ews_diam_inv);

                    // The frequency rate won't change from that point, so check for the cutoff.
                    if (math::dot(freq_3d, freq_3d) > cutoff)
                        continue;

                    // Handle the non-redundancy in x.
                    [[maybe_unused]] real_t conj = 1;
                    if (freq_3d[2] < 0) {
                        freq_3d = -freq_3d;
                        if constexpr(traits::is_complex_v<T>)
                            conj = -1;
                    }

                    // Scale back to the target shape.
                    freq_3d *= f_target_shape;

                    // At this point, we know we are going to use the slice value.
                    T value = slice(i, y, u);
                    if constexpr(traits::is_complex_v<T>)
                        value.imag *= conj;

                    addByGridding_<IS_DST_CENTERED>(grid, grid_shape, value, freq_3d);
                }
            }
        }
    }

    // Direct Fourier insertion, but this time looping through the grid:
    // In practice, it allows to give an explicit "thickness" to the central slices.
    // One limitation is that it requires the slices to be centered.
    template<bool IS_DST_CENTERED, bool APPLY_EWS, bool APPLY_SCALE, typename T, typename S, typename R>
    void fourierInsertThick_(AccessorRestrict<const T, 3, dim_t> slice, long3_t slice_shape,
                             AccessorRestrict<T, 3, dim_t> grid, long3_t grid_shape,
                             S fwd_scaling_matrices, R inv_rotation_matrices,
                             float cutoff, long4_t target_shape, float2_t ews_radius,
                             float slice_z_radius, dim_t threads) {
        using real_t = traits::value_type_t<T>;
        const auto count = static_cast<size_t>(slice_shape[0]);
        const auto l_shape = long2_t(slice_shape.get(1));
        const auto f_slice_shape = float2_t(l_shape / 2 * 2 + long2_t(l_shape == 1));
        const auto l_offset = slice_shape[1] / 2; // slice Y center
        const auto f_offset = static_cast<float>(l_offset);

        // Use the grid shape as backup.
        const auto l_target_shape = any(target_shape == 0) ? grid_shape : long3_t(target_shape.get(1));
        const auto f_target_shape = float3_t(l_target_shape / 2 * 2 + long3_t(l_target_shape == 1));

        // Using the small-angle approximation, Z = wavelength / 2 * (X^2 + Y^2).
        // See doi:10.1016/S0304-3991(99)00120-5 for a derivation.
        const float2_t ews_diam_inv = any(ews_radius != 0) ? 1 / (2 * ews_radius) : float2_t{};

        cutoff = math::clamp(cutoff, 0.f, 0.5f);
        cutoff *= cutoff;

        cpu::geometry::Interpolator2D interp(slice[0], dim2_t(l_shape.fft()), T{0});
        const dim_t count_offset = slice.stride(0);

        #pragma omp parallel for collapse(3) default(none) num_threads(threads)     \
        shared(grid, grid_shape, cutoff, slice_z_radius, f_offset, interp,          \
               count, count_offset, ews_diam_inv, f_slice_shape, f_target_shape,    \
               fwd_scaling_matrices, inv_rotation_matrices)

        for (int64_t z = 0; z < grid_shape[0]; ++z) {
            for (int64_t y = 0; y < grid_shape[1]; ++y) {
                for (int64_t u = 0; u < grid_shape[2] / 2 + 1; ++u) {

                    const int64_t w = index2frequency_<IS_DST_CENTERED>(z, grid_shape[0]);
                    const int64_t v = index2frequency_<IS_DST_CENTERED>(y, grid_shape[1]);
                    const auto orig_freq = float3_t{w, v, u} / f_target_shape;
                    if (math::dot(orig_freq, orig_freq) > cutoff)
                        continue;

                    for (size_t i = 0; i < count; ++i) {
                        auto [freq_z, freq_2d] = transformGridToSlice_<APPLY_EWS, APPLY_SCALE>(
                                orig_freq, fwd_scaling_matrices, inv_rotation_matrices, i, ews_diam_inv);

                        // If voxel is not affected by the slice, skip.
                        T value{0};
                        if (freq_z <= slice_z_radius && freq_z >= -slice_z_radius) {
                            value = interpolateSliceValue_(freq_2d, f_slice_shape, f_offset, interp, i * count_offset);
                            const auto weight = sliceZWeight(freq_z, slice_z_radius);
                            value *= static_cast<real_t>(weight);
                        }

                        // The transformation preserves the hermitian symmetry, so there's nothing else to do.
                        grid(z, y, u) += value;
                    }
                }
            }
        }
    }

    // The exact same transformation as fourierInsert_ is applied here, but instead of inserting the transformed
    // slice(s) into the grid, the transformed slice(s) is extracted from the grid.
    template<bool IS_DST_CENTERED, bool APPLY_EWS, bool APPLY_SCALE, typename T, typename S, typename R>
    void fourierExtract_(AccessorRestrict<const T, 3, dim_t> grid, long3_t grid_shape,
                         AccessorRestrict<T, 3, dim_t> slice, long3_t slice_shape,
                         S inv_scaling_matrices, R fwd_rotation_matrices,
                         float cutoff, long4_t target_shape, float2_t ews_radius, dim_t threads) {
        const auto l_shape = long2_t(slice_shape.get(1));
        const auto f_slice_shape = float2_t(l_shape / 2 * 2 + long2_t(l_shape == 1));

        // Use the grid shape as backup.
        const auto target_shape_ = any(target_shape == 0) ? grid_shape : long3_t(target_shape.get(1));
        const auto f_target_shape = float3_t(target_shape_ / 2 * 2 + long3_t(target_shape_ == 1));
        const auto f_offset = float2_t(grid_shape[0] / 2, grid_shape[1] / 2); // grid ZY center

        // Using the small-angle approximation, Z = wavelength / 2 * (X^2 + Y^2).
        // See doi:10.1016/S0304-3991(99)00120-5 for a derivation.
        const float2_t ews_diam_inv = any(ews_radius != 0) ? 1 / (2 * ews_radius) : float2_t{};

        cutoff = math::clamp(cutoff, 0.f, 0.5f);
        cutoff *= cutoff;

        const cpu::geometry::Interpolator3D interp(grid, dim3_t(grid_shape).fft(), T{0});

        #pragma omp parallel for collapse(3) default(none) num_threads(threads)                     \
        shared(slice, slice_shape, grid, grid_shape, inv_scaling_matrices, fwd_rotation_matrices,   \
               cutoff, ews_diam_inv, f_slice_shape, f_target_shape, f_offset, interp)

        for (int64_t i = 0; i < slice_shape[0]; ++i) {
            for (int64_t y = 0; y < slice_shape[1]; ++y) {
                for (int64_t u = 0; u < slice_shape[2] / 2 + 1; ++u) { // x == u

                    // Transform slice into the grid.
                    const int64_t v = index2frequency_<IS_DST_CENTERED>(y, slice_shape[1]);
                    const float2_t freq_2d = float2_t{v, u} / f_slice_shape;
                    const float3_t freq_3d = transformSliceToGrid_<APPLY_EWS, APPLY_SCALE>(
                            freq_2d, inv_scaling_matrices, fwd_rotation_matrices, i, ews_diam_inv);

                    // Interpolate grid values at slice location.
                    slice(i, y, u) = math::dot(freq_3d, freq_3d) > cutoff ?
                                     T{0} :
                                     interpolateGridValue_(freq_3d, f_target_shape, f_offset, interp);
                }
            }
        }
    }

    template<bool IS_DST_CENTERED, bool APPLY_EWS, bool APPLY_SCALE,
             typename T, typename S0, typename S1, typename R0, typename R1>
    void fourierInsertExtract_(AccessorRestrict<const T, 3, dim_t> input_slices, long3_t input_shape,
                               AccessorRestrict<T, 3, dim_t> output_slices, long3_t output_shape,
                               const S0& insert_fwd_scaling_matrices, const R0& insert_inv_rotation_matrices,
                               const S1& extract_inv_scaling_matrices, const R1& extract_fwd_rotation_matrices,
                               float cutoff, float2_t ews_radius, float slice_z_radius, dim_t) {

        using real_t = traits::value_type_t<T>;
        const auto l_output_shape = long2_t(output_shape.get(1));
        const auto f_output_shape = float2_t(l_output_shape / 2 * 2 + long2_t(l_output_shape == 1));
        const auto l_input_shape = long2_t(input_shape.get(1));
        const auto f_input_shape = float2_t(l_input_shape / 2 * 2 + long2_t(l_input_shape == 1));
        const auto l_input_offset = l_input_shape[1] / 2; // slice Y center
        const auto f_input_offset = static_cast<float>(l_input_offset);
        const auto input_count = static_cast<size_t>(input_shape[0]);
        const auto input_count_offset = input_slices.stride(0);

        // Using the small-angle approximation, Z = wavelength / 2 * (X^2 + Y^2).
        // See doi:10.1016/S0304-3991(99)00120-5 for a derivation.
        const float2_t ews_diam_inv = any(ews_radius != 0) ? 1 / (2 * ews_radius) : float2_t{};

        cutoff = math::clamp(cutoff, 0.f, 0.5f);
        cutoff *= cutoff;

        const cpu::geometry::Interpolator2D interp(input_slices[0], dim2_t(l_input_shape.fft()), T{0});

        for (int64_t oi = 0; oi < output_shape[0]; ++oi) {
            for (int64_t y = 0; y < output_shape[1]; ++y) {
                for (int64_t u = 0; u < output_shape[2] / 2 + 1; ++u) { // x == u

                    // First, compute the 3D frequency of the current slice i to extract.
                    const int64_t v = index2frequency_<IS_DST_CENTERED>(y, output_shape[1]);
                    float2_t freq_2d = float2_t{v, u} / f_output_shape;
                    float3_t freq_3d = transformSliceToGrid_<APPLY_EWS, true>(
                            freq_2d, extract_inv_scaling_matrices, extract_fwd_rotation_matrices, oi, ews_diam_inv);

                    if (math::dot(freq_3d, freq_3d) > cutoff) {
                        output_slices(oi, y, u) = T{0};
                        continue;
                    }

                    // Then, insert the input slices.
                    for (size_t ii = 0; ii < input_count; ++ii) {
                        auto [freq_z, freq_2d_] = transformGridToSlice_<APPLY_EWS, APPLY_SCALE>(
                                freq_3d, insert_fwd_scaling_matrices, insert_inv_rotation_matrices, ii, ews_diam_inv);

                        // If voxel is not affected by the slice, skip.
                        T value{0};
                        if (freq_z <= slice_z_radius && freq_z >= -slice_z_radius) {
                            value = interpolateSliceValue_(freq_2d, f_input_shape, f_input_offset,interp,
                                                           ii * input_count_offset);
                            const auto weight = sliceZWeight(freq_z, slice_z_radius);
                            value *= static_cast<real_t>(weight);
                        }

                        // The transformation preserves the hermitian symmetry, so there's nothing else to do.
                        output_slices(oi, y, u) += value;
                    }
                }
            }
        }
    }

    template<bool POST_CORRECTION, typename T>
    void correctGriddingSinc2_(Accessor<const T, 4, dim_t> input,
                               Accessor<T, 4, dim_t> output,
                               dim4_t shape, dim_t threads) {
        constexpr float PI = math::Constants<float>::PI;
        const long3_t l_shape(shape.get(1));
        const float3_t f_shape(l_shape);
        const float3_t half(f_shape / 2 * float3_t(l_shape != 1)); // if size == 1, half should be 0

        #pragma omp parallel for collapse(4) num_threads(threads) default(none) \
        shared(input, output, shape, f_shape, half)

        for (dim_t i = 0; i < shape[0]; ++i) {
            for (dim_t j = 0; j < shape[1]; ++j) {
                for (dim_t k = 0; k < shape[2]; ++k) {
                    for (dim_t l = 0; l < shape[3]; ++l) {

                        float3_t dist{j, k, l};
                        dist -= half;
                        dist /= f_shape;

                        const float radius = math::sqrt(math::dot(dist, dist));
                        const float sinc = math::sinc(PI * radius);
                        const T sinc2 = static_cast<T>(sinc * sinc); // > 0.05

                        const T value = input(i, j, k, l);
                        output(i, j, k, l) = POST_CORRECTION ? value / sinc2 : value * sinc2;
                    }
                }
            }
        }
    }

    template<typename T>
    auto matrixOrRawConstPtr(T v) {
        using clean_t = traits::remove_ref_cv_t<T>;
        if constexpr (traits::is_floatXX_v<T>) {
            return clean_t(v);
        } else {
            using raw_const_ptr_t = const typename clean_t::element_type*;
            return static_cast<raw_const_ptr_t>(v.get());
        }
    }

    template<typename matrix_wrapper_t, typename matrix_value_t>
    auto inverseMatrices_(matrix_wrapper_t matrices, size_t count,
                          std::unique_ptr<matrix_value_t[]>& buffer) {
        if constexpr (traits::is_floatXX_v<matrix_wrapper_t>) {
            return math::inverse(matrices);
        } else {
            buffer = std::make_unique<matrix_value_t[]>(count);
            for (size_t i = 0; i < count; ++i)
                buffer[i] = math::inverse(matrices[i]);
            return buffer.get();
        }
    }
}

namespace noa::cpu::geometry::fft {
    template<Remap REMAP, typename T, typename S, typename R, typename>
    void insert3D(const shared_t<T[]>& slice, dim4_t slice_strides, dim4_t slice_shape,
                  const shared_t<T[]>& grid, dim4_t grid_strides, dim4_t grid_shape,
                  const S& inv_scaling_matrices, const R& fwd_rotation_matrices,
                  float cutoff, dim4_t target_shape, float2_t ews_radius, Stream& stream) {

        using Layout = ::noa::fft::Layout;
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL)
            static_assert(traits::always_false_v<T>);

        NOA_ASSERT(slice && grid && slice.get() != grid.get() &&
                   all(slice_shape > 0) && all(grid_shape > 0));
        NOA_ASSERT(slice_shape[1] == 1);
        NOA_ASSERT(grid_shape[0] == 1);

        const long3_t slice_shape_{slice_shape[0], slice_shape[2], slice_shape[3]};
        const long3_t grid_shape_{grid_shape[1], grid_shape[2], grid_shape[3]};
        const dim3_t slice_strides_{slice_strides[0], slice_strides[2], slice_strides[3]};
        const dim3_t grid_strides_{grid_strides[1], grid_strides[2], grid_strides[3]};

        const dim_t threads = stream.threads();
        stream.enqueue([=]() {
            const auto inv_scaling_matrices_ = matrixOrRawConstPtr(inv_scaling_matrices);
            const auto fwd_rotation_matrices_ = matrixOrRawConstPtr(fwd_rotation_matrices);
            const auto target_shape_ = static_cast<long4_t>(target_shape);

            const auto apply_ews = any(ews_radius != 0);
            bool apply_scale;
            if constexpr (traits::is_float22_v<S>)
                apply_scale = float22_t{} == inv_scaling_matrices;
            else
                apply_scale = inv_scaling_matrices_ != nullptr;

            if (apply_ews && apply_scale) {
                fourierInsert_<IS_SRC_CENTERED, IS_DST_CENTERED, true, true, T>(
                        {slice.get(), slice_strides_}, slice_shape_,
                        {grid.get(), grid_strides_}, grid_shape_,
                        inv_scaling_matrices_, fwd_rotation_matrices_,
                        cutoff, target_shape_, ews_radius, threads);
            } else if (apply_ews) {
                fourierInsert_<IS_SRC_CENTERED, IS_DST_CENTERED, true, false, T>(
                        {slice.get(), slice_strides_}, slice_shape_,
                        {grid.get(), grid_strides_}, grid_shape_,
                        inv_scaling_matrices_, fwd_rotation_matrices_,
                        cutoff, target_shape_, ews_radius, threads);
            } else if (apply_scale) {
                fourierInsert_<IS_SRC_CENTERED, IS_DST_CENTERED, false, true, T>(
                        {slice.get(), slice_strides_}, slice_shape_,
                        {grid.get(), grid_strides_}, grid_shape_,
                        inv_scaling_matrices_, fwd_rotation_matrices_,
                        cutoff, target_shape_, ews_radius, threads);
            } else {
                fourierInsert_<IS_SRC_CENTERED, IS_DST_CENTERED, false, false, T>(
                        {slice.get(), slice_strides_}, slice_shape_,
                        {grid.get(), grid_strides_}, grid_shape_,
                        inv_scaling_matrices_, fwd_rotation_matrices_,
                        cutoff, target_shape_, ews_radius, threads);
            }
        });
    }

    template<Remap REMAP, typename T, typename S, typename R, typename>
    void insert3D(const shared_t<T[]>& slice, dim4_t slice_strides, dim4_t slice_shape,
                  const shared_t<T[]>& grid, dim4_t grid_strides, dim4_t grid_shape,
                  const S& inv_scaling_matrices, const R& fwd_rotation_matrices,
                  float cutoff, dim4_t target_shape, float2_t ews_radius,
                  float slice_z_radius, Stream& stream) {

        using Layout = ::noa::fft::Layout;
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_NON_CENTERED || REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL)
            static_assert(traits::always_false_v<T>);

        NOA_ASSERT(slice && grid && slice.get() != grid.get() &&
                   all(slice_shape > 0) && all(grid_shape > 0));
        NOA_ASSERT(slice_shape[1] == 1);
        NOA_ASSERT(grid_shape[0] == 1);

        const size_t slice_count = slice_shape[0];
        const long3_t slice_shape_{slice_shape[0], slice_shape[2], slice_shape[3]};
        const long3_t grid_shape_{grid_shape[1], grid_shape[2], grid_shape[3]};
        const dim3_t slice_strides_{slice_strides[0], slice_strides[2], slice_strides[3]};
        const dim3_t grid_strides_{grid_strides[1], grid_strides[2], grid_strides[3]};
        const auto target_shape_ = static_cast<long4_t>(target_shape);

        const dim_t threads = stream.threads();
        stream.enqueue([=]() {
            const auto apply_ews = any(ews_radius != 0);
            bool apply_scale;
            if constexpr (traits::is_float22_v<S>)
                apply_scale = float22_t{} == inv_scaling_matrices;
            else
                apply_scale = inv_scaling_matrices != nullptr;

            constexpr bool SINGLE_SCALING = traits::is_float22_v<S>;
            std::unique_ptr<float22_t[]> fwd_scaling_matrices_buffer;
            using scaling_t = std::conditional_t<SINGLE_SCALING, float22_t, const float22_t*>;
            const scaling_t fwd_scaling_matrices = inverseMatrices_(
                    matrixOrRawConstPtr(inv_scaling_matrices), slice_count, fwd_scaling_matrices_buffer);

            constexpr bool SINGLE_ROTATION = traits::is_float33_v<R>;
            std::unique_ptr<float33_t[]> inv_rotation_matrices_buffer;
            using rotation_t = std::conditional_t<SINGLE_ROTATION, float33_t, const float33_t*>;
            const rotation_t inv_rotation_matrices = inverseMatrices_(
                    matrixOrRawConstPtr(fwd_rotation_matrices), slice_count, inv_rotation_matrices_buffer);

            if (apply_ews && apply_scale) {
                fourierInsertThick_<IS_DST_CENTERED, true, true, T>(
                        {slice.get(), slice_strides_}, slice_shape_,
                        {grid.get(), grid_strides_}, grid_shape_,
                        fwd_scaling_matrices, inv_rotation_matrices,
                        cutoff, target_shape_, ews_radius, slice_z_radius, threads);
            } else if (apply_ews) {
                fourierInsertThick_<IS_DST_CENTERED, true, false, T>(
                        {slice.get(), slice_strides_}, slice_shape_,
                        {grid.get(), grid_strides_}, grid_shape_,
                        fwd_scaling_matrices, inv_rotation_matrices,
                        cutoff, target_shape_, ews_radius, slice_z_radius, threads);
            } else if (apply_scale) {
                fourierInsertThick_<IS_DST_CENTERED, false, true, T>(
                        {slice.get(), slice_strides_}, slice_shape_,
                        {grid.get(), grid_strides_}, grid_shape_,
                        fwd_scaling_matrices, inv_rotation_matrices,
                        cutoff, target_shape_, ews_radius, slice_z_radius, threads);
            } else {
                fourierInsertThick_<IS_DST_CENTERED, false, false, T>(
                        {slice.get(), slice_strides_}, slice_shape_,
                        {grid.get(), grid_strides_}, grid_shape_,
                        fwd_scaling_matrices, inv_rotation_matrices,
                        cutoff, target_shape_, ews_radius, slice_z_radius, threads);
            }
        });
    }

    template<Remap REMAP, typename T, typename S, typename R, typename>
    void extract3D(const shared_t<T[]>& grid, dim4_t grid_strides, dim4_t grid_shape,
                   const shared_t<T[]>& slice, dim4_t slice_strides, dim4_t slice_shape,
                   const S& inv_scaling_matrices, const R& fwd_rotation_matrices,
                   float cutoff, dim4_t target_shape, float2_t ews_radius, Stream& stream) {

        using Layout = ::noa::fft::Layout;
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_NON_CENTERED ||
                      REMAP_ & Layout::SRC_FULL ||
                      REMAP_ & Layout::DST_FULL)
            static_assert(traits::always_false_v<T>);

        NOA_ASSERT(slice && grid && slice.get() != grid.get() &&
                   all(slice_shape > 0) && all(grid_shape > 0));
        NOA_ASSERT(slice_shape[1] == 1);
        NOA_ASSERT(grid_shape[0] == 1);

        const long3_t slice_shape_{slice_shape[0], slice_shape[2], slice_shape[3]};
        const long3_t grid_shape_{grid_shape[1], grid_shape[2], grid_shape[3]};
        const dim3_t slice_strides_{slice_strides[0], slice_strides[2], slice_strides[3]};
        const dim3_t grid_strides_{grid_strides[1], grid_strides[2], grid_strides[3]};

        const dim_t threads = stream.threads();
        stream.enqueue([=]() {
            const auto inv_scaling_matrices_ = matrixOrRawConstPtr(inv_scaling_matrices);
            const auto fwd_rotation_matrices_ = matrixOrRawConstPtr(fwd_rotation_matrices);
            const auto target_shape_ = static_cast<long4_t>(target_shape);

            const auto apply_ews = any(ews_radius != 0);
            bool apply_scale;
            if constexpr (traits::is_float22_v<S>)
                apply_scale = float22_t{} == inv_scaling_matrices;
            else
                apply_scale = inv_scaling_matrices_ != nullptr;

            if (apply_ews && apply_scale) {
                fourierExtract_<IS_DST_CENTERED, true, true, T>(
                        {grid.get(), grid_strides_}, grid_shape_,
                        {slice.get(), slice_strides_}, slice_shape_,
                        inv_scaling_matrices_, fwd_rotation_matrices_,
                        cutoff, target_shape_, ews_radius, threads);
            } else if (apply_ews) {
                fourierExtract_<IS_DST_CENTERED, true, false, T>(
                        {grid.get(), grid_strides_}, grid_shape_,
                        {slice.get(), slice_strides_}, slice_shape_,
                        inv_scaling_matrices_, fwd_rotation_matrices_,
                        cutoff, target_shape_, ews_radius, threads);
            } else if (apply_scale) {
                fourierExtract_<IS_DST_CENTERED, false, true, T>(
                        {grid.get(), grid_strides_}, grid_shape_,
                        {slice.get(), slice_strides_}, slice_shape_,
                        inv_scaling_matrices_, fwd_rotation_matrices_,
                        cutoff, target_shape_, ews_radius, threads);
            } else {
                fourierExtract_<IS_DST_CENTERED, false, false, T>(
                        {grid.get(), grid_strides_}, grid_shape_,
                        {slice.get(), slice_strides_}, slice_shape_,
                        inv_scaling_matrices_, fwd_rotation_matrices_,
                        cutoff, target_shape_, ews_radius, threads);
            }
        });
    }

    template<Remap REMAP, typename T, typename S0, typename S1, typename R0, typename R1, typename>
    void extract3D(const shared_t<T[]>& input_slices, dim4_t input_slices_strides, dim4_t input_slices_shape,
                   const shared_t<T[]>& output_slices, dim4_t output_slices_strides, dim4_t output_slices_shape,
                   const S0& insert_inv_scaling_matrices, const R0& insert_fwd_rotation_matrices,
                   const S1& extract_inv_scaling_matrices, const R1& extract_fwd_rotation_matrices,
                   float cutoff, float2_t ews_radius, float slice_z_radius, Stream& stream) {

        using Layout = ::noa::fft::Layout;
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_NON_CENTERED || REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL)
            static_assert(traits::always_false_v<T>);

        NOA_ASSERT(input_slices && output_slices && input_slices.get() != output_slices.get() &&
                   all(input_slices_shape > 0) && all(output_slices_shape > 0));
        NOA_ASSERT(input_slices_shape[1] == 1 && output_slices_shape[1] == 1);

        const size_t insert_slice_count = input_slices_shape[0];
        const long3_t input_slice_shape_{input_slices_shape[0], input_slices_shape[2], input_slices_shape[3]};
        const long3_t output_slice_shape_{output_slices_shape[0], output_slices_shape[2], output_slices_shape[3]};
        const dim3_t input_slices_strides_{input_slices_strides[0], input_slices_strides[2], input_slices_strides[3]};
        const dim3_t output_slices_strides_{output_slices_strides[0], output_slices_strides[2], output_slices_strides[3]};

        const dim_t threads = stream.threads();
        stream.enqueue([=]() {
            const auto apply_ews = any(ews_radius != 0);
            bool apply_scale;
            if constexpr (traits::is_float22_v<S0>)
                apply_scale = float22_t{} == insert_inv_scaling_matrices;
            else
                apply_scale = insert_inv_scaling_matrices != nullptr;

            // The transformation for the insertion needs to be inverted.
            constexpr bool SINGLE_SCALING = traits::is_float22_v<S0>;
            std::unique_ptr<float22_t[]> insert_fwd_scaling_matrices_buffer;
            using scaling_t = std::conditional_t<SINGLE_SCALING, float22_t, const float22_t*>;
            const scaling_t insert_fwd_scaling_matrices = inverseMatrices_(
                    matrixOrRawConstPtr(insert_inv_scaling_matrices),
                    insert_slice_count, insert_fwd_scaling_matrices_buffer);

            constexpr bool SINGLE_ROTATION = traits::is_float33_v<R0>;
            std::unique_ptr<float33_t[]> insert_inv_rotation_matrices_buffer;
            using rotation_t = std::conditional_t<SINGLE_ROTATION, float33_t, const float33_t*>;
            const rotation_t insert_inv_rotation_matrices = inverseMatrices_(
                    matrixOrRawConstPtr(insert_fwd_rotation_matrices),
                    insert_slice_count, insert_inv_rotation_matrices_buffer);

            auto extract_inv_scaling_matrices_ = matrixOrRawConstPtr(extract_inv_scaling_matrices);
            auto extract_fwd_rotation_matrices_ = matrixOrRawConstPtr(extract_fwd_rotation_matrices);

            if (apply_ews && apply_scale) {
                fourierInsertExtract_<IS_DST_CENTERED, true, true, T>(
                        {input_slices.get(), input_slices_strides_}, input_slice_shape_,
                        {output_slices.get(), output_slices_strides_}, output_slice_shape_,
                        insert_fwd_scaling_matrices, insert_inv_rotation_matrices,
                        extract_inv_scaling_matrices_, extract_fwd_rotation_matrices_,
                        cutoff, ews_radius, slice_z_radius, threads);
            } else if (apply_ews) {
                fourierInsertExtract_<IS_DST_CENTERED, true, false, T>(
                        {input_slices.get(), input_slices_strides_}, input_slice_shape_,
                        {output_slices.get(), output_slices_strides_}, output_slice_shape_,
                        insert_fwd_scaling_matrices, insert_inv_rotation_matrices,
                        extract_inv_scaling_matrices_, extract_fwd_rotation_matrices_,
                        cutoff, ews_radius, slice_z_radius, threads);
            } else if (apply_scale) {
                fourierInsertExtract_<IS_DST_CENTERED, false, true, T>(
                        {input_slices.get(), input_slices_strides_}, input_slice_shape_,
                        {output_slices.get(), output_slices_strides_}, output_slice_shape_,
                        insert_fwd_scaling_matrices, insert_inv_rotation_matrices,
                        extract_inv_scaling_matrices_, extract_fwd_rotation_matrices_,
                        cutoff, ews_radius, slice_z_radius, threads);
            } else {
                fourierInsertExtract_<IS_DST_CENTERED, false, false, T>(
                        {input_slices.get(), input_slices_strides_}, input_slice_shape_,
                        {output_slices.get(), output_slices_strides_}, output_slice_shape_,
                        insert_fwd_scaling_matrices, insert_inv_rotation_matrices,
                        extract_inv_scaling_matrices_, extract_fwd_rotation_matrices_,
                        cutoff, ews_radius, slice_z_radius, threads);
            }
        });
    }

    template<typename T, typename>
    void griddingCorrection(const shared_t<T[]>& input, dim4_t input_strides,
                            const shared_t<T[]>& output, dim4_t output_strides,
                            dim4_t shape, bool post_correction, Stream& stream) {
        NOA_ASSERT(input && input && all(shape > 0));

        const dim_t threads = stream.threads();
        stream.enqueue([=]() {
            if (post_correction) {
                correctGriddingSinc2_<true, T>(
                        {input.get(), input_strides},
                        {output.get(), output_strides},
                        shape, threads);
            } else {
                correctGriddingSinc2_<false, T>(
                        {input.get(), input_strides},
                        {output.get(), output_strides},
                        shape, threads);
            }
        });
    }

    #define NOA_INSTANTIATE_INSERT_(T, REMAP, S, R) \
    template void insert3D<REMAP, T, S, R, void>(   \
        const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const S&, const R&, float, dim4_t, float2_t, Stream&)

    #define NOA_INSTANTIATE_INSERT_THICK_(T, REMAP, S, R) \
    template void insert3D<REMAP, T, S, R, void>(         \
        const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const S&, const R&, float, dim4_t, float2_t, float, Stream&)

    #define NOA_INSTANTIATE_EXTRACT_(T, REMAP, S, R)    \
    template void extract3D<REMAP, T, S, R, void>(      \
        const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const S&, const R&, float, dim4_t, float2_t, Stream&)

    #define NOA_INSTANTIATE_INSERT_EXTRACT_(T, REMAP, S0, S1, R0, R1)   \
    template void extract3D<REMAP, T, S0, S1, R0, R1, void>(            \
        const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const S0&, const R0&, const S1&, const R1&, float, float2_t, float, Stream&)

    #define NOA_INSTANTIATE_PROJECT_ALL_REMAP(T, S, R)      \
    NOA_INSTANTIATE_INSERT_(T, Remap::H2H, S, R);           \
    NOA_INSTANTIATE_INSERT_(T, Remap::H2HC, S, R);          \
    NOA_INSTANTIATE_INSERT_(T, Remap::HC2H, S, R);          \
    NOA_INSTANTIATE_INSERT_(T, Remap::HC2HC, S, R);         \
    NOA_INSTANTIATE_INSERT_THICK_(T, Remap::HC2H, S, R);    \
    NOA_INSTANTIATE_INSERT_THICK_(T, Remap::HC2HC, S, R);   \
    NOA_INSTANTIATE_EXTRACT_(T, Remap::HC2H, S, R);         \
    NOA_INSTANTIATE_EXTRACT_(T, Remap::HC2HC, S, R)

    #define NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP(T, S0, S1, R0, R1)  \
    NOA_INSTANTIATE_INSERT_EXTRACT_(T, Remap::HC2H, S0, S1, R0, R1);    \
    NOA_INSTANTIATE_INSERT_EXTRACT_(T, Remap::HC2HC, S0, S1, R0, R1)

    #define NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE(T, R0, R1)                              \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP(T, float22_t, float22_t, R0, R1);               \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP(T, shared_t<float22_t[]>, float22_t, R0, R1);   \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP(T, float22_t, shared_t<float22_t[]>, R0, R1);   \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP(T, shared_t<float22_t[]>, shared_t<float22_t[]>, R0, R1)

    #define NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE(T)                             \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE(T, float33_t, float33_t);               \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE(T, shared_t<float33_t[]>, float33_t);   \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE(T, float33_t, shared_t<float33_t[]>);   \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE(T, shared_t<float33_t[]>, shared_t<float33_t[]>)

    #define NOA_INSTANTIATE_PROJECT_ALL_(T)                                             \
    NOA_INSTANTIATE_PROJECT_ALL_REMAP(T, float22_t, float33_t);                         \
    NOA_INSTANTIATE_PROJECT_ALL_REMAP(T, shared_t<float22_t[]>, float33_t);             \
    NOA_INSTANTIATE_PROJECT_ALL_REMAP(T, float22_t, shared_t<float33_t[]>);             \
    NOA_INSTANTIATE_PROJECT_ALL_REMAP(T, shared_t<float22_t[]>, shared_t<float33_t[]>); \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE(T);                                        \
    template void griddingCorrection<T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, bool, Stream&)

    NOA_INSTANTIATE_PROJECT_ALL_(float);
    NOA_INSTANTIATE_PROJECT_ALL_(double);
    NOA_INSTANTIATE_PROJECT_ALL_(cfloat_t);
    NOA_INSTANTIATE_PROJECT_ALL_(cdouble_t);
}
