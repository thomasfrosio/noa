#include "noa/common/Types.h"
#include "noa/common/geometry/Interpolate.h"

// Implementation for backward and forward projections using Fourier insertion and extraction.
// Can be called from the CPU (serial/OpenMP) or CUDA backend.

namespace noa::geometry::fft::details {
    using Remap = ::noa::fft::Remap;

    // The input frequency should be in-bound, i.e. -size/2 <= frequency <= (size-1)/2
    template<bool IS_CENTERED, typename index_t>
    [[nodiscard]] NOA_FHD index_t frequency2index(index_t frequency, index_t size) {
        if constexpr (IS_CENTERED)
            return frequency + size / 2;
        else
            return frequency < 0 ? frequency + size : frequency;
        return index_t{}; // unreachable
    }

    template<bool IS_CENTERED, typename index_t>
    [[nodiscard]] NOA_FHD index_t index2frequency(index_t index, index_t size) {
        if constexpr (IS_CENTERED)
            return index - size / 2;
        else
            return index < (size + 1) / 2 ? index : index - size;
        return index_t{}; // unreachable
    }

    // TODO Move atomic to separate common header!
    template<typename grid_accessor_t, typename index_t, typename data_t>
    NOA_FHD void atomicAdd(const grid_accessor_t& grid,
                           index_t idx_w, index_t idx_v, index_t idx_u,
                           data_t value) {
        #if defined(__CUDA_ARCH__)
        ::noa::cuda::utils::atomic::add(grid.offsetPointer(grid.get(), idx_w, idx_v, idx_u), value);
        #else
        data_t& grid_value = grid(idx_w, idx_v, idx_u);
        if constexpr (traits::is_complex_v<data_t>) {
            #pragma omp atomic
            grid_value[0] += value[0];
            #pragma omp atomic
            grid_value[1] += value[1];
        } else {
            #pragma omp atomic
            grid_value += value;
        }
        #endif
    }

    // The gridding kernel is a tri-linear pulse. The total weight within the 2x2x2 cube is 1.
    template<typename index_t>
    NOA_IHD void setGriddingWeights(Int3<index_t> base0, float3_t freq, float o_weights[2][2][2]) {
        // So if the coordinate is centered in the bottom left corner of the cube (base0),
        // i.e. its decimal is 0, the corresponding fraction for this element should be 1.
        float3_t fraction[2];
        fraction[1] = freq - float3_t(base0);
        fraction[0] = 1.f - fraction[1];
        for (index_t w = 0; w < 2; ++w)
            for (index_t v = 0; v < 2; ++v)
                for (index_t u = 0; u < 2; ++u)
                    o_weights[w][v][u] = fraction[w][0] * fraction[v][1] * fraction[u][2];
    }

    // Spread the value within a 2x2x2 centered on a particular frequency using linear interpolation.
    // "frequency" is the frequency, in samples, centered on DC, with negative frequencies on the left.
    template<bool IS_GRID_CENTERED, typename grid_accessor_t, typename index_t, typename data_t>
    NOA_IHD void addByGridding(const grid_accessor_t& grid, Int3<index_t> grid_shape,
                               data_t data, float3_t frequency) {
        using real_t = traits::value_type_t<data_t>;
        const auto base0 = Int3<index_t>(math::floor(frequency));

        float kernel[2][2][2]; // 2x2x2 tri-linear weights
        setGriddingWeights(base0, frequency, kernel);

        for (index_t w = 0; w < 2; ++w) {
            for (index_t v = 0; v < 2; ++v) {
                for (index_t u = 0; u < 2; ++u) {
                    const index_t idx_w = frequency2index<IS_GRID_CENTERED>(base0[0] + w, grid_shape[0]);
                    const index_t idx_v = frequency2index<IS_GRID_CENTERED>(base0[1] + v, grid_shape[1]);
                    const index_t idx_u = base0[2] + u;

                    if (idx_w >= 0 && idx_w < grid_shape[0] &&
                        idx_v >= 0 && idx_v < grid_shape[1] &&
                        idx_u >= 0 && idx_u < grid_shape[2]) {
                        const auto fraction = static_cast<real_t>(kernel[w][v][u]);
                        noa::geometry::fft::details::atomicAdd(grid, idx_w, idx_v, idx_u, data * fraction);
                    }
                }
            }
        }

        // The gridding doesn't preserve the hermitian symmetry, so enforce it on the redundant X==0 ZY plane.
        // So if a side of this plane was modified, add the conjugate at (x=0, -y, -z) with the same fraction.
        if (base0[2] == 0) {
            if constexpr (traits::is_complex_v<data_t>)
                data.imag = -data.imag;
            for (index_t w = 0; w < 2; ++w) {
                for (index_t v = 0; v < 2; ++v) {
                    const index_t idx_w = frequency2index<IS_GRID_CENTERED>(-(base0[0] + w), grid_shape[0]);
                    const index_t idx_v = frequency2index<IS_GRID_CENTERED>(-(base0[1] + v), grid_shape[1]);

                    if (idx_w >= 0 && idx_w < grid_shape[0] &&
                        idx_v >= 0 && idx_v < grid_shape[1]) {
                        const auto fraction = static_cast<real_t>(kernel[w][v][0]);
                        noa::geometry::fft::details::atomicAdd(grid, idx_w, idx_v, index_t{0}, data * fraction);
                    }
                }
            }
        }
    }

    // Transforms a 2D normalized frequency representing the slice to a 3D normalized
    // frequency representing the grid. This is a forward transformation of the frequency,
    // but because it is in Fourier-space, the real-space scaling is inverted.
    template<typename scale_or_empty_t, typename rotate_t,
             typename index_t, typename ews_or_empty_t>
    constexpr NOA_IHD float3_t
    transformSliceToGrid(float2_t frequency,
                         const scale_or_empty_t& inv_scaling_matrices,
                         const rotate_t& fwd_rotation_matrices,
                         index_t index,
                         ews_or_empty_t inv_ews_diameter) {
        // If we apply the EWS curvature, the scaling factors should be corrected
        // before applying the curvature, and therefore before applying the rotation.
        // That way, we use the correct frequencies to compute the EWS, e.g. resulting
        // in a spherical EWS even under anisotropic magnification.
        if constexpr (std::is_pointer_v<scale_or_empty_t>)
            frequency = inv_scaling_matrices[index] * frequency;
        else if constexpr (!std::is_empty_v<scale_or_empty_t>)
            frequency = inv_scaling_matrices * frequency;

        // TODO We use the Small Angle Approximation to compute the EWS curvature,
        //      so the frequency (u,v) is unchanged. Look at the cisTEM implementation
        //      to remove this approximation? RELION shows that even for low voltages
        //      and large boxes, it is probably not worth it though.
        float3_t freq_3d{0, frequency[0], frequency[1]};
        if constexpr (!std::is_empty_v<ews_or_empty_t>)
            freq_3d[0] = math::sum(inv_ews_diameter * frequency * frequency);

        if constexpr (std::is_pointer_v<rotate_t>)
            return fwd_rotation_matrices[index] * freq_3d;
        else
            return fwd_rotation_matrices * freq_3d;
    }

    // Same as above, but in the other direction.
    template<typename scale_or_empty_t, typename rotate_t,
             typename index_t, typename ews_or_empty_t>
    constexpr NOA_IHD Pair<float, float2_t>
    transformGridToSlice(float3_t frequency,
                         const scale_or_empty_t& fwd_scaling_matrices,
                         const rotate_t& inv_rotation_matrices,
                         index_t index,
                         ews_or_empty_t inv_ews_diameter) {
        if constexpr (std::is_pointer_v<rotate_t>)
            frequency = inv_rotation_matrices[index] * frequency;
        else
            frequency = inv_rotation_matrices * frequency;

        float2_t freq_2d{frequency[1], frequency[2]};
        float freq_z = frequency[0];
        if constexpr (!std::is_empty_v<ews_or_empty_t>)
            freq_z -= math::sum(inv_ews_diameter * freq_2d * freq_2d);

        // Same reason as for the forward transformation.
        // Here the grid is correct, so rotate the EWS, then compute
        // the curvature and only then we can scale the slice.
        if constexpr (std::is_pointer_v<scale_or_empty_t>)
            freq_2d = fwd_scaling_matrices[index] * freq_2d;
        else if constexpr (!std::is_empty_v<scale_or_empty_t>)
            freq_2d = fwd_scaling_matrices * freq_2d;

        return {freq_z, freq_2d};
    }

    // Interpolates the value at a given frequency, in cycle per pixel.
    template<typename interpolator_t, typename index_t>
    constexpr NOA_HD auto interpolateSliceValue(float2_t frequency, float2_t slice_shape, float center_y,
                                                const interpolator_t& interpolator, index_t batch) {
        using data_t = typename interpolator_t::data_t;
        using real_t = traits::value_type_t<data_t>;
        constexpr bool IS_COMPLEX = traits::is_complex_v<data_t>;

        // From the normalized frequency to the multidimensional index.
        real_t conj = 1;
        if (frequency[1] < 0) {
            frequency = -frequency;
            if constexpr (IS_COMPLEX)
                conj = -1;
        }
        frequency *= slice_shape;
        frequency[0] += center_y;

        data_t value = interpolator(frequency, batch);
        if constexpr (IS_COMPLEX)
            value.imag *= conj;
        else
            (void) conj;
        return value;
    }

    // Interpolates the value at a given frequency, in cycle per pixel.
    template<typename interpolator_t>
    constexpr NOA_HD auto interpolateGridValue(float3_t frequency, float3_t target_shape,
                                               float2_t grid_center_zy,
                                               const interpolator_t& interpolator) {
        using data_t = typename interpolator_t::data_t;
        using real_t = traits::value_type_t<data_t>;
        constexpr bool IS_COMPLEX = traits::is_complex_v<data_t>;

        real_t conj = 1;
        if (frequency[2] < 0) {
            frequency = -frequency;
            if constexpr(IS_COMPLEX)
                conj = -1;
        }
        frequency *= target_shape;
        frequency[0] += grid_center_zy[0];
        frequency[1] += grid_center_zy[1];

        data_t value = interpolator(frequency);
        if constexpr(IS_COMPLEX)
            value.imag *= conj;
        else
            (void) conj;
        return value;
    }

    // Z weighting for slice thickness.
    // The total weight along a Z slab of a slice is not normalized to 1,
    // because the projected data is supposed to be normalized by the project weights.
    template<int i = 2>
    NOA_FHD float sliceZWeight(float freq_z, float freq_z_radius) {
        // https://www.desmos.com/calculator/ulcxogyr72
        freq_z = math::abs(freq_z) / freq_z_radius;
        if constexpr (i == 1) {
            return geometry::interpolate::lerp1D<float>(1, 0, freq_z);
        } else if constexpr (i == 2) {
            constexpr float PI = math::Constants<float>::PI;
            return math::sinc(PI * freq_z);
        } else if constexpr (i == 3) {
            constexpr float PI_HALF = math::Constants<float>::PI / 2;
            return math::cos(PI_HALF * freq_z);
        }
    }
}

namespace noa::geometry::fft::details {
    // Direct Fourier insertion, using data-driven interpolation (gridding).
    template<Remap REMAP,
             typename index_t, typename data_t, typename offset_t,
             typename scale_or_empty_t, typename rotate_t, typename ews_or_empty_t>
    class FourierInsertionByGridding {
        using Layout = ::noa::fft::Layout;
        static constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        static constexpr bool IS_SLICE_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        static constexpr bool IS_GRID_CENTERED = REMAP_ & Layout::DST_CENTERED;
        static_assert(!(REMAP_ & Layout::SRC_FULL) && !(REMAP_ & Layout::DST_FULL));

        static_assert(traits::is_any_v<data_t, float, double, cfloat_t, cdouble_t>);
        static_assert(traits::is_any_v<scale_or_empty_t, float22_t, const float22_t*> || std::is_empty_v<scale_or_empty_t>);
        static_assert(traits::is_any_v<ews_or_empty_t, float, float2_t> || std::is_empty_v<ews_or_empty_t>);
        static_assert(traits::is_any_v<rotate_t, float33_t, const float33_t*>);
        static_assert(traits::is_any_v<index_t, int32_t, int64_t>);

        using slice_accessor_t = AccessorRestrict<const data_t, 3, offset_t>;
        using grid_accessor_t = AccessorRestrict<data_t, 3, offset_t>;
        using index2_t = Int2<index_t>;
        using index3_t = Int3<index_t>;
        using real_t = traits::value_type_t<data_t>;

    public:
        FourierInsertionByGridding(slice_accessor_t slice, dim4_t slice_shape,
                                   grid_accessor_t grid, dim4_t grid_shape,
                                   scale_or_empty_t inv_scaling_matrices, rotate_t fwd_rotation_matrices,
                                   float cutoff, dim4_t target_shape, ews_or_empty_t ews_radius)
                : m_slice(slice),
                  m_grid(grid),
                  m_fwd_rotation_matrices(fwd_rotation_matrices),
                  m_inv_scaling_matrices(inv_scaling_matrices) {

            NOA_ASSERT(all(slice_shape > 0) && all(grid_shape > 0));
            NOA_ASSERT(slice_shape[1] == 1);
            NOA_ASSERT(grid_shape[0] == 1);

            const auto slice_shape_2d = safe_cast<index2_t>(dim2_t(slice_shape.get(2)));
            m_slice_size_y = slice_shape_2d[0];
            m_grid_shape = safe_cast<index3_t>(dim3_t(grid_shape.get(1)));
            m_f_slice_shape = float2_t(slice_shape_2d / 2 * 2 + index2_t(slice_shape_2d == 1));

            // Use the grid shape as backup.
            const auto target_shape_3d = any(target_shape == 0) ?
                    m_grid_shape : safe_cast<index3_t>(dim3_t(target_shape.get(1)));
            m_f_target_shape = float3_t(target_shape_3d / 2 * 2 + index3_t(target_shape_3d == 1));

            // Using the small-angle approximation, Z = wavelength / 2 * (X^2 + Y^2).
            // See doi:10.1016/S0304-3991(99)00120-5 for a derivation.
            if constexpr (!std::is_empty_v<ews_or_empty_t>)
                m_ews_diam_inv = any(ews_radius != 0) ? 1 / (2 * ews_radius) : ews_or_empty_t{};

            m_cutoff = math::clamp(cutoff, 0.f, 0.5f);
            m_cutoff *= m_cutoff;
        }

        // For every pixel of every central slice to insert.
        NOA_HD void operator()(index_t i, index_t y, index_t u) const noexcept { // x == u
            // We compute the forward transformation and use normalized frequencies.
            // The oversampling is implicitly handled when scaling back to the target shape.
            const index_t v = index2frequency<IS_SLICE_CENTERED>(y, m_slice_size_y);
            const float2_t freq_2d = float2_t{v, u} / m_f_slice_shape;
            float3_t freq_3d = transformSliceToGrid(
                    freq_2d, m_inv_scaling_matrices, m_fwd_rotation_matrices, i, m_ews_diam_inv);

            // The frequency rate won't change from that point, so check for the cutoff.
            if (math::dot(freq_3d, freq_3d) > m_cutoff)
                return;

            // Handle the non-redundancy in x.
            real_t conj = 1;
            if (freq_3d[2] < 0) {
                freq_3d = -freq_3d;
                if constexpr(traits::is_complex_v<data_t>)
                    conj = -1;
            }

            // Scale back to the target shape.
            freq_3d *= m_f_target_shape;

            // At this point, we know we are going to use the slice value.
            data_t value = m_slice(i, y, u);
            if constexpr(traits::is_complex_v<data_t>)
                value.imag *= conj;
            else
                (void) conj;

            addByGridding<IS_GRID_CENTERED>(m_grid, m_grid_shape, value, freq_3d);
        }

    private:
        slice_accessor_t m_slice;
        grid_accessor_t m_grid;
        rotate_t m_fwd_rotation_matrices;
        scale_or_empty_t m_inv_scaling_matrices;

        index3_t m_grid_shape;
        index_t m_slice_size_y;
        float3_t m_f_target_shape;
        float2_t m_f_slice_shape;
        ews_or_empty_t m_ews_diam_inv{};
        float m_cutoff;
    };

    template<Remap REMAP,
             typename index_t, typename data_t, typename offset_t,
             typename scale_or_empty_t, typename rotate_t, typename ews_or_empty_t>
    auto fourierInsertionByGridding(
            const AccessorRestrict<const data_t, 3, offset_t>& slice, dim4_t slice_shape,
            const AccessorRestrict<data_t, 3, offset_t>& grid, dim4_t grid_shape,
            scale_or_empty_t inv_scaling_matrices, rotate_t fwd_rotation_matrices,
            float cutoff, dim4_t target_shape, ews_or_empty_t ews_radius) {
        return FourierInsertionByGridding<
                REMAP, index_t, data_t, offset_t, scale_or_empty_t, rotate_t, ews_or_empty_t>(
                slice, slice_shape, grid, grid_shape,
                inv_scaling_matrices, fwd_rotation_matrices,
                cutoff, target_shape, ews_radius);
    }
}

namespace noa::geometry::fft::details {
    // Direct Fourier insertion, but this time looping through the grid.
    // In practice, it allows to give an explicit "thickness" to the central slices.
    // It also "looks" better (more symmetric) than gridding, but it's much slower.
    // One limitation is that it requires the slices to be centered.
    template<Remap REMAP,
             typename index_t, typename data_t, typename offset_t,
             typename scale_or_empty_t, typename rotate_t, typename ews_or_empty_t,
             typename slice_interpolator_t>
    class FourierInsertionExplicitThickness {
        using Layout = ::noa::fft::Layout;
        static constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        static constexpr bool IS_GRID_CENTERED = REMAP_ & Layout::DST_CENTERED;
        static_assert(REMAP_ & Layout::SRC_CENTERED &&
                      !(REMAP_ & Layout::SRC_FULL) &&
                      !(REMAP_ & Layout::DST_FULL));

        static_assert(traits::is_any_v<data_t, float, double, cfloat_t, cdouble_t>);
        static_assert(traits::is_any_v<scale_or_empty_t, float22_t, const float22_t*> || std::is_empty_v<scale_or_empty_t>);
        static_assert(traits::is_any_v<ews_or_empty_t, float, float2_t> || std::is_empty_v<ews_or_empty_t>);
        static_assert(traits::is_any_v<rotate_t, float33_t, const float33_t*>);
        static_assert(traits::is_any_v<index_t, int32_t, int64_t>);

        using grid_accessor_t = AccessorRestrict<data_t, 3, offset_t>;
        using index2_t = Int2<index_t>;
        using index3_t = Int3<index_t>;
        using real_t = traits::value_type_t<data_t>;

    public:
        FourierInsertionExplicitThickness(slice_interpolator_t slice, dim4_t slice_shape,
                                          grid_accessor_t grid, dim4_t grid_shape,
                                          scale_or_empty_t fwd_scaling_matrices, rotate_t inv_rotation_matrices,
                                          float cutoff, dim4_t target_shape, ews_or_empty_t ews_radius,
                                          float slice_z_radius)
                : m_slice(slice), m_grid(grid),
                  m_inv_rotation_matrices(inv_rotation_matrices),
                  m_fwd_scaling_matrices(fwd_scaling_matrices),
                  m_slice_z_radius(slice_z_radius) {

            NOA_ASSERT(all(slice_shape > 0) && all(grid_shape > 0));
            NOA_ASSERT(slice_shape[1] == 1);
            NOA_ASSERT(grid_shape[0] == 1);

            const auto slice_shape_2d = safe_cast<index2_t>(dim2_t(slice_shape.get(2)));
            m_grid_shape = safe_cast<index3_t>(dim3_t(grid_shape.get(1)));

            m_slice_count = static_cast<index_t>(slice_shape[0]);
            m_f_slice_shape = float2_t(slice_shape_2d / 2 * 2 + index2_t(slice_shape_2d == 1));
            m_f_slice_y_center = static_cast<float>(slice_shape_2d[0] / 2);

            const auto l_target_shape = any(target_shape == 0) ?
                    m_grid_shape : safe_cast<index3_t>(dim3_t(target_shape.get(1)));
            m_f_target_shape = float3_t(l_target_shape / 2 * 2 + index3_t(l_target_shape == 1));

            // Using the small-angle approximation, Z = wavelength / 2 * (X^2 + Y^2).
            // See doi:10.1016/S0304-3991(99)00120-5 for a derivation.
            if constexpr (!std::is_empty_v<ews_or_empty_t>)
                m_ews_diam_inv = any(ews_radius != 0) ? 1 / (2 * ews_radius) : ews_or_empty_t{};

            m_cutoff = math::clamp(cutoff, 0.f, 0.5f);
            m_cutoff *= m_cutoff;
        }

        // For every voxel of the grid.
        NOA_HD void operator()(index_t z, index_t y, index_t u) const noexcept { // x == u
            const index_t w = index2frequency<IS_GRID_CENTERED>(z, m_grid_shape[0]);
            const index_t v = index2frequency<IS_GRID_CENTERED>(y, m_grid_shape[1]);
            const auto orig_freq = float3_t{w, v, u} / m_f_target_shape;
            if (math::dot(orig_freq, orig_freq) > m_cutoff)
                return;

            for (index_t i = 0; i < m_slice_count; ++i) {
                auto [freq_z, freq_2d] = transformGridToSlice(
                        orig_freq, m_fwd_scaling_matrices, m_inv_rotation_matrices, i, m_ews_diam_inv);

                // If voxel is not affected by the slice, skip.
                data_t value{0};
                if (freq_z <= m_slice_z_radius && freq_z >= -m_slice_z_radius) {
                    value = interpolateSliceValue(freq_2d, m_f_slice_shape, m_f_slice_y_center, m_slice, i);
                    const auto weight = sliceZWeight(freq_z, m_slice_z_radius);
                    value *= static_cast<real_t>(weight);
                }

                // The transformation preserves the hermitian symmetry, so there's nothing else to do.
                m_grid(z, y, u) += value;
            }
        }

    private:
        slice_interpolator_t m_slice;
        grid_accessor_t m_grid;
        rotate_t m_inv_rotation_matrices;
        scale_or_empty_t m_fwd_scaling_matrices;

        index3_t m_grid_shape;
        index_t m_slice_count;
        float3_t m_f_target_shape;
        float2_t m_f_slice_shape;
        ews_or_empty_t m_ews_diam_inv{};
        float m_slice_z_radius;
        float m_f_slice_y_center;
        float m_cutoff;
    };

    template<Remap REMAP,
            typename index_t, typename data_t, typename offset_t,
            typename scale_or_empty_t, typename rotate_t, typename ews_or_empty_t,
            typename slice_interpolator>
    auto fourierInsertionExplicitThickness(
            const slice_interpolator& slice, dim4_t slice_shape,
            const AccessorRestrict<data_t, 3, offset_t>& grid, dim4_t grid_shape,
            scale_or_empty_t inv_scaling_matrices, rotate_t fwd_rotation_matrices,
            float cutoff, dim4_t target_shape, ews_or_empty_t ews_radius,
            float slice_z_radius) {

        return FourierInsertionExplicitThickness<
                REMAP, index_t, data_t, offset_t,
                scale_or_empty_t, rotate_t,
                ews_or_empty_t, slice_interpolator>(
                slice, slice_shape, grid, grid_shape,
                inv_scaling_matrices, fwd_rotation_matrices,
                cutoff, target_shape, ews_radius,
                slice_z_radius);
    }
}

namespace noa::geometry::fft::details {
    // The exact same transformation as insertion with gridding is applied here,
    // but instead of inserting the transformed slices into the grid,
    // the transformed slices are extracted from the grid.
    template<Remap REMAP,
            typename index_t, typename data_t, typename offset_t,
            typename scale_or_empty_t, typename rotate_t, typename ews_or_empty_t,
            typename grid_interpolator_t>
    class FourierExtraction {
        using Layout = ::noa::fft::Layout;
        static constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        static constexpr bool IS_SLICE_CENTERED = REMAP_ & Layout::DST_CENTERED;
        static_assert(REMAP_ & Layout::SRC_CENTERED &&
                      !(REMAP_ & Layout::SRC_FULL) &&
                      !(REMAP_ & Layout::DST_FULL));

        static_assert(traits::is_any_v<data_t, float, double, cfloat_t, cdouble_t>);
        static_assert(traits::is_any_v<scale_or_empty_t, float22_t, const float22_t*> || std::is_empty_v<scale_or_empty_t>);
        static_assert(traits::is_any_v<ews_or_empty_t, float, float2_t> || std::is_empty_v<ews_or_empty_t>);
        static_assert(traits::is_any_v<rotate_t, float33_t, const float33_t*>);
        static_assert(traits::is_any_v<index_t, int32_t, int64_t>);

        using slice_accessor_t = AccessorRestrict<data_t, 3, offset_t>;
        using index2_t = Int2<index_t>;
        using index3_t = Int3<index_t>;
        using real_t = traits::value_type_t<data_t>;

    public:
        FourierExtraction(
                grid_interpolator_t grid, dim4_t grid_shape,
                slice_accessor_t slice, dim4_t slice_shape,
                scale_or_empty_t inv_scaling_matrices, rotate_t fwd_rotation_matrices,
                float cutoff, dim4_t target_shape, ews_or_empty_t ews_radius)

                : m_grid(grid), m_slice(slice),
                  m_inv_scaling_matrices(inv_scaling_matrices),
                  m_fwd_rotation_matrices(fwd_rotation_matrices) {

            NOA_ASSERT(all(slice_shape > 0) && all(grid_shape > 0));
            NOA_ASSERT(slice_shape[1] == 1);
            NOA_ASSERT(grid_shape[0] == 1);

            const auto slice_shape_2d = safe_cast<index2_t>(dim2_t(slice_shape.get(2)));
            m_slice_size_y = slice_shape_2d[0];
            m_f_slice_shape = float2_t(slice_shape_2d / 2 * 2 + index2_t(slice_shape_2d == 1));

            // Use the grid shape as backup.
            const auto grid_shape_3d = safe_cast<index3_t>(dim3_t(grid_shape.get(1)));
            const auto target_shape_3d = any(target_shape == 0) ? grid_shape_3d : index3_t(target_shape.get(1));
            m_f_target_shape = float3_t(target_shape_3d / 2 * 2 + index3_t(target_shape_3d == 1));
            m_f_grid_zy_center = float2_t(grid_shape_3d[0] / 2, grid_shape_3d[1] / 2); // grid ZY center

            // Using the small-angle approximation, Z = wavelength / 2 * (X^2 + Y^2).
            // See doi:10.1016/S0304-3991(99)00120-5 for a derivation.
            if constexpr (!std::is_empty_v<ews_or_empty_t>)
                m_ews_diam_inv = any(ews_radius != 0) ? 1 / (2 * ews_radius) : ews_or_empty_t{};

            m_cutoff = math::clamp(cutoff, 0.f, 0.5f);
            m_cutoff *= m_cutoff;
        }

        // For every pixel of every slice to extract.
        NOA_HD void operator()(index_t i, index_t y, index_t u) const noexcept {
            // Transform slice onto the grid.
            const int64_t v = index2frequency<IS_SLICE_CENTERED>(y, m_slice_size_y);
            const float2_t freq_2d = float2_t{v, u} / m_f_slice_shape;
            const float3_t freq_3d = transformSliceToGrid(
                    freq_2d, m_inv_scaling_matrices, m_fwd_rotation_matrices, i, m_ews_diam_inv);

            // Interpolate grid values at slice location.
            m_slice(i, y, u) = math::dot(freq_3d, freq_3d) > m_cutoff ?
                               data_t{0} :
                               interpolateGridValue(freq_3d, m_f_target_shape, m_f_grid_zy_center, m_grid);
        }

    private:
        grid_interpolator_t m_grid;
        slice_accessor_t m_slice;
        scale_or_empty_t m_inv_scaling_matrices;
        rotate_t m_fwd_rotation_matrices;

        float3_t m_f_target_shape;
        float2_t m_f_slice_shape;
        float2_t m_f_grid_zy_center;
        index_t m_slice_size_y;
        ews_or_empty_t m_ews_diam_inv{};
        float m_cutoff;
    };

    template<Remap REMAP,
             typename index_t, typename data_t, typename offset_t,
             typename scale_or_empty_t, typename rotate_t, typename ews_or_empty_t,
             typename grid_interpolator_t>
    auto fourierExtraction(
            const grid_interpolator_t& grid, dim4_t grid_shape,
            const AccessorRestrict<data_t, 3, offset_t>& slice, dim4_t slice_shape,
            scale_or_empty_t inv_scaling_matrices, rotate_t fwd_rotation_matrices,
            float cutoff, dim4_t target_shape, ews_or_empty_t ews_radius) {
        return FourierExtraction<
               REMAP, index_t, data_t, offset_t,
               scale_or_empty_t, rotate_t,
               ews_or_empty_t, grid_interpolator_t>(
               grid, grid_shape, slice, slice_shape,
               inv_scaling_matrices, fwd_rotation_matrices,
               cutoff, target_shape, ews_radius);
    }
}

namespace noa::geometry::fft::details {
    template<Remap REMAP,
            typename index_t, typename data_t, typename offset_t,
            typename input_scale_or_empty_t, typename input_rotate_t,
            typename output_scale_or_empty_t, typename output_rotate_t,
            typename ews_or_empty_t, typename input_slice_interpolator_t>
    class FourierInsertExtraction {
        using Layout = ::noa::fft::Layout;
        static constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        static constexpr bool IS_OUTPUT_SLICE_CENTERED = REMAP_ & Layout::DST_CENTERED;
        static_assert(REMAP_ & Layout::SRC_CENTERED &&
                      !(REMAP_ & Layout::SRC_FULL) &&
                      !(REMAP_ & Layout::DST_FULL));

        static_assert(traits::is_any_v<data_t, float, double, cfloat_t, cdouble_t>);
        static_assert(traits::is_any_v<input_scale_or_empty_t, float22_t, const float22_t*> ||
                      std::is_empty_v<input_scale_or_empty_t>);
        static_assert(traits::is_any_v<output_scale_or_empty_t, float22_t, const float22_t*> ||
                      std::is_empty_v<output_scale_or_empty_t>);
        static_assert(traits::is_any_v<ews_or_empty_t, float, float2_t> || std::is_empty_v<ews_or_empty_t>);
        static_assert(traits::is_any_v<input_rotate_t, float33_t, const float33_t*>);
        static_assert(traits::is_any_v<output_rotate_t, float33_t, const float33_t*>);
        static_assert(traits::is_any_v<index_t, int32_t, int64_t>);

        using output_slice_accessor_t = AccessorRestrict<data_t, 3, offset_t>;
        using index2_t = Int2<index_t>;
        using index3_t = Int3<index_t>;
        using real_t = traits::value_type_t<data_t>;

    public:
        FourierInsertExtraction(input_slice_interpolator_t input_slices, dim4_t input_shape,
                                output_slice_accessor_t output_slices, dim4_t output_shape,
                                const input_scale_or_empty_t& insert_fwd_scaling_matrices,
                                const input_rotate_t& insert_inv_rotation_matrices,
                                const output_scale_or_empty_t& extract_inv_scaling_matrices,
                                const output_rotate_t& extract_fwd_rotation_matrices,
                                float cutoff, ews_or_empty_t ews_radius, float slice_z_radius)
                : m_input_slices(input_slices),
                  m_output_slices(output_slices),
                  m_insert_inv_rotation_matrices(insert_inv_rotation_matrices),
                  m_extract_fwd_rotation_matrices(extract_fwd_rotation_matrices),
                  m_insert_fwd_scaling_matrices(insert_fwd_scaling_matrices),
                  m_extract_inv_scaling_matrices(extract_inv_scaling_matrices),
                  m_slice_z_radius(slice_z_radius) {

//            NOA_ASSERT(input_slices && output_slices && input_slices.get() != output_slices.get() &&
//                       all(input_slices_shape > 0) && all(output_slices_shape > 0));
//            NOA_ASSERT(input_slices_shape[1] == 1 && output_slices_shape[1] == 1);

            const auto l_input_shape = index2_t(input_shape.get(2));
            const auto l_output_shape = index2_t(output_shape.get(2));

            m_f_input_shape = float2_t(l_input_shape / 2 * 2 + index2_t(l_input_shape == 1));
            m_f_output_shape = float2_t(l_output_shape / 2 * 2 + index2_t(l_output_shape == 1));
            m_output_slice_size_y = static_cast<index_t>(l_output_shape[0]);
            m_f_input_center_y = static_cast<float>(l_input_shape[0] / 2); // slice Y center
            m_input_count = static_cast<index_t>(input_shape[0]);

            // Using the small-angle approximation, Z = wavelength / 2 * (X^2 + Y^2).
            // See doi:10.1016/S0304-3991(99)00120-5 for a derivation.
            if constexpr (!std::is_empty_v<ews_or_empty_t>)
                m_ews_diam_inv = any(ews_radius != 0) ? 1 / (2 * ews_radius) : ews_or_empty_t{};

            m_cutoff = math::clamp(cutoff, 0.f, 0.5f);
            m_cutoff *= m_cutoff;
        }

        // Should be called for every pixel of every slice to extract.
        NOA_HD void operator()(index_t oi, index_t y, index_t u) const noexcept {
            // First, compute the 3D frequency of the current slice i to extract.
            const int64_t v = index2frequency<IS_OUTPUT_SLICE_CENTERED>(y, m_output_slice_size_y);
            float2_t freq_2d = float2_t{v, u} / m_f_output_shape;
            float3_t freq_3d = transformSliceToGrid(
                    freq_2d, m_extract_inv_scaling_matrices, m_extract_fwd_rotation_matrices, oi, m_ews_diam_inv);

            if (math::dot(freq_3d, freq_3d) > m_cutoff) {
                m_output_slices(oi, y, u) = data_t{0};
                return;
            }

            // Then, insert the input slices.
            for (int64_t ii = 0; ii < m_input_count; ++ii) {
                auto [freq_z, freq_2d_] = transformGridToSlice(
                        freq_3d, m_insert_fwd_scaling_matrices, m_insert_inv_rotation_matrices, ii, m_ews_diam_inv);

                // If voxel is not affected by the slice, skip.
                data_t value{0};
                if (freq_z <= m_slice_z_radius && freq_z >= -m_slice_z_radius) {
                    value = interpolateSliceValue(freq_2d, m_f_input_shape, m_f_input_center_y, m_input_slices, ii);
                    const auto weight = sliceZWeight(freq_z, m_slice_z_radius);
                    value *= static_cast<real_t>(weight);
                }

                // The transformation preserves the hermitian symmetry, so there's nothing else to do.
                m_output_slices(oi, y, u) += value;
            }
        }

    private:
        input_slice_interpolator_t m_input_slices;
        output_slice_accessor_t m_output_slices;
        input_rotate_t m_insert_inv_rotation_matrices;
        output_rotate_t m_extract_fwd_rotation_matrices;
        input_scale_or_empty_t m_insert_fwd_scaling_matrices;
        output_scale_or_empty_t m_extract_inv_scaling_matrices;

        float2_t m_f_output_shape;
        float2_t m_f_input_shape;
        float m_f_input_center_y;
        index_t m_input_count;
        index_t m_output_slice_size_y;
        ews_or_empty_t m_ews_diam_inv{};
        float m_cutoff;
        float m_slice_z_radius;
    };

    template<Remap REMAP,
             typename index_t, typename data_t, typename offset_t,
             typename input_scale_or_empty_t, typename input_rotate_t,
             typename output_scale_or_empty_t, typename output_rotate_t,
             typename ews_or_empty_t, typename input_slice_interpolator_t>
    auto fourierInsertExtraction(
            input_slice_interpolator_t input_slices, dim4_t input_shape,
            const AccessorRestrict<data_t, 3, offset_t> output_slices, dim4_t output_shape,
            const input_scale_or_empty_t& insert_fwd_scaling_matrices,
            const input_rotate_t& insert_inv_rotation_matrices,
            const output_scale_or_empty_t& extract_inv_scaling_matrices,
            const output_rotate_t& extract_fwd_rotation_matrices,
            float cutoff, ews_or_empty_t ews_radius, float slice_z_radius) {
        return FourierInsertExtraction<
                REMAP, index_t, data_t, offset_t,
                input_scale_or_empty_t, input_rotate_t,
                output_scale_or_empty_t, output_rotate_t,
                ews_or_empty_t, input_slice_interpolator_t>(
                input_slices, input_shape,
                output_slices, output_shape,
                insert_fwd_scaling_matrices,
                insert_inv_rotation_matrices,
                extract_inv_scaling_matrices,
                extract_fwd_rotation_matrices,
                cutoff, ews_radius, slice_z_radius);
    }
}

namespace noa::geometry::fft::details {
    template<bool POST_CORRECTION, typename index_t, typename data_t, typename coord_t>
    class GriddingCorrection {
        static_assert(traits::is_int_v<index_t>);
        static_assert(traits::are_float_v<data_t, coord_t>);

        using input_accessor_t = Accessor<const data_t, 4, index_t>;
        using output_accessor_t = Accessor<data_t, 4, index_t>;
        using index3_t = Int3<index_t>;
        using coord3_t = Float3<coord_t>;

    public:
        GriddingCorrection(input_accessor_t input,
                           output_accessor_t output,
                           dim4_t shape)
                           : m_input(input), m_output(output) {
            const index3_t l_shape(shape.get(1));
            m_f_shape = coord3_t(l_shape);
            m_half = m_f_shape / 2 * coord3_t(l_shape != 1); // if size == 1, half should be 0
        }

        NOA_HD void operator()(index_t i, index_t j, index_t k, index_t l) const noexcept {
            coord3_t dist{j, k, l};
            dist -= m_half;
            dist /= m_f_shape;

            constexpr coord_t PI = math::Constants<coord_t>::PI;
            const coord_t radius = math::sqrt(math::dot(dist, dist));
            const coord_t sinc = math::sinc(PI * radius);
            const auto sinc2 = static_cast<data_t>(sinc * sinc); // > 0.05

            const data_t value = m_input(i, j, k, l);
            m_output(i, j, k, l) = POST_CORRECTION ? value / sinc2 : value * sinc2;
        }

    private:
        input_accessor_t m_input;
        output_accessor_t m_output;
        coord3_t m_f_shape;
        coord3_t m_half;
    };

    template<bool POST_CORRECTION, typename coord_t = float, typename index_t, typename data_t>
    auto griddingCorrection(const Accessor<const data_t, 4, index_t>& input,
                            const Accessor<data_t, 4, index_t>& output,
                            dim4_t shape) {
        return GriddingCorrection<POST_CORRECTION, index_t, data_t, coord_t>(input, output, shape);
    }
}
