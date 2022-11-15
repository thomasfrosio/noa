#pragma once

#include "noa/common/Types.h"
#include "noa/common/geometry/Interpolate.h"
#include "noa/common/geometry/details/Utilities.h"

// Implementation for backward and forward projections using Fourier insertion and extraction.
// Can be called from the CPU (serial/OpenMP) or CUDA backend.

namespace noa::geometry::fft::details {
    using Remap = ::noa::fft::Remap;

    // TODO Move atomic to separate common header!
    template<typename GridAccessor, typename Int, typename Data>
    NOA_FHD void atomicAdd(const GridAccessor& grid,
                           Int idx_w, Int idx_v, Int idx_u,
                           Data value) {
        #if defined(__CUDA_ARCH__)
        ::noa::cuda::utils::atomic::add(grid.offsetPointer(grid.get(), idx_w, idx_v, idx_u), value);
        #else
        Data& grid_value = grid(idx_w, idx_v, idx_u);
        if constexpr (traits::is_complex_v<Data>) {
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
    template<typename Int>
    NOA_IHD void setGriddingWeights(Int3<Int> base0, float3_t freq, float o_weights[2][2][2]) {
        // So if the coordinate is centered in the bottom left corner of the cube (base0),
        // i.e. its decimal is 0, the corresponding fraction for this element should be 1.
        float3_t fraction[2];
        fraction[1] = freq - float3_t(base0);
        fraction[0] = 1.f - fraction[1];
        for (Int w = 0; w < 2; ++w)
            for (Int v = 0; v < 2; ++v)
                for (Int u = 0; u < 2; ++u)
                    o_weights[w][v][u] = fraction[w][0] * fraction[v][1] * fraction[u][2];
    }

    // Spread the value within a 2x2x2 centered on a particular frequency using linear interpolation.
    // "frequency" is the frequency, in samples, centered on DC, with negative frequencies on the left.
    template<bool IS_GRID_CENTERED, typename GridAccessor, typename SInt, typename Data>
    NOA_IHD void addByGridding(const GridAccessor& grid, Int3<SInt> grid_shape,
                               Data data, float3_t frequency) {
        using real_t = traits::value_type_t<Data>;
        const auto base0 = Int3<SInt>(math::floor(frequency));

        float kernel[2][2][2]; // 2x2x2 tri-linear weights
        setGriddingWeights(base0, frequency, kernel);

        for (SInt w = 0; w < 2; ++w) {
            for (SInt v = 0; v < 2; ++v) {
                for (SInt u = 0; u < 2; ++u) {
                    const SInt idx_w = frequency2index<IS_GRID_CENTERED>(base0[0] + w, grid_shape[0]);
                    const SInt idx_v = frequency2index<IS_GRID_CENTERED>(base0[1] + v, grid_shape[1]);
                    const SInt idx_u = base0[2] + u;

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
            if constexpr (traits::is_complex_v<Data>)
                data.imag = -data.imag;
            for (SInt w = 0; w < 2; ++w) {
                for (SInt v = 0; v < 2; ++v) {
                    const SInt idx_w = frequency2index<IS_GRID_CENTERED>(-(base0[0] + w), grid_shape[0]);
                    const SInt idx_v = frequency2index<IS_GRID_CENTERED>(-(base0[1] + v), grid_shape[1]);

                    if (idx_w >= 0 && idx_w < grid_shape[0] &&
                        idx_v >= 0 && idx_v < grid_shape[1]) {
                        const auto fraction = static_cast<real_t>(kernel[w][v][0]);
                        noa::geometry::fft::details::atomicAdd(grid, idx_w, idx_v, SInt{0}, data * fraction);
                    }
                }
            }
        }
    }

    // Transforms a 2D normalized frequency representing the slice to a 3D normalized
    // frequency representing the grid. This is a forward transformation of the frequency,
    // but because it is in Fourier-space, the real-space scaling is inverted.
    template<typename ScaleOrEmtpy, typename Rotate, typename Int, typename EWSOrEmpty>
    constexpr NOA_IHD float3_t
    transformSliceToGrid(float2_t frequency,
                         const ScaleOrEmtpy& inv_scaling_matrices,
                         const Rotate& fwd_rotation_matrices,
                         Int index,
                         EWSOrEmpty inv_ews_diameter) {
        // If we apply the EWS curvature, the scaling factors should be corrected
        // before applying the curvature, and therefore before applying the rotation.
        // That way, we use the correct frequencies to compute the EWS, e.g. resulting
        // in a spherical EWS even under anisotropic magnification.
        if constexpr (std::is_pointer_v<ScaleOrEmtpy>)
            frequency = inv_scaling_matrices[index] * frequency;
        else if constexpr (!std::is_empty_v<ScaleOrEmtpy>)
            frequency = inv_scaling_matrices * frequency;

        // TODO We use the Small Angle Approximation to compute the EWS curvature,
        //      so the frequency (u,v) is unchanged. Look at the cisTEM implementation
        //      to remove this approximation? RELION shows that even for low voltages
        //      and large boxes, it is probably not worth it though.
        float3_t freq_3d{0, frequency[0], frequency[1]};
        if constexpr (!std::is_empty_v<EWSOrEmpty>)
            freq_3d[0] = math::sum(inv_ews_diameter * frequency * frequency);

        if constexpr (std::is_pointer_v<Rotate>)
            return fwd_rotation_matrices[index] * freq_3d;
        else
            return fwd_rotation_matrices * freq_3d;
    }

    // Same as above, but in the other direction.
    template<typename ScaleOrEmpty, typename Rotate, typename Int, typename EWSOrEmpty>
    constexpr NOA_IHD Pair<float, float2_t>
    transformGridToSlice(float3_t frequency,
                         const ScaleOrEmpty& fwd_scaling_matrices,
                         const Rotate& inv_rotation_matrices,
                         Int index,
                         EWSOrEmpty inv_ews_diameter) {
        if constexpr (std::is_pointer_v<Rotate>)
            frequency = inv_rotation_matrices[index] * frequency;
        else
            frequency = inv_rotation_matrices * frequency;

        float2_t freq_2d{frequency[1], frequency[2]};
        float freq_z = frequency[0];
        if constexpr (!std::is_empty_v<EWSOrEmpty>)
            freq_z -= math::sum(inv_ews_diameter * freq_2d * freq_2d);

        // Same reason as for the forward transformation.
        // Here the grid is correct, so rotate the EWS, then compute
        // the curvature and only then we can scale the slice.
        if constexpr (std::is_pointer_v<ScaleOrEmpty>)
            freq_2d = fwd_scaling_matrices[index] * freq_2d;
        else if constexpr (!std::is_empty_v<ScaleOrEmpty>)
            freq_2d = fwd_scaling_matrices * freq_2d;

        return {freq_z, freq_2d};
    }

    // Interpolates the value at a given frequency, in cycle per pixel.
    template<typename Interpolator, typename Int>
    constexpr NOA_HD auto interpolateSliceValue(float2_t frequency, float2_t slice_shape, float center_y,
                                                const Interpolator& interpolator, Int batch) {
        using data_t = typename Interpolator::data_type;
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
    template<typename Interpolator>
    constexpr NOA_HD auto interpolateGridValue(float3_t frequency, float3_t target_shape,
                                               float2_t grid_center_zy,
                                               const Interpolator& interpolator) {
        using data_t = typename Interpolator::data_type;
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
             typename Index, typename Data, typename Offset,
             typename ScaleOrEmpty, typename Rotate, typename EWSOrEmpty>
    class FourierInsertionByGridding {
        using Layout = ::noa::fft::Layout;
        static constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        static constexpr bool IS_SLICE_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        static constexpr bool IS_GRID_CENTERED = REMAP_ & Layout::DST_CENTERED;
        static_assert(!(REMAP_ & Layout::SRC_FULL) && !(REMAP_ & Layout::DST_FULL));

        static_assert(traits::is_any_v<Data, float, double, cfloat_t, cdouble_t>);
        static_assert(traits::is_any_v<ScaleOrEmpty, float22_t, const float22_t*> || std::is_empty_v<ScaleOrEmpty>);
        static_assert(traits::is_any_v<EWSOrEmpty, float, float2_t> || std::is_empty_v<EWSOrEmpty>);
        static_assert(traits::is_any_v<Rotate, float33_t, const float33_t*>);
        static_assert(traits::is_any_v<Index, int32_t, int64_t>);

        using index_type = Index;
        using data_type = Data;
        using offset_type = Offset;
        using scale_or_empty_type = ScaleOrEmpty;
        using rotate_type = Rotate;
        using ews_or_empty_type = EWSOrEmpty;

        using slice_accessor_type = AccessorRestrict<const data_type, 3, offset_type>;
        using grid_accessor_type = AccessorRestrict<data_type, 3, offset_type>;
        using index2_type = Int2<index_type>;
        using index3_type = Int3<index_type>;
        using real_type = traits::value_type_t<data_type>;

    public:
        FourierInsertionByGridding(
                slice_accessor_type slice, dim4_t slice_shape,
                grid_accessor_type grid, dim4_t grid_shape,
                scale_or_empty_type inv_scaling_matrices, rotate_type fwd_rotation_matrices,
                float cutoff, dim4_t target_shape, ews_or_empty_type ews_radius)
                : m_slice(slice),
                  m_grid(grid),
                  m_fwd_rotation_matrices(fwd_rotation_matrices),
                  m_inv_scaling_matrices(inv_scaling_matrices) {

            NOA_ASSERT(all(slice_shape > 0) && all(grid_shape > 0));
            NOA_ASSERT(slice_shape[1] == 1);
            NOA_ASSERT(grid_shape[0] == 1);

            const auto slice_shape_2d = safe_cast<index2_type>(dim2_t(slice_shape.get(2)));
            m_slice_size_y = slice_shape_2d[0];
            m_grid_shape = safe_cast<index3_type>(dim3_t(grid_shape.get(1)));
            m_f_slice_shape = float2_t(slice_shape_2d / 2 * 2 + index2_type(slice_shape_2d == 1));

            // Use the grid shape as backup.
            const auto target_shape_3d = any(target_shape == 0) ?
                    m_grid_shape : safe_cast<index3_type>(dim3_t(target_shape.get(1)));
            m_f_target_shape = float3_t(target_shape_3d / 2 * 2 + index3_type(target_shape_3d == 1));

            // Using the small-angle approximation, Z = wavelength / 2 * (X^2 + Y^2).
            // See doi:10.1016/S0304-3991(99)00120-5 for a derivation.
            if constexpr (!std::is_empty_v<ews_or_empty_type>)
                m_ews_diam_inv = any(ews_radius != 0) ? 1 / (2 * ews_radius) : ews_or_empty_type{};

            m_cutoff = math::clamp(cutoff, 0.f, 0.5f);
            m_cutoff *= m_cutoff;
        }

        // For every pixel of every central slice to insert.
        NOA_HD void operator()(index_type batch, index_type y, index_type u) const noexcept { // x == u
            // We compute the forward transformation and use normalized frequencies.
            // The oversampling is implicitly handled when scaling back to the target shape.
            const index_type v = index2frequency<IS_SLICE_CENTERED>(y, m_slice_size_y);
            const float2_t freq_2d = float2_t{v, u} / m_f_slice_shape;
            float3_t freq_3d = transformSliceToGrid(
                    freq_2d, m_inv_scaling_matrices, m_fwd_rotation_matrices, batch, m_ews_diam_inv);

            // The frequency rate won't change from that point, so check for the cutoff.
            if (math::dot(freq_3d, freq_3d) > m_cutoff)
                return;

            // Handle the non-redundancy in x.
            real_type conj = 1;
            if (freq_3d[2] < 0) {
                freq_3d = -freq_3d;
                if constexpr(traits::is_complex_v<data_type>)
                    conj = -1;
            }

            // Scale back to the target shape.
            freq_3d *= m_f_target_shape;

            // At this point, we know we are going to use the slice value.
            data_type value = m_slice(batch, y, u);
            if constexpr(traits::is_complex_v<data_type>)
                value.imag *= conj;
            else
                (void) conj;

            addByGridding<IS_GRID_CENTERED>(m_grid, m_grid_shape, value, freq_3d);
        }

    private:
        slice_accessor_type m_slice;
        grid_accessor_type m_grid;
        rotate_type m_fwd_rotation_matrices;
        scale_or_empty_type m_inv_scaling_matrices;

        index3_type m_grid_shape;
        index_type m_slice_size_y;
        float3_t m_f_target_shape;
        float2_t m_f_slice_shape;
        ews_or_empty_type m_ews_diam_inv{};
        float m_cutoff;
    };

    template<Remap REMAP,
             typename Index, typename Data, typename Offset,
             typename ScaleOrEmpty, typename Rotate, typename EWSOrEmpty>
    auto fourierInsertionByGridding(
            const AccessorRestrict<const Data, 3, Offset>& slice, dim4_t slice_shape,
            const AccessorRestrict<Data, 3, Offset>& grid, dim4_t grid_shape,
            ScaleOrEmpty inv_scaling_matrices, Rotate fwd_rotation_matrices,
            float cutoff, dim4_t target_shape, EWSOrEmpty ews_radius) {
        return FourierInsertionByGridding<
                REMAP, Index, Data, Offset, ScaleOrEmpty, Rotate, EWSOrEmpty>(
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
             typename Index, typename Data, typename Offset,
             typename ScaleOrEmpty, typename Rotate, typename EWSOrEmpty,
             typename SliceInterpolator>
    class FourierInsertionExplicitThickness {
        using Layout = ::noa::fft::Layout;
        static constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        static constexpr bool IS_GRID_CENTERED = REMAP_ & Layout::DST_CENTERED;
        static_assert(REMAP_ & Layout::SRC_CENTERED &&
                      !(REMAP_ & Layout::SRC_FULL) &&
                      !(REMAP_ & Layout::DST_FULL));

        static_assert(traits::is_any_v<Data, float, double, cfloat_t, cdouble_t>);
        static_assert(traits::is_any_v<ScaleOrEmpty, float22_t, const float22_t*> || std::is_empty_v<ScaleOrEmpty>);
        static_assert(traits::is_any_v<EWSOrEmpty, float, float2_t> || std::is_empty_v<EWSOrEmpty>);
        static_assert(traits::is_any_v<Rotate, float33_t, const float33_t*>);
        static_assert(traits::is_any_v<Index, int32_t, int64_t>);

        using index_type = Index;
        using data_type = Data;
        using offset_type = Offset;
        using scale_or_empty_type = ScaleOrEmpty;
        using rotate_type = Rotate;
        using ews_or_empty_type = EWSOrEmpty;
        using slice_interpolator_type = SliceInterpolator;

        using grid_accessor_type = AccessorRestrict<data_type, 3, offset_type>;
        using index2_type = Int2<index_type>;
        using index3_type = Int3<index_type>;
        using real_type = traits::value_type_t<data_type>;

    public:
        FourierInsertionExplicitThickness(
                slice_interpolator_type slice, dim4_t slice_shape,
                grid_accessor_type grid, dim4_t grid_shape,
                scale_or_empty_type fwd_scaling_matrices, rotate_type inv_rotation_matrices,
                float cutoff, dim4_t target_shape, ews_or_empty_type ews_radius,
                float slice_z_radius)
                : m_slice(slice), m_grid(grid),
                  m_inv_rotation_matrices(inv_rotation_matrices),
                  m_fwd_scaling_matrices(fwd_scaling_matrices),
                  m_slice_z_radius(slice_z_radius) {

            NOA_ASSERT(all(slice_shape > 0) && all(grid_shape > 0));
            NOA_ASSERT(slice_shape[1] == 1);
            NOA_ASSERT(grid_shape[0] == 1);

            const auto slice_shape_2d = safe_cast<index2_type>(dim2_t(slice_shape.get(2)));
            m_grid_shape = safe_cast<index3_type>(dim3_t(grid_shape.get(1)));

            m_slice_count = static_cast<index_type>(slice_shape[0]);
            m_f_slice_shape = float2_t(slice_shape_2d / 2 * 2 + index2_type(slice_shape_2d == 1));
            m_f_slice_y_center = static_cast<float>(slice_shape_2d[0] / 2);

            const auto l_target_shape = any(target_shape == 0) ?
                    m_grid_shape : safe_cast<index3_type>(dim3_t(target_shape.get(1)));
            m_f_target_shape = float3_t(l_target_shape / 2 * 2 + index3_type(l_target_shape == 1));

            // Using the small-angle approximation, Z = wavelength / 2 * (X^2 + Y^2).
            // See doi:10.1016/S0304-3991(99)00120-5 for a derivation.
            if constexpr (!std::is_empty_v<ews_or_empty_type>)
                m_ews_diam_inv = any(ews_radius != 0) ? 1 / (2 * ews_radius) : ews_or_empty_type{};

            m_cutoff = math::clamp(cutoff, 0.f, 0.5f);
            m_cutoff *= m_cutoff;
        }

        // For every voxel of the grid.
        NOA_HD void operator()(index_type z, index_type y, index_type u) const noexcept { // x == u
            const index_type w = index2frequency<IS_GRID_CENTERED>(z, m_grid_shape[0]);
            const index_type v = index2frequency<IS_GRID_CENTERED>(y, m_grid_shape[1]);
            const auto orig_freq = float3_t{w, v, u} / m_f_target_shape;
            if (math::dot(orig_freq, orig_freq) > m_cutoff)
                return;

            for (index_type i = 0; i < m_slice_count; ++i) {
                auto [freq_z, freq_2d] = transformGridToSlice(
                        orig_freq, m_fwd_scaling_matrices, m_inv_rotation_matrices, i, m_ews_diam_inv);

                // If voxel is not affected by the slice, skip.
                data_type value{0};
                if (freq_z <= m_slice_z_radius && freq_z >= -m_slice_z_radius) {
                    value = interpolateSliceValue(freq_2d, m_f_slice_shape, m_f_slice_y_center, m_slice, i);
                    const auto weight = sliceZWeight(freq_z, m_slice_z_radius);
                    value *= static_cast<real_type>(weight);
                }

                // The transformation preserves the hermitian symmetry, so there's nothing else to do.
                m_grid(z, y, u) += value;
            }
        }

    private:
        slice_interpolator_type m_slice;
        grid_accessor_type m_grid;
        rotate_type m_inv_rotation_matrices;
        scale_or_empty_type m_fwd_scaling_matrices;

        index3_type m_grid_shape;
        index_type m_slice_count;
        float3_t m_f_target_shape;
        float2_t m_f_slice_shape;
        ews_or_empty_type m_ews_diam_inv{};
        float m_slice_z_radius;
        float m_f_slice_y_center;
        float m_cutoff;
    };

    template<Remap REMAP,
             typename Index, typename Data, typename Offset,
             typename ScaleOrEmpty, typename Rotate, typename EWSOrEmpty,
             typename SliceInterpolator>
    auto fourierInsertionExplicitThickness(
            const SliceInterpolator& slice, dim4_t slice_shape,
            const AccessorRestrict<Data, 3, Offset>& grid, dim4_t grid_shape,
            ScaleOrEmpty inv_scaling_matrices, Rotate fwd_rotation_matrices,
            float cutoff, dim4_t target_shape, EWSOrEmpty ews_radius,
            float slice_z_radius) {

        return FourierInsertionExplicitThickness<
                REMAP, Index, Data, Offset,
                ScaleOrEmpty, Rotate,
                EWSOrEmpty, SliceInterpolator>(
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
             typename Index, typename Data, typename Offset,
             typename ScaleOrEmpty, typename Rotate, typename EWSOrEmpty,
             typename GridInterpolator>
    class FourierExtraction {
        using Layout = ::noa::fft::Layout;
        static constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        static constexpr bool IS_SLICE_CENTERED = REMAP_ & Layout::DST_CENTERED;
        static_assert(REMAP_ & Layout::SRC_CENTERED &&
                      !(REMAP_ & Layout::SRC_FULL) &&
                      !(REMAP_ & Layout::DST_FULL));

        static_assert(traits::is_any_v<Data, float, double, cfloat_t, cdouble_t>);
        static_assert(traits::is_any_v<ScaleOrEmpty, empty_t, float22_t, const float22_t*>);
        static_assert(traits::is_any_v<EWSOrEmpty, empty_t, float, float2_t>);
        static_assert(traits::is_any_v<Rotate, float33_t, const float33_t*>);
        static_assert(traits::is_any_v<Index, int32_t, int64_t>);

        using index_type = Index;
        using data_type = Data;
        using offset_type = Offset;
        using scale_or_empty_type = ScaleOrEmpty;
        using rotate_type = Rotate;
        using ews_or_empty_type = EWSOrEmpty;
        using grid_interpolator_type = GridInterpolator;

        using slice_accessor_type = AccessorRestrict<data_type, 3, offset_type>;
        using index2_type = Int2<index_type>;
        using index3_type = Int3<index_type>;
        using real_type = traits::value_type_t<data_type>;

    public:
        FourierExtraction(
                grid_interpolator_type grid, dim4_t grid_shape,
                slice_accessor_type slice, dim4_t slice_shape,
                scale_or_empty_type inv_scaling_matrices, rotate_type fwd_rotation_matrices,
                float cutoff, dim4_t target_shape, ews_or_empty_type ews_radius)

                : m_grid(grid), m_slice(slice),
                  m_inv_scaling_matrices(inv_scaling_matrices),
                  m_fwd_rotation_matrices(fwd_rotation_matrices) {

            NOA_ASSERT(all(slice_shape > 0) && all(grid_shape > 0));
            NOA_ASSERT(slice_shape[1] == 1);
            NOA_ASSERT(grid_shape[0] == 1);

            const auto slice_shape_2d = safe_cast<index2_type>(dim2_t(slice_shape.get(2)));
            m_slice_size_y = slice_shape_2d[0];
            m_f_slice_shape = float2_t(slice_shape_2d / 2 * 2 + index2_type(slice_shape_2d == 1));

            // Use the grid shape as backup.
            const auto grid_shape_3d = safe_cast<index3_type>(dim3_t(grid_shape.get(1)));
            const auto target_shape_3d = any(target_shape == 0) ? grid_shape_3d : index3_type(target_shape.get(1));
            m_f_target_shape = float3_t(target_shape_3d / 2 * 2 + index3_type(target_shape_3d == 1));
            m_f_grid_zy_center = float2_t(grid_shape_3d[0] / 2, grid_shape_3d[1] / 2); // grid ZY center

            // Using the small-angle approximation, Z = wavelength / 2 * (X^2 + Y^2).
            // See doi:10.1016/S0304-3991(99)00120-5 for a derivation.
            if constexpr (!std::is_empty_v<ews_or_empty_type>)
                m_ews_diam_inv = any(ews_radius != 0) ? 1 / (2 * ews_radius) : ews_or_empty_type{};

            m_cutoff = math::clamp(cutoff, 0.f, 0.5f);
            m_cutoff *= m_cutoff;
        }

        // For every pixel of every slice to extract.
        NOA_HD void operator()(index_type batch, index_type y, index_type u) const noexcept {
            // Transform slice onto the grid.
            const int64_t v = index2frequency<IS_SLICE_CENTERED>(y, m_slice_size_y);
            const float2_t freq_2d = float2_t{v, u} / m_f_slice_shape;
            const float3_t freq_3d = transformSliceToGrid(
                    freq_2d, m_inv_scaling_matrices, m_fwd_rotation_matrices, batch, m_ews_diam_inv);

            // Interpolate grid values at slice location.
            m_slice(batch, y, u) = math::dot(freq_3d, freq_3d) > m_cutoff ?
                                   data_type{0} :
                                   interpolateGridValue(freq_3d, m_f_target_shape, m_f_grid_zy_center, m_grid);
        }

    private:
        grid_interpolator_type m_grid;
        slice_accessor_type m_slice;
        scale_or_empty_type m_inv_scaling_matrices;
        rotate_type m_fwd_rotation_matrices;

        float3_t m_f_target_shape;
        float2_t m_f_slice_shape;
        float2_t m_f_grid_zy_center;
        index_type m_slice_size_y;
        ews_or_empty_type m_ews_diam_inv{};
        float m_cutoff;
    };

    template<Remap REMAP,
             typename Index, typename Data, typename Offset,
             typename ScaleOrEmpty, typename Rotate, typename EWSOrEmpty,
             typename GridInterpolator>
    auto fourierExtraction(
            const GridInterpolator& grid, dim4_t grid_shape,
            const AccessorRestrict<Data, 3, Offset>& slice, dim4_t slice_shape,
            ScaleOrEmpty inv_scaling_matrices, Rotate fwd_rotation_matrices,
            float cutoff, dim4_t target_shape, EWSOrEmpty ews_radius) {
        return FourierExtraction<
               REMAP, Index, Data, Offset,
               ScaleOrEmpty, Rotate,
               EWSOrEmpty, GridInterpolator>(
               grid, grid_shape, slice, slice_shape,
               inv_scaling_matrices, fwd_rotation_matrices,
               cutoff, target_shape, ews_radius);
    }
}

namespace noa::geometry::fft::details {
    template<Remap REMAP,
             typename Index, typename Data, typename Offset,
             typename InputScaleOrEmpty, typename InputRotate,
             typename OutputScaleOrEmpty, typename OutputRotate,
             typename EWSOrEmpty, typename InputSliceInterpolator>
    class FourierInsertExtraction {
        using Layout = ::noa::fft::Layout;
        static constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        static constexpr bool IS_OUTPUT_SLICE_CENTERED = REMAP_ & Layout::DST_CENTERED;
        static_assert(REMAP_ & Layout::SRC_CENTERED &&
                      !(REMAP_ & Layout::SRC_FULL) &&
                      !(REMAP_ & Layout::DST_FULL));

        static_assert(traits::is_any_v<Data, float, double, cfloat_t, cdouble_t>);
        static_assert(traits::is_any_v<InputScaleOrEmpty, empty_t, float22_t, const float22_t*>);
        static_assert(traits::is_any_v<OutputScaleOrEmpty, float22_t, const float22_t*> ||
                      std::is_empty_v<OutputScaleOrEmpty>);
        static_assert(traits::is_any_v<EWSOrEmpty, empty_t, float, float2_t>);
        static_assert(traits::is_any_v<InputRotate, float33_t, const float33_t*>);
        static_assert(traits::is_any_v<OutputRotate, float33_t, const float33_t*>);
        static_assert(traits::is_any_v<Index, int32_t, int64_t>);

        using index_type = Index;
        using data_type = Data;
        using offset_type = Offset;
        using input_scale_or_empty_type = InputScaleOrEmpty;
        using input_rotate_type = InputRotate;
        using output_scale_or_empty_type = OutputScaleOrEmpty;
        using output_rotate_type = OutputRotate;
        using ews_or_empty_type = EWSOrEmpty;
        using input_slice_interpolator_type = InputSliceInterpolator;

        using output_slice_accessor_type = AccessorRestrict<data_type, 3, offset_type>;
        using index2_type = Int2<index_type>;
        using index3_type = Int3<index_type>;
        using real_type = traits::value_type_t<data_type>;

    public:
        FourierInsertExtraction(
                input_slice_interpolator_type input_slices, dim4_t input_shape,
                output_slice_accessor_type output_slices, dim4_t output_shape,
                const input_scale_or_empty_type& insert_fwd_scaling_matrices,
                const input_rotate_type& insert_inv_rotation_matrices,
                const output_scale_or_empty_type& extract_inv_scaling_matrices,
                const output_rotate_type& extract_fwd_rotation_matrices,
                float cutoff, ews_or_empty_type ews_radius, float slice_z_radius)
                : m_input_slices(input_slices),
                  m_output_slices(output_slices),
                  m_insert_inv_rotation_matrices(insert_inv_rotation_matrices),
                  m_extract_fwd_rotation_matrices(extract_fwd_rotation_matrices),
                  m_insert_fwd_scaling_matrices(insert_fwd_scaling_matrices),
                  m_extract_inv_scaling_matrices(extract_inv_scaling_matrices),
                  m_slice_z_radius(slice_z_radius) {

            NOA_ASSERT(all(input_shape > 0) && all(output_shape > 0));
            NOA_ASSERT(input_shape[1] == 1 && output_shape[1] == 1);

            const auto l_input_shape = index2_type(input_shape.get(2));
            const auto l_output_shape = index2_type(output_shape.get(2));

            m_f_input_shape = float2_t(l_input_shape / 2 * 2 + index2_type(l_input_shape == 1));
            m_f_output_shape = float2_t(l_output_shape / 2 * 2 + index2_type(l_output_shape == 1));
            m_output_slice_size_y = static_cast<index_type>(l_output_shape[0]);
            m_f_input_center_y = static_cast<float>(l_input_shape[0] / 2); // slice Y center
            m_input_count = static_cast<index_type>(input_shape[0]);

            // Using the small-angle approximation, Z = wavelength / 2 * (X^2 + Y^2).
            // See doi:10.1016/S0304-3991(99)00120-5 for a derivation.
            if constexpr (!std::is_empty_v<ews_or_empty_type>)
                m_ews_diam_inv = any(ews_radius != 0) ? 1 / (2 * ews_radius) : ews_or_empty_type{};

            m_cutoff = math::clamp(cutoff, 0.f, 0.5f);
            m_cutoff *= m_cutoff;
        }

        // Should be called for every pixel of every slice to extract.
        NOA_HD void operator()(index_type output_batch, index_type y, index_type u) const noexcept {
            // First, compute the 3D frequency of the current slice i to extract.
            const int64_t v = index2frequency<IS_OUTPUT_SLICE_CENTERED>(y, m_output_slice_size_y);
            float2_t freq_2d = float2_t{v, u} / m_f_output_shape;
            float3_t freq_3d = transformSliceToGrid(
                    freq_2d, m_extract_inv_scaling_matrices, m_extract_fwd_rotation_matrices,
                    output_batch, m_ews_diam_inv);

            if (math::dot(freq_3d, freq_3d) > m_cutoff) {
                m_output_slices(output_batch, y, u) = data_type{0};
                return;
            }

            // Then, insert the input slices.
            for (int64_t i = 0; i < m_input_count; ++i) {
                auto [freq_z, freq_2d_] = transformGridToSlice(
                        freq_3d, m_insert_fwd_scaling_matrices, m_insert_inv_rotation_matrices, i, m_ews_diam_inv);

                // If voxel is not affected by the slice, skip.
                data_type value{0};
                if (freq_z <= m_slice_z_radius && freq_z >= -m_slice_z_radius) {
                    value = interpolateSliceValue(freq_2d, m_f_input_shape, m_f_input_center_y, m_input_slices, i);
                    const auto weight = sliceZWeight(freq_z, m_slice_z_radius);
                    value *= static_cast<real_type>(weight);
                }

                // The transformation preserves the hermitian symmetry, so there's nothing else to do.
                m_output_slices(output_batch, y, u) += value;
            }
        }

    private:
        input_slice_interpolator_type m_input_slices;
        output_slice_accessor_type m_output_slices;
        input_rotate_type m_insert_inv_rotation_matrices;
        output_rotate_type m_extract_fwd_rotation_matrices;
        input_scale_or_empty_type m_insert_fwd_scaling_matrices;
        output_scale_or_empty_type m_extract_inv_scaling_matrices;

        float2_t m_f_output_shape;
        float2_t m_f_input_shape;
        float m_f_input_center_y;
        index_type m_input_count;
        index_type m_output_slice_size_y;
        ews_or_empty_type m_ews_diam_inv{};
        float m_cutoff;
        float m_slice_z_radius;
    };

    template<Remap REMAP,
             typename Index, typename Data, typename Offset,
             typename InputScaleOrEmpty, typename InputRotate,
             typename OutputScaleOrEmpty, typename OutputRotate,
             typename EWSOrEmpty, typename InputSliceInterpolator>
    auto fourierInsertExtraction(
            InputSliceInterpolator input_slices, dim4_t input_shape,
            const AccessorRestrict<Data, 3, Offset> output_slices, dim4_t output_shape,
            const InputScaleOrEmpty& insert_fwd_scaling_matrices,
            const InputRotate& insert_inv_rotation_matrices,
            const OutputScaleOrEmpty& extract_inv_scaling_matrices,
            const OutputRotate& extract_fwd_rotation_matrices,
            float cutoff, EWSOrEmpty ews_radius, float slice_z_radius) {
        return FourierInsertExtraction<
                REMAP, Index, Data, Offset,
                InputScaleOrEmpty, InputRotate,
                OutputScaleOrEmpty, OutputRotate,
                EWSOrEmpty, InputSliceInterpolator>(
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
    template<bool POST_CORRECTION, typename Coord, typename Index, typename Data>
    class GriddingCorrection {
        static_assert(traits::is_int_v<Index>);
        static_assert(traits::are_float_v<Data, Coord>);

        using index_type = Index;
        using data_type = Data;
        using coord_type = Coord;

        using input_accessor_type = Accessor<const data_type, 4, index_type>;
        using output_accessor_type = Accessor<data_type, 4, index_type>;
        using index3_type = Int3<index_type>;
        using coord3_type = Float3<coord_type>;

    public:
        GriddingCorrection(input_accessor_type input,
                           output_accessor_type output,
                           dim4_t shape)
                           : m_input(input), m_output(output) {
            const index3_type l_shape(shape.get(1));
            m_f_shape = coord3_type(l_shape);
            m_half = m_f_shape / 2 * coord3_type(l_shape != 1); // if size == 1, half should be 0
        }

        NOA_HD void operator()(index_type i, index_type j, index_type k, index_type l) const noexcept {
            coord3_type dist{j, k, l};
            dist -= m_half;
            dist /= m_f_shape;

            constexpr coord_type PI = math::Constants<coord_type>::PI;
            const coord_type radius = math::sqrt(math::dot(dist, dist));
            const coord_type sinc = math::sinc(PI * radius);
            const auto sinc2 = static_cast<data_type>(sinc * sinc); // > 0.05

            const data_type value = m_input(i, j, k, l);
            m_output(i, j, k, l) = POST_CORRECTION ? value / sinc2 : value * sinc2;
        }

    private:
        input_accessor_type m_input;
        output_accessor_type m_output;
        coord3_type m_f_shape;
        coord3_type m_half;
    };

    template<bool POST_CORRECTION, typename Coord = float, typename Index, typename Data>
    auto griddingCorrection(const Accessor<const Data, 4, Index>& input,
                            const Accessor<Data, 4, Index>& output,
                            dim4_t shape) {
        return GriddingCorrection<POST_CORRECTION, Coord, Index, Data>(input, output, shape);
    }
}
