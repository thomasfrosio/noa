#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/geometry/Interpolate.hpp"
#include "noa/core/utils/Atomic.hpp"
#include "noa/core/fft/Frequency.hpp"
#include "noa/algorithms/Algorithms.hpp"

// CPU (serial/OpenMP) or GPU (CUDA) implementation for backward/forward projections
// using Fourier insertion/extraction.

namespace noa::algorithm::geometry::details {
    using Remap = ::noa::fft::Remap;

    template<Remap REMAP, typename Value, typename Scale, typename Rotate>
    constexpr bool is_valid_insert_rasterize_v =
            nt::is_any_v<Value, f32, f64, c32, c64> &&
            nt::is_any_v<Scale, const Float22*, Float22> &&
            nt::is_any_v<Rotate, const Float33*, Float33> &&
            (REMAP == Remap::H2H || REMAP == Remap::H2HC ||
             REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    template<Remap REMAP, typename Value, typename Scale, typename Rotate>
    constexpr bool is_valid_insert_interpolate_v =
            nt::is_any_v<Value, f32, f64, c32, c64> &&
            nt::is_any_v<Scale, const Float22*, Float22> &&
            nt::is_any_v<Rotate, const Float33*, Float33> &&
            (REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    template<Remap REMAP, typename Value, typename Scale, typename Rotate>
    constexpr bool is_valid_extract_v = is_valid_insert_interpolate_v<REMAP, Value, Scale, Rotate>;

    template<Remap REMAP, typename Value, typename Scale0, typename Scale1, typename Rotate0, typename Rotate1>
    constexpr bool is_valid_insert_insert_extract_v =
            nt::is_any_v<Value, f32, f64, c32, c64> &&
            nt::is_any_v<Scale0, const Float22*, Float22> &&
            nt::is_any_v<Scale1, const Float22*, Float22> &&
            nt::is_any_v<Rotate0, const Float33*, Float33> &&
            nt::is_any_v<Rotate1, const Float33*, Float33> &&
            (REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    template<Remap REMAP, typename Value, typename Scale, typename Rotate>
    constexpr bool is_valid_insert_interpolate_texture_v =
            nt::is_any_v<Value, f32, c32> &&
            nt::is_any_v<Scale, const Float22*, Float22> &&
            nt::is_any_v<Rotate, const Float33*, Float33> &&
            (REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    template<Remap REMAP, typename Value, typename Scale, typename Rotate>
    constexpr bool is_valid_extract_texture_v =
            is_valid_extract_v<REMAP, Value, Scale, Rotate> &&
            !nt::is_any_v<Value, f64, c64>;

    template<Remap REMAP, typename Value, typename Scale0, typename Scale1, typename Rotate0, typename Rotate1>
    constexpr bool is_valid_insert_insert_extract_texture_v =
            nt::is_any_v<Value, f32, c32> &&
            nt::is_any_v<Scale0, const Float22*, Float22> &&
            nt::is_any_v<Scale1, const Float22*, Float22> &&
            nt::is_any_v<Rotate0, const Float33*, Float33> &&
            nt::is_any_v<Rotate1, const Float33*, Float33> &&
            (REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    // The gridding/rasterization kernel is a trilinear pulse.
    // The total weight within the 2x2x2 cube is 1.
    template<typename Int, typename Coord>
    NOA_IHD constexpr void set_rasterization_weights(
            const Vec3<Int>& base0,
            const Vec3<Coord>& freq,
            Coord o_weights[2][2][2]
    ) {
        // So if the coordinate is centered in the bottom left corner of the cube (base0),
        // i.e., its decimal is 0, the corresponding fraction for this element should be 1.
        Vec3<Coord> fraction[2];
        fraction[1] = freq - base0.template as<Coord>();
        fraction[0] = 1.f - fraction[1];
        for (Int w = 0; w < 2; ++w)
            for (Int v = 0; v < 2; ++v)
                for (Int u = 0; u < 2; ++u)
                    o_weights[w][v][u] = fraction[w][0] * fraction[v][1] * fraction[u][2];
    }

    // Spread the value within a 2x2x2 centered on a particular frequency using linear interpolation.
    // This is called gridding, but is also referred as rasterization with antialiasing.
    // "frequency" is the frequency, in samples, centered on DC, with negative frequencies on the left.
    template<bool IS_GRID_CENTERED, typename GridAccessor, typename SInt, typename Value, typename Coord>
    NOA_IHD void rasterize_on_3d_grid(
            const GridAccessor& grid,
            const Shape3<SInt>& grid_shape,
            Value value,
            const Vec3<Coord>& frequency // in samples
    ) {
        using real_t = nt::value_type_t<Value>;
        const auto base0 = noa::math::floor(frequency).template as<SInt>();

        Coord kernel[2][2][2]; // 2x2x2 trilinear weights
        set_rasterization_weights(base0, frequency, kernel);

        for (SInt w = 0; w < 2; ++w) {
            for (SInt v = 0; v < 2; ++v) {
                for (SInt u = 0; u < 2; ++u) {
                    const SInt idx_w = noa::fft::frequency2index<IS_GRID_CENTERED>(base0[0] + w, grid_shape[0]);
                    const SInt idx_v = noa::fft::frequency2index<IS_GRID_CENTERED>(base0[1] + v, grid_shape[1]);
                    const SInt idx_u = base0[2] + u;

                    if (idx_w >= 0 && idx_w < grid_shape[0] &&
                        idx_v >= 0 && idx_v < grid_shape[1] &&
                        idx_u >= 0 && idx_u < grid_shape[2]) {
                        const auto fraction = static_cast<real_t>(kernel[w][v][u]);
                        noa::details::atomic_add(grid, value * fraction, idx_w, idx_v, idx_u);
                    }
                }
            }
        }

        // The gridding doesn't preserve the hermitian symmetry, so enforce it on the redundant X==0 ZY plane.
        // So if a side of this plane was modified, add the conjugate at (x=0, -y, -z) with the same fraction.
        if (base0[2] == 0) {
            if constexpr (nt::is_complex_v<Value>)
                value.imag = -value.imag;
            for (SInt w = 0; w < 2; ++w) {
                for (SInt v = 0; v < 2; ++v) {
                    const SInt idx_w = noa::fft::frequency2index<IS_GRID_CENTERED>(-(base0[0] + w), grid_shape[0]);
                    const SInt idx_v = noa::fft::frequency2index<IS_GRID_CENTERED>(-(base0[1] + v), grid_shape[1]);

                    if (idx_w >= 0 && idx_w < grid_shape[0] &&
                        idx_v >= 0 && idx_v < grid_shape[1]) {
                        const auto fraction = static_cast<real_t>(kernel[w][v][0]);
                        noa::details::atomic_add(grid, value * fraction, idx_w, idx_v, SInt{0});
                    }
                }
            }
        }
    }

    // Transforms a 3d fftfreq representing the slice, to its 3d fftfreq in the grid.
    // This is a forward transformation of the frequency, but because it is in Fourier-space,
    // the real-space scaling is inverted.
    template<typename Coord, typename ScaleOrEmtpy, typename Rotate, typename Int, typename EWSOrEmpty>
    constexpr NOA_IHD Vec3<Coord>
    transform_slice2grid(
            Vec2<Coord> fftfreq,
            const ScaleOrEmtpy& inv_scaling_matrices, // 2x2
            const Rotate& fwd_rotation_matrices, // 3x3
            Int batch_index,
            EWSOrEmpty inv_ews_diameter
    ) {
        // If we apply the EWS curvature, the scaling factors should be corrected
        // before applying the curvature, and therefore before applying the rotation.
        // That way, we use the correct frequencies to compute the EWS, e.g., resulting
        // in a spherical EWS even under anisotropic magnification.
        if constexpr (std::is_pointer_v<ScaleOrEmtpy>)
            fftfreq = inv_scaling_matrices[batch_index] * fftfreq;
        else if constexpr (!std::is_empty_v<ScaleOrEmtpy>)
            fftfreq = inv_scaling_matrices * fftfreq;

        // TODO We use the Small Angle Approximation to compute the EWS curvature,
        //      so the frequency (u,v) is unchanged. Look at the cisTEM implementation
        //      to remove this approximation? RELION shows that even for low voltages
        //      and large boxes, it is probably not worth it though.
        Vec3<Coord> fftfreq_3d{0, fftfreq[0], fftfreq[1]};
        if constexpr (!std::is_empty_v<EWSOrEmpty>)
            fftfreq_3d[0] = noa::math::sum(inv_ews_diameter * fftfreq * fftfreq);

        if constexpr (std::is_pointer_v<Rotate>)
            return fwd_rotation_matrices[batch_index] * fftfreq_3d;
        else
            return fwd_rotation_matrices * fftfreq_3d;
    }

    // Same as above, but in the other direction.
    template<typename Coord, typename ScaleOrEmpty, typename Rotate, typename Int, typename EWSOrEmpty>
    constexpr NOA_IHD Pair<float, Vec2<Coord>>
    transform_grid2slice(
            Vec3<Coord> frequency,
            const ScaleOrEmpty& fwd_scaling_matrices,
            const Rotate& inv_rotation_matrices,
            Int index, EWSOrEmpty inv_ews_diameter
    ) {
        if constexpr (std::is_pointer_v<Rotate>)
            frequency = inv_rotation_matrices[index] * frequency;
        else
            frequency = inv_rotation_matrices * frequency;

        Vec2<Coord> freq_2d{frequency[1], frequency[2]};
        Coord freq_z = frequency[0];
        if constexpr (!std::is_empty_v<EWSOrEmpty>)
            freq_z -= noa::math::sum(inv_ews_diameter * freq_2d * freq_2d);

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
    template<typename Interpolator, typename Int, typename Coord>
    constexpr NOA_HD auto interpolate_slice_value(
            Vec2<Coord> frequency, const Vec2<Coord>& slice_shape, Coord center_y,
            const Interpolator& interpolator, Int batch
    ) {
        using value_t = typename Interpolator::value_type;
        using real_t = nt::value_type_t<value_t>;
        constexpr bool IS_COMPLEX = nt::is_complex_v<value_t>;

        // From the normalized frequency to the multidimensional index.
        real_t conj = 1;
        if (frequency[1] < 0) {
            frequency = -frequency;
            if constexpr (IS_COMPLEX)
                conj = -1;
        }
        frequency *= slice_shape;
        frequency[0] += center_y;

        value_t value = interpolator(frequency, batch);
        if constexpr (IS_COMPLEX)
            value.imag *= conj;
        else
            (void) conj;
        return value;
    }

    // Interpolates the value at a given frequency, in cycle per pixel.
    template<typename Interpolator, typename Coord>
    constexpr NOA_HD auto interpolate_grid_value(
            Vec3<Coord> frequency, const Vec3<Coord>& target_shape,
            const Vec2<Coord>& grid_center_zy, const Interpolator& interpolator
    ) {
        using value_t = typename Interpolator::value_type;
        using real_t = nt::value_type_t<value_t>;
        constexpr bool IS_COMPLEX = nt::is_complex_v<value_t>;

        real_t conj = 1;
        if (frequency[2] < 0) {
            frequency = -frequency;
            if constexpr(IS_COMPLEX)
                conj = -1;
        }
        frequency *= target_shape;
        frequency[0] += grid_center_zy[0];
        frequency[1] += grid_center_zy[1];

        value_t value = interpolator(frequency);
        if constexpr(IS_COMPLEX)
            value.imag *= conj;
        else
            (void) conj;
        return value;
    }

    // Windowed-sinc. This function assumes fftfreq <= fftfreq_blackman,
    // above that the blackman window will start again.
    template<typename Coord>
    NOA_FHD Coord windowed_sinc(Coord fftfreq, Coord fftfreq_sinc, Coord fftfreq_blackman) {
        // https://www.desmos.com/calculator/tu5b8aqg2e
        constexpr Coord PI = noa::math::Constant<Coord>::PI;
        fftfreq *= PI;
        const auto sinc = noa::math::sinc(fftfreq / fftfreq_sinc);
        const auto blackman_cutoff = fftfreq / fftfreq_blackman;
        const auto blackman = static_cast<Coord>(0.42) +
                              static_cast<Coord>(0.5) * noa::math::cos(blackman_cutoff) +
                              static_cast<Coord>(0.08) * noa::math::cos(2 * blackman_cutoff);
        return sinc * blackman;
    }

    // This is only used for the Fourier extraction step.
    // The window is always an odd-numbered size.
    template<typename Int, typename Coord, typename = std::enable_if_t<nt::is_real_v<Coord> && nt::is_int_v<Int>>>
    Int blackman_window_size(Coord fftfreq_blackman, Coord spectrum_size) {
        // Given a blackman window in range [0, fftfreq_blackman] and a spectrum logical-size
        // (the z size in our case), what is the size of the blackman window, in elements.
        // For instance:
        //  spectrum_size=10, fftfreq_blackman=0.23
        //  rfftfreq=[0.,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
        //  rfftfreq_samples=4.6->5, window_size=11
        //  computed_window=[-0.25,-0.2,-0.15,-0.1,-0.05,0.,0.05,0.1,0.15,0.2,0.25]
        auto rfftfreq_samples = static_cast<f64>(spectrum_size) * static_cast<f64>(fftfreq_blackman);
        rfftfreq_samples = noa::math::ceil(rfftfreq_samples); // include last fraction
        const auto rfftfreq_samples_int = noa::math::max(Int{1}, static_cast<Int>(rfftfreq_samples));
        auto window_size = 2 * (rfftfreq_samples_int) + 1;

        // Truncate the edges because at these indexes, the window is 0, so there's no need to compute it.
        // So using the same example, computed_window=[-0.2,-0.15,-0.1,-0.05,0.,0.05,0.1,0.15,0.2]
        return window_size - 2;
    }

    // This is only used for the Fourier extraction step.
    // Given the iwise-index w and the blackman window size, return the fftfreq offset. For instance,
    // window_size=9: w=[0..8] -> [-0.2,-0.15,-0.1,-0.05,0.,0.05,0.1,0.15,0.2]
    template<typename Coord, typename Int, typename = std::enable_if_t<nt::is_real_v<Coord> && nt::is_int_v<Int>>>
    NOA_FHD Coord w_index_to_fftfreq_offset(Int w, Int window_size, Coord spectrum_size) {
        return static_cast<Coord>(w - window_size / 2) / spectrum_size;
    }
}

namespace noa::algorithm::geometry {
    // Direct Fourier insertion, using rasterization.
    template<noa::fft::Remap REMAP,
            typename Index, typename Offset, typename Value, typename Coord,
            typename ScaleOrEmpty, typename Rotate, typename EWSOrEmpty,
            typename SliceAccessorOrValue>
    class FourierInsertRasterize {
        using Layout = ::noa::fft::Layout;
        static constexpr auto u8_REMAP = static_cast<uint8_t>(REMAP);
        static constexpr bool IS_SLICE_CENTERED = u8_REMAP & Layout::SRC_CENTERED;
        static constexpr bool IS_GRID_CENTERED = u8_REMAP & Layout::DST_CENTERED;
        static_assert(u8_REMAP & Layout::SRC_HALF && u8_REMAP & Layout::DST_HALF);

        static_assert(nt::is_real_or_complex_v<Value>);
        static_assert(nt::is_real_v<Coord>);
        static_assert(nt::is_any_v<ScaleOrEmpty, Mat22<Coord>, const Mat22<Coord>*, Empty>);
        static_assert(nt::is_any_v<EWSOrEmpty, Coord, Vec2<Coord>, Empty>);
        static_assert(nt::is_any_v<Rotate, Mat33<Coord>, const Mat33<Coord>*>);
        static_assert(nt::is_any_v<Index, i32, i64>);
        static_assert(nt::is_any_v<SliceAccessorOrValue, AccessorRestrict<const Value, 3, Offset>, Value>);

        using index_type = Index;
        using offset_type = Offset;
        using value_type = Value;
        using coord_type = Coord;
        using scale_or_empty_type = ScaleOrEmpty;
        using rotate_type = Rotate;
        using ews_or_empty_type = EWSOrEmpty;

        using coord2_type = Vec2<coord_type>;
        using coord3_type = Vec3<coord_type>;
        using shape3_type = Shape3<index_type>;

        using slice_accessor_or_value_type = SliceAccessorOrValue;
        using grid_accessor_type = AccessorRestrict<value_type, 3, offset_type>;
        using real_value_type = nt::value_type_t<value_type>;

    public:
        FourierInsertRasterize(
                const slice_accessor_or_value_type& slice,
                const Shape4<index_type>& slice_shape,
                const grid_accessor_type& grid,
                const Shape4<index_type>& grid_shape,
                const scale_or_empty_type& inv_scaling_matrices,
                const rotate_type& fwd_rotation_matrices,
                coord_type fftfreq_cutoff,
                const Shape4<index_type>& target_shape,
                const ews_or_empty_type& ews_radius
        )
                : m_slice(slice),
                  m_grid(grid),
                  m_fwd_rotation_matrices(fwd_rotation_matrices),
                  m_inv_scaling_matrices(inv_scaling_matrices),
                  m_grid_shape(grid_shape.pop_front()) {

            NOA_ASSERT(noa::all(slice_shape > 0) && noa::all(grid_shape > 0));
            NOA_ASSERT(slice_shape[1] == 1);
            NOA_ASSERT(grid_shape[0] == 1);
            if constexpr (std::is_pointer_v<rotate_type>) {
                NOA_ASSERT(fwd_rotation_matrices != nullptr);
            }

            const auto slice_shape_2d = slice_shape.filter(2, 3);
            m_slice_size_y = slice_shape_2d[0];
            m_f_slice_shape = coord2_type(slice_shape_2d.vec());

            // Use the grid shape as backup.
            const auto target_shape_3d = noa::any(target_shape == 0) ? m_grid_shape : target_shape.pop_front();
            m_f_target_shape = coord3_type(target_shape_3d.vec());

            // Using the small-angle approximation, Z = wavelength / 2 * (X^2 + Y^2).
            // See doi:10.1016/S0304-3991(99)00120-5 for a derivation.
            if constexpr (!std::is_empty_v<ews_or_empty_type>)
                m_ews_diam_inv = noa::any(ews_radius != 0) ? 1 / (2 * ews_radius) : ews_or_empty_type{};

            m_fftfreq_cutoff_sqd = noa::math::max(fftfreq_cutoff, coord_type{0});
            m_fftfreq_cutoff_sqd *= m_fftfreq_cutoff_sqd;
        }

        // For every pixel of every central slice to insert.
        NOA_HD void operator()(index_type batch, index_type y, index_type u) const noexcept { // x == u
            // We compute the forward transformation and use normalized frequencies.
            // The oversampling is implicitly handled when scaling back to the target shape.
            const index_type v = noa::fft::index2frequency<IS_SLICE_CENTERED>(y, m_slice_size_y);
            const auto freq_2d = coord2_type{v, u} / m_f_slice_shape;
            coord3_type freq_3d = details::transform_slice2grid(
                    freq_2d, m_inv_scaling_matrices, m_fwd_rotation_matrices, batch, m_ews_diam_inv);

            // The frequency rate won't change from that point, so check for the cutoff.
            if (noa::math::dot(freq_3d, freq_3d) > m_fftfreq_cutoff_sqd)
                return;

            // Handle the non-redundancy in x.
            real_value_type conj = 1;
            if (freq_3d[2] < 0) {
                freq_3d = -freq_3d;
                if constexpr (nt::is_complex_v<value_type>)
                    conj = -1;
            }

            // Scale back to the target shape.
            freq_3d *= m_f_target_shape;

            // At this point, we know we are going to use the slice value.
            value_type value;
            if constexpr (std::is_same_v<value_type, slice_accessor_or_value_type>)
                value = m_slice;
            else
                value = m_slice(batch, y, u);
            if constexpr (nt::is_complex_v<value_type>)
                value.imag *= conj;
            else
                (void) conj;

            details::rasterize_on_3d_grid<IS_GRID_CENTERED>(m_grid, m_grid_shape, value, freq_3d);
        }

    private:
        slice_accessor_or_value_type m_slice;
        grid_accessor_type m_grid;
        rotate_type m_fwd_rotation_matrices;
        NOA_NO_UNIQUE_ADDRESS scale_or_empty_type m_inv_scaling_matrices;

        shape3_type m_grid_shape;
        index_type m_slice_size_y;
        coord3_type m_f_target_shape;
        coord2_type m_f_slice_shape;
        NOA_NO_UNIQUE_ADDRESS ews_or_empty_type m_ews_diam_inv{};
        coord_type m_fftfreq_cutoff_sqd;
    };

    // Direct Fourier insertion, but this time looping through the grid.
    // In practice, it allows to give an explicit "thickness" to the central slices.
    // It also "looks" better (more symmetric; better/smoother aliasing) than rasterization, but it's much slower.
    // One limitation is that it requires the input slices to be centered.
    template<noa::fft::Remap REMAP,
             typename Index, typename Value, typename Offset,
             typename ScaleOrEmpty, typename Rotate, typename EWSOrEmpty,
             typename SliceInterpolator>
    class FourierInsertInterpolate {
        using Layout = ::noa::fft::Layout;
        static constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        static constexpr bool IS_GRID_CENTERED = REMAP_ & Layout::DST_CENTERED;
        static_assert(REMAP_ & Layout::SRC_CENTERED &&
                      REMAP_ & Layout::SRC_HALF &&
                      REMAP_ & Layout::DST_HALF);

        static_assert(nt::is_real_or_complex_v<Value>);
        static_assert(nt::is_any_v<Index, i32, i64>);

        using index_type = Index;
        using value_type = Value;
        using offset_type = Offset;
        using scale_or_empty_type = ScaleOrEmpty;
        using rotate_type = Rotate;
        using ews_or_empty_type = EWSOrEmpty;
        using slice_interpolator_type = SliceInterpolator;

        using coord_type = typename SliceInterpolator::coord_type;
        using coord2_type = Vec2<coord_type>;
        using coord3_type = Vec3<coord_type>;
        using shape3_type = Shape3<index_type>;

        static_assert(nt::is_any_v<ScaleOrEmpty, Mat22<coord_type>, const Mat22<coord_type>*, Empty>);
        static_assert(nt::is_any_v<EWSOrEmpty, coord_type, Vec2<coord_type>, Empty>);
        static_assert(nt::is_any_v<Rotate, Mat33<coord_type>, const Mat33<coord_type>*>);

        using grid_accessor_type = AccessorRestrict<value_type, 3, offset_type>;
        using real_type = nt::value_type_t<value_type>;

    public:
        FourierInsertInterpolate(
                const slice_interpolator_type& slice,
                const Shape4<index_type>& slice_shape,
                const grid_accessor_type& grid,
                const Shape4<index_type>& grid_shape,
                const scale_or_empty_type& fwd_scaling_matrices,
                const rotate_type& inv_rotation_matrices,
                coord_type fftfreq_cutoff,
                coord_type fftfreq_sinc,
                coord_type fftfreq_blackman,
                const Shape4<index_type>& target_shape,
                const ews_or_empty_type& ews_radius
        )
                : m_slice(slice), m_grid(grid),
                  m_inv_rotation_matrices(inv_rotation_matrices),
                  m_fwd_scaling_matrices(fwd_scaling_matrices),
                  m_grid_shape(grid_shape.pop_front()),
                  m_slice_count(slice_shape[0]) {

            NOA_ASSERT(noa::all(slice_shape > 0) && noa::all(grid_shape > 0));
            NOA_ASSERT(slice_shape[1] == 1);
            NOA_ASSERT(grid_shape[0] == 1);
            if constexpr (std::is_pointer_v<rotate_type>) {
                NOA_ASSERT(inv_rotation_matrices != nullptr);
            }

            const auto slice_shape_2d = slice_shape.filter(2, 3);
            m_f_slice_shape = coord2_type(slice_shape_2d.vec());
            m_f_slice_y_center = static_cast<coord_type>(slice_shape_2d[0] / 2);

            const auto l_target_shape = noa::any(target_shape == 0) ? m_grid_shape : target_shape.pop_front();
            m_f_target_shape = coord3_type(l_target_shape.vec());

            // Using the small-angle approximation, Z = wavelength / 2 * (X^2 + Y^2).
            // See doi:10.1016/S0304-3991(99)00120-5 for a derivation.
            if constexpr (!std::is_empty_v<ews_or_empty_type>)
                m_ews_diam_inv = noa::any(ews_radius != 0) ? 1 / (2 * ews_radius) : ews_or_empty_type{};

            m_fftfreq_cutoff_sqd = noa::math::max(fftfreq_cutoff, coord_type{0});
            m_fftfreq_cutoff_sqd *= m_fftfreq_cutoff_sqd;

            // Clamp the windowed-sinc to ensure it's at least one pixel thick.
            const auto max_output_size = static_cast<coord_type>(noa::math::min(l_target_shape));
            m_fftfreq_sinc = noa::math::max(fftfreq_sinc, 1 / max_output_size);
            m_fftfreq_blackman = noa::math::max(fftfreq_blackman, 1 / max_output_size);
        }

        // For every voxel of the grid.
        NOA_HD void operator()(index_type z, index_type y, index_type u) const noexcept { // x == u
            const index_type w = noa::fft::index2frequency<IS_GRID_CENTERED>(z, m_grid_shape[0]);
            const index_type v = noa::fft::index2frequency<IS_GRID_CENTERED>(y, m_grid_shape[1]);
            const auto orig_freq = coord3_type{w, v, u} / m_f_target_shape;
            if (noa::math::dot(orig_freq, orig_freq) > m_fftfreq_cutoff_sqd)
                return;

            value_type value{0};
            for (index_type i = 0; i < m_slice_count; ++i) {
                const auto [freq_z, freq_2d] = details::transform_grid2slice(
                        orig_freq, m_fwd_scaling_matrices, m_inv_rotation_matrices, i, m_ews_diam_inv);

                // If voxel is not affected by the slice, skip.
                value_type i_value{0};
                if (noa::math::abs(freq_z) <= m_fftfreq_blackman) {
                    i_value = details::interpolate_slice_value(
                            freq_2d, m_f_slice_shape, m_f_slice_y_center, m_slice, i);
                    const auto weight = details::windowed_sinc(
                            freq_z, m_fftfreq_sinc, m_fftfreq_blackman);
                    i_value *= static_cast<real_type>(weight);
                }
                value += i_value;
            }

            // The transformation preserves the hermitian symmetry, so there's nothing else to do.
            m_grid(z, y, u) += value;
        }

    private:
        slice_interpolator_type m_slice;
        grid_accessor_type m_grid;
        rotate_type m_inv_rotation_matrices;
        NOA_NO_UNIQUE_ADDRESS scale_or_empty_type m_fwd_scaling_matrices;

        shape3_type m_grid_shape;
        index_type m_slice_count;
        coord3_type m_f_target_shape;
        coord2_type m_f_slice_shape;
        NOA_NO_UNIQUE_ADDRESS ews_or_empty_type m_ews_diam_inv{};
        coord_type m_f_slice_y_center;
        coord_type m_fftfreq_cutoff_sqd;
        coord_type m_fftfreq_sinc;
        coord_type m_fftfreq_blackman;
    };

    // The exact same transformation as insertion with gridding is applied here,
    // but instead of inserting the transformed slices into the grid,
    // the transformed slices are extracted from the grid.
    template<noa::fft::Remap REMAP,
             typename Index, typename Value, typename Offset,
             typename ScaleOrEmpty, typename Rotate, typename EWSOrEmpty,
             typename GridInterpolator>
    class FourierExtract {
        using Layout = ::noa::fft::Layout;
        static constexpr auto u8_REMAP = static_cast<uint8_t>(REMAP);
        static constexpr bool IS_SLICE_CENTERED = u8_REMAP & Layout::DST_CENTERED;
        static_assert(u8_REMAP & Layout::SRC_CENTERED &&
                      u8_REMAP & Layout::SRC_HALF &&
                      u8_REMAP & Layout::DST_HALF);

        static_assert(nt::is_real_or_complex_v<Value>);
        static_assert(nt::is_any_v<Index, i32, i64>);

        using index_type = Index;
        using value_type = Value;
        using offset_type = Offset;
        using scale_or_empty_type = ScaleOrEmpty;
        using rotate_type = Rotate;
        using ews_or_empty_type = EWSOrEmpty;
        using grid_interpolator_type = GridInterpolator;

        using coord_type = typename GridInterpolator::coord_type;
        using coord2_type = Vec2<coord_type>;
        using coord3_type = Vec3<coord_type>;
        using slice_accessor_type = AccessorRestrict<value_type, 3, offset_type>;
        using real_type = nt::value_type_t<value_type>;

        static_assert(nt::is_any_v<ScaleOrEmpty, Empty, Mat22<coord_type>, const Mat22<coord_type>*>);
        static_assert(nt::is_any_v<EWSOrEmpty, Empty, coord_type, coord2_type>);
        static_assert(nt::is_any_v<Rotate, Mat33<coord_type>, const Mat33<coord_type>*>);

    public:
        FourierExtract(
                const grid_interpolator_type& grid,
                const Shape4<index_type>& grid_shape,
                const slice_accessor_type& slice,
                const Shape4<index_type>& slice_shape,
                const scale_or_empty_type& inv_scaling_matrices,
                const rotate_type& fwd_rotation_matrices,
                coord_type fftfreq_cutoff,
                coord_type fftfreq_sinc,
                coord_type fftfreq_blackman,
                const Shape4<index_type>& target_shape,
                const ews_or_empty_type& ews_radius
        )
                : m_grid(grid), m_slice(slice),
                  m_inv_scaling_matrices(inv_scaling_matrices),
                  m_fwd_rotation_matrices(fwd_rotation_matrices) {

            NOA_ASSERT(noa::all(slice_shape > 0) && noa::all(grid_shape > 0));
            NOA_ASSERT(slice_shape[1] == 1);
            NOA_ASSERT(grid_shape[0] == 1);
            if constexpr (std::is_pointer_v<rotate_type>) {
                NOA_ASSERT(fwd_rotation_matrices != nullptr);
            }

            const auto slice_shape_2d = slice_shape.filter(2, 3);
            m_slice_size_y = slice_shape_2d[0];
            m_f_slice_shape = coord2_type(slice_shape_2d.vec());

            // Use the grid shape as backup.
            const auto grid_shape_3d = grid_shape.pop_front();
            const auto target_shape_3d = noa::any(target_shape == 0) ? grid_shape_3d : target_shape.pop_front();
            m_f_target_shape = coord3_type(target_shape_3d.vec());
            m_f_grid_zy_center = coord2_type((grid_shape_3d.filter(0, 1) / 2).vec()); // grid ZY center

            // Using the small-angle approximation, Z = wavelength / 2 * (X^2 + Y^2).
            // See doi:10.1016/S0304-3991(99)00120-5 for a derivation.
            if constexpr (!std::is_empty_v<ews_or_empty_type>)
                m_ews_diam_inv = noa::any(ews_radius != 0) ? 1 / (2 * ews_radius) : ews_or_empty_type{};

            m_fftfreq_cutoff_sqd = noa::math::max(fftfreq_cutoff, coord_type{0});
            m_fftfreq_cutoff_sqd *= m_fftfreq_cutoff_sqd;

            // This is along the z of the grid.
            m_fftfreq_sinc = noa::math::max(fftfreq_sinc, 1 / m_f_target_shape[0]);
            m_fftfreq_blackman = noa::math::max(fftfreq_blackman, 1 / m_f_target_shape[0]);
            m_blackman_size = details::blackman_window_size<index_type>(m_fftfreq_blackman, m_f_target_shape[0]);
        }

        [[nodiscard]] constexpr index_type windowed_sinc_size() const noexcept { return m_blackman_size; }

        // For every pixel of every slice to extract.
        NOA_HD void operator()(index_type batch, index_type y, index_type u) const noexcept {
            const coord3_type freq_3d = compute_fftfreq_in_volume_(batch, y, u);
            m_slice(batch, y, u) =
                    noa::math::dot(freq_3d, freq_3d) > m_fftfreq_cutoff_sqd ?
                    value_type{0} :
                    details::interpolate_grid_value(freq_3d, m_f_target_shape, m_f_grid_zy_center, m_grid);
        }

        // For every pixel of every slice to extract.
        // w is the index within the windowed-sinc convolution along the z of the grid.
        NOA_HD void operator()(index_type batch, index_type w, index_type y, index_type u) const noexcept {
            coord3_type freq_3d = compute_fftfreq_in_volume_(batch, y, u);

            // Additional z component, within the grid coordinate system.
            const auto z_fftfreq_offset = details::w_index_to_fftfreq_offset(w, m_blackman_size, m_f_target_shape[0]);
            freq_3d[0] += z_fftfreq_offset;

            if (noa::math::dot(freq_3d, freq_3d) > m_fftfreq_cutoff_sqd)
                return;

            const auto value = details::interpolate_grid_value(
                    freq_3d, m_f_target_shape, m_f_grid_zy_center, m_grid);
            const auto weight = details::windowed_sinc(
                    z_fftfreq_offset, m_fftfreq_sinc, m_fftfreq_blackman);
            noa::details::atomic_add(m_slice, value * static_cast<real_type>(weight), batch, y, u); // weighted-mean
        }

    private:
        // The indexes give us the fftfreq in the coordinate system of the slice to extract.
        // This function transforms this fftfreq to the coordinate system of the volume.
        NOA_HD coord3_type compute_fftfreq_in_volume_(
                index_type batch, index_type y, index_type u
        ) const noexcept {
            const index_type v = noa::fft::index2frequency<IS_SLICE_CENTERED>(y, m_slice_size_y);
            const auto fftfreq_2d = coord2_type{v, u} / m_f_slice_shape;
            coord3_type fftfreq_3d = details::transform_slice2grid(
                    fftfreq_2d, m_inv_scaling_matrices, m_fwd_rotation_matrices, batch, m_ews_diam_inv);
            return fftfreq_3d;
        }

    private:
        grid_interpolator_type m_grid;
        slice_accessor_type m_slice;
        NOA_NO_UNIQUE_ADDRESS scale_or_empty_type m_inv_scaling_matrices;
        rotate_type m_fwd_rotation_matrices;

        coord3_type m_f_target_shape;
        coord2_type m_f_slice_shape;
        coord2_type m_f_grid_zy_center;
        index_type m_slice_size_y;
        NOA_NO_UNIQUE_ADDRESS ews_or_empty_type m_ews_diam_inv{};

        coord_type m_fftfreq_cutoff_sqd;
        coord_type m_fftfreq_sinc;
        coord_type m_fftfreq_blackman;
        index_type m_blackman_size;
    };

    // Extracts central-slices from a virtual volume made of (other) central-slices.
    // The extracted slices can be convolved with a windowed-sinc along the z axis of the volume (note that the
    // convolution is reduced to a simple weighted-sum), effectively applying a (smooth) rectangular mask along
    // the z axis and centered on the ifft of the virtual volume.
    template<noa::fft::Remap REMAP,
             typename Index, typename Value, typename Offset,
             typename InputScaleOrEmpty, typename InputRotate,
             typename OutputScaleOrEmpty, typename OutputRotate,
             typename EWSOrEmpty, typename InputSliceInterpolator>
    class FourierInsertExtract {
        using Layout = ::noa::fft::Layout;
        static constexpr auto u8_REMAP = static_cast<uint8_t>(REMAP);
        static constexpr bool IS_OUTPUT_SLICE_CENTERED = u8_REMAP & Layout::DST_CENTERED;
        static_assert((u8_REMAP & Layout::SRC_CENTERED) &&
                      (u8_REMAP & Layout::SRC_HALF) &&
                      (u8_REMAP & Layout::DST_HALF));

        static_assert(nt::is_real_or_complex_v<Value>);
        static_assert(nt::is_any_v<Index, i32, i64>);

        using index_type = Index;
        using value_type = Value;
        using offset_type = Offset;
        using input_scale_or_empty_type = InputScaleOrEmpty;
        using input_rotate_type = InputRotate;
        using output_scale_or_empty_type = OutputScaleOrEmpty;
        using output_rotate_type = OutputRotate;
        using ews_or_empty_type = EWSOrEmpty;
        using input_slice_interpolator_type = InputSliceInterpolator;

        using coord_type = typename InputSliceInterpolator::coord_type;
        using coord2_type = Vec2<coord_type>;
        using coord3_type = Vec3<coord_type>;

        static_assert(nt::is_any_v<InputScaleOrEmpty, Empty, Mat22<coord_type>, const Mat22<coord_type>*>);
        static_assert(nt::is_any_v<OutputScaleOrEmpty, Empty, Mat22<coord_type>, const Mat22<coord_type>*>);
        static_assert(nt::is_any_v<EWSOrEmpty, Empty, coord_type, coord2_type>);
        static_assert(nt::is_any_v<InputRotate, Mat33<coord_type>, const Mat33<coord_type>*>);
        static_assert(nt::is_any_v<OutputRotate, Mat33<coord_type>, const Mat33<coord_type>*>);

        using output_slice_accessor_type = AccessorRestrict<value_type, 3, offset_type>;
        using real_type = nt::value_type_t<value_type>;

    public:
        FourierInsertExtract(
                const input_slice_interpolator_type& input_slices, const Shape4<index_type>& input_shape,
                const output_slice_accessor_type& output_slices, const Shape4<index_type>& output_shape,
                const input_scale_or_empty_type& insert_fwd_scaling_matrices,
                const input_rotate_type& insert_inv_rotation_matrices,
                const output_scale_or_empty_type& extract_inv_scaling_matrices,
                const output_rotate_type& extract_fwd_rotation_matrices,
                coord_type fftfreq_cutoff,
                coord_type insert_fftfreq_sinc,
                coord_type insert_fftfreq_blackman,
                coord_type extract_fftfreq_sinc,
                coord_type extract_fftfreq_blackman,
                bool add_to_output,
                const ews_or_empty_type& ews_radius
        )
                : m_input_slices(input_slices),
                  m_output_slices(output_slices),
                  m_insert_inv_rotation_matrices(insert_inv_rotation_matrices),
                  m_extract_fwd_rotation_matrices(extract_fwd_rotation_matrices),
                  m_insert_fwd_scaling_matrices(insert_fwd_scaling_matrices),
                  m_extract_inv_scaling_matrices(extract_inv_scaling_matrices),
                  m_input_count(input_shape[0]),
                  m_add_to_output(add_to_output) {

            NOA_ASSERT(noa::all(input_shape > 0) && noa::all(output_shape > 0));
            NOA_ASSERT(input_shape[1] == 1 && output_shape[1] == 1);
            if constexpr (std::is_pointer_v<input_rotate_type>) {
                NOA_ASSERT(insert_inv_rotation_matrices != nullptr);
            }
            if constexpr (std::is_pointer_v<output_rotate_type>) {
                NOA_ASSERT(extract_fwd_rotation_matrices != nullptr);
            }

            const auto l_input_shape = input_shape.filter(2, 3);
            const auto l_output_shape = output_shape.filter(2, 3);

            m_f_input_shape = coord2_type(l_input_shape.vec());
            m_f_output_shape = coord2_type(l_output_shape.vec());
            m_output_slice_size_y = l_output_shape[0];
            m_f_input_center_y = static_cast<coord_type>(l_input_shape[0] / 2); // slice Y center

            // Using the small-angle approximation, Z = wavelength / 2 * (X^2 + Y^2).
            // See doi:10.1016/S0304-3991(99)00120-5 for a derivation.
            if constexpr (!std::is_empty_v<ews_or_empty_type>)
                m_ews_diam_inv = noa::any(ews_radius != 0) ? 1 / (2 * ews_radius) : ews_or_empty_type{};

            m_fftfreq_cutoff_sqd = noa::math::max(fftfreq_cutoff, coord_type{0});
            m_fftfreq_cutoff_sqd *= m_fftfreq_cutoff_sqd;

            // Clamp the slice is thick enough for these dimensions.
            // Of course, we have no z here, but using the smallest axis as z-size is a good fallback.
            m_volume_z = static_cast<coord_type>(noa::math::min(l_output_shape));
            m_insert_fftfreq_sinc = noa::math::max(insert_fftfreq_sinc, 1 / m_volume_z);
            m_insert_fftfreq_blackman = noa::math::max(insert_fftfreq_blackman, 1 / m_volume_z);
            m_extract_fftfreq_sinc = noa::math::max(extract_fftfreq_sinc, 1 / m_volume_z);
            m_extract_fftfreq_blackman = noa::math::max(extract_fftfreq_blackman, 1 / m_volume_z);
            m_extract_blackman_size = details::blackman_window_size<index_type>(
                    extract_fftfreq_blackman, m_volume_z);
        }

        [[nodiscard]] constexpr auto windowed_sinc_size() const noexcept -> index_type {
            return m_extract_blackman_size;
        }

        // Should be called for every pixel of every slice to extract.
        NOA_HD void operator()(index_type batch, index_type y, index_type u) const noexcept {
            const coord3_type fftfreq_3d = compute_fftfreq_in_volume_(batch, y, u);

            if (noa::math::dot(fftfreq_3d, fftfreq_3d) > m_fftfreq_cutoff_sqd) {
                if (!m_add_to_output)
                    m_output_slices(batch, y, u) = value_type{0};
                return;
            }

            const value_type value = sample_virtual_volume_(fftfreq_3d);

            // The transformation preserves the hermitian symmetry, so there's nothing else to do.
            if (m_add_to_output)
                m_output_slices(batch, y, u) += value;
            else
                m_output_slices(batch, y, u) = value;
        }

        // Should be called for every pixel of every slice to extract and for every element in the z-windowed-sinc.
        NOA_HD void operator()(index_type batch, index_type w, index_type y, index_type u) const noexcept {
            coord3_type fftfreq_3d = compute_fftfreq_in_volume_(batch, y, u);

            // Get and add the volume z-offset for the z-windowed-sinc.
            const auto fftfreq_z_offset = details::w_index_to_fftfreq_offset(
                    w, m_extract_blackman_size, m_volume_z);
            fftfreq_3d[0] += fftfreq_z_offset;

            if (noa::math::dot(fftfreq_3d, fftfreq_3d) > m_fftfreq_cutoff_sqd)
                return;

            // Sample the value at this frequency, as well as the value of the z-windowed-sinc.
            const value_type value = sample_virtual_volume_(fftfreq_3d);
            const auto weight = details::windowed_sinc(
                    fftfreq_z_offset, m_extract_fftfreq_sinc, m_extract_fftfreq_blackman);

            // The transformation preserves the hermitian symmetry, so there's nothing else to do.
            noa::details::atomic_add(m_output_slices, value * static_cast<real_type>(weight), batch, y, u);
        }

    private:
        // The indexes give us the fftfreq in the coordinate system of the slice to extract.
        // This function transforms this fftfreq to the coordinate system of the virtual volume.
        NOA_HD coord3_type compute_fftfreq_in_volume_(
                index_type batch, index_type y, index_type u
        ) const noexcept {
            const index_type v = noa::fft::index2frequency<IS_OUTPUT_SLICE_CENTERED>(y, m_output_slice_size_y);
            auto fftfreq_2d = coord2_type{v, u} / m_f_output_shape;
            coord3_type fftfreq_3d = details::transform_slice2grid(
                    fftfreq_2d, m_extract_inv_scaling_matrices, m_extract_fwd_rotation_matrices,
                    batch, m_ews_diam_inv);
            return fftfreq_3d;
        }

        // Add the contribution of every input central-slice.
        // In other words, sample the signal at this 3d frequency within the virtual volume.
        NOA_HD value_type sample_virtual_volume_(coord3_type fftfreq_3d) const noexcept {
            value_type value{0};
            // 0. For every slice to insert...
            for (index_type i = 0; i < m_input_count; ++i) {
                // 1. Project the 3d frequency onto that input-slice.
                //    fftfreq_z is along the normal of that input-slice.
                const auto [fftfreq_z, fftfreq_yx] = details::transform_grid2slice(
                        fftfreq_3d, m_insert_fwd_scaling_matrices, m_insert_inv_rotation_matrices, i, m_ews_diam_inv);

                // 2. Sample the input-slice at fftfreq_yx and weight using a windowed-sinc at fftfreq_z.
                //    Compute only if voxel is affected by this slice.
                //    If we fall exactly at the blackman cutoff, the value is 0, so exclude the equality case.
                value_type i_value{0};
                if (noa::math::abs(fftfreq_z) < m_insert_fftfreq_blackman) {
                    i_value = details::interpolate_slice_value(
                            fftfreq_yx, m_f_input_shape, m_f_input_center_y, m_input_slices, i);
                    const auto weight = details::windowed_sinc(
                            fftfreq_z, m_insert_fftfreq_sinc, m_insert_fftfreq_blackman);
                    i_value *= static_cast<real_type>(weight);
                }
                value += i_value;
            }
            return value;
        }

    private:
        input_slice_interpolator_type m_input_slices;
        output_slice_accessor_type m_output_slices;
        input_rotate_type m_insert_inv_rotation_matrices;
        output_rotate_type m_extract_fwd_rotation_matrices;
        NOA_NO_UNIQUE_ADDRESS input_scale_or_empty_type m_insert_fwd_scaling_matrices;
        NOA_NO_UNIQUE_ADDRESS output_scale_or_empty_type m_extract_inv_scaling_matrices;

        coord2_type m_f_output_shape;
        coord2_type m_f_input_shape;
        coord_type m_f_input_center_y;
        index_type m_input_count;
        index_type m_output_slice_size_y;
        NOA_NO_UNIQUE_ADDRESS ews_or_empty_type m_ews_diam_inv{};

        coord_type m_volume_z;
        coord_type m_fftfreq_cutoff_sqd;
        coord_type m_insert_fftfreq_sinc;
        coord_type m_insert_fftfreq_blackman;
        coord_type m_extract_fftfreq_sinc;
        coord_type m_extract_fftfreq_blackman;
        index_type m_extract_blackman_size;

        bool m_add_to_output;
    };

    template<bool POST_CORRECTION, typename Coord, typename Index, typename Value>
    class GriddingCorrection {
        static_assert(nt::is_int_v<Index>);
        static_assert(nt::are_real_v<Value, Coord>);

        using index_type = Index;
        using value_type = Value;
        using coord_type = Coord;

        using input_accessor_type = Accessor<const value_type, 4, index_type>;
        using output_accessor_type = Accessor<value_type, 4, index_type>;
        using shape3_type = Shape3<index_type>;
        using coord3_type = Vec3<coord_type>;

    public:
        GriddingCorrection(
                const input_accessor_type& input,
                const output_accessor_type& output,
                const Shape4<index_type>& shape
        )
                : m_input(input), m_output(output)
        {
            const shape3_type l_shape = shape.pop_front();
            m_f_shape = coord3_type(l_shape.vec());
            m_half = m_f_shape / 2 * coord3_type(l_shape != 1); // if size == 1, half should be 0
        }

        NOA_HD void operator()(index_type i, index_type j, index_type k, index_type l) const noexcept {
            coord3_type dist{j, k, l};
            dist -= m_half;
            dist /= m_f_shape;

            constexpr coord_type PI = noa::math::Constant<coord_type>::PI;
            const coord_type radius = noa::math::sqrt(noa::math::dot(dist, dist));
            const coord_type sinc = noa::math::sinc(PI * radius);
            const auto sinc2 = static_cast<value_type>(sinc * sinc); // > 0.05

            const value_type value = m_input(i, j, k, l);
            m_output(i, j, k, l) = POST_CORRECTION ? value / sinc2 : value * sinc2;
        }

    private:
        input_accessor_type m_input;
        output_accessor_type m_output;
        coord3_type m_f_shape;
        coord3_type m_half;
    };
}

namespace noa::algorithm::geometry {
    template<noa::fft::Remap REMAP,
             typename Index, typename Offset, typename Value, typename Coord,
             typename ScaleOrEmpty, typename Rotate, typename EWSOrEmpty>
    auto fourier_insert_rasterize(
            const AccessorRestrict<const Value, 3, Offset>& slice, const Shape4<Index>& slice_shape,
            const AccessorRestrict<Value, 3, Offset>& grid, const Shape4<Index>& grid_shape,
            const ScaleOrEmpty& inv_scaling_matrices, const Rotate& fwd_rotation_matrices,
            Coord fftfreq_cutoff, const Shape4<Index>& target_shape, EWSOrEmpty ews_radius
    ) {
        using slice_accessor_t = AccessorRestrict<const Value, 3, Offset>;
        return FourierInsertRasterize<
                REMAP, Index, Offset, Value, Coord, ScaleOrEmpty, Rotate, EWSOrEmpty, slice_accessor_t>(
                slice, slice_shape, grid, grid_shape,
                inv_scaling_matrices, fwd_rotation_matrices,
                fftfreq_cutoff, target_shape, ews_radius);
    }

    template<noa::fft::Remap REMAP,
             typename Index, typename Offset, typename Value, typename Coord,
             typename ScaleOrEmpty, typename Rotate, typename EWSOrEmpty>
    auto fourier_insert_rasterize(
            const Value& slice, const Shape4<Index>& slice_shape,
            const AccessorRestrict<Value, 3, Offset>& grid, const Shape4<Index>& grid_shape,
            const ScaleOrEmpty& inv_scaling_matrices, const Rotate& fwd_rotation_matrices,
            Coord fftfreq_cutoff, const Shape4<Index>& target_shape, EWSOrEmpty ews_radius
    ) {
        return FourierInsertRasterize<
                REMAP, Index, Offset, Value, Coord, ScaleOrEmpty, Rotate, EWSOrEmpty, Value>(
                slice, slice_shape, grid, grid_shape,
                inv_scaling_matrices, fwd_rotation_matrices,
                fftfreq_cutoff, target_shape, ews_radius);
    }

    template<noa::fft::Remap REMAP,
             typename Index, typename Value, typename Offset, typename Coord,
             typename ScaleOrEmpty, typename Rotate, typename EWSOrEmpty,
             typename SliceInterpolator>
    auto fourier_insert_interpolate(
            const SliceInterpolator& slice, const Shape4<Index>& slice_shape,
            const AccessorRestrict<Value, 3, Offset>& grid, const Shape4<Index>& grid_shape,
            const ScaleOrEmpty& inv_scaling_matrices, const Rotate& fwd_rotation_matrices,
            Coord fftfreq_cutoff, Coord fftfreq_sinc, Coord fftfreq_blackman,
            const Shape4<Index>& target_shape, EWSOrEmpty ews_radius
    ) {
        return FourierInsertInterpolate<
                REMAP, Index, Value, Offset,
                ScaleOrEmpty, Rotate,
                EWSOrEmpty, SliceInterpolator>(
                slice, slice_shape, grid, grid_shape,
                inv_scaling_matrices, fwd_rotation_matrices,
                fftfreq_cutoff, fftfreq_sinc, fftfreq_blackman,
                target_shape, ews_radius);
    }

    template<noa::fft::Remap REMAP,
             typename Index, typename Value, typename Offset, typename Coord,
             typename ScaleOrEmpty, typename Rotate, typename EWSOrEmpty,
             typename GridInterpolator>
    auto fourier_extract(
            const GridInterpolator& grid, const Shape4<Index>& grid_shape,
            const AccessorRestrict<Value, 3, Offset>& slice, const Shape4<Index>& slice_shape,
            const ScaleOrEmpty& inv_scaling_matrices, const Rotate& fwd_rotation_matrices,
            Coord fftfreq_cutoff, Coord fftfreq_z_sinc, Coord fftfreq_z_blackman,
            const Shape4<Index>& target_shape, EWSOrEmpty ews_radius
    ) {
        return FourierExtract<
                REMAP, Index, Value, Offset,
                ScaleOrEmpty, Rotate,
                EWSOrEmpty, GridInterpolator>(
                grid, grid_shape, slice, slice_shape,
                inv_scaling_matrices, fwd_rotation_matrices,
                fftfreq_cutoff, fftfreq_z_sinc, fftfreq_z_blackman,
                target_shape, ews_radius);
    }

    template<noa::fft::Remap REMAP,
             typename Index, typename Value, typename Offset, typename Coord,
             typename InputScaleOrEmpty, typename InputRotate,
             typename OutputScaleOrEmpty, typename OutputRotate,
             typename EWSOrEmpty, typename InputSliceInterpolator>
    auto fourier_insert_and_extract(
            const InputSliceInterpolator& input_slices, const Shape4<Index>& input_shape,
            const AccessorRestrict<Value, 3, Offset> output_slices, const Shape4<Index>& output_shape,
            const InputScaleOrEmpty& insert_fwd_scaling_matrices,
            const InputRotate& insert_inv_rotation_matrices,
            const OutputScaleOrEmpty& extract_inv_scaling_matrices,
            const OutputRotate& extract_fwd_rotation_matrices,
            Coord fftfreq_cutoff,
            Coord insert_fftfreq_sinc,
            Coord insert_fftfreq_blackman,
            Coord extract_fftfreq_sinc,
            Coord extract_fftfreq_blackman,
            bool add_to_output,
            EWSOrEmpty ews_radius
    ) {
        return FourierInsertExtract<
                REMAP, Index, Value, Offset,
                InputScaleOrEmpty, InputRotate,
                OutputScaleOrEmpty, OutputRotate,
                EWSOrEmpty, InputSliceInterpolator>(
                input_slices, input_shape,
                output_slices, output_shape,
                insert_fwd_scaling_matrices,
                insert_inv_rotation_matrices,
                extract_inv_scaling_matrices,
                extract_fwd_rotation_matrices,
                fftfreq_cutoff,
                insert_fftfreq_sinc,
                insert_fftfreq_blackman,
                extract_fftfreq_sinc,
                extract_fftfreq_blackman,
                add_to_output,
                ews_radius
        );
    }

    template<bool POST_CORRECTION, typename Coord = f32, typename Index, typename Value>
    auto gridding_correction(
            const Accessor<const Value, 4, Index>& input,
            const Accessor<Value, 4, Index>& output,
            const Shape4<Index>& shape
    ) {
        return GriddingCorrection<POST_CORRECTION, Coord, Index, Value>(input, output, shape);
    }
}
