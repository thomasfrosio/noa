#pragma once

#include "noa/core/Enums.hpp"
#include "noa/core/utils/Atomic.hpp"
#include "noa/core/fft/Frequency.hpp"
#include "noa/unified/Array.hpp"
#include "noa/unified/Ewise.hpp"
#include "noa/unified/Iwise.hpp"

namespace noa::signal {
    constexpr auto n_shells(const Shape4<i64>& shape) -> i64 {
        switch (shape.ndim()) {
            case 1:
                return (shape[3] > 1 ? shape[3] : shape[2]) / 2 + 1;
            case 2:
                return min(shape.filter(2, 3)) / 2 + 1;
            case 3:
                return min(shape.pop_front()) / 2 + 1;
            default:
                panic("BUG: this should not have happened");
        }
        return 1;
    }
}


namespace noa::signal::guts {
    /// 4d iwise operator to compute an isotropic FSC.
    /// * A lerp is used to add frequencies in its two neighbour shells, instead of rounding to the nearest shell.
    /// * The frequencies are normalized, so rectangular volumes can be passed.
    /// * The number of shells is fixed by the input shape: min(shape) // 2 + 1
    template<Remap REMAP,
             nt::real Coord,
             nt::sinteger Index,
             nt::readable_nd<4> Input,
             nt::atomic_addable_nd<2> Output>
    class FSCIsotropic {
    public:
        static_assert(not REMAP.has_layout_change());
        static constexpr bool IS_CENTERED = REMAP.is_xx2xc();
        static constexpr bool IS_RFFT = REMAP.is_hx2hx();

        using index_type = Index;
        using coord_type = Coord;
        using coord3_type = Vec3<coord_type>;
        using shape3_type = Shape3<index_type>;
        using shape_nd_type = Shape<index_type, 3 - IS_RFFT>;

        using input_type = Input;
        using output_type = Output;
        using input_value_type = nt::mutable_value_type_t<input_type>;
        using input_real_type = nt::value_type_t<input_value_type>;
        using output_value_type = nt::value_type_t<output_type>;
        static_assert(nt::complex<input_value_type> and nt::real<output_value_type>);

    public:
        FSCIsotropic(
            const input_type& lhs,
            const input_type& rhs,
            const shape3_type& shape,
            const output_type& numerator_and_output,
            const output_type& denominator_lhs,
            const output_type& denominator_rhs
        ) : m_lhs(lhs), m_rhs(rhs),
            m_numerator_and_output(numerator_and_output),
            m_denominator_lhs(denominator_lhs),
            m_denominator_rhs(denominator_rhs),
            m_norm(coord_type{1} / coord3_type::from_vec(shape.vec)),
            m_scale(static_cast<coord_type>(min(shape))),
            m_max_shell_index(min(shape) / 2)
        {
            if constexpr (IS_RFFT)
                m_shape = shape.pop_back();
            else
                m_shape = shape;
        }

        // Initial reduction.
        NOA_HD void operator()(index_type batch, index_type z, index_type y, index_type x) const {
            // Compute the normalized frequency.
            auto frequency = noa::fft::index2frequency<IS_CENTERED, IS_RFFT>(Vec{z, y, x}, m_shape);
            auto fftfreq = coord3_type::from_vec(frequency) * m_norm;

            // Shortcut for everything past Nyquist.
            const auto radius_sqd = dot(fftfreq, fftfreq);
            if (radius_sqd > coord_type{0.25})
                return;

            const auto radius = sqrt(radius_sqd) * m_scale;
            const auto fraction = floor(radius);

            // Compute lerp weights.
            const auto shell_low = static_cast<index_type>(floor(radius));
            const auto shell_high = min(m_max_shell_index, shell_low + 1); // clamp for oob
            const auto fraction_high = static_cast<input_real_type>(radius - fraction);
            const auto fraction_low = 1 - fraction_high;

            // Preliminary shell values.
            const auto lhs = m_lhs(batch, z, y, x);
            const auto rhs = m_rhs(batch, z, y, x);
            const auto numerator = dot(lhs, rhs); // sqrt(abs(lhs) * abs(rhs))
            const auto denominator_lhs = abs_squared(lhs);
            const auto denominator_rhs = abs_squared(rhs);

            // Atomic save.
            // TODO In CUDA, we could do the atomic reduction in shared memory to reduce global memory transfers.
            using oreal_t = output_value_type;
            ng::atomic_add(m_numerator_and_output, static_cast<oreal_t>(numerator * fraction_low), batch, shell_low);
            ng::atomic_add(m_numerator_and_output, static_cast<oreal_t>(numerator * fraction_high), batch, shell_high);
            ng::atomic_add(m_denominator_lhs, static_cast<oreal_t>(denominator_lhs * fraction_low), batch, shell_low);
            ng::atomic_add(m_denominator_lhs, static_cast<oreal_t>(denominator_lhs * fraction_high), batch, shell_high);
            ng::atomic_add(m_denominator_rhs, static_cast<oreal_t>(denominator_rhs * fraction_low), batch, shell_low);
            ng::atomic_add(m_denominator_rhs, static_cast<oreal_t>(denominator_rhs * fraction_high), batch, shell_high);
        }

    private:
        input_type m_lhs;
        input_type m_rhs;
        output_type m_numerator_and_output;
        output_type m_denominator_lhs;
        output_type m_denominator_rhs;

        coord3_type m_norm;
        coord_type m_scale;
        shape_nd_type m_shape;
        index_type m_max_shell_index;
    };

    /// Anisotropic/Conical FSC implementation.
    /// * Implementation is same as isotropic FSC expect that there's an additional step where
    ///   we compute the cone mask. Multiple cones can be defined and the innermost loop go through
    ///   the cones.
    /// * Cones are described by their orientation (a 3d vector) and the cone aperture.
    ///   The aperture is fixed for every batch and the angular distance from the cone is
    ///   used to compute the cone mask.
    template<Remap REMAP,
             nt::real Coord,
             nt::sinteger Index,
             nt::readable_nd<4> Input,
             nt::atomic_addable_nd<3> Output,
             nt::readable_pointer_like Direction>
    class FSCAnisotropic {
    public:
        static_assert(not REMAP.has_layout_change());
        static constexpr bool IS_CENTERED = REMAP.is_xx2xc();
        static constexpr bool IS_RFFT = REMAP.is_hx2hx();

        using index_type = Index;
        using coord_type = Coord;
        using coord3_type = Vec3<coord_type>;
        using shape3_type = Shape3<index_type>;
        using shape_nd_type = Shape<index_type, 3 - IS_RFFT>;

        using input_type = Input;
        using output_type = Output;
        using direction_type = Direction;
        using input_value_type = nt::mutable_value_type_t<input_type>;
        using input_real_type = nt::value_type_t<input_value_type>;
        using output_value_type = nt::value_type_t<output_type>;
        using direction_value_type = nt::value_type_t<direction_type>;
        static_assert(nt::complex<input_value_type> and
                      nt::real<output_value_type> and
                      nt::vec_real_size<direction_value_type, 3>);

    public:
        FSCAnisotropic(
            const input_type& lhs,
            const input_type& rhs, const shape3_type& shape,
            const output_type& numerator_and_output,
            const output_type& denominator_lhs,
            const output_type& denominator_rhs,
            const direction_type& normalized_cone_directions,
            index_type cone_count,
            coord_type cone_aperture
        ) : m_lhs(lhs), m_rhs(rhs),
            m_numerator_and_output(numerator_and_output),
            m_denominator_lhs(denominator_lhs),
            m_denominator_rhs(denominator_rhs),
            m_normalized_cone_directions(normalized_cone_directions),
            m_norm(coord_type{1} / coord3_type::from_vec(shape.vec)),
            m_scale(static_cast<coord_type>(min(shape))),
            m_cos_cone_aperture(cos(cone_aperture)),
            m_max_shell_index(min(shape) / 2),
            m_cone_count(cone_count)
        {
            if constexpr (IS_RFFT)
                m_shape = shape.pop_back();
            else
                m_shape = shape;
        }

        // Initial reduction.
        NOA_HD void operator()(index_type batch, index_type z, index_type y, index_type x) const {
            // Compute the normalized frequency.
            auto frequency = noa::fft::index2frequency<IS_CENTERED, IS_RFFT>(Vec{z, y, x}, m_shape);
            auto fftfreq = coord3_type::from_vec(frequency) * m_norm;

            // Shortcut for everything past Nyquist.
            const auto radius_sqd = dot(fftfreq, fftfreq);
            if (radius_sqd > coord_type{0.25})
                return;

            const auto radius = sqrt(radius_sqd) * m_scale;
            const auto fraction = floor(radius);

            // Compute lerp weights.
            const auto shell_low = static_cast<index_type>(floor(radius));
            const auto shell_high = min(m_max_shell_index, shell_low + 1); // clamp for oob
            const auto fraction_high = static_cast<input_real_type>(radius - fraction);
            const auto fraction_low = 1 - fraction_high;

            // Preliminary shell values.
            const auto lhs = m_lhs(batch, z, y, x);
            const auto rhs = m_rhs(batch, z, y, x);
            const auto numerator = dot(lhs, rhs); // sqrt(abs(lhs) * abs(rhs))
            const auto denominator_lhs = abs_squared(lhs);
            const auto denominator_rhs = abs_squared(rhs);

            const auto normalized_direction = coord3_type::from_values(z, y, x) / radius;
            for (index_type cone{}; cone < m_cone_count; ++cone) {
                // angular_difference = acos(dot(a,b))
                // We could compute the angular difference between the current frequency and the cone direction.
                // However, we only want to know if the frequency is inside or outside the cone, so skip the arccos
                // and check abs(cos(angle)) > cos(cone_aperture).

                // TODO In CUDA, try constant memory for the directions, or the less appropriate shared memory.
                const auto normalized_direction_cone = m_normalized_cone_directions[cone].template as<coord_type>();
                const auto cos_angle_difference = dot(normalized_direction, normalized_direction_cone);
                if (abs(cos_angle_difference) > m_cos_cone_aperture)
                    continue;

                // Atomic save.
                // TODO In CUDA, we could do the atomic reduction in shared memory to reduce global memory transfers.
                using oreal_t = output_value_type;
                ng::atomic_add(m_numerator_and_output, static_cast<oreal_t>(numerator * fraction_low), batch, cone, shell_low);
                ng::atomic_add(m_numerator_and_output, static_cast<oreal_t>(numerator * fraction_high), batch, cone, shell_high);
                ng::atomic_add(m_denominator_lhs, static_cast<oreal_t>(denominator_lhs * fraction_low), batch, cone, shell_low);
                ng::atomic_add(m_denominator_lhs, static_cast<oreal_t>(denominator_lhs * fraction_high), batch, cone, shell_high);
                ng::atomic_add(m_denominator_rhs, static_cast<oreal_t>(denominator_rhs * fraction_low), batch, cone, shell_low);
                ng::atomic_add(m_denominator_rhs, static_cast<oreal_t>(denominator_rhs * fraction_high), batch, cone, shell_high);
            }
        }

    private:
        input_type m_lhs;
        input_type m_rhs;
        output_type m_numerator_and_output;
        output_type m_denominator_lhs;
        output_type m_denominator_rhs;
        direction_type m_normalized_cone_directions;

        coord3_type m_norm;
        coord_type m_scale;
        coord_type m_cos_cone_aperture;
        shape_nd_type m_shape;
        index_type m_max_shell_index;
        index_type m_cone_count;
    };

    struct FSCNormalization {
        using enable_vectorization = bool;

        template<typename T>
        NOA_HD void operator()(T lhs, T rhs, T& output) const {
            constexpr auto EPSILON = static_cast<T>(1e-6);
            output /= max(EPSILON, sqrt(lhs * rhs));
        }
    };

    template<typename Lhs, typename Rhs, typename Output, typename Cones = Empty>
    void check_fsc_parameters(
        const Lhs& lhs, const Rhs& rhs, const Output& fsc, const Shape4<i64>& shape,
        const Cones& cone_directions = {}
    ) {
        check(not lhs.is_empty() and not rhs.is_empty() and not fsc.is_empty(), "Empty array detected");
        check(not ni::are_overlapped(lhs, rhs), "Computing the FSC on the same array is not allowed");

        check(vall(Equal{}, rhs.shape(), shape.rfft()),
              "Given the logical shape {}, the expected non-redundant shape should be {}, but got {}",
              shape, shape.rfft(), rhs.shape());
        check(vall(Equal{}, lhs.shape(), rhs.shape()),
              "The two input arrays should have the same shape. Got lhs:{} and rhs:{}",
              lhs.shape(), rhs.shape());

        const Device device = fsc.device();
        check(device == lhs.device() and device == rhs.device(),
              "The input and output arrays must be on the same device, but got lhs:{}, rhs:{}, fsc:{}",
              lhs.device(), rhs.device(), device);

        i64 n_cones{1};
        if constexpr (not nt::empty<Cones>) {
            check(not cone_directions.is_empty(), "Empty array detected");
            check(device == cone_directions.device(),
                  "The input and output arrays must be on the same device, "
                  "but got fsc:{}, cones:{}",
                  device, cone_directions.device());
            check(ni::is_contiguous_vector(cone_directions),
                  "The cone directions should be specified as a contiguous vector, but got shape:{}, strides:{}",
                  cone_directions.shape(), cone_directions.strides());
            n_cones = cone_directions.ssize();
        }

        const auto expected_shape = Shape4<i64>{shape[0], 1, n_cones, n_shells(lhs.shape())};
        check(vall(Equal{}, fsc.shape(), expected_shape) and fsc.are_contiguous(),
              "The FSC does not have the correct shape. Given the input shape {}, and the number of cones ({}),"
              "the expected shape is {}, but got {}",
              shape, n_cones, expected_shape, fsc.shape());
    }
}

namespace noa::signal::fft {
    /// Computes the isotropic Fourier Shell Correlation between \p lhs and \p rhs.
    /// \tparam REMAP   Whether the input rffts are centered. Should be H2H or HC2HC.
    /// \param[in] lhs  Left-hand side.
    /// \param[in] rhs  Right-hand side. Should have the same shape as \p lhs.
    /// \param[out] fsc The output FSC. Should be a (batched) vector of size n_shells(lhs.shape()).
    /// \param shape    Logical shape of \p lhs and \p rhs.
    template<Remap REMAP,
             nt::readable_varray_decay_of_complex Lhs,
             nt::readable_varray_decay_of_complex Rhs,
             nt::writable_varray_decay_of_real Output>
    requires (nt::varray_decay_of_almost_same_type<Lhs, Rhs> and not REMAP.has_layout_change())
    void fsc_isotropic(
        Lhs&& lhs,
        Rhs&& rhs,
        Output&& fsc,
        const Shape4<i64>& shape
    ) {
        guts::check_fsc_parameters(lhs, rhs, fsc, shape);

        using complex_t = nt::const_value_type_t<Lhs>;
        using real_t = nt::value_type_t<Output>;
        using input_accessor_t = AccessorRestrictI64<complex_t, 4>;
        using output_accessor_t = AccessorRestrictContiguousI32<real_t, 2>;

        const auto options = lhs.options().set_allocator(Allocator::DEFAULT_ASYNC);
        const auto denominator = Array<real_t>(fsc.shape().template set<1>(2), options);
        auto denominator_lhs = denominator.subregion(ni::Full{}, 0);
        auto denominator_rhs = denominator.subregion(ni::Full{}, 1);

        const auto reduction_op = FSCIsotropic<REMAP, real_t, i64, input_accessor_t, output_accessor_t>(
                input_accessor_t(lhs.get(), lhs.strides()),
                input_accessor_t(rhs.get(), rhs.strides()), shape.pop_front(),
                output_accessor_t(fsc.get(), fsc.strides().filter(0, 3).template as_safe<i32>()),
                output_accessor_t(denominator_lhs.get(), denominator_lhs.strides().filter(0, 3).template as_safe<i32>()),
                output_accessor_t(denominator_rhs.get(), denominator_rhs.strides().filter(0, 3).template as_safe<i32>()));
        iwise(shape.rfft(), fsc.device(), reduction_op, std::forward<Lhs>(lhs), std::forward<Rhs>(rhs));
        ewise(wrap(std::move(denominator_lhs), std::move(denominator_rhs)),
              std::forward<Output>(fsc),
              guts::FSCNormalization{});
    }

    /// Computes the isotropic Fourier Shell Correlation between \p lhs and \p rhs.
    /// \tparam REMAP   Whether the input rffts are centered. Should be H2H or HC2HC.
    /// \param[in] lhs  Left-hand side.
    /// \param[in] rhs  Right-hand side. Should have the same shape as \p lhs.
    /// \param shape    Logical shape of \p lhs and \p rhs.
    /// \return A (batched) row vector with the FSC. The number of shells is n_shells(lhs.shape()).
    template<Remap REMAP,
             nt::readable_varray_decay_of_complex Lhs,
             nt::readable_varray_decay_of_complex Rhs>
    requires (nt::varray_decay_of_almost_same_type<Lhs, Rhs> and not REMAP.has_layout_change())
    auto fsc_isotropic(
        Lhs&& lhs,
        Rhs&& rhs,
        const Shape4<i64>& shape
    ) {
        using value_t = nt::mutable_value_type_t<Lhs>;
        auto fsc = Array<value_t>({shape[0], 1, 1, n_shells(rhs.shape())}, rhs.options());
        fsc_isotropic<REMAP>(std::forward<Lhs>(lhs), std::forward<Rhs>(rhs), fsc, shape);
        return fsc;
    }

    /// Computes the anisotropic Fourier Shell Correlation between \p lhs and \p rhs.
    /// \tparam REMAP               Whether the input rffts are centered. Should be H2H or HC2HC.
    /// \param[in] lhs              Left-hand side.
    /// \param[in] rhs              Right-hand side. Should have the same shape as \p lhs.
    /// \param[out] fsc             The output FSC. A row-major table of shape (n_batches, 1, n_cones, n_shells).
    ///                             Each row contains the shell values. There's one row per cone.
    ///                             Each column is a shell, with the number of shells set to n_shells(lhs.shape()).
    ///                             There's one table per input batch.
    /// \param shape                Logical shape of \p lhs and \p rhs.
    /// \param[in] cone_directions  DHW normalized direction(s) of the cone(s).
    /// \param cone_aperture        Cone aperture, in radians.
    template<Remap REMAP,
            nt::readable_varray_decay_of_complex Lhs,
            nt::readable_varray_decay_of_complex Rhs,
            nt::writable_varray_decay_of_real Output,
            nt::readable_varray_decay Cones>
    requires (nt::varray_decay_of_almost_same_type<Lhs, Rhs> and
              nt::vec_real_size<nt::value_type_t<Cones>, 3> and
              not REMAP.has_layout_change())
    void fsc_anisotropic(
        Lhs&& lhs,
        Rhs&& rhs,
        Output&& fsc,
        const Shape4<i64>& shape,
        Cones&& cone_directions,
        f64 cone_aperture
    ) {
        guts::check_fsc_parameters(lhs, rhs, fsc, shape, cone_directions);

        using coord_t = nt::mutable_value_type_t<Cones>;
        using real_t = nt::value_type_t<Output>;
        using input_accessor_t = AccessorRestrictI64<nt::const_value_type_t<Lhs>, 4>;
        using output_accessor_t = AccessorRestrictContiguousI32<real_t, 3>;
        using direction_accessor_t = AccessorRestrictContiguousI32<nt::const_value_type_t<Cones>, 1>;

        const auto options = lhs.options().set_allocator(Allocator::DEFAULT_ASYNC);
        const auto denominator = Array<real_t>(fsc.shape().template set<1>(2), options);
        auto denominator_lhs = denominator.subregion(ni::Full{}, 0);
        auto denominator_rhs = denominator.subregion(ni::Full{}, 1);

        auto reduction_op = FSCAnisotropic<REMAP, coord_t, i64, input_accessor_t, output_accessor_t, direction_accessor_t>(
            input_accessor_t(lhs.get(), lhs.strides()),
            input_accessor_t(rhs.get(), rhs.strides()), shape.pop_front(),
            output_accessor_t(fsc.get(), fsc.strides().filter(0, 2, 3).template as_safe<i32>()),
            output_accessor_t(denominator_lhs.get(), denominator_lhs.strides().filter(0, 2, 3).template as_safe<i32>()),
            output_accessor_t(denominator_rhs.get(), denominator_rhs.strides().filter(0, 2, 3).template as_safe<i32>()),
            direction_accessor_t(cone_directions.get()),
            cone_directions.ssize(),
            static_cast<coord_t>(cone_aperture));
        iwise(shape.rfft(), fsc.device(), reduction_op,
              std::forward<Lhs>(lhs),
              std::forward<Rhs>(rhs),
              std::forward<Cones>(cone_directions));
        ewise(wrap(std::move(denominator_lhs), std::move(denominator_rhs)),
              std::forward<Output>(fsc),
              guts::FSCNormalization{});
    }

    /// Computes the anisotropic/conical Fourier Shell Correlation between \p lhs and \p rhs.
    /// \tparam REMAP               Whether the input rffts are centered. Should be H2H or HC2HC.
    /// \param[in] lhs              Left-hand side.
    /// \param[in] rhs              Right-hand side. Should have the same shape as \p lhs.
    /// \param shape                Logical shape of \p lhs and \p rhs.
    /// \param[in] cone_directions  DHW normalized direction(s) of the cone(s).
    /// \param cone_aperture        Cone aperture, in radians.
    /// \return A row-major (batched) table with the FSC, of shape (n_batches, 1, n_cones, n_shells).
    ///         Each row contains the shell values. There's one row per cone.
    ///         Each column is a shell, with the number of shells set to n_shells(lhs.shape()).
    ///         There's one table per input batch.
    template<Remap REMAP,
             nt::readable_varray_decay_of_complex Lhs,
             nt::readable_varray_decay_of_complex Rhs,
             nt::readable_varray_decay Cones>
    requires (nt::varray_decay_of_almost_same_type<Lhs, Rhs> and
              nt::vec_real_size<nt::value_type_t<Cones>, 3> and
              not REMAP.has_layout_change())
    auto fsc_anisotropic(
        Lhs&& lhs,
        Rhs&& rhs,
        const Shape4<i64>& shape,
        Cones&& cone_directions,
        f32 cone_aperture
    ) {
        using value_t = nt::mutable_value_type_t<Lhs>;
        auto fsc = Array<value_t>({shape[0], 1, cone_directions.ssize(), n_shells(lhs.shape())}, rhs.options());
        fsc_anisotropic<REMAP>(std::forward<Lhs>(lhs), std::forward<Rhs>(rhs), fsc, shape,
                               std::forward<Cones>(cone_directions), cone_aperture);
        return fsc;
    }
}
