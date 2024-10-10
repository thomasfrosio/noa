#pragma once

#include "noa/core/Enums.hpp"
#include "noa/core/signal/FSC.hpp"
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
        auto denominator_lhs = denominator.subregion(ni::FullExtent{}, 0);
        auto denominator_rhs = denominator.subregion(ni::FullExtent{}, 1);

        const auto reduction_op = FSCIsotropic<REMAP, real_t, i64, input_accessor_t, output_accessor_t>(
                input_accessor_t(lhs.get(), lhs.strides()),
                input_accessor_t(rhs.get(), rhs.strides()), shape.pop_front(),
                output_accessor_t(fsc.get(), fsc.strides().filter(0, 3).template as_safe<i32>()),
                output_accessor_t(denominator_lhs.get(), denominator_lhs.strides().filter(0, 3).template as_safe<i32>()),
                output_accessor_t(denominator_rhs.get(), denominator_rhs.strides().filter(0, 3).template as_safe<i32>()));
        iwise(shape.rfft(), fsc.device(), reduction_op, std::forward<Lhs>(lhs), std::forward<Rhs>(rhs));
        ewise(wrap(std::move(denominator_lhs), std::move(denominator_rhs)),
              std::forward<Output>(fsc),
              FSCNormalization{});
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
        auto denominator_lhs = denominator.subregion(ni::FullExtent{}, 0);
        auto denominator_rhs = denominator.subregion(ni::FullExtent{}, 1);

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
              FSCNormalization{});
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
