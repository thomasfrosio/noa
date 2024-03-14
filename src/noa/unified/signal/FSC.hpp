#pragma once

#include "noa/core/fft/RemapInterface.hpp"
#include "noa/core/signal/FSC.hpp"
#include "noa/unified/Array.hpp"
#include "noa/unified/Ewise.hpp"
#include "noa/unified/Iwise.hpp"

namespace noa::signal::guts {
    template<typename Lhs, typename Rhs, typename Output, typename Cones = Empty>
    void check_fsc_parameters(
            const Lhs& lhs, const Rhs& rhs, const Output& fsc, const Shape4<i64>& shape,
            const Cones& cone_directions = {}
    ) {
        check(not lhs.is_empty() and not rhs.is_empty() and not fsc.is_empty(), "Empty array detected");
        check(lhs.get() != rhs.get(), "Computing the FSC on the same array is not allowed");

        check(all(rhs.shape() == shape.rfft()),
              "Given the logical shape {}, the expected non-redundant shape should be {}, but got {}",
              shape, shape.rfft(), rhs.shape());
        check(all(lhs.shape() == rhs.shape()),
              "The two input arrays should have the same shape. Got lhs:{} and rhs:{}",
              lhs.shape(), rhs.shape());

        const Device device = fsc.device();
        check(device == lhs.device() and device == rhs.device(),
              "The input and output arrays must be on the same device, but got lhs:{}, rhs:{}, fsc:{}",
              lhs.device(), rhs.device(), device);

        i64 n_cones{1};
        if constexpr (not std::is_empty_v<Cones>) {
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

        const auto n_shells = min(lhs.shape().pop_front()) / 2 + 1;
        const auto expected_shape = Shape4<i64>{shape[0], 1, n_cones, n_shells};
        check(all(fsc.shape() == expected_shape) and fsc.are_contiguous(),
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
    /// \param[out] fsc The output FSC. Should be a (batched) vector of size min(shape) // 2 + 1.
    /// \param shape    Logical shape of \p lhs and \p rhs.
    template<noa::fft::RemapInterface REMAP, typename Lhs, typename Rhs, typename Output>
    requires (nt::are_varray_of_complex_v<Lhs, Rhs> and
              nt::are_almost_same_value_type_v<Lhs, Rhs> and
              nt::is_varray_of_real_v<Output> and
              (REMAP.remap == noa::fft::Remap::H2H or REMAP.remap == noa::fft::Remap::HC2HC))
    void fsc_isotropic(
            const Lhs& lhs,
            const Rhs& rhs,
            const Output& fsc,
            const Shape4<i64>& shape
    ) {
        guts::check_fsc_parameters(lhs, rhs, fsc, shape);

        using complex_t = nt::mutable_value_type_t<Lhs>;
        using real_t = nt::value_type_t<Output>;
        using input_accessor_t = AccessorRestrictI64<const complex_t, 4>;
        using output_accessor_t = AccessorRestrictContiguousI32<real_t, 2>;

        const auto options = lhs.options().set_allocator(Allocator(MemoryResource::DEFAULT_ASYNC));
        const auto denominator_lhs = Array<real_t>(fsc.shape(), options);
        const auto denominator_rhs = Array<real_t>(fsc.shape(), options);

        const auto shell_strides = fsc.strides().filter(2, 3).template as<i32>();
        const auto reduction_op = FSCIsotropic<REMAP.remap, real_t, i64, input_accessor_t, output_accessor_t>(
                input_accessor_t(lhs, lhs.strides()),
                input_accessor_t(rhs, rhs.strides()), shape.pop_front(),
                output_accessor_t(fsc.get(), shell_strides),
                output_accessor_t(denominator_lhs.get(), shell_strides),
                output_accessor_t(denominator_rhs.get(), shell_strides));
        iwise(shape.rfft(), fsc.device(), reduction_op, lhs, rhs);
        ewise(wrap(denominator_lhs, denominator_rhs), fsc, FSCNormalization{});
    }

    /// Computes the isotropic Fourier Shell Correlation between \p lhs and \p rhs.
    /// \tparam REMAP   Whether the input rffts are centered. Should be H2H or HC2HC.
    /// \param[in] lhs  Left-hand side.
    /// \param[in] rhs  Right-hand side. Should have the same shape as \p lhs.
    /// \param shape    Logical shape of \p lhs and \p rhs.
    /// \return A (batched) row vector with the FSC. The number of shells is min(shape) // 2 + 1.
    template<noa::fft::RemapInterface REMAP, typename Lhs, typename Rhs>
    requires (nt::are_varray_of_complex_v<Lhs, Rhs> and
              nt::are_almost_same_value_type_v<Lhs, Rhs> and
              (REMAP.remap == noa::fft::Remap::H2H or REMAP.remap == noa::fft::Remap::HC2HC))
    auto fsc_isotropic(
            const Lhs& lhs,
            const Rhs& rhs,
            const Shape4<i64>& shape
    ) {
        const auto n_shells = min(lhs.shape()) / 2 + 1;
        const auto expected_shape = Shape4<i64>{shape[0], 1, 1, n_shells};
        using value_t = Lhs::mutable_value_type;
        Array<value_t> fsc(expected_shape, rhs.options());
        fsc_isotropic<REMAP>(lhs, rhs, fsc, shape);
        return fsc;
    }

    /// Computes the anisotropic Fourier Shell Correlation between \p lhs and \p rhs.
    /// \tparam REMAP               Whether the input rffts are centered. Should be H2H or HC2HC.
    /// \param[in] lhs              Left-hand side.
    /// \param[in] rhs              Right-hand side. Should have the same shape as \p lhs.
    /// \param[out] fsc             The output FSC. A row-major table of shape (n_batches, 1, n_cones, n_shells).
    ///                             Each row contains the shell values. There's one row per cone.
    ///                             Each column is a shell, with the number of shells set to min(shape_3d) // 2 + 1.
    ///                             There's one table per input batch.
    /// \param shape                Logical shape of \p lhs and \p rhs.
    /// \param[in] cone_directions  DHW normalized direction(s) of the cone(s).
    /// \param cone_aperture        Cone aperture, in radians.
    template<noa::fft::RemapInterface REMAP, typename Lhs, typename Rhs, typename Output, typename Cones>
    requires (nt::are_varray_of_complex_v<Lhs, Rhs> and
              nt::are_almost_same_value_type_v<Lhs, Rhs> and
              nt::is_varray_of_any_v<Output, f32, f64> and
              nt::is_varray_of_almost_any_v<Cones, Vec3<f32>, Vec3<f64>> and
              (REMAP.remap == noa::fft::Remap::H2H or REMAP.remap == noa::fft::Remap::HC2HC))
    void fsc_anisotropic(
            const Lhs& lhs,
            const Rhs& rhs,
            const Output& fsc,
            const Shape4<i64>& shape,
            const Cones& cone_directions,
            f64 cone_aperture
    ) {
        guts::check_fsc_parameters(lhs, rhs, fsc, shape, cone_directions);

        using complex_t = nt::mutable_value_type_t<Lhs>;
        using real_t = nt::value_type_t<Output>;
        using direction_t = nt::mutable_value_type_t<Cones>;
        using input_accessor_t = AccessorRestrictI64<const complex_t, 4>;
        using output_accessor_t = AccessorRestrictContiguousI32<real_t, 3>;
        using direction_accessor_t = AccessorRestrictContiguousI32<const direction_t, 1>;

        const auto options = lhs.options().set_allocator(Allocator(MemoryResource::DEFAULT_ASYNC));
        const auto denominator_lhs = Array<real_t>(fsc.shape(), options);
        const auto denominator_rhs = Array<real_t>(fsc.shape(), options);

        const auto shell_strides = fsc.strides().pop_front().template as<i32>();
        const auto reduction_op =
                FSCAnisotropic<REMAP.remap, real_t, i64, input_accessor_t, output_accessor_t, direction_accessor_t>(
                        input_accessor_t(lhs, lhs.strides()),
                        input_accessor_t(rhs, rhs.strides()), shape.pop_front(),
                        output_accessor_t(fsc.get(), shell_strides),
                        output_accessor_t(denominator_lhs.get(), shell_strides),
                        output_accessor_t(denominator_rhs.get(), shell_strides),
                        direction_accessor_t(cone_directions.get()),
                        cone_directions.ssize(),
                        static_cast<real_t>(cone_aperture));
        iwise(shape.rfft(), fsc.device(), reduction_op, lhs, rhs);
        ewise(wrap(denominator_lhs, denominator_rhs), fsc, FSCNormalization{});
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
    ///         Each column is a shell, with the number of shells set to min(shape_3d) // 2 + 1.
    ///         There's one table per input batch.
    template<noa::fft::RemapInterface REMAP, typename Lhs, typename Rhs, typename Cones>
    requires (nt::are_varray_of_complex_v<Lhs, Rhs> and
              nt::are_almost_same_value_type_v<Lhs, Rhs> and
              nt::is_varray_of_almost_any_v<Cones, Vec3<f32>, Vec3<f64>> and
              (REMAP.remap == noa::fft::Remap::H2H or REMAP.remap == noa::fft::Remap::HC2HC))
    auto fsc_anisotropic(
            const Lhs& lhs,
            const Rhs& rhs,
            const Shape4<i64>& shape,
            const Cones& cone_directions,
            f32 cone_aperture
    ) {
        const auto n_shelss = min(lhs.shape().pop_front()) / 2 + 1;
        const auto expected_shape = Shape4<i64>{shape[0], 1, cone_directions.ssize(), n_shelss};
        using value_t = Lhs::mutable_value_type;
        Array<value_t> fsc(expected_shape, rhs.options());
        fsc_anisotropic<REMAP>(lhs, rhs, fsc, shape, cone_directions, cone_aperture);
        return fsc;
    }
}
