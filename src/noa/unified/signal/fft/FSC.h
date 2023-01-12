#pragma once

#include "noa/Array.h"

namespace noa::signal::fft {
    /// Computes the isotropic Fourier Shell Correlation between \p lhs and \p rhs.
    /// \tparam REMAP   Whether the input non-redundant FFTs are centered. Should be H2H or HC2HC.
    /// \tparam Real    float or double.
    /// \param[in] lhs  Left-hand side.
    /// \param[in] rhs  Right-hand side. Should have the same shape as \p lhs.
    /// \param[out] fsc The output FSC. Should be a (batched) vector of size min(shape) // 2 + 1.
    /// \param shape    Logical shape of \p lhs and \p rhs.
    ///                 It should be a cubic or rectangular (batched) volume.
    template<noa::fft::Remap REMAP, typename Real,
             typename = std::enable_if_t<noa::traits::is_any_v<Real, float, double>>>
    void isotropicFSC(const Array<Complex<Real>>& lhs,
                      const Array<Complex<Real>>& rhs,
                      const Array<Real>& fsc,
                      dim4_t shape);

    /// Computes the isotropic Fourier Shell Correlation between \p lhs and \p rhs.
    /// \tparam REMAP   Whether the input non-redundant FFTs are centered. Should be H2H or HC2HC.
    /// \tparam Real    float or double.
    /// \param[in] lhs  Left-hand side.
    /// \param[in] rhs  Right-hand side. Should have the same shape as \p lhs.
    /// \param shape    Logical shape of \p lhs and \p rhs.
    ///                 It should be a cubic or rectangular (batched) volume.
    /// \return A (batched) row vector with the FSC. The number of shells is min(shape) // 2 + 1.
    template<noa::fft::Remap REMAP, typename Real,
             typename = std::enable_if_t<noa::traits::is_any_v<Real, float, double>>>
    Array<Real> isotropicFSC(const Array<Complex<Real>>& lhs,
                             const Array<Complex<Real>>& rhs,
                             dim4_t shape);

    /// Computes the anisotropic Fourier Shell Correlation between \p lhs and \p rhs.
    /// \tparam REMAP               Whether the input non-redundant FFTs are centered. Should be H2H or HC2HC.
    /// \tparam Real                float or double.
    /// \param[in] lhs              Left-hand side.
    /// \param[in] rhs              Right-hand side. Should have the same shape as \p lhs.
    /// \param shape                Logical shape of \p lhs and \p rhs.
    ///                             It should be a cubic or rectangular (batched) volume.
    /// \param[in] cone_directions  DHW normalized direction(s) of the cone(s).
    /// \param cone_aperture        Cone aperture, in radians.
    /// \return A row-major table with the FSC of shape (batch, 1, cones, shells).
    ///         Each row contains the shell values. There's one row per cone.
    ///         Each column is a shell, with the number of shells set to min(shape) // 2 + 1.
    ///         There's one table per input batch.
    template<noa::fft::Remap REMAP, typename Real,
             typename = std::enable_if_t<noa::traits::is_any_v<Real, float, double>>>
    void anisotropicFSC(const Array<Complex<Real>>& lhs,
                        const Array<Complex<Real>>& rhs,
                        const Array<Real>& fsc, dim4_t shape,
                        const Array<float3_t>& cone_directions, float cone_aperture);

    /// Computes the anisotropic/conical Fourier Shell Correlation between \p lhs and \p rhs.
    /// \tparam REMAP               Whether the input non-redundant FFTs are centered. Should be H2H or HC2HC.
    /// \tparam Real                float or double.
    /// \param[in] lhs              Left-hand side.
    /// \param[in] rhs              Right-hand side. Should have the same shape as \p lhs.
    /// \param shape                Logical shape of \p lhs and \p rhs.
    ///                             It should be a cubic or rectangular (batched) volume.
    /// \param[in] cone_directions  DHW normalized direction(s) of the cone(s).
    /// \param cone_aperture        Cone aperture, in radians.
    /// \return A row-major (batched) table with the FSC of shape (batch, 1, cones, shells).
    ///         Each row contains the shell values. There's one row per cone.
    ///         Each column is a shell, with the number of shells set to min(shape) // 2 + 1.
    ///         There's one table per input batch.
    template<noa::fft::Remap REMAP, typename Real,
             typename = std::enable_if_t<noa::traits::is_any_v<Real, float, double>>>
    Array<Real> anisotropicFSC(const Array<Complex<Real>>& lhs,
                               const Array<Complex<Real>>& rhs, dim4_t shape,
                               const Array<float3_t>& cone_directions, float cone_aperture);
}
