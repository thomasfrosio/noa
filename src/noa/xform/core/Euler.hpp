#pragma once

#include "noa/base/Mat.hpp"
#include "noa/runtime/core/Span.hpp"

namespace noa::xform {
    struct EulerOptions {
        std::string_view axes{"zyx"};
        bool intrinsic{true};
        bool right_handed{true};
    };

    /// Extracts the 3x3 rotation matrix from the Euler angles.
    /// \param angles       Euler angles, in radians.
    /// \param options      Euler angles options.
    template<nt::any_of<f32, f64> T>
    auto euler2matrix(Vec<T, 3> angles, const EulerOptions& options = {}) -> Mat33<T>;

    /// Derives the Euler angles, in radians, from the rotation matrix.
    /// \param rotation     Rotation (orthogonal) matrix to decompose.
    /// \param options      Euler angles options.
    template<nt::any_of<f32, f64> T>
    auto matrix2euler(const Mat33<T>& rotation, const EulerOptions& options = {}) -> Vec<T, 3>;

    /// Extracts a set of 3x3 rotation matrices from the Euler angles.
    /// \tparam T               f32 or f64.
    /// \param[in] angles       Euler angles, in radians, to convert.
    /// \param[out] matrices    Output 3x3 matrices.
    /// \param options          Euler angles options.
    template<nt::any_of<f32, f64> T,
             typename I0, StridesTraits S0,
             typename I1, StridesTraits S1>
    void euler2matrix(
        const Span<const Vec<T, 3>, 1, I0, S0>& angles,
        const Span<Mat33<T>, 1, I1, S1>& matrices,
        const EulerOptions& options = {}
    ) {
        check(angles.size() == matrices.size(), "angles and matrices don't have the same size");
        for (usize batch{}; batch < angles.size(); ++batch)
            matrices[batch] = euler2matrix(angles[batch], options);
    }

    /// Derives a set of Euler angles, in radians, from a set of rotation matrices.
    /// \tparam T           f32 or f64.
    /// \param[in] matrices 3x3 matrices to convert.
    /// \param[out] angles  Output Euler angles.
    /// \param options      Euler angles options.
    template<nt::any_of<f32, f64> T,
             typename I0, StridesTraits S0,
             typename I1, StridesTraits S1>
    void matrix2euler(
        const Span<Mat33<T>, 1, I0, S0>& matrices,
        const Span<const Vec<T, 3>, 1, I1, S1>& angles,
        const EulerOptions& options = {}
    ) {
        check(angles.size() == matrices.size(), "angles and matrices don't have the same size");
        for (usize batch{}; batch < angles.size(); ++batch)
            angles[batch] = matrix2euler(matrices[batch], options);
    }
}
