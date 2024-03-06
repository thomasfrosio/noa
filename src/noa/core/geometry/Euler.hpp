#pragma once

#include "noa/core/Config.hpp"

#ifdef NOA_IS_OFFLINE
#include "noa/core/types/Mat.hpp"
#include "noa/core/types/Span.hpp"

namespace noa::geometry {
    struct EulerOptions {
        std::string_view axes{"zyx"};
        bool intrinsic{true};
        bool right_handed{true};
    };

    /// Extracts the 3x3 rotation matrix from the Euler angles.
    /// \tparam T           f32 or double.
    /// \param angles       Euler angles, in radians.
    /// \param options      Euler angles options.
    template<typename T>
    auto euler2matrix(Vec3<T> angles, const EulerOptions& options = {}) -> Mat33<T>;

    /// Derives the Euler angles, in radians, from the rotation matrix.
    /// \tparam T           f32 or f64.
    /// \param rotation     Rotation (orthogonal) matrix to decompose.
    /// \param options      Euler angles options.
    template<typename T>
    auto matrix2euler(const Mat33<T>& rotation, const EulerOptions& options = {}) -> Vec3<T>;

    /// Extracts a set of 3x3 rotation matrices from the Euler angles.
    /// \tparam T               f32 or f64.
    /// \param[in] angles       Euler angles, in radians, to convert.
    /// \param[out] matrices    Output 3x3 matrices.
    /// \param options          Euler angles options.
    template<typename T> requires nt::is_any_v<T, f32, f64>
    void euler2matrix(Span<const Vec3<T>> angles, Span<Mat33<T>> matrices, const EulerOptions& options = {}) {
        check(angles.size() == matrices.size(), "angles and matrices don't have the same size");
        for (size_t batch = 0; batch < angles.size(); ++batch)
            matrices[batch] = euler2matrix(angles[batch], options);
    }

    /// Derives a set of Euler angles, in radians, from a set of rotation matrices.
    /// \tparam T           f32 or f64.
    /// \param[in] matrices 3x3 matrices to convert.
    /// \param[out] angles  Output Euler angles.
    /// \param options      Euler angles options.
    template<typename T> requires nt::is_any_v<T, f32, f64>
    void matrix2euler(Span<const Mat33<T>> matrices, Span<Vec3<T>> angles, const EulerOptions& options = {}) {
        check(angles.size() == matrices.size(), "angles and matrices don't have the same size");
        for (size_t batch = 0; batch < angles.size(); ++batch)
            angles[batch] = matrix2euler(matrices[batch], options);
    }
}
#endif
