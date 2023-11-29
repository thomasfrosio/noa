#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/types/Mat.hpp"
#include "noa/core/types/Span.hpp"

namespace noa::geometry {
    /// Extracts the 3x3 rotation matrix from the Euler angles.
    /// \tparam T           f32 or double.
    /// \param angles       Euler angles, in radians.
    /// \param axes         Euler angles axes.
    /// \param intrinsic    Whether the Euler angles are interpreted as intrinsic or extrinsic rotations.
    /// \param right_handed Whether the Euler angles are interpreted as right or left handed rotations.
    /// \note Rotation matrices are orthogonal, so to take the inverse, a simple transposition is enough.
    template<typename T>
    Mat33<T> euler2matrix(Vec3<T> angles,
                          std::string_view axes = "ZYZ",
                          bool intrinsic = true,
                          bool right_handed = true);

    /// Derives the Euler angles, in radians, from the rotation matrix.
    /// \tparam T           f32 or f64.
    /// \param rotation     Rotation (orthogonal) matrix to decompose.
    /// \param axes         Euler angles axes.
    /// \param intrinsic    Whether the Euler angles are for intrinsic or extrinsic rotations.
    /// \param right_handed Whether the Euler angles are for right or left handed rotations.
    /// \note Rotation matrices are assumed to be orthogonal, so to take the inverse, a simple transposition is enough.
    template<typename T>
    Vec3<T> matrix2euler(const Mat33<T>& rotation,
                         std::string_view axes = "ZYZ",
                         bool intrinsic = true,
                         bool right_handed = true);

    /// Extracts a set of 3x3 rotation matrices from the Euler angles.
    /// \tparam T                   f32 or f64.
    /// \param[in] angles           Euler angles, in radians, to convert.
    /// \param[out] matrices        Output 3x3 matrices.
    /// \param axes                 Euler angles axes.
    /// \param intrinsic            Whether the Euler angles are interpreted as intrinsic or extrinsic rotations.
    /// \param right_handed         Whether the Euler angles are interpreted as right or left handed rotations.
    /// \note Rotation matrices are orthogonal, so to take the inverse, a simple transposition is enough.
    template<typename T>
    void euler2matrix(
            const Span<const Vec3<T>>& angles,
            const Span<Mat33<T>>& matrices,
            std::string_view axes = "ZYZ",
            bool intrinsic = true,
            bool right_handed = true
    ) {
        check(angles.size() == matrices.size(), "angles and matrices don't have the same size");
        for (size_t batch = 0; batch < angles.size(); ++batch)
            matrices[batch] = euler2matrix(angles[batch], axes, intrinsic, right_handed);
    }

    /// Derives a set of Euler angles, in radians, from a set of rotation matrices.
    /// \tparam T                   f32 or f64.
    /// \param[in] matrices         3x3 matrices to convert.
    /// \param[out] angles          Output Euler angles.
    /// \param axes                 Euler angles axes.
    /// \param intrinsic            Whether the Euler angles are for intrinsic or extrinsic rotations.
    /// \param right_handed         Whether the Euler angles are for right or left handed rotations.
    /// \note Rotation matrices are orthogonal, so to take the inverse, a simple transposition is enough.
    template<typename T>
    void matrix2euler(
            const Span<const Mat33<T>>& matrices,
            const Span<Vec3<T>>& angles,
            std::string_view axes = "ZYZ",
            bool intrinsic = true,
            bool right_handed = true
    ) {
        check(angles.size() == matrices.size(), "angles and matrices don't have the same size");
        for (size_t batch = 0; batch < angles.size(); ++batch)
            angles[batch] = matrix2euler(matrices[batch], axes, intrinsic, right_handed);
    }
}
