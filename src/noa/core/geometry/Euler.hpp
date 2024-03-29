#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/core/Math.hpp"
#include "noa/core/Types.hpp"

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
    /// \param batches              Number of sets to convert.
    /// \param axes                 Euler angles axes.
    /// \param intrinsic            Whether the Euler angles are interpreted as intrinsic or extrinsic rotations.
    /// \param right_handed         Whether the Euler angles are interpreted as right or left handed rotations.
    /// \note Rotation matrices are orthogonal, so to take the inverse, a simple transposition is enough.
    template<typename T>
    void euler2matrix(const Vec3<T>* angles, Mat33<T>* matrices, size_t batches,
                      std::string_view axes = "ZYZ",
                      bool intrinsic = true,
                      bool right_handed = true) {
        for (size_t batch = 0; batch < batches; ++batch)
            matrices[batch] = euler2matrix(angles[batch], axes, intrinsic, right_handed);
    }

    /// Derives a set of Euler angles, in radians, from a set of rotation matrices.
    /// \tparam T                   f32 or f64.
    /// \param[in] matrices         3x3 matrices to convert.
    /// \param[out] angles          Output Euler angles.
    /// \param batches              Number of sets to convert.
    /// \param axes                 Euler angles axes.
    /// \param intrinsic            Whether the Euler angles are for intrinsic or extrinsic rotations.
    /// \param right_handed         Whether the Euler angles are for right or left handed rotations.
    /// \note Rotation matrices are orthogonal, so to take the inverse, a simple transposition is enough.
    template<typename T>
    void matrix2euler(const Mat33<T>* matrices, Vec3<T>* angles, size_t batches,
                      std::string_view axes = "ZYZ",
                      bool intrinsic = true,
                      bool right_handed = true) {
        for (size_t batch = 0; batch < batches; ++batch)
            angles[batch] = matrix2euler(matrices[batch], axes, intrinsic, right_handed);
    }
}
