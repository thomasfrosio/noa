#pragma once

#include "noa/Definitions.h"
#include "noa/Types.h"
#include "noa/Math.h"

// Links:
//  - https://eulerangles.readthedocs.io/en/latest/usage/quick_start.html
//  - https://www.geometrictools.com/Documentation/EulerAngles.pdf
//
// Convention:
//  - The default Euler angles are ZYZ intrinsic (coordinates are attached to the body), i.e. rotate around
//    the Z axis, then around the new Y axis, then around the new Z axis.
//
// TODO Add more Euler angles conversions, e.g. `Float3<T> convert(Float3<T> angles, Convention in, Convention out)`

namespace Noa::Transform {
    /// Extracts the 3x3 rotation matrix from the Euler angles.
    /// \tparam T       float or double.
    /// \param angles   ZYZ intrinsic angles.
    /// \return         3x3 rotation matrix.
    template<typename T>
    NOA_HD Mat3<T> toMatrix(const Float3<T>& angles) {
        // ZYZ intrinsic: Rz(1) * Ry(2) * Rz(3)
        const T c1 = Math::cos(angles.x);
        const T s1 = Math::sin(angles.x);
        const T c2 = Math::cos(angles.y);
        const T s2 = Math::sin(angles.y);
        const T c3 = Math::cos(angles.z);
        const T s3 = Math::sin(angles.z);

        const T A = c1 * c2;
        const T B = s1 * c3;
        const T C = s1 * s3;

        return Mat3<T>(c3 * A - C, -s3 * A - B, c1 * s2,
                       c2 * B + s3 * c1, -c2 * C + c1 * c3, s1 * s2,
                       -s2 * c3, s2 * s3, c2);
    }

    /// Extracts the Euler angles from the 3x3 rotation matrix.
    /// \tparam T   float or double.
    /// \param rm   3x3 rotation matrix.
    /// \return     ZYZ intrinsic angles.
    template<typename T>
    NOA_HD Float3<T> toEuler(const Mat3<T>& rm) {
        // From https://github.com/3dem/relion/blob/master/src/euler.cpp
        T alpha, beta, gamma;
        T abs_sb, sign_sb;
        T float_epsilon = static_cast<T>(Math::Limits<float>::epsilon());

        abs_sb = Math::sqrt(rm[0][2] * rm[0][2] + rm[1][2] * rm[1][2]);
        if (abs_sb > 16 * float_epsilon) {
            gamma = Math::atan2(rm[1][2], -rm[0][2]);
            alpha = Math::atan2(rm[2][1], rm[2][0]);
            if (Math::abs(Math::sin(gamma)) < float_epsilon)
                sign_sb = Math::sign(-rm[0][2] / Math::cos(gamma));
            else
                sign_sb = (Math::sin(gamma) > 0) ? Math::sign(rm[1][2]) : -Math::sign(rm[1][2]);
            beta = Math::atan2(sign_sb * abs_sb, rm[2][2]);
        } else {
            alpha = 0;
            if (Math::sign(rm[2][2]) > 0) {
                beta = 0;
                gamma = Math::atan2(-rm[1][0], rm[0][0]);
            } else {
                beta = Math::Constants<T>::PI;
                gamma = Math::atan2(rm[1][0], -rm[0][0]);
            }
        }

        return Float3<T>(alpha, beta, gamma);
    }
}
