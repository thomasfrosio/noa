/**
 *
 * Links
 * =====
 *
 *  - https://eulerangles.readthedocs.io/en/latest/usage/quick_start.html
 *  - https://rock-learning.github.io/pytransform3d/transformation_ambiguities.html
 *  - https://www.geometrictools.com/Documentation/EulerAngles.pdf
 *
 * Conventions
 * ===========
 *
 *  - Transformations are active (alibi), i.e. body rotates about the origin of the coordinate system.
 *  - Transformations assume a right handed coordinate system.
 *  - Angles are given in radians by default.
 *  - Positive angles specify a counter-clockwise rotation when looking at the origin.
 *  - Rotation matrices pre-multiply column vectors to produce transformed column vectors: M * v = v'
 *  - The default Euler angles are ZYZ intrinsic (coordinates are attached to the body), i.e. rotate around
 *    the Z axis, then around the new Y axis, then around the new Z axis.
 *
 * TODO Add more Euler angles conversions.
 */

#pragma once

#include "noa/Definitions.h"
#include "noa/Types.h"

namespace Noa::Euler {
    enum Convention { I_ZYZ, E_ZYZ, I_ZXZ, E_ZXZ };

    NOA_HOST float3_t convert(const float3_t& in_angles, Convention in_convention, Convention out_convention);

    NOA_HOST float33_t toMatrix3(const float3_t& angles);

    NOA_HOST float44_t toMatrix4(const float3_t& angles);

    NOA_HOST float3_t toEuler(const float44_t& rm);

    NOA_HOST float3_t toEuler(const float33_t& rm);

    NOA_HOST std::vector<float3_t> getEqualAngularSpacing(const float2_t& range_phi,
                                                          const float2_t& range_theta,
                                                          const float2_t& range_psi);
}
