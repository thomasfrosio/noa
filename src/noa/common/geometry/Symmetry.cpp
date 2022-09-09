#include "noa/common/Math.h"
#include "noa/common/string/Parse.h"
#include "noa/common/geometry/Symmetry.h"
#include "noa/common/geometry/Transform.h"

// The symmetry operators are taken from cisTEM:
// https://github.com/timothygrant80/cisTEM/blob/master/src/core/symmetry_matrix.cpp
// Note that their SetToValues() function specify values in column-major order, and we use row-major order,
// not that this should matter. RELION's approach might be better, but I don't understand it, so never mind.

// TODO Does the forward or inverse matter here? I would assume it does not, but test by transposing.

namespace {
    using namespace ::noa;

    constexpr float33_t s_matrices_O[] = {
            { 0.000000, 0.000000,-1.000000, 1.000000, 0.000000, 0.000000, 0.000000,-1.000000, 0.000000},
            { 0.000000, 0.000000,-1.000000, 0.000000,-1.000000, 0.000000,-1.000000, 0.000000, 0.000000},
            { 0.000000, 0.000000,-1.000000,-1.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000},
            { 0.000000, 0.000000,-1.000000, 0.000000, 1.000000, 0.000000, 1.000000, 0.000000, 0.000000},
            { 0.000000, 0.000000, 1.000000, 0.000000, 1.000000, 0.000000,-1.000000, 0.000000, 0.000000},
            { 0.000000, 0.000000, 1.000000,-1.000000, 0.000000, 0.000000, 0.000000,-1.000000, 0.000000},
            { 0.000000, 0.000000, 1.000000, 0.000000,-1.000000, 0.000000, 1.000000, 0.000000, 0.000000},
            { 0.000000, 0.000000, 1.000000, 1.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000},
            { 0.000000,-1.000000, 0.000000, 0.000000, 0.000000, 1.000000,-1.000000, 0.000000, 0.000000},
            { 0.000000,-1.000000, 0.000000,-1.000000, 0.000000, 0.000000, 0.000000, 0.000000,-1.000000},
            { 0.000000,-1.000000, 0.000000, 0.000000, 0.000000,-1.000000, 1.000000, 0.000000, 0.000000},
            { 0.000000,-1.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000},
            { 0.000000, 1.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000,-1.000000},
            { 0.000000, 1.000000, 0.000000, 0.000000, 0.000000,-1.000000,-1.000000, 0.000000, 0.000000},
            { 0.000000, 1.000000, 0.000000,-1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000},
            { 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 1.000000, 1.000000, 0.000000, 0.000000},
            {-1.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000,-1.000000},
            {-1.000000, 0.000000, 0.000000, 0.000000, 0.000000,-1.000000, 0.000000,-1.000000, 0.000000},
            {-1.000000, 0.000000, 0.000000, 0.000000,-1.000000, 0.000000, 0.000000, 0.000000, 1.000000},
            {-1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 1.000000, 0.000000},
            { 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000,-1.000000, 0.000000},
            { 1.000000, 0.000000, 0.000000, 0.000000,-1.000000, 0.000000, 0.000000, 0.000000,-1.000000},
            { 1.000000, 0.000000, 0.000000, 0.000000, 0.000000,-1.000000, 0.000000, 1.000000, 0.000000}
    };

    constexpr float33_t s_matrices_I1[] = {
            { 0.309017, 0.500000,-0.809017, 0.500000,-0.809017,-0.309017,-0.809017,-0.309017,-0.500000},
            { 0.809017, 0.309017,-0.500000,-0.309017,-0.500000,-0.809017,-0.500000, 0.809017,-0.309017},
            { 0.809017,-0.309017,-0.500000,-0.309017, 0.500000,-0.809017, 0.500000, 0.809017, 0.309017},
            { 0.309017,-0.500000,-0.809017, 0.500000, 0.809017,-0.309017, 0.809017,-0.309017, 0.500000},
            { 0.000000, 0.000000,-1.000000, 1.000000, 0.000000, 0.000000, 0.000000,-1.000000, 0.000000},
            {-0.500000,-0.809017,-0.309017, 0.809017,-0.309017,-0.500000, 0.309017,-0.500000, 0.809017},
            {-0.309017,-0.500000,-0.809017, 0.500000,-0.809017, 0.309017,-0.809017,-0.309017, 0.500000},
            { 0.309017,-0.500000,-0.809017,-0.500000,-0.809017, 0.309017,-0.809017, 0.309017,-0.500000},
            { 0.500000,-0.809017,-0.309017,-0.809017,-0.309017,-0.500000, 0.309017, 0.500000,-0.809017},
            { 0.000000,-1.000000, 0.000000, 0.000000, 0.000000,-1.000000, 1.000000, 0.000000, 0.000000},
            { 0.809017,-0.309017, 0.500000, 0.309017,-0.500000,-0.809017, 0.500000, 0.809017,-0.309017},
            { 0.500000,-0.809017, 0.309017, 0.809017, 0.309017,-0.500000, 0.309017, 0.500000, 0.809017},
            { 0.500000,-0.809017,-0.309017, 0.809017, 0.309017, 0.500000,-0.309017,-0.500000, 0.809017},
            { 0.809017,-0.309017,-0.500000, 0.309017,-0.500000, 0.809017,-0.500000,-0.809017,-0.309017},
            { 1.000000, 0.000000, 0.000000, 0.000000,-1.000000, 0.000000, 0.000000, 0.000000,-1.000000},
            {-0.309017,-0.500000,-0.809017,-0.500000, 0.809017,-0.309017, 0.809017, 0.309017,-0.500000},
            {-0.809017,-0.309017,-0.500000, 0.309017, 0.500000,-0.809017, 0.500000,-0.809017,-0.309017},
            {-0.809017, 0.309017,-0.500000, 0.309017,-0.500000,-0.809017,-0.500000,-0.809017, 0.309017},
            {-0.309017, 0.500000,-0.809017,-0.500000,-0.809017,-0.309017,-0.809017, 0.309017, 0.500000},
            { 0.000000, 0.000000,-1.000000,-1.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000},
            { 0.500000, 0.809017,-0.309017,-0.809017, 0.309017,-0.500000,-0.309017, 0.500000, 0.809017},
            { 0.309017, 0.500000,-0.809017,-0.500000, 0.809017, 0.309017, 0.809017, 0.309017, 0.500000},
            {-0.309017, 0.500000,-0.809017, 0.500000, 0.809017, 0.309017, 0.809017,-0.309017,-0.500000},
            {-0.500000, 0.809017,-0.309017, 0.809017, 0.309017,-0.500000,-0.309017,-0.500000,-0.809017},
            { 0.000000, 1.000000, 0.000000, 0.000000, 0.000000,-1.000000,-1.000000, 0.000000, 0.000000},
            {-0.809017, 0.309017, 0.500000,-0.309017, 0.500000,-0.809017,-0.500000,-0.809017,-0.309017},
            {-0.500000, 0.809017, 0.309017,-0.809017,-0.309017,-0.500000,-0.309017,-0.500000, 0.809017},
            {-0.500000, 0.809017,-0.309017,-0.809017,-0.309017, 0.500000, 0.309017, 0.500000, 0.809017},
            {-0.809017, 0.309017,-0.500000,-0.309017, 0.500000, 0.809017, 0.500000, 0.809017,-0.309017},
            {-1.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000,-1.000000},
            {-0.309017, 0.500000, 0.809017,-0.500000,-0.809017, 0.309017, 0.809017,-0.309017, 0.500000},
            {-0.809017, 0.309017, 0.500000, 0.309017,-0.500000, 0.809017, 0.500000, 0.809017, 0.309017},
            {-0.809017,-0.309017, 0.500000, 0.309017, 0.500000, 0.809017,-0.500000, 0.809017,-0.309017},
            {-0.309017,-0.500000, 0.809017,-0.500000, 0.809017, 0.309017,-0.809017,-0.309017,-0.500000},
            { 0.000000, 0.000000, 1.000000,-1.000000, 0.000000, 0.000000, 0.000000,-1.000000, 0.000000},
            { 0.500000,-0.809017, 0.309017,-0.809017,-0.309017, 0.500000,-0.309017,-0.500000,-0.809017},
            { 0.309017,-0.500000, 0.809017,-0.500000,-0.809017,-0.309017, 0.809017,-0.309017,-0.500000},
            {-0.309017,-0.500000, 0.809017, 0.500000,-0.809017,-0.309017, 0.809017, 0.309017, 0.500000},
            {-0.500000,-0.809017, 0.309017, 0.809017,-0.309017, 0.500000,-0.309017, 0.500000, 0.809017},
            { 0.000000,-1.000000, 0.000000, 0.000000, 0.000000, 1.000000,-1.000000, 0.000000, 0.000000},
            {-0.809017,-0.309017,-0.500000,-0.309017,-0.500000, 0.809017,-0.500000, 0.809017, 0.309017},
            {-0.500000,-0.809017,-0.309017,-0.809017, 0.309017, 0.500000,-0.309017, 0.500000,-0.809017},
            {-0.500000,-0.809017, 0.309017,-0.809017, 0.309017,-0.500000, 0.309017,-0.500000,-0.809017},
            {-0.809017,-0.309017, 0.500000,-0.309017,-0.500000,-0.809017, 0.500000,-0.809017, 0.309017},
            {-1.000000, 0.000000, 0.000000, 0.000000,-1.000000, 0.000000, 0.000000, 0.000000, 1.000000},
            { 0.309017,-0.500000, 0.809017, 0.500000, 0.809017, 0.309017,-0.809017, 0.309017, 0.500000},
            { 0.809017,-0.309017, 0.500000,-0.309017, 0.500000, 0.809017,-0.500000,-0.809017, 0.309017},
            { 0.809017, 0.309017, 0.500000,-0.309017,-0.500000, 0.809017, 0.500000,-0.809017,-0.309017},
            { 0.309017, 0.500000, 0.809017, 0.500000,-0.809017, 0.309017, 0.809017, 0.309017,-0.500000},
            { 0.000000, 0.000000, 1.000000, 1.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000},
            {-0.500000, 0.809017, 0.309017, 0.809017, 0.309017, 0.500000, 0.309017, 0.500000,-0.809017},
            {-0.309017, 0.500000, 0.809017, 0.500000, 0.809017,-0.309017,-0.809017, 0.309017,-0.500000},
            { 0.309017, 0.500000, 0.809017,-0.500000, 0.809017,-0.309017,-0.809017,-0.309017, 0.500000},
            { 0.500000, 0.809017, 0.309017,-0.809017, 0.309017, 0.500000, 0.309017,-0.500000, 0.809017},
            { 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 1.000000, 1.000000, 0.000000, 0.000000},
            { 0.809017, 0.309017,-0.500000, 0.309017, 0.500000, 0.809017, 0.500000,-0.809017, 0.309017},
            { 0.500000, 0.809017,-0.309017, 0.809017,-0.309017, 0.500000, 0.309017,-0.500000,-0.809017},
            { 0.500000, 0.809017, 0.309017, 0.809017,-0.309017,-0.500000,-0.309017, 0.500000,-0.809017},
            { 0.809017, 0.309017, 0.500000, 0.309017, 0.500000,-0.809017,-0.500000, 0.809017, 0.309017}
    };

    constexpr float33_t s_matrices_I2[] = {
            {-0.809017,-0.500000,-0.309017, 0.500000,-0.309017,-0.809017, 0.309017,-0.809017, 0.500000},
            {-0.500000,-0.309017, 0.809017,-0.309017,-0.809017,-0.500000, 0.809017,-0.500000, 0.309017},
            { 0.500000, 0.309017, 0.809017,-0.309017,-0.809017, 0.500000, 0.809017,-0.500000,-0.309017},
            { 0.809017, 0.500000,-0.309017, 0.500000,-0.309017, 0.809017, 0.309017,-0.809017,-0.500000},
            { 0.000000, 0.000000,-1.000000, 1.000000, 0.000000, 0.000000, 0.000000,-1.000000, 0.000000},
            { 0.500000,-0.309017, 0.809017, 0.309017,-0.809017,-0.500000, 0.809017, 0.500000,-0.309017},
            { 0.309017, 0.809017, 0.500000, 0.809017,-0.500000, 0.309017, 0.500000, 0.309017,-0.809017},
            {-0.309017, 0.809017,-0.500000, 0.809017, 0.500000, 0.309017, 0.500000,-0.309017,-0.809017},
            {-0.500000,-0.309017,-0.809017, 0.309017, 0.809017,-0.500000, 0.809017,-0.500000,-0.309017},
            { 0.000000,-1.000000, 0.000000, 0.000000, 0.000000,-1.000000, 1.000000, 0.000000, 0.000000},
            { 0.309017, 0.809017,-0.500000, 0.809017,-0.500000,-0.309017,-0.500000,-0.309017,-0.809017},
            {-0.809017, 0.500000,-0.309017, 0.500000, 0.309017,-0.809017,-0.309017,-0.809017,-0.500000},
            {-0.809017,-0.500000, 0.309017,-0.500000, 0.309017,-0.809017, 0.309017,-0.809017,-0.500000},
            { 0.309017,-0.809017, 0.500000,-0.809017,-0.500000,-0.309017, 0.500000,-0.309017,-0.809017},
            { 1.000000, 0.000000, 0.000000, 0.000000,-1.000000, 0.000000, 0.000000, 0.000000,-1.000000},
            { 0.809017, 0.500000,-0.309017,-0.500000, 0.309017,-0.809017,-0.309017, 0.809017, 0.500000},
            { 0.500000, 0.309017, 0.809017, 0.309017, 0.809017,-0.500000,-0.809017, 0.500000, 0.309017},
            {-0.500000,-0.309017, 0.809017, 0.309017, 0.809017, 0.500000,-0.809017, 0.500000,-0.309017},
            {-0.809017,-0.500000,-0.309017,-0.500000, 0.309017, 0.809017,-0.309017, 0.809017,-0.500000},
            { 0.000000, 0.000000,-1.000000,-1.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000},
            {-0.500000, 0.309017, 0.809017,-0.309017, 0.809017,-0.500000,-0.809017,-0.500000,-0.309017},
            {-0.309017,-0.809017, 0.500000,-0.809017, 0.500000, 0.309017,-0.500000,-0.309017,-0.809017},
            { 0.309017,-0.809017,-0.500000,-0.809017,-0.500000, 0.309017,-0.500000, 0.309017,-0.809017},
            { 0.500000, 0.309017,-0.809017,-0.309017,-0.809017,-0.500000,-0.809017, 0.500000,-0.309017},
            { 0.000000, 1.000000, 0.000000, 0.000000, 0.000000,-1.000000,-1.000000, 0.000000, 0.000000},
            {-0.309017,-0.809017,-0.500000,-0.809017, 0.500000,-0.309017, 0.500000, 0.309017,-0.809017},
            { 0.809017,-0.500000,-0.309017,-0.500000,-0.309017,-0.809017, 0.309017, 0.809017,-0.500000},
            { 0.809017, 0.500000, 0.309017, 0.500000,-0.309017,-0.809017,-0.309017, 0.809017,-0.500000},
            {-0.309017, 0.809017, 0.500000, 0.809017, 0.500000,-0.309017,-0.500000, 0.309017,-0.809017},
            {-1.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000,-1.000000},
            { 0.809017,-0.500000, 0.309017,-0.500000,-0.309017, 0.809017,-0.309017,-0.809017,-0.500000},
            { 0.500000,-0.309017,-0.809017, 0.309017,-0.809017, 0.500000,-0.809017,-0.500000,-0.309017},
            {-0.500000, 0.309017,-0.809017, 0.309017,-0.809017,-0.500000,-0.809017,-0.500000, 0.309017},
            {-0.809017, 0.500000, 0.309017,-0.500000,-0.309017,-0.809017,-0.309017,-0.809017, 0.500000},
            { 0.000000, 0.000000, 1.000000,-1.000000, 0.000000, 0.000000, 0.000000,-1.000000, 0.000000},
            {-0.500000,-0.309017,-0.809017,-0.309017,-0.809017, 0.500000,-0.809017, 0.500000, 0.309017},
            {-0.309017, 0.809017,-0.500000,-0.809017,-0.500000,-0.309017,-0.500000, 0.309017, 0.809017},
            { 0.309017, 0.809017, 0.500000,-0.809017, 0.500000,-0.309017,-0.500000,-0.309017, 0.809017},
            { 0.500000,-0.309017, 0.809017,-0.309017, 0.809017, 0.500000,-0.809017,-0.500000, 0.309017},
            { 0.000000,-1.000000, 0.000000, 0.000000, 0.000000, 1.000000,-1.000000, 0.000000, 0.000000},
            {-0.309017, 0.809017, 0.500000,-0.809017,-0.500000, 0.309017, 0.500000,-0.309017, 0.809017},
            { 0.809017, 0.500000, 0.309017,-0.500000, 0.309017, 0.809017, 0.309017,-0.809017, 0.500000},
            { 0.809017,-0.500000,-0.309017, 0.500000, 0.309017, 0.809017,-0.309017,-0.809017, 0.500000},
            {-0.309017,-0.809017,-0.500000, 0.809017,-0.500000, 0.309017,-0.500000,-0.309017, 0.809017},
            {-1.000000, 0.000000, 0.000000, 0.000000,-1.000000, 0.000000, 0.000000, 0.000000, 1.000000},
            {-0.809017, 0.500000, 0.309017, 0.500000, 0.309017, 0.809017, 0.309017, 0.809017,-0.500000},
            {-0.500000, 0.309017,-0.809017,-0.309017, 0.809017, 0.500000, 0.809017, 0.500000,-0.309017},
            { 0.500000,-0.309017,-0.809017,-0.309017, 0.809017,-0.500000, 0.809017, 0.500000, 0.309017},
            { 0.809017,-0.500000, 0.309017, 0.500000, 0.309017,-0.809017, 0.309017, 0.809017, 0.500000},
            { 0.000000, 0.000000, 1.000000, 1.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000},
            { 0.500000, 0.309017,-0.809017, 0.309017, 0.809017, 0.500000, 0.809017,-0.500000, 0.309017},
            { 0.309017,-0.809017,-0.500000, 0.809017, 0.500000,-0.309017, 0.500000,-0.309017, 0.809017},
            {-0.309017,-0.809017, 0.500000, 0.809017,-0.500000,-0.309017, 0.500000, 0.309017, 0.809017},
            {-0.500000, 0.309017, 0.809017, 0.309017,-0.809017, 0.500000, 0.809017, 0.500000, 0.309017},
            { 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 1.000000, 1.000000, 0.000000, 0.000000},
            { 0.309017,-0.809017, 0.500000, 0.809017, 0.500000, 0.309017,-0.500000, 0.309017, 0.809017},
            {-0.809017,-0.500000, 0.309017, 0.500000,-0.309017, 0.809017,-0.309017, 0.809017, 0.500000},
            {-0.809017, 0.500000,-0.309017,-0.500000,-0.309017, 0.809017, 0.309017, 0.809017, 0.500000},
            { 0.309017, 0.809017,-0.500000,-0.809017, 0.500000, 0.309017, 0.500000, 0.309017, 0.809017}
    };

    // Axial on Z.
    constexpr void setCX(float33_t* rotm, size_t order) {
        NOA_ASSERT(order > 0);
        const float angle = math::Constants<float>::PI2 / static_cast<float>(order);
        for (size_t i = 1; i < order; ++i) // skip the identity
            rotm[i - 1] = geometry::rotateZ(static_cast<float>(i) * angle);
    }

    // Axial on Z, plus 2-fold on X
    constexpr void setDX(float33_t* rotm, size_t order) {
        NOA_ASSERT(order > 0);
        setCX(rotm, order);
        float33_t two_fold_x(-1, +1, +1,
                             +1, -1, -1,
                             +1, +1, +1); // TODO I don't entirely get this...
        rotm[order - 1] = math::elementMultiply(float33_t(), two_fold_x); // -1 since the identity isn't there
        for (size_t i = 0; i < order - 1; ++i)
            rotm[order + i] = math::elementMultiply(rotm[i], two_fold_x);
    }
}

namespace noa::geometry {
    Symmetry::Symbol Symmetry::parse(std::string_view symbol) {
        Symbol out = parseSymbol_(symbol);
        switch (out.type) {
            case 'C':
            case 'D':
                if (out.order > 0)
                    return out;
                break;
            case 'I':
                if (out.order == 1 || out.order == 2)
                    return out;
                break;
            case 'O':
                if (out.order == 0)
                    return out;
                break;
            default:
                break;
        }
        NOA_THROW("Failed to parse \"{}\" to a valid symmetry. Should be CX, DX, O, I1 or I2", symbol);
    }

    Symmetry::Symbol Symmetry::parseSymbol_(std::string_view symbol) {
        symbol = string::trim(symbol);
        if (symbol.empty())
            NOA_THROW("Input symmetry string is empty");

        Symmetry::Symbol out{};
        try {
            out.type = static_cast<char>(std::toupper(static_cast<unsigned char>(symbol[0])));

            if (symbol.size() > 1) {
                std::string number(symbol, 1, symbol.length()); // offset by 1
                out.order = string::parse<ushort>(number);
            } else {
                out.order = 0;
            }
        } catch (...) {
            NOA_THROW("Failed to parse \"{}\" to a symmetry symbol", symbol);
        }
        return out;
    }

    void Symmetry::parseAndSetMatrices_(std::string_view symbol) {
        m_symbol = parseSymbol_(symbol);

        // Set the "interface" pointer.
        constexpr int IDENTITY = 1; // remove the identity from the matrices
        switch (m_symbol.type) {
            case 'C':
                if (m_symbol.order > 0) {
                    m_count = m_symbol.order - IDENTITY;
                    m_buffer = std::make_unique<float33_t[]>(m_count);
                    setCX(m_buffer.get(), m_symbol.order);
                    m_matrices = m_buffer.get();
                    return;
                }
                break;
            case 'D':
                if (m_symbol.order > 0) {
                    m_count = 2 * m_symbol.order - IDENTITY;
                    m_buffer = std::make_unique<float33_t[]>(m_count);
                    setDX(m_buffer.get(), m_symbol.order);
                    m_matrices = m_buffer.get();
                    return;
                }
                break;
            case 'I':
                if (m_symbol.order == 1) {
                    m_count = 60 - IDENTITY;
                    m_matrices = s_matrices_I1;
                    return;
                } else if (m_symbol.order == 2) {
                    m_count = 60 - IDENTITY;
                    m_matrices = s_matrices_I2;
                    return;
                }
                break;
            case 'O':
                if (m_symbol.order == 0) {
                    m_count = 24 - IDENTITY;
                    m_matrices = s_matrices_O;
                    return;
                }
                break;
        }
        NOA_THROW("Failed to parse \"{}\" to a valid symmetry. Should be CX, DX, O, I1 or I2", symbol);
    }
}
