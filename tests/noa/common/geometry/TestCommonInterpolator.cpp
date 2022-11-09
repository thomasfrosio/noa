#include "noa/common/geometry/Interpolator.h"

#include <catch2/catch.hpp>

// This is tested extensively in cpu::geometry, and this was
// just added during the development to check if it compiled.
TEST_CASE("common::geometry, Interpolator", "[.]") {
    using namespace ::noa;

    const int3_t shape0;
    Accessor<float, 4, uint32_t> accessor0;
    geometry::interpolator3D<BORDER_MIRROR, INTERP_CUBIC_BSPLINE>(accessor0, shape0, 0.f)({}, 0);
    geometry::interpolator3D<BORDER_MIRROR, INTERP_CUBIC>(accessor0, shape0, 0.f)({}, 0);
    geometry::interpolator3D<BORDER_MIRROR, INTERP_NEAREST>(accessor0, shape0, 0.f)({}, 0);

    Accessor<float, 3, int32_t> accessor1;
    geometry::interpolator3D<BORDER_MIRROR, INTERP_CUBIC_BSPLINE>(accessor1, shape0, 0.f)({});
    geometry::interpolator3D<BORDER_MIRROR, INTERP_CUBIC>(accessor1, shape0, 0.f)({});
    geometry::interpolator3D<BORDER_MIRROR, INTERP_NEAREST>(accessor1, shape0, 0.f)({});

    const int2_t shape1;
    Accessor<const float, 3, uint32_t> accessor2;
    geometry::interpolator2D<BORDER_MIRROR, INTERP_CUBIC_BSPLINE>(accessor2, shape1, 0.f)({});
    geometry::interpolator2D<BORDER_MIRROR, INTERP_CUBIC>(accessor2, shape1, 0.f)({}, 0);
    geometry::interpolator2D<BORDER_MIRROR, INTERP_NEAREST>(accessor2, shape1, 0.f)({}, 0);

    Accessor<float, 2, int32_t> accessor3;
    geometry::interpolator2D<BORDER_MIRROR, INTERP_CUBIC_BSPLINE>(accessor3, shape1, 0.f)({});
    geometry::interpolator2D<BORDER_MIRROR, INTERP_CUBIC>(accessor3, shape1, 0.f)({});
    geometry::interpolator2D<BORDER_MIRROR, INTERP_NEAREST>(accessor3, shape1, 0.f)({});
}
