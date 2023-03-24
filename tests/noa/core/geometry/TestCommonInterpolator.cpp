#include "noa/core/geometry/Interpolator.hpp"

#include <catch2/catch.hpp>

// The interpolators are tested extensively already, and this was
// just added during the development to check if it compiled.
TEST_CASE("core::geometry, Interpolator", "[noa][core]") {
    using namespace noa;

    const Shape3<int32_t> shape0;
    const Accessor<float, 4, uint32_t> accessor0;
    noa::geometry::interpolator_3d<BorderMode::MIRROR, InterpMode::CUBIC_BSPLINE>(accessor0, shape0, 0.f);
    noa::geometry::interpolator_3d<BorderMode::MIRROR, InterpMode::CUBIC>(accessor0, shape0, 0.f);
    noa::geometry::interpolator_3d<BorderMode::MIRROR, InterpMode::NEAREST>(accessor0, shape0, 0.f);

    const Accessor<float, 3, int32_t> accessor1;
    noa::geometry::interpolator_3d<BorderMode::MIRROR, InterpMode::CUBIC_BSPLINE>(accessor1, shape0, 0.f);
    noa::geometry::interpolator_3d<BorderMode::MIRROR, InterpMode::CUBIC>(accessor1, shape0, 0.f);
    noa::geometry::interpolator_3d<BorderMode::MIRROR, InterpMode::NEAREST>(accessor1, shape0, 0.f);

    const Shape2<int64_t> shape1;
    const Accessor<const float, 3, uint64_t> accessor2;
    noa::geometry::interpolator_2d<BorderMode::MIRROR, InterpMode::CUBIC_BSPLINE>(accessor2, shape1, 0.f);
    noa::geometry::interpolator_2d<BorderMode::MIRROR, InterpMode::CUBIC>(accessor2, shape1, 0.f);
    noa::geometry::interpolator_2d<BorderMode::MIRROR, InterpMode::NEAREST>(accessor2, shape1, 0.f);

    const Accessor<float, 2, int64_t> accessor3;
    noa::geometry::interpolator_2d<BorderMode::MIRROR, InterpMode::CUBIC_BSPLINE>(accessor3, shape1, 0.f);
    noa::geometry::interpolator_2d<BorderMode::MIRROR, InterpMode::CUBIC>(accessor3, shape1, 0.f);
    noa::geometry::interpolator_2d<BorderMode::MIRROR, InterpMode::NEAREST>(accessor3, shape1, 0.f);
}
