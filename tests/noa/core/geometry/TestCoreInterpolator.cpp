#include "noa/core/geometry/Interpolator.hpp"

#include <catch2/catch.hpp>

// The interpolators are tested extensively already, and this was
// just added during the development to check if it compiled.
TEST_CASE("core::geometry, Interpolator", "[noa][core]") {
    using namespace noa;
    using namespace noa::geometry;

    const Shape3<int32_t> shape0{};
    const Accessor<float, 4, uint32_t> accessor0;
    interpolator_3d<Border::MIRROR, Interp::CUBIC_BSPLINE>(accessor0, shape0, 0.f);
    interpolator_3d<Border::MIRROR, Interp::CUBIC>(accessor0, shape0, 0.f);
    interpolator_3d<Border::MIRROR, Interp::NEAREST>(accessor0, shape0, 0.f);

    const Accessor<float, 3, int32_t> accessor1;
    interpolator_3d<Border::MIRROR, Interp::CUBIC_BSPLINE>(accessor1, shape0, 0.f);
    interpolator_3d<Border::MIRROR, Interp::CUBIC>(accessor1, shape0, 0.f);
    interpolator_3d<Border::MIRROR, Interp::NEAREST>(accessor1, shape0, 0.f);

    const Shape2<int64_t> shape1{};
    const Accessor<const float, 3, uint64_t> accessor2;
    interpolator_2d<Border::MIRROR, Interp::CUBIC_BSPLINE>(accessor2, shape1, 0.f);
    interpolator_2d<Border::MIRROR, Interp::CUBIC>(accessor2, shape1, 0.f);
    interpolator_2d<Border::MIRROR, Interp::NEAREST>(accessor2, shape1, 0.f);

    const Accessor<float, 2, int64_t> accessor3;
    interpolator_2d<Border::MIRROR, Interp::CUBIC_BSPLINE>(accessor3, shape1, 0.f);
    interpolator_2d<Border::MIRROR, Interp::CUBIC>(accessor3, shape1, 0.f);
    interpolator_2d<Border::MIRROR, Interp::NEAREST>(accessor3, shape1, 0.f);

    AccessorValue<const f32, i64> accessor_value(1.f);
    interpolator_2d<Border::MIRROR, Interp::NEAREST>(accessor_value, shape1, 0.f);
}
