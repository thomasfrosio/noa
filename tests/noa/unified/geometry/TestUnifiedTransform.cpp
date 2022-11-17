#include <noa/common/io/MRCFile.h>
#include "noa/common/geometry/Transform.h"
#include <noa/unified/math/Ewise.h>
#include <noa/unified/math/Random.h>
#include <noa/unified/math/Reduce.h>
#include <noa/unified/geometry/Transform.h>
#include <noa/unified/geometry/Prefilter.h>
#include <noa/unified/signal/Shape.h>

#include <catch2/catch.hpp>
#include "Helpers.h"

using namespace ::noa;

TEMPLATE_TEST_CASE("unified::geometry::transform", "[noa][unified]", float, cfloat_t, double, cdouble_t) {
//    Array<float> a;
//    Array<float23_t> a0;
//    Texture<float> b;
//    geometry::transform2D(a, a, float23_t{});
//    geometry::transform2D(b, a, a0);
//
//    geometry::transform2D(b, a, float2_t{}, float22_t{}, geometry::Symmetry{}, float2_t{}, true);
    // Test GPU against CPU, with batches, textures, multiple matrices.

}

