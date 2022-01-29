#include <noa/common/types/IntX.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("strides(), pitches()", "[noa][common]") {
    {
        const size4_t shape{2, 128, 64, 65};
        const size4_t stride{shape.strides()};
        const size3_t pitch{stride.pitches()};
        REQUIRE(all(size4_t{532480, 4160, 65, 1} == stride));
        REQUIRE(all(size3_t{128, 64, 65} == pitch));
    }
    {
        const size4_t shape{3, 128, 64, 64};
        const size4_t stride{shape.strides() * 2}; // stride in x of 2
        const size3_t pitch{stride.pitches()};
        REQUIRE(all(size4_t{1048576, 8192, 128, 2} == stride));
        REQUIRE(all(size3_t{128, 64, 128} == pitch));
    }
    {
        const size4_t shape{1, 65, 48, 71};
        size4_t stride{221520, 3408, 71, 1};
        REQUIRE(all(shape.strides() == stride));
        stride *= 4; // stride in x
        size3_t pitch{stride.pitches()};
        REQUIRE(all(size3_t{65, 48, 284} == pitch));
    }
    {
        const uint ndim = GENERATE(1u, 2u, 3u);
        for (size_t i = 0; i < 20; ++i) {
            size4_t shape = test::getRandomShapeBatched(ndim);
            size4_t stride = shape.strides();
            REQUIRE(all(size3_t{shape.get() + 1} == stride.pitches()));

            stride *= 3;
            shape[3] *= 3;
            REQUIRE(all(size3_t{shape.get() + 1} == stride.pitches()));
        }
    }
}
