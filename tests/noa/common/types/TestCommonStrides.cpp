#include <noa/common/types/Int3.h>
#include <noa/common/types/Int4.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("stride(), pitch()", "[noa][common]") {
    AND_THEN("contiguous") {
        const size4_t shape{2, 128, 64, 65};
        const size4_t stride{shape.stride()};
        const size3_t pitch{stride.pitch()};
        REQUIRE(all(indexing::isContiguous(stride, shape)));
        REQUIRE(all(size4_t{532480, 4160, 65, 1} == stride));
        REQUIRE(all(size3_t{128, 64, 65} == pitch));
    }

    AND_THEN("stride X") {
        const size4_t shape{3, 128, 64, 64};
        const size4_t stride{shape.stride() * 2};
        const size3_t pitch{stride.pitch()};
        REQUIRE(all(indexing::isContiguous(stride, shape) == bool4_t{1, 1, 1, 0}));
        REQUIRE(all(size4_t{1048576, 8192, 128, 2} == stride));
        REQUIRE(all(size3_t{128, 64, 128} == pitch));
    }

    AND_THEN("indexing::isContiguous") {
        AND_THEN("Y") {
            const size4_t shape{3, 128, 64, 64};
            size4_t stride{shape.stride()};
            stride[0] *= 2;
            stride[1] *= 2;
            stride[2] *= 2;
            REQUIRE(all(indexing::isContiguous(stride, shape) == bool4_t{1, 1, 0, 1}));
        }

        AND_THEN("YX") {
            const size4_t shape{3, 128, 64, 64};
            size4_t stride{shape.stride()};
            stride[0] *= 6;
            stride[1] *= 6;
            stride[2] *= 6;
            stride[3] *= 3;
            REQUIRE(all(indexing::isContiguous(stride, shape) == bool4_t{1, 1, 0, 0}));
        }

        AND_THEN("Z") {
            const size4_t shape{3, 128, 64, 64};
            size4_t stride{shape.stride()};
            stride[0] *= 2;
            stride[1] *= 2;
            REQUIRE(all(indexing::isContiguous(stride, shape) == bool4_t{1, 0, 1, 1}));
        }

        AND_THEN("W") {
            const size4_t shape{3, 128, 64, 64};
            size4_t stride{shape.stride()};
            stride[0] *= 2;
            REQUIRE(all(indexing::isContiguous(stride, shape) == bool4_t{0, 1, 1, 1}));
        }

        AND_THEN("WZ") {
            const size4_t shape{3, 128, 64, 64};
            size4_t stride{shape.stride()};
            stride[0] *= 4;
            stride[1] *= 2;
            REQUIRE(all(indexing::isContiguous(stride, shape) == bool4_t{0, 0, 1, 1}));
        }

        AND_THEN("ZX, empty dim") {
            // For now, I think empty dimensions should not be contiguous by default because it
            // can affect other (outer) dimensions...
            const size4_t shape{3, 128, 1, 64};
            size4_t stride{shape.stride()};
            stride[0] *= 4;
            stride[1] *= 4;
            stride[2] *= 4;
            stride[3] *= 2;
            REQUIRE(all(indexing::isContiguous(stride, shape) == bool4_t{1, 1, 0, 0}));

            stride[0] *= 8;
            stride[1] *= 8;
            stride[2] *= 4;
            stride[3] *= 2;
            REQUIRE(all(indexing::isContiguous(stride, shape) == bool4_t{1, 0, 0, 0}));
        }
    }

    AND_THEN("compute pitches from strides") {
        const uint ndim = GENERATE(1u, 2u, 3u);
        for (size_t i = 0; i < 20; ++i) {
            size4_t shape = test::getRandomShapeBatched(ndim);
            size4_t stride = shape.stride();
            REQUIRE(all(size3_t{shape.get() + 1} == stride.pitch()));

            stride *= 3;
            shape[3] *= 3;
            REQUIRE(all(size3_t{shape.get() + 1} == stride.pitch()));
        }
    }
}
