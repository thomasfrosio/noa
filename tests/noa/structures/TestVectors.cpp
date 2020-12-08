#include <catch2/catch.hpp>
#include "../../Helpers.h"

#include "noa/structures/Vectors.h"

using namespace ::Noa;

TEMPLATE_TEST_CASE("Vectors: Int2", "[noa][vectors]",
                   int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t) {

    Int2<TestType> test;
    REQUIRE(test == 0);

}
