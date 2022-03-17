#include <noa/common/Types.h>
#include <noa/cpu/memory/PtrHost.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("common::BorderMode, getBorderIndex()", "[noa][common]") {
    AND_THEN("BORDER_PERIODIC") {
        int expected_odd[55] = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
                                0, 1, 2, 3, 4,
                                0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4};
        int starts_at = 25;
        int len = 5;
        cpu::memory::PtrHost<int> data(55);
        for (size_t idx = 0; idx < data.size(); ++idx)
            data[idx] = getBorderIndex<BORDER_PERIODIC>(static_cast<int>(idx) - starts_at, len);

        int diff = test::getDifference(expected_odd, data.get(), data.size());
        REQUIRE(diff == 0);

        int expected_even[36] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                                 0, 1, 2, 3,
                                 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
        starts_at = 16;
        len = 4;
        data.reset(32);
        for (size_t idx = 0; idx < data.size(); ++idx)
            data[idx] = getBorderIndex<BORDER_PERIODIC>(static_cast<int>(idx) - starts_at, len);

        diff = test::getDifference(expected_even, data.get(), data.size());
        REQUIRE(diff == 0);
    }

    AND_THEN("BORDER_CLAMP") {
        int expected_odd[35] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 1, 2, 3, 4,
                                4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
        int starts_at = 15;
        int len = 5;
        cpu::memory::PtrHost<int> data(35);
        for (size_t idx = 0; idx < data.size(); ++idx)
            data[idx] = getBorderIndex<BORDER_CLAMP>(static_cast<int>(idx) - starts_at, len);

        int diff = test::getDifference(expected_odd, data.get(), data.size());
        REQUIRE(diff == 0);

        int expected_even[34] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 1, 2, 3,
                                 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
        starts_at = 15;
        len = 4;
        data.reset(34);
        for (size_t idx = 0; idx < data.size(); ++idx)
            data[idx] = getBorderIndex<BORDER_CLAMP>(static_cast<int>(idx) - starts_at, len);

        diff = test::getDifference(expected_even, data.get(), data.size());
        REQUIRE(diff == 0);
    }

    AND_THEN("BORDER_MIRROR") {
        int expected_odd[45] = {0, 1, 2, 3, 4, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 4, 3, 2, 1, 0,
                                0, 1, 2, 3, 4,
                                4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4};
        int starts_at = 20;
        int len = 5;
        cpu::memory::PtrHost<int> data(45);
        for (size_t idx = 0; idx < data.size(); ++idx)
            data[idx] = getBorderIndex<BORDER_MIRROR>(static_cast<int>(idx) - starts_at, len);

        int diff = test::getDifference(expected_odd, data.get(), data.size());
        REQUIRE(diff == 0);

        int expected_even[52] = {0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0,
                                 0, 1, 2, 3,
                                 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3};
        starts_at = 24;
        len = 4;
        data.reset(52);
        for (size_t idx = 0; idx < data.size(); ++idx)
            data[idx] = getBorderIndex<BORDER_MIRROR>(static_cast<int>(idx) - starts_at, len);

        diff = test::getDifference(expected_even, data.get(), data.size());
        REQUIRE(diff == 0);
    }

    AND_THEN("BORDER_REFLECT") {
        int expected_odd[53] = {0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1,
                                0, 1, 2, 3, 4,
                                3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4};
        int starts_at = 24;
        int len = 5;
        cpu::memory::PtrHost<int> data(53);
        for (size_t idx = 0; idx < data.size(); ++idx)
            data[idx] = getBorderIndex<BORDER_REFLECT>(static_cast<int>(idx) - starts_at, len);

        int diff = test::getDifference(expected_odd, data.get(), data.size());
        REQUIRE(diff == 0);

        int expected_even[40] = {0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1,
                                 0, 1, 2, 3,
                                 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3};
        starts_at = 18;
        len = 4;
        data.reset(40);
        for (size_t idx = 0; idx < data.size(); ++idx)
            data[idx] = getBorderIndex<BORDER_REFLECT>(static_cast<int>(idx) - starts_at, len);

        diff = test::getDifference(expected_even, data.get(), data.size());
        REQUIRE(diff == 0);
    }
}
