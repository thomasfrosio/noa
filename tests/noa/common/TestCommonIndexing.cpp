#include <noa/common/types/Int3.h>
#include <noa/common/types/Int4.h>
#include <noa/cpu/memory/PtrHost.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("common: indexing::at<BorderMode>()", "[noa][common]") {
    AND_THEN("BORDER_PERIODIC") {
        int expected_odd[55] = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
                                0, 1, 2, 3, 4,
                                0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4};
        int starts_at = 25;
        int len = 5;
        cpu::memory::PtrHost<int> data(55);
        for (size_t idx = 0; idx < data.size(); ++idx)
            data[idx] = indexing::at<BORDER_PERIODIC>(static_cast<int>(idx) - starts_at, len);

        int diff = test::getDifference(expected_odd, data.get(), data.size());
        REQUIRE(diff == 0);

        int expected_even[36] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                                 0, 1, 2, 3,
                                 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
        starts_at = 16;
        len = 4;
        data = cpu::memory::PtrHost<int>(32);
        for (size_t idx = 0; idx < data.size(); ++idx)
            data[idx] = indexing::at<BORDER_PERIODIC>(static_cast<int>(idx) - starts_at, len);

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
            data[idx] = indexing::at<BORDER_CLAMP>(static_cast<int>(idx) - starts_at, len);

        int diff = test::getDifference(expected_odd, data.get(), data.size());
        REQUIRE(diff == 0);

        int expected_even[34] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 1, 2, 3,
                                 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
        starts_at = 15;
        len = 4;
        data = cpu::memory::PtrHost<int>(34);
        for (size_t idx = 0; idx < data.size(); ++idx)
            data[idx] = indexing::at<BORDER_CLAMP>(static_cast<int>(idx) - starts_at, len);

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
            data[idx] = indexing::at<BORDER_MIRROR>(static_cast<int>(idx) - starts_at, len);

        int diff = test::getDifference(expected_odd, data.get(), data.size());
        REQUIRE(diff == 0);

        int expected_even[52] = {0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0,
                                 0, 1, 2, 3,
                                 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3};
        starts_at = 24;
        len = 4;
        data = cpu::memory::PtrHost<int>(52);
        for (size_t idx = 0; idx < data.size(); ++idx)
            data[idx] = indexing::at<BORDER_MIRROR>(static_cast<int>(idx) - starts_at, len);

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
            data[idx] = indexing::at<BORDER_REFLECT>(static_cast<int>(idx) - starts_at, len);

        int diff = test::getDifference(expected_odd, data.get(), data.size());
        REQUIRE(diff == 0);

        int expected_even[40] = {0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1,
                                 0, 1, 2, 3,
                                 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3};
        starts_at = 18;
        len = 4;
        data = cpu::memory::PtrHost<int>(40);
        for (size_t idx = 0; idx < data.size(); ++idx)
            data[idx] = indexing::at<BORDER_REFLECT>(static_cast<int>(idx) - starts_at, len);

        diff = test::getDifference(expected_even, data.get(), data.size());
        REQUIRE(diff == 0);
    }
}


TEST_CASE("common: shape, strides, pitch", "[noa][common]") {
    AND_THEN("C- contiguous") {
        const size4_t shape{2, 128, 64, 65};
        const size4_t strides = shape.strides();
        const size3_t pitches = strides.pitches();
        REQUIRE(all(indexing::isContiguous(strides, shape)));
        REQUIRE(indexing::areContiguous(strides, shape));
        REQUIRE(all(size4_t{532480, 4160, 65, 1} == strides));
        REQUIRE(all(size3_t{128, 64, 65} == pitches));
    }

    AND_THEN("F- contiguous") {
        const size4_t shape{2, 128, 64, 65};
        const size4_t strides = shape.strides<'F'>();
        const size3_t pitches = strides.pitches<'F'>();
        REQUIRE(all(indexing::isContiguous<'F'>(strides, shape)));
        REQUIRE(indexing::areContiguous<'F'>(strides, shape));
        REQUIRE(all(size4_t{532480, 4160, 1, 64} == strides));
        REQUIRE(all(size3_t{128, 64, 65} == pitches));
    }

    AND_THEN("C- inner stride") {
        const size4_t shape{3, 128, 64, 64};
        const size4_t strides = shape.strides() * 2;
        const size3_t pitches = strides.pitches();
        REQUIRE(all(indexing::isContiguous(strides, shape) == bool4_t{1, 1, 1, 0}));
        REQUIRE(all(size4_t{1048576, 8192, 128, 2} == strides));
        REQUIRE(all(size3_t{128, 64, 128} == pitches));
    }

    AND_THEN("F- inner stride") {
        const size4_t shape{3, 128, 64, 64};
        const size4_t strides = shape.strides<'F'>() * 2;
        const size3_t pitches = strides.pitches<'F'>();
        REQUIRE(all(indexing::isContiguous<'F'>(strides, shape) == bool4_t{1, 1, 0, 1}));
        REQUIRE(all(size4_t{1048576, 8192, 2, 128} == strides));
        REQUIRE(all(size3_t{128, 128, 64} == pitches));
    }

    AND_THEN("isContiguous<'C'>") {
        size4_t shape{3, 128, 64, 64};
        size4_t strides = shape.strides();
        strides[0] *= 2;
        strides[1] *= 2;
        strides[2] *= 2;
        REQUIRE(all(indexing::isContiguous(strides, shape) == bool4_t{1, 1, 0, 1}));

        strides = shape.strides();
        strides[0] *= 6;
        strides[1] *= 6;
        strides[2] *= 6;
        strides[3] *= 3;
        REQUIRE(all(indexing::isContiguous(strides, shape) == bool4_t{1, 1, 0, 0}));

        strides = shape.strides();
        strides[0] *= 2;
        strides[1] *= 2;
        REQUIRE(all(indexing::isContiguous(strides, shape) == bool4_t{1, 0, 1, 1}));

        strides = shape.strides();
        strides[0] *= 2;
        REQUIRE(all(indexing::isContiguous(strides, shape) == bool4_t{0, 1, 1, 1}));

        strides = shape.strides();
        strides[0] *= 4;
        strides[1] *= 2;
        REQUIRE(all(indexing::isContiguous(strides, shape) == bool4_t{0, 0, 1, 1}));

        shape = size4_t{3, 128, 1, 64};
        strides = size4_t{8192, 64, 64, 1};
        REQUIRE(all(indexing::isContiguous(strides, shape) == bool4_t{1, 1, 1, 1}));
        strides *= 2;
        REQUIRE(all(indexing::isContiguous(strides, shape) == bool4_t{1, 1, 1, 0}));
        strides[2] *= 2;
        REQUIRE(all(indexing::isContiguous(strides, shape) == bool4_t{1, 1, 1, 0}));
        strides[0] *= 2;
        strides[1] *= 2;
        strides[2] *= 2;
        REQUIRE(all(indexing::isContiguous(strides, shape) == bool4_t{1, 0, 1, 0}));
    }

    AND_THEN("isContiguous<'C'> - broadcast") {
        const size4_t original_shape{3, 128, 1, 64};
        const size4_t broadcast_shape{3, 128, 64, 64};
        size4_t strides = original_shape.strides();
        REQUIRE(all(indexing::isContiguous(strides, original_shape) == bool4_t{1, 1, 1, 1}));

        indexing::broadcast(original_shape, strides, size4_t{3, 128, 64, 64});
        REQUIRE(all(indexing::isContiguous(strides, broadcast_shape) == bool4_t{1, 1, 0, 1}));
    }

    AND_THEN("isContiguous<'F'>") {
        size4_t shape{3, 128, 64, 64};
        size4_t strides = shape.strides<'F'>();
        strides[0] *= 2;
        strides[1] *= 2;
        strides[3] *= 2;
        REQUIRE(all(indexing::isContiguous<'F'>(strides, shape) == bool4_t{1, 1, 1, 0}));

        strides = shape.strides<'F'>();
        strides[0] *= 6;
        strides[1] *= 6;
        strides[2] *= 3;
        strides[3] *= 6;
        REQUIRE(all(indexing::isContiguous<'F'>(strides, shape) == bool4_t{1, 1, 0, 0}));

        strides = shape.strides<'F'>();
        strides[0] *= 2;
        strides[1] *= 2;
        REQUIRE(all(indexing::isContiguous<'F'>(strides, shape) == bool4_t{1, 0, 1, 1}));

        strides = shape.strides<'F'>();
        strides[0] *= 2;
        REQUIRE(all(indexing::isContiguous<'F'>(strides, shape) == bool4_t{0, 1, 1, 1}));

        strides = shape.strides<'F'>();
        strides[0] *= 4;
        strides[1] *= 2;
        REQUIRE(all(indexing::isContiguous<'F'>(strides, shape) == bool4_t{0, 0, 1, 1}));

        shape = size4_t{3, 128, 64, 1};
        strides = size4_t{8192, 64, 1, 64};
        REQUIRE(all(indexing::isContiguous<'F'>(strides, shape) == bool4_t{1, 1, 1, 1}));
        strides *= 2;
        REQUIRE(all(indexing::isContiguous<'F'>(strides, shape) == bool4_t{1, 1, 0, 1}));
        strides[3] *= 2;
        REQUIRE(all(indexing::isContiguous<'F'>(strides, shape) == bool4_t{1, 1, 0, 1}));
        strides[0] *= 2;
        strides[1] *= 2;
        strides[3] *= 2;
        REQUIRE(all(indexing::isContiguous<'F'>(strides, shape) == bool4_t{1, 0, 0, 1}));
    }

    AND_THEN("isContiguous<'F'> - broadcast") {
        const size4_t original_shape{3, 128, 1, 64};
        const size4_t broadcast_shape{3, 128, 64, 64};
        size4_t strides = original_shape.strides<'F'>();
        REQUIRE(all(indexing::isContiguous<'F'>(strides, original_shape) == bool4_t{1, 1, 1, 1}));

        indexing::broadcast(original_shape, strides, size4_t{3, 128, 64, 64});
        REQUIRE(all(indexing::isContiguous<'F'>(strides, broadcast_shape) == bool4_t{1, 1, 0, 1}));
    }

    AND_THEN("pitches from strides") {
        const uint ndim = GENERATE(1u, 2u, 3u);
        for (size_t i = 0; i < 20; ++i) {
            size4_t shape = test::getRandomShapeBatched(ndim);
            size4_t strides = shape.strides();
            REQUIRE(all(size3_t{shape.get() + 1} == strides.pitches()));

            strides *= 3;
            shape[3] *= 3;
            REQUIRE(all(size3_t{shape.get() + 1} == strides.pitches()));
        }
    }

    AND_THEN("isRightmost") {
        size4_t shape{3, 128, 65, 64};
        REQUIRE(indexing::isRightmost(shape.strides()));
        REQUIRE_FALSE(indexing::isRightmost(shape.strides<'F'>()));

        shape = size4_t{3, 128, 1, 1};
        REQUIRE(indexing::isRightmost(shape.strides()));
        REQUIRE(indexing::isRightmost(shape.strides<'F'>()));
    }

    AND_THEN("isVector") {
        size4_t shape{3, 128, 65, 64};
        REQUIRE_FALSE(indexing::isVector(shape));

        shape = size4_t{3, 128, 1, 1};
        REQUIRE_FALSE(indexing::isVector(shape));

        shape = size4_t{1, 1, 1, 128};
        REQUIRE(indexing::isVector(shape));
        shape = size4_t{1, 1, 128, 1};
        REQUIRE(indexing::isVector(shape));
        shape = size4_t{1, 128, 1, 1};
        REQUIRE(indexing::isVector(shape));
        shape = size4_t{128, 1, 1, 1};
        REQUIRE(indexing::isVector(shape));

        shape = size4_t{3, 1, 1, 128};
        REQUIRE(indexing::isVector(shape, true));
        REQUIRE_FALSE(indexing::isVector(shape, false));
        shape = size4_t{3, 128, 1, 1};
        REQUIRE(indexing::isVector(shape, true));
        REQUIRE_FALSE(indexing::isVector(shape, false));
    }

    AND_THEN("effectiveShape") {
        const size4_t original_shape{3, 128, 1, 64};
        const size4_t broadcast_shape{3, 128, 64, 64};
        size4_t strides = original_shape.strides();
        indexing::broadcast(original_shape, strides, size4_t{3, 128, 64, 64});
        REQUIRE(all(indexing::effectiveShape(broadcast_shape, strides) == original_shape));
    }
}

TEST_CASE("common: order(), squeeze()", "[noa][common]") {
    size4_t shape{2, 32, 64, 128};
    REQUIRE(all(indexing::order(shape.strides(), shape) == size4_t{0, 1, 2, 3}));
    REQUIRE(all(indexing::order(shape.strides<'F'>(), shape) == size4_t{0, 1, 3, 2}));
    REQUIRE(all(indexing::order(size4_t{8192, 0, 128, 1}, shape) == size4_t{0, 2, 3, 1}));

    REQUIRE(all(indexing::order(size4_t{8192, 0, 128, 1}, size4_t{2, 1, 64, 128}) == size4_t{1, 0, 2, 3}));
    REQUIRE(all(indexing::order(size4_t{8192, 8192, 8192, 1}, size4_t{1, 1, 1, 8192}) == size4_t{0, 1, 2, 3}));
    REQUIRE(all(indexing::order(size4_t{4096, 128, 128, 1}, size4_t{2, 32, 1, 128}) == size4_t{2, 0, 1, 3}));
    REQUIRE(all(indexing::order(size4_t{4096, 4096, 128, 1}, size4_t{2, 32, 1, 128}) == size4_t{2, 0, 1, 3}));
    REQUIRE(all(indexing::order(size4_t{128, 4096, 128, 1}, size4_t{2, 32, 1, 128}) == size4_t{2, 1, 0, 3}));

    REQUIRE(all(indexing::squeeze(shape) == size4_t{0, 1, 2, 3}));
    REQUIRE(all(indexing::squeeze(size4_t{1, 1, 3, 4}) == size4_t{0, 1, 2, 3}));
    REQUIRE(all(indexing::squeeze(size4_t{1, 1, 3, 1}) == size4_t{0, 1, 3, 2}));
    REQUIRE(all(indexing::squeeze(size4_t{1, 1, 1, 1}) == size4_t{0, 1, 2, 3}));
    REQUIRE(all(indexing::squeeze(size4_t{5, 1, 3, 1}) == size4_t{1, 3, 0, 2}));
    REQUIRE(all(indexing::squeeze(size4_t{5, 1, 1, 1}) == size4_t{1, 2, 3, 0}));
}

TEST_CASE("common: (row/col)Major", "[noa][common]") {
    size4_t shape{2, 32, 64, 128};
    size4_t strides = shape.strides();
    REQUIRE(indexing::isRowMajor(strides));
    REQUIRE(indexing::isRowMajor(strides, shape));
    REQUIRE_FALSE(indexing::isColMajor(strides));
    REQUIRE_FALSE(indexing::isColMajor(strides, shape));

    strides = shape.strides<'F'>();
    REQUIRE_FALSE(indexing::isRowMajor(strides));
    REQUIRE_FALSE(indexing::isRowMajor(strides, shape));
    REQUIRE(indexing::isColMajor(strides));
    REQUIRE(indexing::isColMajor(strides, shape));

    // Even in the 'F' version, we still follow the BDHW order, so in our case, 'F' is not synonym of left-most.
    // Indeed, only the height/rows and width/columns are affected by 'C' vs 'F'. As such, if these two dimensions
    // are empty, 'C' and 'F' makes no difference.
    shape = size4_t{64, 32, 1, 1};
    REQUIRE(all(shape.strides<'C'>() == shape.strides<'F'>()));

    // For column and row vectors (or any vector actually), none of this really matters since we can just squeeze
    // the empty dimensions and get the rightmost order (i.e. row vectors).
    // contiguous row vector: {1,1,1,5} -> C strides: {5,5,5,1} and F strides: {5,5,1,1}
    // contiguous col vector: {1,1,5,1} -> C strides: {5,5,1,1} and F strides: {5,5,1,5}

    // Squeeze vector and check it is contiguous
    for (int i = 0; i < 4; ++i) {
        shape = 1;
        shape[i] = 5;
        strides = shape.strides();

        // The isXMajor overload taking a shape squeezes everything before checking the order and because it is
        // a vector and empty dimensions are contiguous, vectors are row-major and column-major.
        REQUIRE(indexing::isRowMajor(strides, shape));
        REQUIRE(indexing::isColMajor(strides, shape));

        auto order = indexing::squeeze(shape);
        shape = indexing::reorder(shape, order);
        strides = indexing::reorder(strides, order);
        REQUIRE(all(indexing::isContiguous(strides, shape)));
    }
}
