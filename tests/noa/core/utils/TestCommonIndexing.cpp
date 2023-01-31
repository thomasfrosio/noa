#include <noa/core/types/Vec.hpp>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("core::indexing::at<BorderMode>()", "[noa][common]") {
    AND_THEN("BorderMode::PERIODIC") {
        int expected_odd[55] = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
                                0, 1, 2, 3, 4,
                                0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4};
        int starts_at = 25;
        int len = 5;
        std::vector<int> data(55);
        for (size_t idx = 0; idx < data.size(); ++idx)
            data[idx] = indexing::at<BorderMode::PERIODIC>(static_cast<int>(idx) - starts_at, len);

        int diff = test::get_difference(expected_odd, data.data(), data.size());
        REQUIRE(diff == 0);

        int expected_even[36] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                                 0, 1, 2, 3,
                                 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
        starts_at = 16;
        len = 4;
        data = std::vector<int>(32);
        for (size_t idx = 0; idx < data.size(); ++idx)
            data[idx] = indexing::at<BorderMode::PERIODIC>(static_cast<int>(idx) - starts_at, len);

        diff = test::get_difference(expected_even, data.data(), data.size());
        REQUIRE(diff == 0);
    }

    AND_THEN("BorderMode::CLAMP") {
        int expected_odd[35] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 1, 2, 3, 4,
                                4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
        int starts_at = 15;
        int len = 5;
        std::vector<int> data(35);
        for (size_t idx = 0; idx < data.size(); ++idx)
            data[idx] = indexing::at<BorderMode::CLAMP>(static_cast<int>(idx) - starts_at, len);

        int diff = test::get_difference(expected_odd, data.data(), data.size());
        REQUIRE(diff == 0);

        int expected_even[34] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 1, 2, 3,
                                 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
        starts_at = 15;
        len = 4;
        data = std::vector<int>(34);
        for (size_t idx = 0; idx < data.size(); ++idx)
            data[idx] = indexing::at<BorderMode::CLAMP>(static_cast<int>(idx) - starts_at, len);

        diff = test::get_difference(expected_even, data.data(), data.size());
        REQUIRE(diff == 0);
    }

    AND_THEN("BorderMode::MIRROR") {
        int expected_odd[45] = {0, 1, 2, 3, 4, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 4, 3, 2, 1, 0,
                                0, 1, 2, 3, 4,
                                4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4};
        int starts_at = 20;
        int len = 5;
        std::vector<int> data(45);
        for (size_t idx = 0; idx < data.size(); ++idx)
            data[idx] = indexing::at<BorderMode::MIRROR>(static_cast<int>(idx) - starts_at, len);

        int diff = test::get_difference(expected_odd, data.data(), data.size());
        REQUIRE(diff == 0);

        int expected_even[52] = {0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0,
                                 0, 1, 2, 3,
                                 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3};
        starts_at = 24;
        len = 4;
        data = std::vector<int>(52);
        for (size_t idx = 0; idx < data.size(); ++idx)
            data[idx] = indexing::at<BorderMode::MIRROR>(static_cast<int>(idx) - starts_at, len);

        diff = test::get_difference(expected_even, data.data(), data.size());
        REQUIRE(diff == 0);
    }

    AND_THEN("BorderMode::REFLECT") {
        int expected_odd[53] = {0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1,
                                0, 1, 2, 3, 4,
                                3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4};
        int starts_at = 24;
        int len = 5;
        std::vector<int> data(53);
        for (size_t idx = 0; idx < data.size(); ++idx)
            data[idx] = indexing::at<BorderMode::REFLECT>(static_cast<int>(idx) - starts_at, len);

        int diff = test::get_difference(expected_odd, data.data(), data.size());
        REQUIRE(diff == 0);

        int expected_even[40] = {0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1,
                                 0, 1, 2, 3,
                                 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3};
        starts_at = 18;
        len = 4;
        data = std::vector<int>(40);
        for (size_t idx = 0; idx < data.size(); ++idx)
            data[idx] = indexing::at<BorderMode::REFLECT>(static_cast<int>(idx) - starts_at, len);

        diff = test::get_difference(expected_even, data.data(), data.size());
        REQUIRE(diff == 0);
    }
}

TEST_CASE("core:: shape, strides", "[noa][common]") {
    AND_THEN("C- contiguous") {
        const Shape4<u64> shape{2, 128, 64, 65};
        const auto strides = shape.strides();
        const auto physical_shape = strides.physical_shape();
        REQUIRE(all(indexing::is_contiguous(strides, shape)));
        REQUIRE(indexing::are_contiguous(strides, shape));
        REQUIRE(all(Strides4<u64>{532480, 4160, 65, 1} == strides));
        REQUIRE(all(Shape3<u64>{128, 64, 65} == physical_shape));
    }

    AND_THEN("F- contiguous") {
        const Shape4<u64> shape{2, 128, 64, 65};
        const auto strides = shape.strides<'F'>();
        const auto physical_shape = strides.physical_shape<'F'>();
        REQUIRE(all(indexing::is_contiguous<'F'>(strides, shape)));
        REQUIRE(indexing::are_contiguous<'F'>(strides, shape));
        REQUIRE(all(Strides4<u64>{532480, 4160, 1, 64} == strides));
        REQUIRE(all(Shape3<u64>{128, 64, 65} == physical_shape));
    }

    AND_THEN("C- inner stride") {
        const Shape4<u64> shape{3, 128, 64, 64};
        const auto strides = shape.strides() * 2;
        const auto physical_shape = strides.physical_shape();
        REQUIRE(all(indexing::is_contiguous(strides, shape) == Vec4<bool>{1, 1, 1, 0}));
        REQUIRE(all(Strides4<u64>{1048576, 8192, 128, 2} == strides));
        REQUIRE(all(Shape3<u64>{128, 64, 128} == physical_shape));
    }

    AND_THEN("F- inner stride") {
        const Shape4<u64> shape{3, 128, 64, 64};
        const auto strides = shape.strides<'F'>() * 2;
        const auto physical_shape = strides.physical_shape<'F'>();
        REQUIRE(all(indexing::is_contiguous<'F'>(strides, shape) == Vec4<bool>{1, 1, 0, 1}));
        REQUIRE(all(Strides4<u64>{1048576, 8192, 2, 128} == strides));
        REQUIRE(all(Shape3<u64>{128, 128, 64} == physical_shape));
    }

    AND_THEN("is_contiguous<'C'>") {
        Shape4<u64> shape{3, 128, 64, 64};
        auto strides = shape.strides();
        strides[0] *= 2;
        strides[1] *= 2;
        strides[2] *= 2;
        REQUIRE(all(indexing::is_contiguous(strides, shape) == Vec4<bool>{1, 1, 0, 1}));

        strides = shape.strides();
        strides[0] *= 6;
        strides[1] *= 6;
        strides[2] *= 6;
        strides[3] *= 3;
        REQUIRE(all(indexing::is_contiguous(strides, shape) == Vec4<bool>{1, 1, 0, 0}));

        strides = shape.strides();
        strides[0] *= 2;
        strides[1] *= 2;
        REQUIRE(all(indexing::is_contiguous(strides, shape) == Vec4<bool>{1, 0, 1, 1}));

        strides = shape.strides();
        strides[0] *= 2;
        REQUIRE(all(indexing::is_contiguous(strides, shape) == Vec4<bool>{0, 1, 1, 1}));

        strides = shape.strides();
        strides[0] *= 4;
        strides[1] *= 2;
        REQUIRE(all(indexing::is_contiguous(strides, shape) == Vec4<bool>{0, 0, 1, 1}));

        // Empty is contiguous by definition.
        shape = {3, 128, 1, 64};
        strides = {8192, 64, 64, 1};
        REQUIRE(all(indexing::is_contiguous(strides, shape) == Vec4<bool>{1, 1, 1, 1}));
        strides *= 2;
        REQUIRE(all(indexing::is_contiguous(strides, shape) == Vec4<bool>{1, 1, 1, 0}));
        strides[2] *= 2;
        REQUIRE(all(indexing::is_contiguous(strides, shape) == Vec4<bool>{1, 1, 1, 0}));
        strides[0] *= 2;
        strides[1] *= 2;
        strides[2] *= 2;
        REQUIRE(all(indexing::is_contiguous(strides, shape) == Vec4<bool>{1, 0, 1, 0}));
    }

    AND_THEN("is_contiguous<'C'> - broadcast") {
        const Shape4<u64> original_shape{3, 128, 1, 64};
        const Shape4<u64> broadcast_shape{3, 128, 32, 64};
        auto strides = original_shape.strides();
        REQUIRE(all(indexing::is_contiguous(strides, original_shape) == Vec4<bool>{1, 1, 1, 1}));

        indexing::broadcast(original_shape, strides, broadcast_shape);
        REQUIRE(all(indexing::is_contiguous(strides, broadcast_shape) == Vec4<bool>{1, 1, 0, 1}));

        REQUIRE(all(indexing::is_contiguous({0, 0, 0, 1}, broadcast_shape) == Vec4<bool>{0, 0, 0, 1}));
        REQUIRE(all(indexing::is_contiguous({0, 64, 0, 1}, broadcast_shape) == Vec4<bool>{0, 1, 0, 1}));
        REQUIRE(all(indexing::is_contiguous({128 * 64, 64, 0, 1}, broadcast_shape) == Vec4<bool>{1, 1, 0, 1}));
        REQUIRE(all(indexing::is_contiguous({64, 0, 0, 1}, broadcast_shape) == Vec4<bool>{1, 0, 0, 1}));
    }

    AND_THEN("is_contiguous<'F'>") {
        Shape4<u64> shape{3, 128, 64, 64};
        auto strides = shape.strides<'F'>();
        strides[0] *= 2;
        strides[1] *= 2;
        strides[3] *= 2;
        REQUIRE(all(indexing::is_contiguous<'F'>(strides, shape) == Vec4<bool>{1, 1, 1, 0}));

        strides = shape.strides<'F'>();
        strides[0] *= 6;
        strides[1] *= 6;
        strides[2] *= 3;
        strides[3] *= 6;
        REQUIRE(all(indexing::is_contiguous<'F'>(strides, shape) == Vec4<bool>{1, 1, 0, 0}));

        strides = shape.strides<'F'>();
        strides[0] *= 2;
        strides[1] *= 2;
        REQUIRE(all(indexing::is_contiguous<'F'>(strides, shape) == Vec4<bool>{1, 0, 1, 1}));

        strides = shape.strides<'F'>();
        strides[0] *= 2;
        REQUIRE(all(indexing::is_contiguous<'F'>(strides, shape) == Vec4<bool>{0, 1, 1, 1}));

        strides = shape.strides<'F'>();
        strides[0] *= 4;
        strides[1] *= 2;
        REQUIRE(all(indexing::is_contiguous<'F'>(strides, shape) == Vec4<bool>{0, 0, 1, 1}));

        shape = {3, 128, 64, 1};
        strides = {8192, 64, 1, 64};
        REQUIRE(all(indexing::is_contiguous<'F'>(strides, shape) == Vec4<bool>{1, 1, 1, 1}));
        strides *= 2;
        REQUIRE(all(indexing::is_contiguous<'F'>(strides, shape) == Vec4<bool>{1, 1, 0, 1}));
        strides[3] *= 2;
        REQUIRE(all(indexing::is_contiguous<'F'>(strides, shape) == Vec4<bool>{1, 1, 0, 1}));
        strides[0] *= 2;
        strides[1] *= 2;
        strides[3] *= 2;
        REQUIRE(all(indexing::is_contiguous<'F'>(strides, shape) == Vec4<bool>{1, 0, 0, 1}));
    }

    AND_THEN("is_contiguous<'F'> - broadcast") {
        const Shape4<u64> original_shape{3, 128, 1, 64};
        const Shape4<u64> broadcast_shape{3, 128, 64, 64};
        auto strides = original_shape.strides<'F'>();
        REQUIRE(all(indexing::is_contiguous<'F'>(strides, original_shape) == Vec4<bool>{1, 1, 1, 1}));

        indexing::broadcast(original_shape, strides, broadcast_shape);
        REQUIRE(all(indexing::is_contiguous<'F'>(strides, broadcast_shape) == Vec4<bool>{1, 1, 0, 1}));
    }

    AND_THEN("physical shape") {
        const uint ndim = GENERATE(1u, 2u, 3u);
        for (size_t i = 0; i < 20; ++i) {
            auto shape = test::get_random_shape4_batched(ndim);
            auto strides = shape.strides();
            REQUIRE(all(shape.pop_front() == strides.physical_shape()));

            strides *= 3;
            shape[3] *= 3;
            REQUIRE(all(shape.pop_front() == strides.physical_shape()));
        }
    }

    AND_THEN("is_rightmost") {
        Shape4<u64> shape{3, 128, 65, 64};
        REQUIRE(indexing::is_rightmost(shape.strides()));
        REQUIRE_FALSE(indexing::is_rightmost(shape.strides<'F'>()));

        shape = {3, 128, 1, 1};
        REQUIRE(indexing::is_rightmost(shape.strides()));
        REQUIRE(indexing::is_rightmost(shape.strides<'F'>()));
    }

    AND_THEN("is_vector") {
        Shape4<u64> shape{3, 128, 65, 64};
        REQUIRE_FALSE(indexing::is_vector(shape));

        shape = {3, 128, 1, 1};
        REQUIRE_FALSE(indexing::is_vector(shape));

        shape = {1, 1, 1, 128};
        REQUIRE(indexing::is_vector(shape));
        shape = {1, 1, 128, 1};
        REQUIRE(indexing::is_vector(shape));
        shape = {1, 128, 1, 1};
        REQUIRE(indexing::is_vector(shape));
        shape = {128, 1, 1, 1};
        REQUIRE(indexing::is_vector(shape));

        shape = {3, 1, 1, 128};
        REQUIRE(indexing::is_vector(shape, true));
        REQUIRE_FALSE(indexing::is_vector(shape, false));
        shape = {3, 128, 1, 1};
        REQUIRE(indexing::is_vector(shape, true));
        REQUIRE_FALSE(indexing::is_vector(shape, false));
    }

    AND_THEN("effective_shape") {
        const Shape4<u64> original_shape{3, 128, 1, 64};
        const Shape4<u64> broadcast_shape{3, 128, 64, 64};
        auto strides = original_shape.strides();
        indexing::broadcast(original_shape, strides, broadcast_shape);
        REQUIRE(all(indexing::effective_shape(broadcast_shape, strides) == original_shape));
    }
}

TEST_CASE("core::indexing:: order(), squeeze()", "[noa][common]") {
    Shape4<u64> shape{2, 32, 64, 128};
    REQUIRE(all(indexing::order(shape.strides(), shape) == Vec4<u64>{0, 1, 2, 3}));
    REQUIRE(all(indexing::order(shape.strides<'F'>(), shape) == Vec4<u64>{0, 1, 3, 2}));

    // Order will move the broadcast dimensions to the right,
    // because they are indeed the dimensions with the smallest strides.
    REQUIRE(all(indexing::order(Strides4<u64>{8192, 0, 128, 1}, shape) == Vec4<u64>{0, 2, 3, 1}));

    // We almost always call order() on the output array and rearrange the layout to make the output
    // as rightmost as possible. The output is rarely broadcast and in most cases it is actually invalid
    // to have a stride of 0 in the output (because unless the write operator is atomic, it will create
    // a data-race). Anyway, broadcast dimensions cause an issue here, because they are not contiguous
    // and order() will put them to the right. In practice, this is solved by calling effective_shape()
    // on the output shape beforehand, "tagging" these broadcast dimension as empty, so that order()
    // moves them to the left.
    // TL-DR: If is often best to call effective_shape() on the layout that is about to be reordered.
    const auto strides = Strides4<u64>{8192, 0, 128, 1};
    REQUIRE(all(indexing::order(strides, indexing::effective_shape(shape, strides)) == Vec4<u64>{1, 0, 2, 3}));

    REQUIRE(all(indexing::order(Strides4<i64>{8192, 0, 128, 1}, Shape4<i64>{2, 1, 64, 128}) == Vec4<i64>{1, 0, 2, 3}));
    REQUIRE(all(indexing::order(Strides4<i64>{8192, 8192, 8192, 1}, Shape4<i64>{1, 1, 1, 8192}) == Vec4<i64>{0, 1, 2, 3}));
    REQUIRE(all(indexing::order(Strides4<i64>{4096, 128, 128, 1}, Shape4<i64>{2, 32, 1, 128}) == Vec4<i64>{2, 0, 1, 3}));
    REQUIRE(all(indexing::order(Strides4<i64>{4096, 4096, 128, 1}, Shape4<i64>{2, 32, 1, 128}) == Vec4<i64>{2, 0, 1, 3}));
    REQUIRE(all(indexing::order(Strides4<i64>{128, 4096, 128, 1}, Shape4<i64>{2, 32, 1, 128}) == Vec4<i64>{2, 1, 0, 3}));
    REQUIRE(all(indexing::order(Strides4<i64>{2, 2, 2, 2}, Shape4<i64>{1, 1, 1, 1}) == Vec4<i64>{0, 1, 2, 3}));

    REQUIRE(all(indexing::squeeze(shape) == Vec4<u64>{0, 1, 2, 3}));
    REQUIRE(all(indexing::squeeze(Shape4<u64>{1, 1, 3, 4}) == Vec4<u64>{0, 1, 2, 3}));
    REQUIRE(all(indexing::squeeze(Shape4<u64>{1, 1, 3, 1}) == Vec4<u64>{0, 1, 3, 2}));
    REQUIRE(all(indexing::squeeze(Shape4<u64>{1, 1, 1, 1}) == Vec4<u64>{0, 1, 2, 3}));
    REQUIRE(all(indexing::squeeze(Shape4<u64>{5, 1, 3, 1}) == Vec4<u64>{1, 3, 0, 2}));
    REQUIRE(all(indexing::squeeze(Shape4<u64>{5, 1, 1, 1}) == Vec4<u64>{1, 2, 3, 0}));
}

TEST_CASE("core::indexing:: memory layouts", "[noa][common]") {
    Shape4<u64> shape{2, 32, 64, 128};
    auto strides = shape.strides();
    REQUIRE(indexing::is_row_major(strides));
    REQUIRE(indexing::is_row_major(strides, shape));
    REQUIRE_FALSE(indexing::is_column_major(strides));
    REQUIRE_FALSE(indexing::is_column_major(strides, shape));

    strides = shape.strides<'F'>();
    REQUIRE_FALSE(indexing::is_row_major(strides));
    REQUIRE_FALSE(indexing::is_row_major(strides, shape));
    REQUIRE(indexing::is_column_major(strides));
    REQUIRE(indexing::is_column_major(strides, shape));

    // Even in the 'F' version, we still follow the BDHW order, so in our case, 'F' is not synonym of left-most.
    // Indeed, only the height/rows and width/columns are affected by 'C' vs 'F'. As such, if these two dimensions
    // are empty, 'C' and 'F' makes no difference.
    shape = {64, 32, 1, 1};
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

        // The is_x_major overload taking a shape squeezes everything before checking the order and because it is
        // a vector and empty dimensions are contiguous, vectors are row-major and column-major.
        REQUIRE(indexing::is_row_major(strides, shape));
        REQUIRE(indexing::is_column_major(strides, shape));

        auto order = indexing::squeeze(shape);
        shape = indexing::reorder(shape, order);
        strides = indexing::reorder(strides, order);
        REQUIRE(all(indexing::is_contiguous(strides, shape)));
    }
}

TEST_CASE("core::indexing:: Reinterpret", "[noa][common]") {
    const auto shape = test::get_random_shape4_batched(3);
    auto strides = shape.strides();
    c32* ptr = nullptr;
    auto real = indexing::Reinterpret(shape, strides, ptr).as<float>();
    REQUIRE(all(real.shape == Shape4<u64>{shape[0], shape[1], shape[2], shape[3] * 2}));
    REQUIRE(all(real.strides == Strides4<u64>{strides[0] * 2, strides[1] * 2, strides[2] * 2, 1}));

    // Reinterpret moves everything to the rightmost order,
    // compute the new shape and strides, then moves back to original order.
    strides = shape.strides<'F'>();
    real = indexing::Reinterpret(shape, strides, ptr).as<float>();
    REQUIRE(all(real.shape == Shape4<u64>{shape[0], shape[1], shape[2] * 2, shape[3]}));
    REQUIRE(all(real.strides == Strides4<u64>{strides[0] * 2, strides[1] * 2, 1, strides[3] * 2}));
}

TEMPLATE_TEST_CASE("core::indexing::offset2index(), 4D", "[noa][common]", Vec4<i64>, Vec4<u64>) {
    const u32 ndim = GENERATE(1u, 2u, 3u);

    using value_t = traits::value_type_t<TestType>;
    test::Randomizer<value_t> randomizer(1, 3);
    test::Randomizer<value_t> idx_randomizer(0, 50);

    for (int i = 0; i < 20; ++i) {
        const auto shape = test::get_random_shape4_batched(ndim).as<value_t>();
        const auto strides = shape.strides() * randomizer.get();

        // Generate a random 4D index.
        auto idx_expected = (shape - 1).vec();
        for (size_t j = 0; j < 4; ++j)
            idx_expected[j] = std::clamp(idx_expected[j] - idx_randomizer.get(), value_t{0}, shape[j] - 1);

        // Now find the 4D index of the offset.
        const value_t offset = indexing::at(idx_expected, strides);
        const auto idx_result = indexing::offset2index(offset, strides, shape);
        INFO(strides);
        INFO(shape);
        INFO(idx_expected);
        INFO(idx_result);

        REQUIRE(all(idx_expected == idx_result));
    }
}

TEMPLATE_TEST_CASE("core::indexing::offset2index(), 3D", "[noa][common]", Vec3<i64>, Vec3<u64>) {
    const u32 ndim = GENERATE(1u, 2u, 3u);

    using value_t = traits::value_type_t<TestType>;
    test::Randomizer<value_t> randomizer(1, 3);
    test::Randomizer<value_t> idx_randomizer(0, 50);

    for (int i = 0; i < 20; ++i) {
        const auto shape4 = test::get_random_shape4_batched(ndim).as<value_t>();
        const auto shape = shape4.pop_back();
        const auto strides = shape.strides() * randomizer.get();

        // Generate a random 4D index.
        auto idx_expected = (shape - 1).vec();
        for (size_t j = 0; j < 3; ++j)
            idx_expected[j] = std::clamp(idx_expected[j] - idx_randomizer.get(), value_t{0}, shape[j] - 1);

        // Now find the 4D index of the offset.
        const value_t offset = indexing::at(idx_expected, strides);
        const auto idx_result = indexing::offset2index(offset, strides, shape);
        INFO(strides);
        INFO(shape);
        INFO(idx_expected);
        INFO(idx_result);

        REQUIRE(all(idx_expected == idx_result));
    }
}

TEMPLATE_TEST_CASE("core::indexing::offset2index(), 2D", "[noa][common]", Vec2<i64>, Vec2<u64>) {
    const u32 ndim = GENERATE(1u, 2u, 3u);

    using value_t = traits::value_type_t<TestType>;
    test::Randomizer<value_t> randomizer(1, 3);
    test::Randomizer<value_t> idx_randomizer(0, 50);

    for (int i = 0; i < 20; ++i) {
        const auto shape4 = test::get_random_shape4_batched(ndim).as<value_t>();
        const auto shape = shape4.pop_back().pop_back();
        const auto strides = shape.strides() * randomizer.get();

        // Generate a random 4D index.
        auto idx_expected = (shape - 1).vec();
        for (size_t j = 0; j < 2; ++j)
            idx_expected[j] = std::clamp(idx_expected[j] - idx_randomizer.get(), value_t{0}, shape[j] - 1);

        // Now find the 4D index of the offset.
        const value_t offset = indexing::at(idx_expected, strides);
        const auto idx_result = indexing::offset2index(offset, strides, shape);
        INFO(strides);
        INFO(shape);
        INFO(idx_expected);
        INFO(idx_result);

        REQUIRE(all(idx_expected == idx_result));
    }
}

TEMPLATE_TEST_CASE("core::indexing:: reorder matrices", "[noa][common]", u32, i32, i64, u64) {
    // When we reorder the array, if it is attached to a matrix, we need to reorder the matrix as well.
    test::Randomizer<double> randomizer(-3, 3);
    SECTION("2D") {
        Double22 matrix;
        for (u64 i = 0; i < 2; ++i)
            for (u64 j = 0; j < 2; ++j)
            matrix[i][j] = randomizer.get();

        Vec2<f64> vector{randomizer.get(), randomizer.get()};
        Vec2<f64> expected = matrix * vector;

        Vec2<TestType> order{1, 0};
        vector = indexing::reorder(vector, order);
        matrix = indexing::reorder(matrix, order);
        Vec2<f64> result = matrix * vector;
        REQUIRE(expected[0] == result[1]);
        REQUIRE(expected[1] == result[0]);
    }

    SECTION("2D affine") {
        Double23 matrix;
        for (u64 i = 0; i < 2; ++i)
            for (u64 j = 0; j < 2; ++j)
                matrix[i][j] = randomizer.get();

        Vec3<f64> vector{randomizer.get(), randomizer.get(), 1};
        Vec2<f64> expected = matrix * vector;

        vector = indexing::reorder(vector, Vec3<TestType>{1, 0, 2});
        matrix = indexing::reorder(matrix, Vec2<TestType>{1, 0});
        Vec2<f64> result = matrix * vector;
        REQUIRE(expected[0] == result[1]);
        REQUIRE(expected[1] == result[0]);
    }

    SECTION("3D") {
        Double33 matrix;
        for (u64 i = 0; i < 3; ++i)
            for (u64 j = 0; j < 3; ++j)
                matrix[i][j] = randomizer.get();

        Vec3<f64> vector{randomizer.get(), randomizer.get(), randomizer.get()};
        Vec3<f64> expected = matrix * vector;

        using order_t = Vec3<TestType>;
        std::vector<order_t> orders{{0, 1, 2}, {0, 2, 1}, {2, 1, 0}, {1, 0, 2}};
        for (const auto& order: orders) {
            vector = indexing::reorder(vector, order);
            matrix = indexing::reorder(matrix, order);
            Vec3<f64> result = matrix * vector;
            expected = indexing::reorder(expected, order);

            INFO(order);
            REQUIRE_THAT(result[0], Catch::WithinAbs(expected[0], 1e-6));
            REQUIRE_THAT(result[1], Catch::WithinAbs(expected[1], 1e-6));
            REQUIRE_THAT(result[2], Catch::WithinAbs(expected[2], 1e-6));
        }
    }
}

TEST_CASE("core::indexing::reshape, broadcasting", "[noa][indexing]") {
    const auto shape1 = Shape4<u64>{30, 20, 10, 5};
    const auto strides1 = Strides4<u64>{1000, 50, 5, 1};

    // Reshape contiguous array to a row vector.
    Strides4<u64> new_strides;
    auto new_shape = Shape4<u64>{1, 1, 1, math::product(shape1)};
    bool success = indexing::reshape(shape1, strides1, new_shape, new_strides);
    REQUIRE(success);
    REQUIRE(all(new_strides == new_shape.strides()));

    // Broadcast the shape2 onto shape1.
    const auto shape2 = Shape4<u64>{30, 1, 10, 5};
    auto strides2 = shape2.strides();
    success = indexing::broadcast(shape2, strides2, shape1);
    REQUIRE(success);

    // Reshape the broadcast array to a row vector.
    new_shape = {1, 1, 1, math::product(shape2)};
    success = indexing::reshape(shape2, strides2, new_shape, new_strides);
    REQUIRE(success);
    REQUIRE(all(new_strides == new_shape.strides()));
}
