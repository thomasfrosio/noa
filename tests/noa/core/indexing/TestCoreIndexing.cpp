#include <noa/core/types/Mat.hpp>
#include <noa/core/types/Shape.hpp>
#include <noa/core/types/Vec.hpp>
#include <noa/core/indexing/Offset.hpp>
#include <noa/core/indexing/Layout.hpp>
#include <noa/core/indexing/Subregion.hpp>

#include "Utils.hpp"
#include <catch2/catch.hpp>

using Border = noa::Border;
using namespace ::noa::types;
using namespace ::noa::indexing;

TEST_CASE("core::indexing::index_at()", "[noa][core]") {
    AND_THEN("Border::PERIODIC") {
        int expected_odd[55] = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
                                0, 1, 2, 3, 4,
                                0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4};
        int starts_at = 25;
        int len = 5;
        std::vector<int> data(55);
        for (size_t idx = 0; idx < data.size(); ++idx)
            data[idx] = index_at<Border::PERIODIC>(static_cast<int>(idx) - starts_at, len);

        REQUIRE(test::allclose_abs(expected_odd, data.data(), static_cast<i64>(data.size())));

        int expected_even[36] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                                 0, 1, 2, 3,
                                 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
        starts_at = 16;
        len = 4;
        data = std::vector<int>(32);
        for (size_t idx = 0; idx < data.size(); ++idx)
            data[idx] = index_at<Border::PERIODIC>(static_cast<int>(idx) - starts_at, len);

        REQUIRE(test::allclose_abs(expected_even, data.data(), static_cast<i64>(data.size())));
    }

    AND_THEN("Border::CLAMP") {
        int expected_odd[35] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 1, 2, 3, 4,
                                4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
        int starts_at = 15;
        int len = 5;
        std::vector<int> data(35);
        for (size_t idx = 0; idx < data.size(); ++idx)
            data[idx] = index_at<Border::CLAMP>(static_cast<int>(idx) - starts_at, len);

        REQUIRE(test::allclose_abs(expected_odd, data.data(), static_cast<i64>(data.size())));

        int expected_even[34] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 1, 2, 3,
                                 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
        starts_at = 15;
        len = 4;
        data = std::vector<int>(34);
        for (size_t idx = 0; idx < data.size(); ++idx)
            data[idx] = index_at<Border::CLAMP>(static_cast<int>(idx) - starts_at, len);

        REQUIRE(test::allclose_abs(expected_even, data.data(), static_cast<i64>(data.size())));
    }

    AND_THEN("Border::MIRROR") {
        int expected_odd[45] = {0, 1, 2, 3, 4, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 4, 3, 2, 1, 0,
                                0, 1, 2, 3, 4,
                                4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4};
        int starts_at = 20;
        int len = 5;
        std::vector<int> data(45);
        for (size_t idx = 0; idx < data.size(); ++idx)
            data[idx] = index_at<Border::MIRROR>(static_cast<int>(idx) - starts_at, len);

        REQUIRE(test::allclose_abs(expected_odd, data.data(), static_cast<i64>(data.size())));

        int expected_even[52] = {0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0,
                                 0, 1, 2, 3,
                                 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3};
        starts_at = 24;
        len = 4;
        data = std::vector<int>(52);
        for (size_t idx = 0; idx < data.size(); ++idx)
            data[idx] = index_at<Border::MIRROR>(static_cast<int>(idx) - starts_at, len);

        REQUIRE(test::allclose_abs(expected_even, data.data(), static_cast<i64>(data.size())));
    }

    AND_THEN("Border::REFLECT") {
        int expected_odd[53] = {0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1,
                                0, 1, 2, 3, 4,
                                3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4};
        int starts_at = 24;
        int len = 5;
        std::vector<int> data(53);
        for (size_t idx = 0; idx < data.size(); ++idx)
            data[idx] = index_at<Border::REFLECT>(static_cast<int>(idx) - starts_at, len);

        REQUIRE(test::allclose_abs(expected_odd, data.data(), static_cast<i64>(data.size())));

        int expected_even[40] = {0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1,
                                 0, 1, 2, 3,
                                 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3};
        starts_at = 18;
        len = 4;
        data = std::vector<int>(40);
        for (size_t idx = 0; idx < data.size(); ++idx)
            data[idx] = index_at<Border::REFLECT>(static_cast<int>(idx) - starts_at, len);

        REQUIRE(test::allclose_abs(expected_even, data.data(), static_cast<i64>(data.size())));
    }
}

TEST_CASE("core:: shape, strides", "[noa][core]") {
    AND_THEN("C- contiguous") {
        const Shape4<u64> shape{2, 128, 64, 65};
        const auto strides = shape.strides();
        const auto physical_shape = strides.physical_shape();
        REQUIRE(noa::all(is_contiguous(strides, shape)));
        REQUIRE(are_contiguous(strides, shape));
        REQUIRE(noa::all(Strides4<u64>{532480, 4160, 65, 1} == strides));
        REQUIRE(noa::all(Shape3<u64>{128, 64, 65} == physical_shape));
    }

    AND_THEN("F- contiguous") {
        const Shape4<u64> shape{2, 128, 64, 65};
        const auto strides = shape.strides<'F'>();
        const auto physical_shape = strides.physical_shape<'F'>();
        REQUIRE(noa::all(is_contiguous<'F'>(strides, shape)));
        REQUIRE(are_contiguous<'F'>(strides, shape));
        REQUIRE(noa::all(Strides4<u64>{532480, 4160, 1, 64} == strides));
        REQUIRE(noa::all(Shape3<u64>{128, 64, 65} == physical_shape));
    }

    AND_THEN("C- inner stride") {
        const Shape4<u64> shape{3, 128, 64, 64};
        const auto strides = shape.strides() * 2;
        const auto physical_shape = strides.physical_shape();
        REQUIRE(noa::all(is_contiguous(strides, shape) == Vec4<bool>{1, 1, 1, 0}));
        REQUIRE(noa::all(Strides4<u64>{1048576, 8192, 128, 2} == strides));
        REQUIRE(noa::all(Shape3<u64>{128, 64, 128} == physical_shape));
    }

    AND_THEN("F- inner stride") {
        const Shape4<u64> shape{3, 128, 64, 64};
        const auto strides = shape.strides<'F'>() * 2;
        const auto physical_shape = strides.physical_shape<'F'>();
        REQUIRE(noa::all(is_contiguous<'F'>(strides, shape) == Vec4<bool>{1, 1, 0, 1}));
        REQUIRE(noa::all(Strides4<u64>{1048576, 8192, 2, 128} == strides));
        REQUIRE(noa::all(Shape3<u64>{128, 128, 64} == physical_shape));
    }

    AND_THEN("is_contiguous<'C'>") {
        Shape4<u64> shape{3, 128, 64, 64};
        auto strides = shape.strides();
        strides[0] *= 2;
        strides[1] *= 2;
        strides[2] *= 2;
        REQUIRE(noa::all(is_contiguous(strides, shape) == Vec4<bool>{1, 1, 0, 1}));

        strides = shape.strides();
        strides[0] *= 6;
        strides[1] *= 6;
        strides[2] *= 6;
        strides[3] *= 3;
        REQUIRE(noa::all(is_contiguous(strides, shape) == Vec4<bool>{1, 1, 0, 0}));

        strides = shape.strides();
        strides[0] *= 2;
        strides[1] *= 2;
        REQUIRE(noa::all(is_contiguous(strides, shape) == Vec4<bool>{1, 0, 1, 1}));

        strides = shape.strides();
        strides[0] *= 2;
        REQUIRE(noa::all(is_contiguous(strides, shape) == Vec4<bool>{0, 1, 1, 1}));

        strides = shape.strides();
        strides[0] *= 4;
        strides[1] *= 2;
        REQUIRE(noa::all(is_contiguous(strides, shape) == Vec4<bool>{0, 0, 1, 1}));

        // Empty is contiguous by definition.
        shape = {3, 128, 1, 64};
        strides = {8192, 64, 64, 1};
        REQUIRE(noa::all(is_contiguous(strides, shape) == Vec4<bool>{1, 1, 1, 1}));
        strides *= 2;
        REQUIRE(noa::all(is_contiguous(strides, shape) == Vec4<bool>{1, 1, 1, 0}));
        strides[2] *= 2;
        REQUIRE(noa::all(is_contiguous(strides, shape) == Vec4<bool>{1, 1, 1, 0}));
        strides[0] *= 2;
        strides[1] *= 2;
        strides[2] *= 2;
        REQUIRE(noa::all(is_contiguous(strides, shape) == Vec4<bool>{1, 0, 1, 0}));
    }

    AND_THEN("is_contiguous<'C'> - broadcast") {
        const Shape4<u64> original_shape{3, 128, 1, 64};
        const Shape4<u64> broadcast_shape{3, 128, 32, 64};
        auto strides = original_shape.strides();
        REQUIRE(noa::all(is_contiguous(strides, original_shape) == Vec4<bool>{1, 1, 1, 1}));

        REQUIRE(broadcast(original_shape, strides, broadcast_shape));
        REQUIRE(noa::all(is_contiguous(strides, broadcast_shape) == Vec4<bool>{1, 1, 0, 1}));

        REQUIRE(noa::all(is_contiguous({0, 0, 0, 1}, broadcast_shape) == Vec4<bool>{0, 0, 0, 1}));
        REQUIRE(noa::all(is_contiguous({0, 64, 0, 1}, broadcast_shape) == Vec4<bool>{0, 1, 0, 1}));
        REQUIRE(noa::all(is_contiguous({128 * 64, 64, 0, 1}, broadcast_shape) == Vec4<bool>{1, 1, 0, 1}));
        REQUIRE(noa::all(is_contiguous({64, 0, 0, 1}, broadcast_shape) == Vec4<bool>{1, 0, 0, 1}));
    }

    AND_THEN("is_contiguous<'F'>") {
        Shape4<u64> shape{3, 128, 64, 64};
        auto strides = shape.strides<'F'>();
        strides[0] *= 2;
        strides[1] *= 2;
        strides[3] *= 2;
        REQUIRE(noa::all(is_contiguous<'F'>(strides, shape) == Vec4<bool>{1, 1, 1, 0}));

        strides = shape.strides<'F'>();
        strides[0] *= 6;
        strides[1] *= 6;
        strides[2] *= 3;
        strides[3] *= 6;
        REQUIRE(noa::all(is_contiguous<'F'>(strides, shape) == Vec4<bool>{1, 1, 0, 0}));

        strides = shape.strides<'F'>();
        strides[0] *= 2;
        strides[1] *= 2;
        REQUIRE(noa::all(is_contiguous<'F'>(strides, shape) == Vec4<bool>{1, 0, 1, 1}));

        strides = shape.strides<'F'>();
        strides[0] *= 2;
        REQUIRE(noa::all(is_contiguous<'F'>(strides, shape) == Vec4<bool>{0, 1, 1, 1}));

        strides = shape.strides<'F'>();
        strides[0] *= 4;
        strides[1] *= 2;
        REQUIRE(noa::all(is_contiguous<'F'>(strides, shape) == Vec4<bool>{0, 0, 1, 1}));

        shape = {3, 128, 64, 1};
        strides = {8192, 64, 1, 64};
        REQUIRE(noa::all(is_contiguous<'F'>(strides, shape) == Vec4<bool>{1, 1, 1, 1}));
        strides *= 2;
        REQUIRE(noa::all(is_contiguous<'F'>(strides, shape) == Vec4<bool>{1, 1, 0, 1}));
        strides[3] *= 2;
        REQUIRE(noa::all(is_contiguous<'F'>(strides, shape) == Vec4<bool>{1, 1, 0, 1}));
        strides[0] *= 2;
        strides[1] *= 2;
        strides[3] *= 2;
        REQUIRE(noa::all(is_contiguous<'F'>(strides, shape) == Vec4<bool>{1, 0, 0, 1}));
    }

    AND_THEN("is_contiguous<'F'> - broadcast") {
        const Shape4<u64> original_shape{3, 128, 1, 64};
        const Shape4<u64> broadcast_shape{3, 128, 64, 64};
        auto strides = original_shape.strides<'F'>();
        REQUIRE(noa::all(is_contiguous<'F'>(strides, original_shape) == Vec4<bool>{1, 1, 1, 1}));

        REQUIRE(broadcast(original_shape, strides, broadcast_shape));
        REQUIRE(noa::all(is_contiguous<'F'>(strides, broadcast_shape) == Vec4<bool>{1, 1, 0, 1}));
    }

    AND_THEN("physical shape") {
        const uint ndim = GENERATE(1u, 2u, 3u);
        for (size_t i = 0; i < 20; ++i) {
            auto shape = test::random_shape(ndim, {.batch_range={1, 5}});
            auto strides = shape.strides();
            REQUIRE(noa::all(shape.pop_front() == strides.physical_shape()));

            strides *= 3;
            shape[3] *= 3;
            REQUIRE(noa::all(shape.pop_front() == strides.physical_shape()));
        }
    }

    AND_THEN("is_rightmost") {
        Shape4<u64> shape{3, 128, 65, 64};
        REQUIRE(is_rightmost(shape.strides()));
        REQUIRE_FALSE(is_rightmost(shape.strides<'F'>()));

        shape = {3, 128, 1, 1};
        REQUIRE(is_rightmost(shape.strides()));
        REQUIRE(is_rightmost(shape.strides<'F'>()));
    }

    AND_THEN("is_vector") {
        Shape4<u64> shape{3, 128, 65, 64};
        REQUIRE_FALSE(is_vector(shape));

        shape = {3, 128, 1, 1};
        REQUIRE_FALSE(is_vector(shape));

        shape = {1, 1, 1, 128};
        REQUIRE(is_vector(shape));
        shape = {1, 1, 128, 1};
        REQUIRE(is_vector(shape));
        shape = {1, 128, 1, 1};
        REQUIRE(is_vector(shape));
        shape = {128, 1, 1, 1};
        REQUIRE(is_vector(shape));

        shape = {3, 1, 1, 128};
        REQUIRE(is_vector(shape, true));
        REQUIRE_FALSE(is_vector(shape, false));
        shape = {3, 128, 1, 1};
        REQUIRE(is_vector(shape, true));
        REQUIRE_FALSE(is_vector(shape, false));
    }

    AND_THEN("effective_shape") {
        const Shape4<u64> original_shape{3, 128, 1, 64};
        const Shape4<u64> broadcast_shape{3, 128, 64, 64};
        auto strides = original_shape.strides();
        REQUIRE(broadcast(original_shape, strides, broadcast_shape));
        REQUIRE(noa::all(effective_shape(broadcast_shape, strides) == original_shape));
    }
}

TEST_CASE("core::indexing:: order(), squeeze()", "[noa][core]") {
    const Shape4<u64> shape{2, 32, 64, 128};
    REQUIRE(noa::all(order(shape.strides(), shape) == Vec4<u64>{0, 1, 2, 3}));
    REQUIRE(noa::all(order(shape.strides<'F'>(), shape) == Vec4<u64>{0, 1, 3, 2}));

    // Order will move the broadcast dimensions to the right,
    // because they are indeed the dimensions with the smallest strides.
    REQUIRE(noa::all(order(Strides4<u64>{8192, 0, 128, 1}, shape) == Vec4<u64>{0, 2, 3, 1}));

    // We almost always call order() on the output array and rearrange the layout to make the output
    // as rightmost as possible. The output is rarely broadcast and in most cases it is actually invalid
    // to have a stride of 0 in the output (because unless the write operator is atomic, it will create
    // a data-race). Anyway, broadcast dimensions cause an issue here, because they are not contiguous
    // and order() will put them to the right. In practice, this is solved by calling effective_shape()
    // on the output shape beforehand, "tagging" these broadcast dimension as empty, so that order()
    // moves them to the left.
    // TL-DR: If is often best to call effective_shape() on the layout that is about to be reordered.
    const auto strides = Strides4<u64>{8192, 0, 128, 1};
    REQUIRE(noa::all(order(strides, effective_shape(shape, strides)) == Vec4<u64>{1, 0, 2, 3}));

    REQUIRE(noa::all(order(Strides4<i64>{8192, 0, 128, 1}, Shape4<i64>{2, 1, 64, 128}) == Vec4<i64>{1, 0, 2, 3}));
    REQUIRE(noa::all(order(Strides4<i64>{8192, 8192, 8192, 1}, Shape4<i64>{1, 1, 1, 8192}) == Vec4<i64>{0, 1, 2, 3}));
    REQUIRE(noa::all(order(Strides4<i64>{4096, 128, 128, 1}, Shape4<i64>{2, 32, 1, 128}) == Vec4<i64>{2, 0, 1, 3}));
    REQUIRE(noa::all(order(Strides4<i64>{4096, 4096, 128, 1}, Shape4<i64>{2, 32, 1, 128}) == Vec4<i64>{2, 0, 1, 3}));
    REQUIRE(noa::all(order(Strides4<i64>{128, 4096, 128, 1}, Shape4<i64>{2, 32, 1, 128}) == Vec4<i64>{2, 1, 0, 3}));
    REQUIRE(noa::all(order(Strides4<i64>{2, 2, 2, 2}, Shape4<i64>{1, 1, 1, 1}) == Vec4<i64>{0, 1, 2, 3}));

    REQUIRE(noa::all(squeeze_left(shape) == Vec4<u64>{0, 1, 2, 3}));
    REQUIRE(noa::all(squeeze_left(Shape4<u64>{1, 1, 3, 4}) == Vec4<u64>{0, 1, 2, 3}));
    REQUIRE(noa::all(squeeze_left(Shape4<u64>{1, 1, 3, 1}) == Vec4<u64>{0, 1, 3, 2}));
    REQUIRE(noa::all(squeeze_left(Shape4<u64>{1, 1, 1, 1}) == Vec4<u64>{0, 1, 2, 3}));
    REQUIRE(noa::all(squeeze_left(Shape4<u64>{5, 1, 3, 1}) == Vec4<u64>{1, 3, 0, 2}));
    REQUIRE(noa::all(squeeze_left(Shape4<u64>{5, 1, 1, 1}) == Vec4<u64>{1, 2, 3, 0}));

    REQUIRE(noa::all(squeeze_right(Shape4<u64>{1, 1, 3, 4}) == Vec4<u64>{2, 3, 0, 1}));
    REQUIRE(noa::all(squeeze_right(Shape4<u64>{1, 1, 3, 1}) == Vec4<u64>{2, 0, 1, 3}));
    REQUIRE(noa::all(squeeze_right(Shape4<u64>{1, 1, 1, 1}) == Vec4<u64>{0, 1, 2, 3}));
    REQUIRE(noa::all(squeeze_right(Shape4<u64>{5, 1, 3, 1}) == Vec4<u64>{0, 2, 1, 3}));
    REQUIRE(noa::all(squeeze_right(Shape4<u64>{5, 1, 1, 1}) == Vec4<u64>{0, 1, 2, 3}));
}

TEST_CASE("core::indexing:: memory layouts", "[noa][core]") {
    Shape4<u64> shape{2, 32, 64, 128};
    auto strides = shape.strides();
    REQUIRE(is_row_major(strides));
    REQUIRE(is_row_major(strides, shape));
    REQUIRE_FALSE(is_column_major(strides));
    REQUIRE_FALSE(is_column_major(strides, shape));

    strides = shape.strides<'F'>();
    REQUIRE_FALSE(is_row_major(strides));
    REQUIRE_FALSE(is_row_major(strides, shape));
    REQUIRE(is_column_major(strides));
    REQUIRE(is_column_major(strides, shape));

    // Even in the 'F' version, we still follow the BDHW order, so in our case, 'F' is not synonym of left-most.
    // Indeed, only the height/rows and width/columns are affected by 'C' vs 'F'. As such, if these two dimensions
    // are empty, 'C' and 'F' makes no difference.
    shape = {64, 32, 1, 1};
    REQUIRE(noa::all(shape.strides<'C'>() == shape.strides<'F'>()));

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
        REQUIRE(is_row_major(strides, shape));
        REQUIRE(is_column_major(strides, shape));

        auto order = squeeze_left(shape);
        shape = reorder(shape, order);
        strides = reorder(strides, order);
        REQUIRE(noa::all(is_contiguous(strides, shape)));
    }
}

TEST_CASE("core::indexing:: Reinterpret", "[noa][core]") {
    const auto shape = test::random_shape(3, {.batch_range={1, 10}});
    auto strides = shape.strides();
    c32* ptr = nullptr;
    auto real = ReinterpretLayout(shape, strides, ptr).as<float>();
    REQUIRE(noa::all(real.shape == Shape4<i64>{shape[0], shape[1], shape[2], shape[3] * 2}));
    REQUIRE(noa::all(real.strides == Strides4<i64>{strides[0] * 2, strides[1] * 2, strides[2] * 2, 1}));

    // Reinterpret moves everything to the rightmost order,
    // compute the new shape and strides, then moves back to original order.
    strides = shape.strides<'F'>();
    real = ReinterpretLayout(shape, strides, ptr).as<float>();
    REQUIRE(noa::all(real.shape == Shape4<i64>{shape[0], shape[1], shape[2] * 2, shape[3]}));
    REQUIRE(noa::all(real.strides == Strides4<i64>{strides[0] * 2, strides[1] * 2, 1, strides[3] * 2}));
}

TEMPLATE_TEST_CASE("core::indexing::offset2index(), 4D", "[noa][core]", Vec4<i64>, Vec4<u64>) {
    const u32 ndim = GENERATE(1u, 2u, 3u);

    using value_t = noa::traits::value_type_t<TestType>;
    test::Randomizer<value_t> randomizer(1, 3);
    test::Randomizer<value_t> idx_randomizer(0, 50);

    for (int i = 0; i < 20; ++i) {
        const auto shape = test::random_shape<value_t>(ndim, {.batch_range={1, 10}});
        const auto strides = shape.strides() * randomizer.get();

        // Generate a random 4D index.
        auto idx_expected = (shape - 1).vec;
        for (size_t j = 0; j < 4; ++j)
            idx_expected[j] = std::clamp(idx_expected[j] - idx_randomizer.get(), value_t{0}, shape[j] - 1);

        // Now find the 4D index of the offset.
        const value_t offset = offset_at(strides, idx_expected);
        const auto idx_result = offset2index(offset, strides, shape);
        INFO(strides);
        INFO(shape);
        INFO(idx_expected);
        INFO(idx_result);

        REQUIRE(all(idx_expected == idx_result));
    }
}

TEMPLATE_TEST_CASE("core::indexing::offset2index()", "[noa][core]",
                   Vec1<i64>, Vec2<i64>, Vec3<i64>, Vec4<i64>) {

    constexpr size_t N = TestType::SIZE;
    using value_t = noa::traits::value_type_t<TestType>;
    test::Randomizer<value_t> randomizer(2, 3);
    test::Randomizer<value_t> idx_randomizer(0, 50);

    for (i32 i = 0; i < 100; ++i) {
        const auto shape_nd = test::random_shape<value_t, N>(3, {.batch_range={1, 10}});

        // Generate a random nd index and its memory offset.
        auto idx_expected = (shape_nd - 1).vec;
        for (size_t j = 0; j < N; ++j)
            idx_expected[j] = std::clamp(idx_expected[j] - idx_randomizer.get(), value_t{0}, shape_nd[j] - 1);

        const auto strides_strided = shape_nd.strides() * randomizer.get();
        const auto strides_contiguous = shape_nd.strides();
        const auto offset_strided = offset_at(strides_strided, idx_expected);
        const auto offset_contiguous = offset_at(strides_contiguous, idx_expected);

        INFO(strides_strided);
        INFO(shape_nd);
        INFO(idx_expected);

        // Now find the nd index back.
        auto idx_result = offset2index(offset_strided, strides_strided, shape_nd);
        REQUIRE(noa::all(idx_expected == idx_result));
        idx_result = offset2index(offset_contiguous, strides_contiguous, shape_nd);
        REQUIRE(noa::all(idx_expected == idx_result));
        idx_result = offset2index(offset_contiguous, shape_nd);
        REQUIRE(noa::all(idx_expected == idx_result));
    }
}

TEMPLATE_TEST_CASE("core::indexing:: reorder matrices", "[noa][core]", u32, i32, i64, u64) {
    // When we reorder the array, if it is attached to a matrix, we need to reorder the matrix as well.
    test::Randomizer<double> randomizer(-3, 3);
    SECTION("2D") {
        Mat22<f64> matrix{};
        for (u64 i = 0; i < 2; ++i)
            for (u64 j = 0; j < 2; ++j)
            matrix[i][j] = randomizer.get();

        Vec2<f64> vector{randomizer.get(), randomizer.get()};
        Vec2<f64> expected = matrix * vector;

        Vec2<TestType> order{1, 0};
        vector = reorder(vector, order);
        matrix = reorder(matrix, order);
        Vec2<f64> result = matrix * vector;
        REQUIRE(expected[0] == result[1]);
        REQUIRE(expected[1] == result[0]);
    }

    SECTION("2D affine") {
        Mat23<f64> matrix{};
        for (u64 i = 0; i < 2; ++i)
            for (u64 j = 0; j < 2; ++j)
                matrix[i][j] = randomizer.get();

        Vec3<f64> vector{randomizer.get(), randomizer.get(), 1};
        Vec2<f64> expected = matrix * vector;

        vector = reorder(vector, Vec3<TestType>{1, 0, 2});
        matrix = reorder(matrix, Vec2<TestType>{1, 0});
        Vec2<f64> result = matrix * vector;
        REQUIRE(expected[0] == result[1]);
        REQUIRE(expected[1] == result[0]);
    }

    SECTION("3D") {
        Mat33<f64> matrix{};
        for (u64 i = 0; i < 3; ++i)
            for (u64 j = 0; j < 3; ++j)
                matrix[i][j] = randomizer.get();

        Vec3<f64> vector{randomizer.get(), randomizer.get(), randomizer.get()};
        Vec3<f64> expected = matrix * vector;

        using order_t = Vec3<TestType>;
        std::vector<order_t> orders{{0, 1, 2}, {0, 2, 1}, {2, 1, 0}, {1, 0, 2}};
        for (const auto& order: orders) {
            vector = reorder(vector, order);
            matrix = reorder(matrix, order);
            Vec3<f64> result = matrix * vector;
            expected = reorder(expected, order);

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
    auto new_shape = Shape4<u64>{1, 1, 1, product(shape1)};
    bool success = reshape(shape1, strides1, new_shape, new_strides);
    REQUIRE(success);
    REQUIRE(noa::all(new_strides == new_shape.strides()));

    // Broadcast the shape2 onto shape1.
    const auto shape2 = Shape4<u64>{30, 1, 10, 5};
    auto strides2 = shape2.strides();
    success = broadcast(shape2, strides2, shape1);
    REQUIRE(success);

    // Reshape the broadcast array to a row vector.
    new_shape = {1, 1, 1, product(shape2)};
    success = reshape(shape2, strides2, new_shape, new_strides);
    REQUIRE(success);
    REQUIRE(noa::all(new_strides == new_shape.strides()));
}

TEST_CASE("core::indexing::Subregion", "[noa][indexing]") {
    constexpr auto shape = Shape4<i64>{30, 20, 10, 5};
    constexpr auto strides = shape.strides();
    constexpr std::uintptr_t offset = 5;

    auto subregion = make_subregion<4>(0, 0).extract_from(shape, strides, offset);
    REQUIRE(noa::all(subregion.shape == Shape4<i64>{1, 1, 10, 5}));
    REQUIRE(noa::all(subregion.strides == strides));
    REQUIRE(noa::all(subregion.offset == offset));

    subregion = make_subregion<4>(Ellipsis{}, 5, 2).extract_from(shape, strides, offset);
    REQUIRE(noa::all(subregion.shape == Shape4<i64>{30, 20, 1, 1}));
    REQUIRE(noa::all(subregion.strides == strides));
    REQUIRE(noa::all(subregion.offset == offset + offset_at(strides, 0, 0, 5, 2)));

    subregion = make_subregion<4>(Slice{10, 20}, FullExtent{}, Slice{2, 5}, 3).extract_from(shape, strides, offset);
    REQUIRE(noa::all(subregion.shape == Shape4<i64>{10, 20, 3, 1}));
    REQUIRE(noa::all(subregion.strides == strides));
    REQUIRE(noa::all(subregion.offset == offset + offset_at(strides, 10, 0, 2, 3)));
}
