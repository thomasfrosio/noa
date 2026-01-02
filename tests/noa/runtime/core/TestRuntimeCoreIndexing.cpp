#include <noa/base/Complex.hpp>
#include <noa/runtime/core/Shape.hpp>
#include <noa/runtime/core/Subregion.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

using Border = noa::Border;
using namespace ::noa::types;

TEST_CASE("runtime::core::index_at()") {
    AND_THEN("Border::PERIODIC") {
        int expected_odd[55] = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
                                0, 1, 2, 3, 4,
                                0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4};
        int starts_at = 25;
        int len = 5;
        std::vector<int> data(55);
        for (size_t idx = 0; idx < data.size(); ++idx)
            data[idx] = noa::index_at<Border::PERIODIC>(static_cast<int>(idx) - starts_at, len);

        REQUIRE(test::allclose_abs(expected_odd, data.data(), static_cast<i64>(data.size())));

        int expected_even[36] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                                 0, 1, 2, 3,
                                 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
        starts_at = 16;
        len = 4;
        data = std::vector<int>(32);
        for (size_t idx = 0; idx < data.size(); ++idx)
            data[idx] = noa::index_at<Border::PERIODIC>(static_cast<int>(idx) - starts_at, len);

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
            data[idx] = noa::index_at<Border::CLAMP>(static_cast<int>(idx) - starts_at, len);

        REQUIRE(test::allclose_abs(expected_odd, data.data(), static_cast<i64>(data.size())));

        int expected_even[34] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 1, 2, 3,
                                 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
        starts_at = 15;
        len = 4;
        data = std::vector<int>(34);
        for (size_t idx = 0; idx < data.size(); ++idx)
            data[idx] = noa::index_at<Border::CLAMP>(static_cast<int>(idx) - starts_at, len);

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
            data[idx] = noa::index_at<Border::MIRROR>(static_cast<int>(idx) - starts_at, len);

        REQUIRE(test::allclose_abs(expected_odd, data.data(), static_cast<i64>(data.size())));

        int expected_even[52] = {0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0,
                                 0, 1, 2, 3,
                                 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3};
        starts_at = 24;
        len = 4;
        data = std::vector<int>(52);
        for (size_t idx = 0; idx < data.size(); ++idx)
            data[idx] = noa::index_at<Border::MIRROR>(static_cast<int>(idx) - starts_at, len);

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
            data[idx] = noa::index_at<Border::REFLECT>(static_cast<int>(idx) - starts_at, len);

        REQUIRE(test::allclose_abs(expected_odd, data.data(), static_cast<i64>(data.size())));

        int expected_even[40] = {0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1,
                                 0, 1, 2, 3,
                                 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3};
        starts_at = 18;
        len = 4;
        data = std::vector<int>(40);
        for (size_t idx = 0; idx < data.size(); ++idx)
            data[idx] = noa::index_at<Border::REFLECT>(static_cast<int>(idx) - starts_at, len);

        REQUIRE(test::allclose_abs(expected_even, data.data(), static_cast<i64>(data.size())));
    }
}

TEST_CASE("runtime::core:: shape, strides") {
    AND_THEN("C- contiguous") {
        const Shape<u64, 4> shape{2, 128, 64, 65};
        const auto strides = shape.strides();
        const auto physical_shape = strides.physical_shape();
        REQUIRE(strides.contiguity(shape) == true);
        REQUIRE(strides.is_contiguous(shape));
        REQUIRE(Strides<u64, 4>{532480, 4160, 65, 1} == strides);
        REQUIRE(Shape<u64, 3>{128, 64, 65} == physical_shape);
    }

    AND_THEN("F- contiguous") {
        const Shape<u64, 4> shape{2, 128, 64, 65};
        const auto strides = shape.strides<'F'>();
        const auto physical_shape = strides.physical_shape<'F'>();
        REQUIRE(strides.contiguity<'F'>(shape) == true);
        REQUIRE(strides.is_contiguous<'F'>(shape));
        REQUIRE(Strides<u64, 4>{532480, 4160, 1, 64} == strides);
        REQUIRE(Shape<u64, 3>{128, 64, 65} == physical_shape);
    }

    AND_THEN("C- inner stride") {
        const Shape<u64, 4> shape{3, 128, 64, 64};
        const auto strides = shape.strides() * 2;
        const auto physical_shape = strides.physical_shape();
        REQUIRE(strides.contiguity(shape) == Vec<bool, 4>{1, 1, 1, 0});
        REQUIRE(Strides<u64, 4>{1048576, 8192, 128, 2} == strides);
        REQUIRE(Shape<u64, 3>{128, 64, 128} == physical_shape);
    }

    AND_THEN("F- inner stride") {
        const Shape<u64, 4> shape{3, 128, 64, 64};
        const auto strides = shape.strides<'F'>() * 2;
        const auto physical_shape = strides.physical_shape<'F'>();
        REQUIRE(strides.contiguity<'F'>(shape) == Vec<bool, 4>{1, 1, 0, 1});
        REQUIRE(Strides<u64, 4>{1048576, 8192, 2, 128} == strides);
        REQUIRE(Shape<u64, 3>{128, 128, 64} == physical_shape);
    }

    AND_THEN("is_contiguous<'C'>") {
        Shape<u64, 4> shape{3, 128, 64, 64};
        auto strides = shape.strides();
        strides[0] *= 2;
        strides[1] *= 2;
        strides[2] *= 2;
        REQUIRE(strides.contiguity(shape) == Vec<bool, 4>{1, 1, 0, 1});

        strides = shape.strides();
        strides[0] *= 6;
        strides[1] *= 6;
        strides[2] *= 6;
        strides[3] *= 3;
        REQUIRE(strides.contiguity(shape) == Vec<bool, 4>{1, 1, 0, 0});

        strides = shape.strides();
        strides[0] *= 2;
        strides[1] *= 2;
        REQUIRE(strides.contiguity(shape) == Vec<bool, 4>{1, 0, 1, 1});

        strides = shape.strides();
        strides[0] *= 2;
        REQUIRE(strides.contiguity(shape) == Vec<bool, 4>{0, 1, 1, 1});

        strides = shape.strides();
        strides[0] *= 4;
        strides[1] *= 2;
        REQUIRE(strides.contiguity(shape) == Vec<bool, 4>{0, 0, 1, 1});

        // Empty is contiguous by definition.
        shape = {3, 128, 1, 64};
        strides = {8192, 64, 64, 1};
        REQUIRE(strides.contiguity(shape) == Vec<bool, 4>{1, 1, 1, 1});
        strides *= 2;
        REQUIRE(strides.contiguity(shape) == Vec<bool, 4>{1, 1, 1, 0});
        strides[2] *= 2;
        REQUIRE(strides.contiguity(shape) == Vec<bool, 4>{1, 1, 1, 0});
        strides[0] *= 2;
        strides[1] *= 2;
        strides[2] *= 2;
        REQUIRE(strides.contiguity(shape) == Vec<bool, 4>{1, 0, 1, 0});
    }

    AND_THEN("is_contiguous<'C'> - broadcast") {
        const Shape4 original_shape{3, 128, 1, 64};
        const Shape4 broadcast_shape{3, 128, 32, 64};
        auto strides = original_shape.strides();
        REQUIRE(strides.contiguity(original_shape) == Vec<bool, 4>{1, 1, 1, 1});

        REQUIRE(broadcast(original_shape, strides, broadcast_shape));
        REQUIRE(strides.contiguity(broadcast_shape) == Vec<bool, 4>{1, 1, 0, 1});

        REQUIRE((Strides4{0, 0, 0, 1}.contiguity(broadcast_shape) == Vec<bool, 4>{0, 0, 0, 1}));
        REQUIRE((Strides4{0, 64, 0, 1}.contiguity(broadcast_shape) == Vec<bool, 4>{0, 1, 0, 1}));
        REQUIRE((Strides4{128 * 64, 64, 0, 1}.contiguity(broadcast_shape) == Vec<bool, 4>{1, 1, 0, 1}));
        REQUIRE((Strides4{64, 0, 0, 1}.contiguity(broadcast_shape) == Vec<bool, 4>{1, 0, 0, 1}));
    }

    AND_THEN("is_contiguous<'F'>") {
        Shape<u64, 4> shape{3, 128, 64, 64};
        auto strides = shape.strides<'F'>();
        strides[0] *= 2;
        strides[1] *= 2;
        strides[3] *= 2;
        REQUIRE(strides.contiguity<'F'>(shape) == Vec{1, 1, 1, 0}.as<bool>());

        strides = shape.strides<'F'>();
        strides[0] *= 6;
        strides[1] *= 6;
        strides[2] *= 3;
        strides[3] *= 6;
        REQUIRE(strides.contiguity<'F'>(shape) == Vec{1, 1, 0, 0}.as<bool>());

        strides = shape.strides<'F'>();
        strides[0] *= 2;
        strides[1] *= 2;
        REQUIRE(strides.contiguity<'F'>(shape) == Vec{1, 0, 1, 1}.as<bool>());

        strides = shape.strides<'F'>();
        strides[0] *= 2;
        REQUIRE(strides.contiguity<'F'>(shape) == Vec{0, 1, 1, 1}.as<bool>());

        strides = shape.strides<'F'>();
        strides[0] *= 4;
        strides[1] *= 2;
        REQUIRE(strides.contiguity<'F'>(shape) == Vec{0, 0, 1, 1}.as<bool>());

        shape = {3, 128, 64, 1};
        strides = {8192, 64, 1, 64};
        REQUIRE(strides.contiguity<'F'>(shape) == Vec{1, 1, 1, 1}.as<bool>());
        strides *= 2;
        REQUIRE(strides.contiguity<'F'>(shape) == Vec{1, 1, 0, 1}.as<bool>());
        strides[3] *= 2;
        REQUIRE(strides.contiguity<'F'>(shape) == Vec{1, 1, 0, 1}.as<bool>());
        strides[0] *= 2;
        strides[1] *= 2;
        strides[3] *= 2;
        REQUIRE(strides.contiguity<'F'>(shape) == Vec{1, 0, 0, 1}.as<bool>());
    }

    AND_THEN("is_contiguous<'F'> - broadcast") {
        const Shape<u64, 4> original_shape{3, 128, 1, 64};
        const Shape<u64, 4> broadcast_shape{3, 128, 64, 64};
        auto strides = original_shape.strides<'F'>();
        REQUIRE(strides.contiguity<'F'>(original_shape) == Vec<bool, 4>{1, 1, 1, 1});

        REQUIRE(broadcast(original_shape, strides, broadcast_shape));
        REQUIRE(strides.contiguity<'F'>(broadcast_shape) == Vec<bool, 4>{1, 1, 0, 1});
    }

    AND_THEN("physical shape") {
        const u32 ndim = GENERATE(1u, 2u, 3u);
        for (size_t i = 0; i < 20; ++i) {
            auto shape = test::random_shape(ndim, {.batch_range={1, 5}});
            auto strides = shape.strides();
            REQUIRE(shape.pop_front() == strides.physical_shape());

            strides *= 3;
            shape[3] *= 3;
            REQUIRE(shape.pop_front() == strides.physical_shape());
        }
    }

    AND_THEN("is_rightmost") {
        Shape<u64, 4> shape{3, 128, 65, 64};
        REQUIRE(shape.strides().is_rightmost());
        REQUIRE_FALSE(shape.strides<'F'>().is_rightmost());

        shape = {3, 128, 1, 1};
        REQUIRE(shape.strides().is_rightmost());
        REQUIRE(shape.strides<'F'>().is_rightmost());
    }

    AND_THEN("is_vector") {
        Shape<u64, 4> shape{3, 128, 65, 64};
        REQUIRE_FALSE(shape.is_vector());

        shape = {3, 128, 1, 1};
        REQUIRE_FALSE(shape.is_vector());

        shape = {1, 1, 1, 128};
        REQUIRE(shape.is_vector());
        shape = {1, 1, 128, 1};
        REQUIRE(shape.is_vector());
        shape = {1, 128, 1, 1};
        REQUIRE(shape.is_vector());
        shape = {128, 1, 1, 1};
        REQUIRE(shape.is_vector());

        shape = {3, 1, 1, 128};
        REQUIRE(shape.is_vector(true));
        REQUIRE_FALSE(shape.is_vector(false));
        shape = {3, 128, 1, 1};
        REQUIRE(shape.is_vector(true));
        REQUIRE_FALSE(shape.is_vector(false));
    }

    AND_THEN("effective_shape") {
        const Shape<u64, 4> original_shape{3, 128, 1, 64};
        const Shape<u64, 4> broadcast_shape{3, 128, 64, 64};
        auto strides = original_shape.strides();
        REQUIRE(broadcast(original_shape, strides, broadcast_shape));
        REQUIRE(strides.effective_shape(broadcast_shape) == original_shape);
    }
}

TEST_CASE("runtime::core:: rightmost_order(), squeeze()") {
    const Shape<u64, 4> shape{2, 32, 64, 128};
    REQUIRE(shape.strides().rightmost_order(shape) == Vec<u64, 4>{0, 1, 2, 3});
    REQUIRE(shape.strides<'F'>().rightmost_order(shape) == Vec<u64, 4>{0, 1, 3, 2});

    // Order will move the broadcast dimensions to the right,
    // because they are indeed the dimensions with the smallest strides.
    REQUIRE(Strides<u64, 4>{8192, 0, 128, 1}.rightmost_order(shape) == Vec<u64, 4>{0, 2, 3, 1});

    // We almost always call order() on the output array and rearrange the layout to make the output
    // as rightmost as possible. The output is rarely broadcast and in most cases it is actually invalid
    // to have a stride of 0 in the output (because unless the write operator is atomic, it will create
    // a data-race). Anyway, broadcast dimensions cause an issue here, because they are not contiguous
    // and order() will put them to the right. In practice, this is solved by calling effective_shape()
    // on the output shape beforehand, "tagging" these broadcast dimension as empty, so that order()
    // moves them to the left.
    // TL-DR: If is often best to call effective_shape() on the layout that is about to be reordered.
    const auto strides = Strides<u64, 4>{8192, 0, 128, 1};
    REQUIRE(strides.rightmost_order(strides.effective_shape(shape)) == Vec<u64, 4>{1, 0, 2, 3});

    REQUIRE((Strides<i64, 4>{8192, 0, 128, 1}.rightmost_order(Shape<i64, 4>{2, 1, 64, 128}) == Vec<i64, 4>{1, 0, 2, 3}));
    REQUIRE((Strides<i64, 4>{8192, 8192, 8192, 1}.rightmost_order(Shape<i64, 4>{1, 1, 1, 8192}) == Vec<i64, 4>{0, 1, 2, 3}));
    REQUIRE((Strides<i64, 4>{4096, 128, 128, 1}.rightmost_order(Shape<i64, 4>{2, 32, 1, 128}) == Vec<i64, 4>{2, 0, 1, 3}));
    REQUIRE((Strides<i64, 4>{4096, 4096, 128, 1}.rightmost_order(Shape<i64, 4>{2, 32, 1, 128}) == Vec<i64, 4>{2, 0, 1, 3}));
    REQUIRE((Strides<i64, 4>{128, 4096, 128, 1}.rightmost_order(Shape<i64, 4>{2, 32, 1, 128}) == Vec<i64, 4>{2, 1, 0, 3}));
    REQUIRE((Strides<i64, 4>{2, 2, 2, 2}.rightmost_order(Shape<i64, 4>{1, 1, 1, 1}) == Vec<i64, 4>{0, 1, 2, 3}));

    REQUIRE(squeeze_left(shape) == Vec<u64, 4>{0, 1, 2, 3});
    REQUIRE(squeeze_left(Shape<u64, 4>{1, 1, 3, 4}) == Vec<u64, 4>{0, 1, 2, 3});
    REQUIRE(squeeze_left(Shape<u64, 4>{1, 1, 3, 1}) == Vec<u64, 4>{0, 1, 3, 2});
    REQUIRE(squeeze_left(Shape<u64, 4>{1, 1, 1, 1}) == Vec<u64, 4>{0, 1, 2, 3});
    REQUIRE(squeeze_left(Shape<u64, 4>{5, 1, 3, 1}) == Vec<u64, 4>{1, 3, 0, 2});
    REQUIRE(squeeze_left(Shape<u64, 4>{5, 1, 1, 1}) == Vec<u64, 4>{1, 2, 3, 0});

    REQUIRE(squeeze_right(Shape<u64, 4>{1, 1, 3, 4}) == Vec<u64, 4>{2, 3, 0, 1});
    REQUIRE(squeeze_right(Shape<u64, 4>{1, 1, 3, 1}) == Vec<u64, 4>{2, 0, 1, 3});
    REQUIRE(squeeze_right(Shape<u64, 4>{1, 1, 1, 1}) == Vec<u64, 4>{0, 1, 2, 3});
    REQUIRE(squeeze_right(Shape<u64, 4>{5, 1, 3, 1}) == Vec<u64, 4>{0, 2, 1, 3});
    REQUIRE(squeeze_right(Shape<u64, 4>{5, 1, 1, 1}) == Vec<u64, 4>{0, 1, 2, 3});
}

TEST_CASE("runtime::core:: memory layouts") {
    Shape<u64, 4> shape{2, 32, 64, 128};
    auto strides = shape.strides();
    REQUIRE(strides.is_row_major());
    REQUIRE(strides.is_row_major(shape));
    REQUIRE_FALSE(strides.is_column_major());
    REQUIRE_FALSE(strides.is_column_major(shape));

    strides = shape.strides<'F'>();
    REQUIRE_FALSE(strides.is_row_major());
    REQUIRE_FALSE(strides.is_row_major(shape));
    REQUIRE(strides.is_column_major());
    REQUIRE(strides.is_column_major(shape));

    // Even in the 'F' version, we still follow the BDHW order, so in our case, 'F' is not synonym of left-most.
    // Indeed, only the height/rows and width/columns are affected by 'C' vs 'F'. As such, if these two dimensions
    // are empty, 'C' and 'F' makes no difference.
    shape = {64, 32, 1, 1};
    REQUIRE(shape.strides<'C'>() == shape.strides<'F'>());

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
        REQUIRE(strides.is_row_major(shape));
        REQUIRE(strides.is_column_major(shape));

        auto order = squeeze_left(shape);
        shape = shape.permute(order);
        strides = strides.permute(order);
        REQUIRE(strides.contiguity(shape) == true);
    }
}

TEST_CASE("runtime::core:: Reinterpret") {
    const auto shape = test::random_shape<i64>(3, {.batch_range={1, 10}});
    auto strides = shape.strides();
    c32* ptr = nullptr;
    auto real = noa::details::ReinterpretLayout(shape, strides, ptr).as<float>();
    REQUIRE(real.shape == Shape<i64, 4>{shape[0], shape[1], shape[2], shape[3] * 2});
    REQUIRE(real.strides == Strides<i64, 4>{strides[0] * 2, strides[1] * 2, strides[2] * 2, 1});

    // Reinterpret moves everything to the rightmost order,
    // compute the new shape and strides, then moves back to original order.
    strides = shape.strides<'F'>();
    real = noa::details::ReinterpretLayout(shape, strides, ptr).as<float>();
    REQUIRE(real.shape == Shape<i64, 4>{shape[0], shape[1], shape[2] * 2, shape[3]});
    REQUIRE(real.strides == Strides<i64, 4>{strides[0] * 2, strides[1] * 2, 1, strides[3] * 2});
}

TEMPLATE_TEST_CASE("runtime::core::offset2index(), 4D", "", (Vec<i64, 4>), (Vec<u64, 4>)) {
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

        REQUIRE(idx_expected == idx_result);
    }
}

TEMPLATE_TEST_CASE("runtime::core::offset2index()", "", (Vec<i64, 1>), (Vec<i64, 2>), (Vec<i64, 3>), (Vec<i64, 4>)) {
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
        REQUIRE(idx_expected == idx_result);
        idx_result = offset2index(offset_contiguous, strides_contiguous, shape_nd);
        REQUIRE(idx_expected == idx_result);
        idx_result = offset2index(offset_contiguous, shape_nd);
        REQUIRE(idx_expected == idx_result);
    }
}

TEST_CASE("runtime::core::reshape, broadcasting") {
    const auto shape1 = Shape<u64, 4>{30, 20, 10, 5};
    const auto strides1 = Strides<u64, 4>{1000, 50, 5, 1};

    // Reshape contiguous array to a row vector.
    Strides<u64, 4> new_strides;
    auto new_shape = Shape<u64, 4>{1, 1, 1, product(shape1)};
    bool success = noa::details::reshape(shape1, strides1, new_shape, new_strides);
    REQUIRE(success);
    REQUIRE(new_strides == new_shape.strides());

    // Broadcast the shape2 onto shape1.
    const auto shape2 = Shape<u64, 4>{30, 1, 10, 5};
    auto strides2 = shape2.strides();
    success = broadcast(shape2, strides2, shape1);
    REQUIRE(success);

    // Reshape the broadcast array to a row vector.
    new_shape = {1, 1, 1, product(shape2)};
    success = noa::details::reshape(shape2, strides2, new_shape, new_strides);
    REQUIRE(success);
    REQUIRE(new_strides == new_shape.strides());
}

TEST_CASE("runtime::core::Subregion") {
    constexpr auto shape = Shape<i64, 4>{30, 20, 10, 5};
    constexpr auto strides = shape.strides();
    constexpr std::uintptr_t offset = 5;

    auto subregion = noa::make_subregion<4>(0, 0).extract_from(shape, strides, offset);
    REQUIRE(subregion.shape == Shape<i64, 4>{1, 1, 10, 5});
    REQUIRE(subregion.strides == strides);
    REQUIRE(subregion.offset == offset);

    subregion = noa::make_subregion<4>(Ellipsis{}, 5, 2).extract_from(shape, strides, offset);
    REQUIRE(subregion.shape == Shape<i64, 4>{30, 20, 1, 1});
    REQUIRE(subregion.strides == strides);
    REQUIRE(subregion.offset == offset + offset_at(strides, 0, 0, 5, 2));

    subregion = noa::make_subregion<4>(Slice{10, 20}, Full{}, Slice{2, 5}, 3).extract_from(shape, strides, offset);
    REQUIRE(subregion.shape == Shape<i64, 4>{10, 20, 3, 1});
    REQUIRE(subregion.strides == strides);
    REQUIRE(subregion.offset == offset + offset_at(strides, 10, 0, 2, 3));
}
