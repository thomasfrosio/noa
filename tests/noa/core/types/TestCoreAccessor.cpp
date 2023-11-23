#include <noa/core/types/Accessor.hpp>
#include <noa/core/utils/Irange.hpp>

#include <catch2/catch.hpp>
#include "Helpers.h"

using namespace ::noa;

TEMPLATE_TEST_CASE("core:: Accessor", "[noa]", i32, i64, u32, u64) {
    const Shape4<u64> shape{2, 30, 50, 60};
    const auto strides = shape.strides().as<TestType>();
    const auto elements = shape.elements();

    std::unique_ptr<int[]> raw_data = std::make_unique<int[]>(elements);
    Accessor<int, 4, TestType> accessor(raw_data.get(), strides);
    for (auto i: irange(4))
        REQUIRE(accessor.strides()[i] == static_cast<TestType>(strides[i]));

    AND_THEN("multidimensional indexing") {
        std::unique_ptr<int[]> expected_data = std::make_unique<int[]>(elements);
        for (size_t i = 0; i < elements; ++i)
            expected_data[i] = static_cast<int>(i);

        int count = 0;
        for (size_t w = 0; w < shape[0]; ++w) {
            const auto batch = accessor[w];
            for (size_t z = 0; z < shape[1]; ++z)
                for (size_t y = 0; y < shape[2]; ++y)
                    for (size_t x = 0; x < shape[3]; ++x)
                        batch[z][y][x] = count++;
        }
        REQUIRE(test::Matcher(test::MATCH_ABS, raw_data.get(), expected_data.get(), elements, 0));

        test::memset(accessor.get(), elements, 0);
        count = 0;
        for (size_t w = 0; w < shape[0]; ++w) {
            for (size_t z = 0; z < shape[1]; ++z)
                for (size_t y = 0; y < shape[2]; ++y)
                    for (size_t x = 0; x < shape[3]; ++x)
                        accessor(w, z, y, x) = count++;
        }
        REQUIRE(test::Matcher(test::MATCH_ABS, raw_data.get(), expected_data.get(), elements, 0));
    }

    AND_THEN("const conversion") {
        Accessor<const int, 4, TestType> a(raw_data.get(), strides);
        Accessor<const int, 4, TestType> b = accessor; // implicit const conversion
        Accessor<const int, 4, TestType> c(accessor);

        REQUIRE((a.get() == b.get() &&
                 a.get() == c.get() &&
                 accessor.get() == a.get()));
    }

    AND_THEN("ND") {
        // This test requires a contiguous layout.
        Accessor<int, 1, TestType> a(raw_data.get(), strides.filter(3));
        Accessor<int, 2, TestType> b(raw_data.get(), strides.filter(2, 3));
        Accessor<int, 3, TestType> c(raw_data.get(), strides.filter(1, 2, 3));
        Accessor<int, 4, TestType> d(raw_data.get(), strides);
        const Vec4<TestType> idx{1, 3, 4, 2};

        int av = a[noa::offset_at(idx, strides)];
        int bv = b[idx[0]][noa::offset_at(idx[1], idx[2], idx[3], strides.pop_front())];
        int cv = c[idx[0]][idx[1]][noa::offset_at(idx[2], idx[3], strides.filter(2, 3))];
        int dv = d[idx[0]][idx[1]][idx[2]][idx[3]];
        int ev = d(idx);
        int fv = d(idx[0], idx[1], idx[2], idx[3]);

        REQUIRE((av == bv && av == cv && av == dv && av == ev && av == fv));
    }
}
