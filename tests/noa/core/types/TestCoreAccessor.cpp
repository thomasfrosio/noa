#include <noa/core/types/Accessor.hpp>
#include <noa/core/utils/Irange.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

using namespace ::noa::types;

TEMPLATE_TEST_CASE("core:: Accessor", "", i32, i64, u32, u64) {
    using noa::indexing::offset_at;

    const auto shape = Shape<u64, 4>{2, 30, 50, 60};
    const auto strides = shape.strides().as<TestType>();
    const auto elements = shape.n_elements();

    auto raw_data = std::make_unique<i32[]>(elements);
    auto accessor = Accessor<i32, 4, TestType>(raw_data.get(), strides);
    for (auto i: noa::irange(4))
        REQUIRE(accessor.strides()[i] == static_cast<TestType>(strides[i]));

    AND_THEN("offset") {
        REQUIRE(accessor.offset_at(1, 2, 3, 4) == offset_at(accessor.strides(), 1, 2, 3, 4));
        REQUIRE(accessor.offset_at(1, 2, 3) == offset_at(accessor.strides(), 1, 2, 3));
        REQUIRE(accessor.offset_at(1, 2) == offset_at(accessor.strides(), 1, 2));
        REQUIRE(accessor.offset_at(1) == offset_at(accessor.strides(), 1));

        REQUIRE(accessor.offset_pointer(accessor.get(), 1, 2, 3, 4) == accessor.get() + offset_at(accessor.strides(), 1, 2, 3, 4));
        REQUIRE(accessor.offset_pointer(accessor.get(), 1, 2, 3) == accessor.get() + offset_at(accessor.strides(), 1, 2, 3));
        REQUIRE(accessor.offset_pointer(accessor.get(), 1, 2) == accessor.get() + offset_at(accessor.strides(), 1, 2));
        REQUIRE(accessor.offset_pointer(accessor.get(), 1) == accessor.get() + offset_at(accessor.strides(), 1));

        auto accessor0 = accessor;
        REQUIRE(accessor0.offset_inplace(1, 2, 3, 4).get() == accessor.offset_pointer(accessor.get(), 1, 2, 3, 4));
        REQUIRE(accessor0.get() == accessor.offset_pointer(accessor.get(), 1, 2, 3, 4));

        auto accessor1 = accessor;
        REQUIRE(accessor1.offset_inplace(Vec{1, 2}).get() == accessor.offset_pointer(accessor.get(), 1, 2));
        REQUIRE(accessor1.offset_inplace(Vec{0, 0, 3, 4}).get() == accessor.offset_pointer(accessor.get(), 1, 2, 3, 4));
    }

    AND_THEN("multidimensional indexing") {
        std::unique_ptr expected_data = std::make_unique<i32[]>(elements);
        for (size_t i{}; i < elements; ++i)
            expected_data[i] = static_cast<i32>(i);

        i32 count{};
        for (size_t w{}; w < shape[0]; ++w) {
            const auto batch = accessor[w];
            for (size_t z{}; z < shape[1]; ++z)
                for (size_t y{}; y < shape[2]; ++y)
                    for (size_t x{}; x < shape[3]; ++x)
                        batch[z][y][x] = count++;
        }
        REQUIRE(test::allclose_abs(raw_data.get(), expected_data.get(), elements, 0));

        test::fill(accessor.get(), elements, 0);
        count = 0;
        for (size_t w{}; w < shape[0]; ++w) {
            for (size_t z{}; z < shape[1]; ++z)
                for (size_t y{}; y < shape[2]; ++y)
                    for (size_t x{}; x < shape[3]; ++x)
                        accessor(w, z, y, x) = count++;
        }
        REQUIRE(test::allclose_abs(raw_data.get(), expected_data.get(), elements, 0));
    }

    AND_THEN("const conversion") {
        Accessor<const i32, 4, TestType> a(raw_data.get(), strides);
        Accessor<const i32, 4, TestType> b = accessor; // implicit const conversion
        Accessor<const i32, 4, TestType> c(accessor);

        REQUIRE((a.get() == b.get() and
                 a.get() == c.get() and
                 accessor.get() == a.get()));
    }

    AND_THEN("nd") {
        // This test requires a contiguous layout.
        Accessor<int, 1, TestType> a(raw_data.get(), strides.filter(3));
        Accessor<int, 2, TestType> b(raw_data.get(), strides.filter(2, 3));
        Accessor<int, 3, TestType> c(raw_data.get(), strides.filter(1, 2, 3));
        Accessor<int, 4, TestType> d(raw_data.get(), strides);
        const auto idx = Vec<TestType, 4>{1, 3, 4, 2};

        int av = a[offset_at(strides, idx)];
        int bv = b[idx[0]][offset_at(strides.pop_front(), idx[1], idx[2], idx[3])];
        int cv = c[idx[0]][idx[1]][offset_at(strides.filter(2, 3), idx[2], idx[3])];
        int dv = d[idx[0]][idx[1]][idx[2]][idx[3]];
        int ev = d(idx);
        int fv = d(idx[0], idx[1], idx[2], idx[3]);

        REQUIRE((av == bv and av == cv and av == dv and av == ev and av == fv));
    }
}

TEMPLATE_TEST_CASE("core:: AccessorValue", "", i32, i64, u32, u64) {
    AccessorValue<TestType> accessor_value; // uninitialized
    accessor_value(0) = 1;
    REQUIRE(accessor_value.ref() == 1);
    accessor_value(0, 0, 0, 0) = 2;
    REQUIRE(accessor_value.ref() == 2);
    accessor_value[2] = 3; // behave like an array broadcasted to any dimension
    REQUIRE(accessor_value.ref() == 3);
    REQUIRE(accessor_value.template stride<0>() == 0);

    AccessorValue<const TestType> accessor_value0 = accessor_value;
    // AccessorValue<TestType> accessor_value1 = accessor_value0; // rightfully does not compile
    REQUIRE(accessor_value.ref() == accessor_value0.ref());
    // accessor_value0.ref() = 4; // rightfully does not compile
    accessor_value0.ref_() = 4; // "private" function to "bypass" the const
    REQUIRE(accessor_value0(0) == 4);
}
