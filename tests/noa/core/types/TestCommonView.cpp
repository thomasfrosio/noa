#include <noa/core/types/View.hpp>
#include <noa/core/utils/Irange.hpp>

#include <catch2/catch.hpp>
#include "Helpers.h"

using namespace ::noa;
using namespace ::noa::indexing;

TEST_CASE("common: View", "[noa][common]") {
    View<int, int64_t> a;
    REQUIRE(a.empty());

    const View<const int, int64_t> b(a.data(), a.shape());
    REQUIRE(b.empty());

    a = View<int, int64_t>(nullptr, long4_t{10, 10, 10, 10});
    REQUIRE(a.contiguous());

    // to const
    View<const int, int64_t> d(a.data(), a.shape(), a.strides());
    View<const int, int64_t> e(a);
    REQUIRE(all(d.shape() == a.shape()));
    REQUIRE(all(d.strides() == a.strides()));
    REQUIRE(d.data() == a.data());
    REQUIRE(e.get() == a.get());

    View<float, int32_t> a1;
    View<const float, int32_t> a2;
    // a1 = a2; // loss const - do not compile
    a2 = a1; // fine
}

TEST_CASE("common: View - indexing", "[noa][common]") {
    const size4_t shape{3, 50, 40, 60};
    const size_t elements = shape.elements();
    std::unique_ptr<int[]> buffer = std::make_unique<int[]>(elements);
    for (auto i: irange(elements))
        buffer[i] = static_cast<int>(i);

    const View<int, size_t> v0(buffer.get(), shape);
    REQUIRE(v0(1, 45, 23, 10) == buffer[at(1, 45, 23, 10, v0.strides())]);
    REQUIRE(v0(1, 45, 23, 10) == v0[at(1, 45, 23, 10, v0.strides())]);

    REQUIRE(v0(1, 45, 23) == buffer[at(1, 45, 23, v0.strides())]);
    REQUIRE(v0(1, 45) == buffer[at(1, 45, v0.strides())]);
    REQUIRE(v0(1) == buffer[at(1, v0.strides())]);

    AND_THEN("full extent") {
        const View<int, size_t> v1 = v0.subregion(ellipsis_t{}, 15);
        REQUIRE(all(v1.shape() == size4_t{3, 50, 40, 1}));
        REQUIRE(v0(0, 0, 0, 15) == v1(0));

        const View<int, size_t> v2 = v0.subregion(ellipsis_t{}, 39, full_extent_t{});
        REQUIRE(all(v2.shape() == size4_t{3, 50, 1, 60}));
        REQUIRE(v0(1, 1, 39, 3) == v2(1, 1, 0, 3));

        const View<int, size_t> v3 = v0.subregion(ellipsis_t{}, 9, full_extent_t{}, full_extent_t{});
        REQUIRE(all(v3.shape() == size4_t{3, 1, 40, 60}));
        REQUIRE(v0(1, 9, 25, 14) == v3(1, 0, 25, 14));

        const View<int, size_t> v4 = v0.subregion(1, full_extent_t{}, full_extent_t{}, full_extent_t{});
        const View<int, size_t> v5 = v0.subregion(1);
        REQUIRE(all(v4.shape() == size4_t{1, 50, 40, 60}));
        REQUIRE(all(v4.shape() == v5.shape()));
        REQUIRE(v4(0) == v5(0));
        REQUIRE(v4(0) == v0(1));
    }

    AND_THEN("slice") {
        const View<int, size_t> v1 = v0.subregion(ellipsis_t{}, slice_t{});
        REQUIRE(all(v1.shape() == v0.shape()));

        const View<int, size_t> v2 = v0.subregion(ellipsis_t{}, slice_t{5, 45});
        const View<int, size_t> v3 = v0.subregion(ellipsis_t{}, slice_t{5, -15});
        REQUIRE(all(v2.shape() == size4_t{3, 50, 40, 40}));
        REQUIRE(all(v3.shape() == size4_t{3, 50, 40, 40}));
        REQUIRE(v2(1, 1, 1, 4) == v0(1, 1, 1, 9));
        REQUIRE(v2(2, 32, 14, 7) == v3(2, 32, 14, 7));

        const View<int, size_t> v4 = v3.subregion(slice_t{0, -1},
                                          slice_t{1, 100}, // clamp
                                          slice_t{10, 15},
                                          slice_t{-25, -10});
        REQUIRE(all(v4.shape() == size4_t{2, 49, 5, 15}));
        REQUIRE(v4(1, 2, 2, 0) == v3(1, 3, 12, 15));
    }

    AND_THEN("stride") {
        const View<int, size_t> v1 = v0.subregion(ellipsis_t{}, slice_t{0, 60, 2});
        REQUIRE(all(v1.shape() == size4_t{3, 50, 40, 30}));
        REQUIRE(v0(1, 6, 3, 10) == v1(1, 6, 3, 5));
        REQUIRE(v0(1, 6, 3, 28) == v1(1, 6, 3, 14));
    }

    AND_THEN("empty or oob") {
        const View<int, size_t> v1 = v0.subregion(slice_t{3, 3},
                                          slice_t{30, 20},
                                          slice_t{10, 10},
                                          slice_t{-10, -20});
        REQUIRE(all(v1.shape() == size4_t{}));
        REQUIRE(v1.empty());

        REQUIRE_THROWS(v0.subregion(2, 30, 10, 60 /* oob */));
        REQUIRE_THROWS(v0.subregion(2, 30, -41 /* oob */, 0));
    }
}

TEMPLATE_TEST_CASE("common: View, shape manipulation", "[noa][common]",
                   int32_t, uint64_t, float, double, cfloat_t, cdouble_t) {
    std::unique_ptr<int[]> buffer = std::make_unique<int[]>(2000);
    for (auto i: irange<size_t>(2000))
        buffer[i] = static_cast<int>(i);

    AND_THEN("as another type") {
        View<int, size_t> c(buffer.get(), size4_t{2, 3, 4, 5});
        View<unsigned char, size_t> d = c.as<unsigned char>();
        REQUIRE(all(d.shape() == size4_t{2, 3, 4, 20}));
        REQUIRE(all(d.strides() == size4_t{240, 80, 20, 1}));

        cdouble_t* ptr{};
        View<cdouble_t, size_t> e(ptr, size4_t{2, 3, 4, 5});
        View<double, size_t> f = e.as<double>();
        REQUIRE(all(f.shape() == size4_t{2, 3, 4, 10}));
        REQUIRE(all(f.strides() == size4_t{120, 40, 10, 1}));

        e = f.as<cdouble_t>();
        REQUIRE(all(e.shape() == size4_t{2, 3, 4, 5}));
        REQUIRE(all(e.strides() == size4_t{60, 20, 5, 1}));

        // const int* ptr0{};
        // View<const int> g{ptr0, size4_t{2, 3, 4, 5}};
        // g.as<unsigned char>(); // does not compile because const is lost
    }

    AND_THEN("reshape") {
        TestType* ptr{};
        View<TestType, int32_t> a(ptr, {4, 10, 50, 30});
        a = a.reshape({1, 1, 1, a.elements()});
        REQUIRE(all(a.strides() == a.shape().strides()));
        a = a.reshape({4, 10, 50, 30});
        REQUIRE(all(a.strides() == a.shape().strides()));
        a = a.reshape({10, 4, 30, 50});
        REQUIRE(all(a.strides() == a.shape().strides()));
        REQUIRE(all(a.shape() == int4_t{10, 4, 30, 50}));
    }

    AND_THEN("permute") {
        TestType* ptr{};
        View<TestType, size_t> a(ptr, size4_t{4, 10, 50, 30});
        View<TestType, size_t> b = a.permute({0, 1, 2, 3});
        REQUIRE(all(b.shape() == size4_t{4, 10, 50, 30}));
        REQUIRE(all(b.strides() == size4_t{15000, 1500, 30, 1}));

        b = a.permute({1, 0, 3, 2});
        REQUIRE(all(b.shape() == size4_t{10, 4, 30, 50}));
        REQUIRE(all(b.strides() == size4_t{1500, 15000, 1, 30}));
    }
}

TEMPLATE_TEST_CASE("common: View, accessor", "[noa][common]",
                   int32_t, uint64_t, float, double, cfloat_t, cdouble_t) {
    const size4_t shape{10, 10, 5, 5};
    const size_t elements = shape.elements();

    // Expected:
    std::unique_ptr<TestType[]> expected = std::make_unique<TestType[]>(elements);
    for (size_t i = 0; i < elements; ++i)
        expected[i] = static_cast<TestType>(i);

    // Test:
    std::unique_ptr<TestType[]> buffer = std::make_unique<TestType[]>(elements);
    View<TestType, size_t> view(buffer.get(), shape);

    AND_THEN("to 1D") {
        test::memset(buffer.get(), elements, 0);

        auto [accessor, size] = view.template accessor<TestType, 1>();
        for (auto i: irange<size_t>(elements))
            accessor[i] = static_cast<TestType>(i);

        REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), accessor.get(), elements, 0));
    }

    AND_THEN("to 2D") {
        test::memset(buffer.get(), elements, 0);

        auto [accessor, shape_2d] = view.template accessor<TestType, 2>();
        for (size_t i = 0; i < shape_2d[0]; ++i)
            for (size_t j = 0; j < shape_2d[1]; ++j)
                accessor[i][j] = static_cast<TestType>(i * shape_2d[1] + j);

        REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), accessor.get(), elements, 0));
    }

    AND_THEN("to 3D") {
        test::memset(buffer.get(), elements, 0);

        auto [accessor, shape_3d] = view.template accessor<TestType, 3>();
        for (size_t i = 0; i < shape_3d[0]; ++i)
            for (size_t j = 0; j < shape_3d[1]; ++j)
                for (size_t k = 0; k < shape_3d[2]; ++k)
                    accessor[i][j][k] = static_cast<TestType>((i * shape_3d[1] + j) * shape_3d[2] + k);

        REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), accessor.get(), elements, 0));
    }

    AND_THEN("to 4D") {
        test::memset(buffer.get(), elements, 0);

        const auto accessor = view.accessor();
        for (size_t i = 0; i < shape[0]; ++i)
            for (size_t j = 0; j < shape[1]; ++j)
                for (size_t k = 0; k < shape[2]; ++k)
                    for (size_t l = 0; l < shape[3]; ++l)
                        accessor[i][j][k][l] = static_cast<TestType>(indexing::at(i, j, k, l, view.strides()));

        REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), accessor.get(), elements, 0));
    }
}
