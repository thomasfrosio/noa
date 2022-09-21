#include <noa/common/types/View.h>
#include <noa/common/utils/Irange.h>

#include <catch2/catch.hpp>

using namespace ::noa;
using namespace ::noa::indexing;

TEST_CASE("View") {
    View<int> a;
    REQUIRE(a.empty());

    const View<const int> b(a.data(), a.shape());
    REQUIRE(a.empty());

    a = View<int>(nullptr, size4_t{10, 10, 10, 10});
    REQUIRE(a.contiguous());

    // to const
    View<const int> d(a.data(), a.shape(), a.strides());
    View<const int> e(a);
    REQUIRE(all(d.shape() == a.shape()));
    REQUIRE(all(d.strides() == a.strides()));
    REQUIRE(d.data() == a.data());

    View<float> a1;
    View<const float> a2;
    // a1 = a2; // loss const - do not compile
    a2 = a1; // fine
}

TEST_CASE("View - indexing") {
    const size4_t shape{3, 50, 40, 60};
    const size4_t stride = shape.strides();
    const size_t elements = stride[0] * shape[0];
    std::unique_ptr<int[]> buffer = std::make_unique<int[]>(elements);
    for (auto i: irange(elements))
        buffer[i] = static_cast<int>(i);

    const View<int> v0(buffer.get(), shape, stride);
    REQUIRE(v0(1, 45, 23, 10) == buffer[at(1, 45, 23, 10, stride)]);

    AND_THEN("full extent") {
        const View<int> v1 = v0.subregion(ellipsis_t{}, 15);
        REQUIRE(all(v1.shape() == size4_t{3, 50, 40, 1}));
        REQUIRE(v0(0, 0, 0, 15) == v1(0));

        const View<int> v2 = v0.subregion(ellipsis_t{}, 39, full_extent_t{});
        REQUIRE(all(v2.shape() == size4_t{3, 50, 1, 60}));
        REQUIRE(v0(1, 1, 39, 3) == v2(1, 1, 0, 3));

        const View<int> v3 = v0.subregion(ellipsis_t{}, 9, full_extent_t{}, full_extent_t{});
        REQUIRE(all(v3.shape() == size4_t{3, 1, 40, 60}));
        REQUIRE(v0(1, 9, 25, 14) == v3(1, 0, 25, 14));

        const View<int> v4 = v0.subregion(1, full_extent_t{}, full_extent_t{}, full_extent_t{});
        const View<int> v5 = v0.subregion(1);
        REQUIRE(all(v4.shape() == size4_t{1, 50, 40, 60}));
        REQUIRE(all(v4.shape() == v5.shape()));
        REQUIRE(v4(0) == v5(0));
        REQUIRE(v4(0) == v0(1));
    }

    AND_THEN("slice") {
        const View<int> v1 = v0.subregion(ellipsis_t{}, slice_t{});
        REQUIRE(all(v1.shape() == v0.shape()));

        const View<int> v2 = v0.subregion(ellipsis_t{}, slice_t{5, 45});
        const View<int> v3 = v0.subregion(ellipsis_t{}, slice_t{5, -15});
        REQUIRE(all(v2.shape() == size4_t{3, 50, 40, 40}));
        REQUIRE(all(v3.shape() == size4_t{3, 50, 40, 40}));
        REQUIRE(v2(1, 1, 1, 4) == v0(1, 1, 1, 9));
        REQUIRE(v2(2, 32, 14, 7) == v3(2, 32, 14, 7));

        const View<int> v4 = v3.subregion(slice_t{0, -1},
                                          slice_t{1, 100}, // clamp
                                          slice_t{10, 15},
                                          slice_t{-25, -10});
        REQUIRE(all(v4.shape() == size4_t{2, 49, 5, 15}));
        REQUIRE(v4(1, 2, 2, 0) == v3(1, 3, 12, 15));
    }

    AND_THEN("stride") {
        const View<int> v1 = v0.subregion(ellipsis_t{}, slice_t{0, 60, 2});
        REQUIRE(all(v1.shape() == size4_t{3, 50, 40, 30}));
        REQUIRE(v0(1, 6, 3, 10) == v1(1, 6, 3, 5));
        REQUIRE(v0(1, 6, 3, 28) == v1(1, 6, 3, 14));
    }

    AND_THEN("empty or oob") {
        const View<int> v1 = v0.subregion(slice_t{3, 3},
                                          slice_t{30, 20},
                                          slice_t{10, 10},
                                          slice_t{-10, -20});
        REQUIRE(all(v1.shape() == size4_t{}));
        REQUIRE(v1.empty());

        REQUIRE_THROWS(v0.subregion(2, 30, 10, 60 /* oob */));
        REQUIRE_THROWS(v0.subregion(2, 30, -41 /* oob */, 0));
    }
}

TEMPLATE_TEST_CASE("View, shape manipulation", "[noa]", int32_t, uint64_t, float, double, cfloat_t, cdouble_t) {
    std::unique_ptr<int[]> buffer = std::make_unique<int[]>(2000);
    for (auto i: irange<size_t>(2000))
        buffer[i] = static_cast<int>(i);

    AND_THEN("as another type") {
        View<int> c(buffer.get(), size4_t{2, 3, 4, 5});
        View<unsigned char> d = c.as<unsigned char>();
        REQUIRE(all(d.shape() == size4_t{2, 3, 4, 20}));
        REQUIRE(all(d.strides() == size4_t{240, 80, 20, 1}));

        cdouble_t* ptr{};
        View<cdouble_t> e(ptr, size4_t{2, 3, 4, 5});
        View<double> f = e.as<double>();
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
        View<TestType> a(ptr, size4_t{4, 10, 50, 30});
        View<TestType> b = a.permute(uint4_t{0, 1, 2, 3});
        REQUIRE(all(b.shape() == size4_t{4, 10, 50, 30}));
        REQUIRE(all(b.strides() == size4_t{15000, 1500, 30, 1}));

        b = a.permute(uint4_t{1, 0, 3, 2});
        REQUIRE(all(b.shape() == size4_t{10, 4, 30, 50}));
        REQUIRE(all(b.strides() == size4_t{1500, 15000, 1, 30}));
    }
}
