#include <noa/runtime/core/Span.hpp>

#include "Catch.hpp"

TEST_CASE("runtime::core::Span") {
    using namespace noa::types;

    // The library doesn't use std::span and std::mdspan, the main reason being that these were not available
    // when I started this project. Instead, we have noa::Span, which is a simpler version of mdspan but
    // matches the library's API and conventions.

    // noa::Span is intended to be used by user (e.g. to iterate through an array) and is sometimes found in the
    // library's frontend. The library's core and backends rarely use it and prefer the lighter noa::Accessor.

    // The type signature looks like:
    // template<typename T, // value type, can be const
    //          size_t N = 1, // number of dimensions
    //          typename I = i64, // indexing/stride integral type.
    //          StridesTraits = StridesTraits::STRIDED> // whether the rightmost/innermost dimension is contiguous
    // class Span;

    // StridesTraits::STRIDED is the default, meaning that every dimension is attached to a stride defined at runtime.
    // StridesTraits::CONTIGUOUS sets the rightmost stride to 1 at compile time, resulting in strides being of type
    // Strides<I, N - 1>. For 1d spans (N=1), this means that the strides are Strides<I,0>, i.e. they are empty,
    // resulting in the Span only storing a pointer and a size (like std::span).
    {
        using t0 = Span<const f64>;
        using t1 = Span<const f64, 3, isize, StridesTraits::CONTIGUOUS>;
        using t2 = Span<const f64, 1, isize, StridesTraits::CONTIGUOUS>;
        static_assert(std::is_same_v<t0, Span<const f64, 1, isize, StridesTraits::STRIDED>>);
        static_assert(std::is_same_v<t1::strides_type, Strides<isize, 3 - 1>>);
        static_assert(std::is_same_v<t2::strides_type, Strides<isize, 0>> and t2{nullptr, 1}.stride<0>() == 1);
    }

    // While StridesTraits::STRIDED is the default, Span comes with deduction guides that make it
    // StridesTraits::CONTIGUOUS by default with CTAD whenever strides are not specified.
    {
        f32* p0{};
        auto s0 = Span{p0, u32{10}};
        (void) s0; // nvcc unused warning
        static_assert(std::is_same_v<decltype(s0), Span<f32, 1, u32, StridesTraits::CONTIGUOUS>>);

        i32 p1[10]{};
        auto s1 = Span{p1};
        static_assert(std::is_same_v<decltype(s1), Span<i32, 1, isize, StridesTraits::CONTIGUOUS>>);
        REQUIRE(s1.size() == 10);

        auto s2 = Span{p0, Shape{10, 10}};
        static_assert(std::is_same_v<decltype(s2), Span<f32, 2, i32, StridesTraits::CONTIGUOUS>>);

        [[maybe_unused]] auto s00 = s2;
        [[maybe_unused]] auto s01(s2);
        [[maybe_unused]] auto s02 = std::move(s2);
        [[maybe_unused]] auto s03(std::move(s2));

        // Contiguous spans can be implicitly converted to non-contiguous spans.
        [[maybe_unused]] Span<f32, 2, i32> s3 = s2;
    }

    // So to wrap a 1d contiguous array, one could do:
    std::vector<f64> buffer{0, 1, 2, 3, 4, 5};
    Span span(buffer.data(), buffer.size());
    static_assert(std::is_same_v<decltype(span), SpanContiguous<f64, 1, size_t>>); // 'I' was deduced to 'size_t'
    // SpanContiguous is an alias of Span with StridesTraits::CONTIGUOUS.

    // 1d spans can be iterated through...
    for (auto& e: span)
        e += 1;
    REQUIRE(span[2] == 3);
    REQUIRE(span(4) == 5);
    // REQUIRE_THROWS_AS(span.at(6), noa::Exception); // fails to compile with new static bounds check!
    REQUIRE(span.size() == buffer.size());
    REQUIRE((span.stride<0>() == 1 and span.strides_full()[0] == 1));

    {
        // Similar to pointers, mutable to const are implicitly convertible, same as void conversion.
        auto s0 = Span<f64>(buffer.data(), static_cast<isize>(buffer.size()));
        static_assert(std::is_same_v<decltype(s0.as_const()), Span<const f64, 1, isize>>); // explicit
        [[maybe_unused]] Span<const f64> s1 = s0; // implicit
        [[maybe_unused]] Span<const void> s2 = s0;
        REQUIRE(s2.shape() == s0.shape()); // conversion to void only reinterprets the pointer

        // Conversion to bytes correctly changes the shape/strides to the new type.
        Span<const std::byte> s4 = s0.as_bytes(); // + implicit conversion to const
        REQUIRE(buffer.size() * sizeof(f64) == s4.size());

        // Type-erasing to void is possible but should be used carefully.
        // s1 = s2; // does not compile, same as f64* = void*
        s1 = s2.as<const f64>(); // reinterprets back to the original type.
        REQUIRE(s1.size() == buffer.size());
    }

    // We can have more dimensions too.
    {
        auto shape_4d = Shape{2, 3, 4, 5};
        auto p0 = std::make_unique<i32[]>(static_cast<size_t>(shape_4d.n_elements()));
        auto s0 = Span(p0.get(), shape_4d); // contiguous
        REQUIRE(s0.n_elements() == shape_4d.n_elements());
        REQUIRE(s0.strides() == shape_4d.strides().pop_back());
        REQUIRE(s0.strides_full() == shape_4d.strides());

        // Multidimensional indexing.
        for (i32 i{}; i < s0.shape()[0]; ++i)
            for (i32 j{}; j < s0.shape()[1]; ++j)
                for (i32 k{}; k < s0.shape()[2]; ++k)
                    for (i32 l{}; l < s0.shape()[3]; ++l)
                        s0(i, j, k, l) = ((i * shape_4d[1] + j) * shape_4d[2] + k) * shape_4d[3] + l;

        // Dimensions can be collapsed.
        auto s1 = s0.as_1d();
        static_assert(std::is_same_v<decltype(s1), SpanContiguous<i32, 1, i32>>);
        REQUIRE(s1.size() == s0.size());

        bool is_ok{true};
        for (i32 i{}; auto& e: s0.as_1d()) {
            if (e != i++)
                is_ok = false;
        }
        REQUIRE(is_ok);

        // C-style nd-indexing.
        static_assert(std::is_same_v<decltype(s0[1][2]), Span<i32, 2, i32, StridesTraits::CONTIGUOUS>>);
        s0[1][2](3, 4) = 11;
        REQUIRE(s0(1, 2, 3, 4) == 11);

        auto s2 = s0.span<const i32, 2, i64, StridesTraits::STRIDED>();
        REQUIRE(s2.get() == s0.get());
        REQUIRE(s2.shape() == Shape<i64, 2>{shape_4d.pop_back().n_elements(), shape_4d[3]});
        REQUIRE(s2.strides() == Strides<i64, 2>{shape_4d[3], 1});

        // Reshape, subregion and dimension conversion.
        REQUIRE_NOTHROW(s0.reshape({1, 6, 1, 20}));
        REQUIRE(s0.flat().permute({3, 0, 1, 2}).shape() == Shape{shape_4d.n_elements(), 1, 1, 1});
        REQUIRE(s0.span<i32, 4, u64>().subregion(1, 1).data() == s0.get() + s0.offset_at(1, 1));
    }
}
