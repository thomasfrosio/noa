#include <noa/runtime/core/Accessor.hpp>
#include <noa/runtime/cuda/Allocators.hpp>
#include <noa/runtime/cuda/Iwise.cuh>
#include <noa/runtime/cuda/Event.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

namespace {
    using namespace noa::types;
    namespace nt = noa::traits;

    struct Tracked {
        Vec<i32, 2> count{};
        constexpr Tracked() = default;
        NOA_HD constexpr Tracked(const Tracked& t) : count(t.count) { count[0] += 1; }
        // NOA_HD constexpr Tracked(Tracked&& t) noexcept : count(t.count) { count[1] += 1; }
    };

    template<usize N>
    struct OffsetOp {
        noa::AccessorContiguous<i32, N> accessor;
        NOA_HD constexpr void operator()(const Vec<isize, N>& indices) const {
            accessor(indices) = static_cast<i32>(accessor.offset_at(indices));
        }
    };

    struct Op {
        Vec<i32, 2>* ptr;
        Tracked tracked;
        NOA_HD constexpr void operator()(i64 i) const {
            ptr[i] = tracked.count;
        }
    };

    template<size_t N>
    struct IwiseCopy {
        SpanContiguous<i64, N> span;
        NOA_HD constexpr void operator()(nt::integer auto... indices) const {
            span(indices...) = noa::is_inbound(span.shape(), indices...) ?
                noa::offset_at(span.strides_full(), indices...) : -1;
        }
    };
}

TEST_CASE("runtime::cuda::iwise") {
    using namespace noa::types;
    using noa::cuda::iwise;
    using noa::cuda::Stream;
    using noa::cuda::Device;
    using noa::cuda::AllocatorManaged;

    Stream stream(Device::current());

    AND_THEN("check operator") {
        constexpr auto shape = Shape1{2};
        auto value = AllocatorManaged::allocate<Vec<i32, 2>>(2, stream);
        auto op0 = Op{value.get(), Tracked{}};

        iwise(shape, op0, stream); // operator is copied once to the kernel
        stream.synchronize();
        REQUIRE((value[0][0] == 1 and value[0][1] == 0));
        REQUIRE((value[1][0] == 1 and value[1][1] == 0));

        iwise(shape, std::move(op0), stream); // operator is moved once into the kernel
        stream.synchronize();
        REQUIRE((value[0][0] == 1 and value[0][1] == 0));
        REQUIRE((value[1][0] == 1 and value[1][1] == 0));
    }

    const auto shapes = noa::make_tuple(
        test::random_shape_batched<isize, 1>(1),
        test::random_shape_batched<isize, 2>(2),
        test::random_shape_batched<isize, 3>(3),
        test::random_shape_batched<isize, 4>(3),
        test::random_shape_batched<isize, 5>(3),
        test::random_shape_batched<isize, 6>(3)
    );
    shapes.for_each([&]<usize N>(const Shape<isize, N>& shape) {
        const auto n_elements = shape.n_elements();
        const auto buffer = AllocatorManaged::allocate<i32>(n_elements, stream);
        const auto accessor = noa::AccessorContiguous<i32, N>(buffer.get(), shape.strides());

        iwise(shape, OffsetOp{accessor}, stream);

        const auto expected = std::make_unique<i32[]>(static_cast<size_t>(n_elements));
        for (i32 i{}; auto& e: Span(expected.get(), n_elements))
            e = i++;

        stream.synchronize();
        REQUIRE(test::allclose_abs(buffer.get(), expected.get(), n_elements, 0));
    });
}

TEST_CASE("runtime::cuda::iwise - multi-launch") {
    using namespace noa::types;
    using noa::cuda::iwise;
    using noa::cuda::Stream;
    using noa::cuda::Device;
    using noa::cuda::AllocatorManaged;

    Stream stream(Device::current(), Stream::DEFAULT);

    {
        auto shape = Shape4{140000, 10, 10, 32};
        auto ptr = AllocatorManaged::allocate<i64>(shape.n_elements(), stream);
        auto span = Span(ptr.get(), shape);

        iwise(shape, IwiseCopy{span}, stream);
        stream.synchronize();

        bool is_ok{true};
        for (i64 i{}; auto& e: span.as_1d()) {
            if (e != i) {
                is_ok = false;
                break;
            }
            ++i;
        }
        REQUIRE(is_ok);
    }
    {
        auto shape = Shape3{70001, 10, 32};
        auto ptr = AllocatorManaged::allocate<i64>(shape.n_elements(), stream);
        auto span = Span(ptr.get(), shape);

        iwise(shape, IwiseCopy{span}, stream);
        stream.synchronize();

        bool is_ok{true};
        for (i64 i{}; auto& e: span.as_1d()) {
            if (e != i) {
                is_ok = false;
                break;
            }
            ++i;
        }
        REQUIRE(is_ok);
    }
}
