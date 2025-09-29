#include <noa/core/types/Accessor.hpp>
#include <noa/gpu/cuda/Allocators.hpp>
#include <noa/gpu/cuda/Iwise.cuh>
#include <noa/gpu/cuda/Event.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

namespace {
    using namespace noa::types;
    namespace ni = noa::indexing;

    struct Tracked {
        Vec<i32, 2> count{};
        constexpr Tracked() = default;
        NOA_HD constexpr Tracked(const Tracked& t) : count(t.count) { count[0] += 1; }
        // NOA_HD constexpr Tracked(Tracked&& t) noexcept : count(t.count) { count[1] += 1; }
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
        NOA_HD constexpr void operator()(auto... indices) const {
            span(indices...) = ni::is_inbound(span.shape(), indices...) ?
                ni::offset_at(span.strides_full(), indices...) : -1;
        }
    };

    TEST_CASE("cuda::iwise") {
        using namespace noa::types;
        using noa::cuda::iwise;
        using noa::cuda::Stream;
        using noa::cuda::Device;
        using noa::cuda::AllocatorManaged;

        Stream stream(Device::current());

        AND_THEN("check operator") {
            constexpr auto shape = Shape1<i64>{2};
            auto value = AllocatorManaged::allocate<Vec2<i32>>(2, stream);
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

        AND_THEN("1d") {
            const auto shapes = std::array{
                Shape1<i64>{1000},
                test::random_shape<i64, 1>(1),
            };
            for (const auto& shape: shapes) {
                const auto n_elements = shape.n_elements();
                const auto buffer = AllocatorManaged::allocate<i32>(n_elements, stream);
                iwise(shape, [ptr = buffer.get()] __device__ (i64 i) { ptr[i] = static_cast<i32>(i); }, stream);

                const auto expected = std::make_unique<i32[]>(static_cast<size_t>(n_elements));
                for (i32 i{}; auto& e: Span(expected.get(), n_elements))
                    e = i++;

                stream.synchronize();
                REQUIRE(test::allclose_abs(buffer.get(), expected.get(), n_elements, 1e-6));
            }
        }

        AND_THEN("2d") {
            const auto shapes = std::array{
                Shape2<i64>{150, 100},
                test::random_shape<i64, 2>(1),
                test::random_shape<i64, 2>(2),
            };
            for (const auto& shape: shapes) {
                const auto n_elements = shape.n_elements();
                const auto buffer = AllocatorManaged::allocate<i32>(n_elements, stream);
                const auto accessor = AccessorContiguousI64<i32, 2>(buffer.get(), shape.strides());
                iwise(shape, [=] __device__(i64 i, i64 j) {
                    accessor(i, j) = static_cast<i32>(accessor.offset_at(i, j));
                }, stream);

                const auto expected = std::make_unique<i32[]>(static_cast<size_t>(n_elements));
                for (i32 i{}; auto& e: Span(expected.get(), n_elements))
                    e = i++;

                stream.synchronize();
                REQUIRE(test::allclose_abs(buffer.get(), expected.get(), n_elements, 0));
            }
        }

        AND_THEN("3d") {
            const auto shapes = std::array{
                Shape3<i64>{25, 15, 35},
                test::random_shape<i64, 3>(1),
                test::random_shape<i64, 3>(2),
                test::random_shape<i64, 3>(3),
            };
            for (const auto& shape: shapes) {
                const auto n_elements = shape.n_elements();

                const auto buffer = AllocatorManaged::allocate<i32>(n_elements, stream);
                const auto accessor = AccessorContiguousI64<i32, 3>(buffer.get(), shape.strides());
                iwise(shape, [=] __device__(const Vec3<i64>& indices) {
                    accessor(indices) = static_cast<i32>(accessor.offset_at(indices));
                }, stream);

                const auto expected = std::make_unique<i32[]>(static_cast<size_t>(n_elements));
                for (i32 i{}; auto& e: Span(expected.get(), n_elements))
                    e = i++;

                stream.synchronize();
                REQUIRE(test::allclose_abs(buffer.get(), expected.get(), n_elements, 0));
            }
        }

        AND_THEN("4d") {
            const auto shapes = std::array{
                Shape4<i64>{3, 5, 25, 60},
                test::random_shape_batched(1),
                test::random_shape_batched(2),
                test::random_shape_batched(3),
                test::random_shape_batched(4),
            };
            for (const auto& shape: shapes) {
                const auto n_elements = shape.n_elements();

                const auto buffer = AllocatorManaged::allocate<i32>(n_elements, stream);
                const auto accessor = AccessorContiguousI64<i32, 4>(buffer.get(), shape.strides());
                iwise(shape, [=] __device__(const Vec4<i64>& indices) {
                    accessor(indices) = static_cast<i32>(accessor.offset_at(indices));
                }, stream);

                const auto expected = std::make_unique<i32[]>(static_cast<size_t>(n_elements));
                for (i32 i{}; auto& e: Span(expected.get(), n_elements))
                    e = i++;

                stream.synchronize();
                REQUIRE(test::allclose_abs(buffer.get(), expected.get(), n_elements, 0));
            }
        }
    }
}


TEST_CASE("cuda::iwise - multi-launch") {
    using namespace noa::types;
    using noa::cuda::iwise;
    using noa::cuda::Stream;
    using noa::cuda::Device;
    using noa::cuda::AllocatorManaged;

    Stream stream(Device::current(), Stream::DEFAULT);

    {
        auto shape = Shape4<i64>{140000, 10, 10, 32};
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
        auto shape = Shape3<i64>{70001, 10, 32};
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
