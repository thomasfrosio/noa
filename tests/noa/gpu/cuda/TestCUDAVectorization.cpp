#include <noa/core/types/Accessor.hpp>
#include <noa/cpu/Allocators.hpp>
#include <noa/gpu/cuda/Pointers.hpp>

#include "Catch.hpp"

TEST_CASE("cuda::ewise vectorization") {
    using namespace noa::types;
    using namespace noa::cuda;

    using aligned_1 = Vec<i8, 1>;
    using aligned_2 = Vec<i8, 2>;
    using aligned_4 = Vec<i8, 4>;
    using aligned_8 = Vec<i8, 8>;
    using aligned_16 = Vec<i8, 16>;
    using unaligned_2 = Vec<f32, 2, 4>;

    using accessor_1 = Accessor<aligned_1, 4, i64>;
    using accessor_2 = Accessor<aligned_2, 4, i64>;
    using accessor_4 = Accessor<aligned_4, 4, i64>;
    using accessor_8 = Accessor<aligned_8, 4, i64>;
    using accessor_16 = Accessor<aligned_16, 4, i64>;
    using value_16 = AccessorValue<aligned_8>;
    using accessor_u2 = Accessor<unaligned_2, 4, i64>;

    {
        using t0 = Tuple<accessor_1, value_16>;
        using t1 = Tuple<accessor_1, accessor_2, value_16>;
        using t2 = Tuple<accessor_1, accessor_2, accessor_4, value_16>;
        using t3 = Tuple<accessor_1, accessor_2, accessor_4, accessor_8, value_16>;
        using t4 = Tuple<accessor_1, accessor_2, accessor_4, accessor_8, accessor_16, value_16>;

        static_assert(maximum_allowed_aligned_buffer_size<16, t0>() == 16);
        static_assert(maximum_allowed_aligned_buffer_size<16, t1>() == 8);
        static_assert(maximum_allowed_aligned_buffer_size<16, t2>() == 4);
        static_assert(maximum_allowed_aligned_buffer_size<16, t3>() == 2);
        static_assert(maximum_allowed_aligned_buffer_size<16, t4>() == 1);
        static_assert(std::same_as<to_aligned_buffer_t<t0, 16, 16>,
                                   Tuple<AlignedBuffer<aligned_1, 16, 16>,
                                         AlignedBuffer<aligned_8, 16, 8>>>);
        static_assert(std::same_as<to_aligned_buffer_t<t1, 16, 8>,
                                   Tuple<AlignedBuffer<aligned_1, 8, 8>,
                                         AlignedBuffer<aligned_2, 8, 16>,
                                         AlignedBuffer<aligned_8, 8, 8>>>);
        static_assert(std::same_as<to_aligned_buffer_t<t2, 16, 4>,
                                   Tuple<AlignedBuffer<aligned_1, 4, 4>,
                                         AlignedBuffer<aligned_2, 4, 8>,
                                         AlignedBuffer<aligned_4, 4, 16>,
                                         AlignedBuffer<aligned_8, 4, 8>>>);
        static_assert(std::same_as<to_aligned_buffer_t<t3, 16, 2>,
                                   Tuple<AlignedBuffer<aligned_1, 2, 2>,
                                         AlignedBuffer<aligned_2, 2, 4>,
                                         AlignedBuffer<aligned_4, 2, 8>,
                                         AlignedBuffer<aligned_8, 2, 16>,
                                         AlignedBuffer<aligned_8, 2, 8>>>);
        static_assert(std::same_as<to_aligned_buffer_t<t4, 16, 1>,
                                   Tuple<AlignedBuffer<aligned_1, 1, 1>,
                                         AlignedBuffer<aligned_2, 1, 2>,
                                         AlignedBuffer<aligned_4, 1, 4>,
                                         AlignedBuffer<aligned_8, 1, 8>,
                                         AlignedBuffer<aligned_16, 1, 16>,
                                         AlignedBuffer<aligned_8, 1, 8>>>);

        static_assert(maximum_allowed_aligned_buffer_size<8, t0>() == 8);
        static_assert(maximum_allowed_aligned_buffer_size<8, t1>() == 4);
        static_assert(maximum_allowed_aligned_buffer_size<8, t2>() == 2);
        static_assert(maximum_allowed_aligned_buffer_size<8, t3>() == 1);
        static_assert(maximum_allowed_aligned_buffer_size<8, t4>() == 1);
        static_assert(std::same_as<to_aligned_buffer_t<t0, 8, 8>,
                                   Tuple<AlignedBuffer<aligned_1, 8, 8>,
                                         AlignedBuffer<aligned_8, 8, 8>>>);
        static_assert(std::same_as<to_aligned_buffer_t<t1, 8, 4>,
                                   Tuple<AlignedBuffer<aligned_1, 4, 4>,
                                         AlignedBuffer<aligned_2, 4, 8>,
                                         AlignedBuffer<aligned_8, 4, 8>>>);
        static_assert(std::same_as<to_aligned_buffer_t<t2, 8, 2>,
                                   Tuple<AlignedBuffer<aligned_1, 2, 2>,
                                         AlignedBuffer<aligned_2, 2, 4>,
                                         AlignedBuffer<aligned_4, 2, 8>,
                                         AlignedBuffer<aligned_8, 2, 8>>>);
        static_assert(std::same_as<to_aligned_buffer_t<t3, 8, 1>,
                                   Tuple<AlignedBuffer<aligned_1, 1, 1>,
                                         AlignedBuffer<aligned_2, 1, 2>,
                                         AlignedBuffer<aligned_4, 1, 4>,
                                         AlignedBuffer<aligned_8, 1, 8>,
                                         AlignedBuffer<aligned_8, 1, 8>>>);
        static_assert(std::same_as<to_aligned_buffer_t<t4, 8, 1>,
                                   Tuple<AlignedBuffer<aligned_1, 1, 1>,
                                         AlignedBuffer<aligned_2, 1, 2>,
                                         AlignedBuffer<aligned_4, 1, 4>,
                                         AlignedBuffer<aligned_8, 1, 8>,
                                         AlignedBuffer<aligned_16, 1, 16>,
                                         AlignedBuffer<aligned_8, 1, 8>>>);

        static_assert(maximum_allowed_aligned_buffer_size<4, t0>() == 4);
        static_assert(maximum_allowed_aligned_buffer_size<4, t1>() == 2);
        static_assert(maximum_allowed_aligned_buffer_size<4, t2>() == 1);
        static_assert(maximum_allowed_aligned_buffer_size<4, t3>() == 1);
        static_assert(maximum_allowed_aligned_buffer_size<4, t4>() == 1);

        static_assert(maximum_allowed_aligned_buffer_size<2, t0>() == 2);
        static_assert(maximum_allowed_aligned_buffer_size<2, t1>() == 1);
        static_assert(maximum_allowed_aligned_buffer_size<2, t2>() == 1);
        static_assert(maximum_allowed_aligned_buffer_size<2, t3>() == 1);
        static_assert(maximum_allowed_aligned_buffer_size<2, t4>() == 1);

        static_assert(maximum_allowed_aligned_buffer_size<1, t0>() == 1);
        static_assert(maximum_allowed_aligned_buffer_size<1, t1>() == 1);
        static_assert(maximum_allowed_aligned_buffer_size<1, t2>() == 1);
        static_assert(maximum_allowed_aligned_buffer_size<1, t3>() == 1);
        static_assert(maximum_allowed_aligned_buffer_size<1, t4>() == 1);
    }

    {
        using t0 = Tuple<value_16, accessor_u2, accessor_8>;
        static_assert(maximum_allowed_aligned_buffer_size<16, t0>() == 2);
        static_assert(maximum_allowed_aligned_buffer_size<8, t0>() == 1);

        static_assert(std::same_as<to_aligned_buffer_t<t0, 16, 2>,
                                   Tuple<AlignedBuffer<aligned_8, 2, 8>,
                                         AlignedBuffer<unaligned_2, 2, 16>,
                                         AlignedBuffer<aligned_8, 2, 16>>>);
        static_assert(std::same_as<to_aligned_buffer_t<t0, 8, 1>,
                                   Tuple<AlignedBuffer<aligned_8, 1, 8>,
                                         AlignedBuffer<unaligned_2, 1, 8>,
                                         AlignedBuffer<aligned_8, 1, 8>>>);
        static_assert(std::same_as<to_aligned_buffer_t<t0, 4, 1>,
                                   Tuple<AlignedBuffer<aligned_8, 1, 8>,
                                         AlignedBuffer<unaligned_2, 1, 4>,
                                         AlignedBuffer<aligned_8, 1, 8>>>);
        static_assert(is_vectorized<to_aligned_buffer_t<t0, 16, 1>>());
        static_assert(is_vectorized<to_aligned_buffer_t<t0, 8, 1>>());
        static_assert(not is_vectorized<to_aligned_buffer_t<t0, 4, 1>>());
    }
    {
        using t0 = Tuple<accessor_16, accessor_8, accessor_4>;
        static_assert(maximum_allowed_aligned_buffer_size<16, t0>() == 1);
        static_assert(maximum_allowed_aligned_buffer_size<8, t0>() == 1);
        static_assert(maximum_allowed_aligned_buffer_size<4, t0>() == 1);
        static_assert(noa::traits::same_as<
            to_aligned_buffer_t<t0, 16, 1>,
            to_aligned_buffer_t<t0, 8, 1>,
            to_aligned_buffer_t<t0, 4, 1>,
            Tuple<AlignedBuffer<aligned_16, 1, 16>,
                  AlignedBuffer<aligned_8, 1, 8>,
                  AlignedBuffer<aligned_4, 1, 4>>>);
        static_assert(not is_vectorized<to_aligned_buffer_t<t0, 16, 1>>());
        static_assert(not is_vectorized<to_aligned_buffer_t<t0, 8, 1>>());
        static_assert(not is_vectorized<to_aligned_buffer_t<t0, 4, 1>>());
    }

    auto d1 = noa::cpu::AllocatorHeap::allocate<aligned_1, 16>(2);
    auto d2 = noa::cpu::AllocatorHeap::allocate<aligned_2, 16>(2);
    auto d4 = noa::cpu::AllocatorHeap::allocate<aligned_4, 16>(2);
    auto d8 = noa::cpu::AllocatorHeap::allocate<aligned_8, 16>(2);
    auto d16 = noa::cpu::AllocatorHeap::allocate<aligned_16, 16>(2);
    {
        const auto shape = Shape<i64, 4>{3, 2, 16, 16};
        const auto strides = shape.strides();
        auto a1 = Accessor<aligned_1, 4, i64>(d1.get(), strides);
        auto a2 = Accessor<aligned_2, 4, i64>(d2.get(), strides);
        auto a4 = Accessor<aligned_4, 4, i64>(d4.get(), strides);
        auto a8 = Accessor<aligned_8, 4, i64>(d8.get(), strides);
        auto a16 = Accessor<aligned_16, 4, i64>(d16.get(), strides);
        REQUIRE(min_address_alignment(noa::make_tuple(a1), shape.pop_back()) == 16);
        REQUIRE(min_address_alignment(noa::make_tuple(a2), shape.pop_back()) == 16);
        REQUIRE(min_address_alignment(noa::make_tuple(a4), shape.pop_back()) == 16);
        REQUIRE(min_address_alignment(noa::make_tuple(a8), shape.pop_back()) == 16);
        REQUIRE(min_address_alignment(noa::make_tuple(a16), shape.pop_back()) == 16);
        REQUIRE(min_address_alignment(noa::make_tuple(a1, a2, a4, a8, a16), shape.pop_back()) == 16);
    }
    {
        const auto shape = Shape<i64, 4>{3, 2, 16, 10};
        const auto strides = shape.strides();
        auto a1 = Accessor<aligned_1, 4, i64>(d1.get(), strides);
        auto a2 = Accessor<aligned_2, 4, i64>(d2.get(), strides);
        auto a4 = Accessor<aligned_4, 4, i64>(d4.get(), strides);
        auto a8 = Accessor<aligned_8, 4, i64>(d8.get(), strides);
        auto a16 = Accessor<aligned_16, 4, i64>(d16.get(), strides);

        REQUIRE(min_address_alignment(noa::make_tuple(a1), shape.pop_back()) == 2);
        REQUIRE(min_address_alignment(noa::make_tuple(a2), shape.pop_back()) == 4);
        REQUIRE(min_address_alignment(noa::make_tuple(a4), shape.pop_back()) == 8);
        REQUIRE(min_address_alignment(noa::make_tuple(a8), shape.pop_back()) == 16);
        REQUIRE(min_address_alignment(noa::make_tuple(a16), shape.pop_back()) == 16);
        REQUIRE(min_address_alignment(noa::make_tuple(a1, a2, a4, a8, a16), shape.pop_back()) == 2);
    }
}
