#include <noa/base/Irange.hpp>
#include <noa/runtime/core/Interfaces.hpp>
#include <noa/runtime/core/Accessor.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

TEST_CASE("runtime::core::IwiseInterface") {
    using namespace noa::types;
    using namespace noa::details;

    constexpr size_t size = 1000;
    const auto b0 = std::make_unique<i32[]>(size);
    const auto b1 = std::make_unique<i32[]>(size);
    const auto b2 = std::make_unique<i32[]>(size);

    AND_THEN("simple 1d") {
        auto op0 = [&](size_t i) { b0[i] = static_cast<i32>(i); };
        for (auto i: noa::irange(size)) {
            IwiseInterface::call(op0, i);
            b1[i] = static_cast<i32>(i);
        }
        REQUIRE(test::allclose_abs(b0.get(), b1.get(), size, 1e-6));
        test::fill(b0.get(), size, 0);
        test::fill(b1.get(), size, 0);
    }

    AND_THEN("simple 3d") {
        auto op1 = [b = b0.get()](Vec<size_t, 3>& indices) {
            const auto offset = indices[0] * 100 + indices[1] * 10 + indices[2];
            b[offset] = static_cast<int>(noa::sum(indices));
        };
        auto op2 = [b = b1.get()](size_t i, size_t j, size_t k) {
            auto o = i + j + k;
            b[i * 100 + j * 10 + k] = static_cast<int>(o);
        };
        for (size_t i{}; i < 10; ++i) {
            for (size_t j{}; j < 10; ++j) {
                for (size_t k{}; k < 10; ++k) {
                    IwiseInterface::call(op1, i, j, k);
                    IwiseInterface::call(op2, i, j, k);
                    b2[i * 100 + j * 10 + k] = static_cast<i32>(i + j + k);
                }
            }
        }
        REQUIRE(test::allclose_abs(b0.get(), b1.get(), size, 1e-6));
        REQUIRE(test::allclose_abs(b0.get(), b2.get(), size, 1e-6));
    }
}

TEST_CASE("runtime::core::EwiseInterface") {
    using namespace noa::types;
    using namespace noa::details;
    using noa::AccessorContiguous;

    constexpr size_t size = 200;
    const auto b0 = std::make_unique<i32[]>(size);
    const auto b1 = std::make_unique<i32[]>(size);
    const auto b2 = std::make_unique<i32[]>(size);

    AND_THEN("simple fill 1d") {
        auto op0 = [](int& b) { b = 1; };
        auto a0 = AccessorContiguous<i32, 1, i64>(b0.get());
        auto input = noa::forward_as_tuple();
        auto output = noa::forward_as_tuple(a0);

        for (auto i: noa::irange(size)) {
            EwiseInterface<false, false>::call(op0, input, output, i);
            b1[i] = 1;
        }
        REQUIRE(test::allclose_abs(b0.get(), b1.get(), size, 1e-6));
    }

    AND_THEN("simple 1d") {
        auto op0 = [](Tuple<const i32&, i32&> a, int& b) {
            const auto& [a0, a1] = a;
            b = a0 + a1;
        };
        const auto strides_2d = Strides<i64, 2>{50, 1};
        auto a0 = AccessorContiguous<const i32, 2, i64>(b0.get(), strides_2d);
        auto a1 = AccessorContiguous<i32, 2, i64>(b0.get(), strides_2d);
        auto a2 = AccessorContiguous<i32, 2, i64>(b1.get(), strides_2d);
        auto input = noa::forward_as_tuple(a0, a1);
        auto output = noa::forward_as_tuple(a2);

        for (size_t i{}; i < 4; ++i) {
            for (size_t j{}; j < 50; ++j) {
                EwiseInterface<true, false>::call(op0, input, output, i, j);
                b2[i * 50 + j] = 2;
            }
        }
        REQUIRE(test::allclose_abs(b0.get(), b1.get(), size, 1e-6));
    }
}

TEST_CASE("runtime::core::ReduceIwiseInterface") {
    using namespace noa::types;
    using noa::AccessorValue;
    using noa::AccessorContiguous;
    using noa::details::ReduceIwiseInterface;

    AND_THEN("simple sum") {
        const auto buffer = std::make_unique<i32[]>(100);
        auto array = AccessorContiguous<i32, 1, i64>(buffer.get());
        std::fill_n(array.get(), 100, 1);

        auto op = [=](u32 i, int& sum) { sum += array[i]; };
        Tuple sum = noa::make_tuple(AccessorValue(0));

        using interface_t = ReduceIwiseInterface<false, false>;
        for (auto i: noa::irange<u32>(100))
            interface_t::init(op, sum, i);

        REQUIRE(noa::get<0>(sum).ref() == 100);
    }

    AND_THEN("serial sum and max") {
        const auto buffer = std::make_unique<i32[]>(100);
        auto array = AccessorContiguous<i32, 1, i64>(buffer.get());
        std::fill_n(array.get(), 100, 1);
        array[50] = 101; // expected max

        auto op0 = [=](Vec<u32, 1> i, i64& sum, i32& max) {
            const auto& value = array[i[0]];
            sum += value;
            max = std::max(value, max);
        };
//        auto op1 = [=](u32 i, i64& sum, i32& max) {
//            const auto& value = array[i];
//            sum += value;
//            max = std::max(value, max);
//        };
        Tuple sum_max = noa::make_tuple(AccessorValue<i64>(0), AccessorValue<i32>(0));

        using interface_t = ReduceIwiseInterface<false, false>;
        for (auto i: noa::irange<u32>(100)) {
            interface_t::init(op0, sum_max, i);
        }
        REQUIRE(noa::get<0>(sum_max).ref() == 200);
        REQUIRE(noa::get<1>(sum_max).ref() == 101);
    }

    AND_THEN("parallel sum") {
        const auto buffer = std::make_unique<i64[]>(100);
        auto array = AccessorContiguous<i64, 1, i64>(buffer.get());
        std::fill_n(array.get(), 100, 1);

        auto init_op = [=](size_t i, i64& sum) { sum += array[i]; };
        auto join_op = [](i64 lhs, i64& rhs) { rhs += lhs; };
        struct final_op_t {
            static void final(i64 lhs, i64& rhs) {
                rhs = lhs + 1;
            }
        } final_op{};

        Tuple sum0 = noa::make_tuple(AccessorValue<i64>(0));
        Tuple sum1 = noa::make_tuple(AccessorValue<i64>(0));
        Tuple sum = noa::make_tuple(AccessorValue<i64>(0));

        using interface_t = ReduceIwiseInterface<false, false>;
        for (auto i: noa::irange<size_t>(50)) { // divide the reduction in two
            interface_t::init(init_op, sum0, i);
            interface_t::init(init_op, sum1, i + 50);
        }
        interface_t::join(join_op, sum0, sum1);
        REQUIRE(noa::get<0>(sum1).ref() == 100);

        interface_t::final(join_op, sum1, sum); // default .final() simply copies
        REQUIRE(noa::get<0>(sum).ref() == 100);

        interface_t::final(final_op, sum1, sum); // check that .final() is detected
        REQUIRE(noa::get<0>(sum).ref() == 101);
    }
}

TEST_CASE("runtime::core::ReduceEwiseInterface") {
    using namespace noa::types;
    using noa::AccessorContiguous;
    using noa::AccessorValue;
    using noa::details::ReduceEwiseInterface;

    AND_THEN("simple sum") {
        const auto b0 = std::make_unique<i32[]>(100);
        const auto b1 = std::make_unique<i32[]>(100);
        std::fill_n(b0.get(), 100, 1);
        std::fill_n(b1.get(), 100, 2);

        auto a0 = AccessorContiguous<const i32, 1, i64>(b0.get());
        auto a1 = AccessorContiguous<i32, 1, i64>(b1.get());

        auto op = [](Tuple<const i32&, i32&> inputs, int& sum) {
            auto[lhs, rhs] = inputs;
            sum += lhs + rhs;
        };
        Tuple input = noa::make_tuple(a0, a1);
        Tuple sum = noa::make_tuple(AccessorValue(0));

        using interface_t = ReduceEwiseInterface<true, false, false>;
        for (auto i: noa::irange<u32>(100))
            interface_t::init(op, input, sum, i);

        REQUIRE(noa::get<0>(sum).ref() == 300);
    }
}
