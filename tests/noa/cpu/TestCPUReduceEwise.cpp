#include <noa/cpu/ReduceEwise.hpp>
#include <catch2/catch.hpp>

namespace {
    struct Tracked {
        std::array<int, 2> count{};
        Tracked() = default;
        Tracked(const Tracked& t) : count(t.count) { count[0] += 1; }
        Tracked(Tracked&& t)  noexcept : count(t.count) { count[1] += 1; }
    };
}

TEST_CASE("cpu::reduce_ewise") {
    using namespace noa::types;
    using noa::cpu::reduce_ewise;
    using noa::cpu::ReduceEwiseConfig;

    AND_THEN("simple sum") {
        const auto shape = Shape4<i64>{1, 50, 1, 100};
        const auto elements = shape.n_elements();

        const auto buffer = std::make_unique<f64[]>(static_cast<size_t>(elements));
        std::fill_n(buffer.get(), elements, 1);
        auto input = noa::make_tuple(AccessorI64<f64, 4>(buffer.get(), shape.strides()));
        auto init = noa::make_tuple(AccessorValue<f64>(0.));
        auto output = noa::make_tuple(AccessorValue<f64>(0.));

        auto reduce_op = [](f64 to_reduce, f64& reduced) { reduced += to_reduce; };
        reduce_ewise(shape, reduce_op, input, init, output);
        REQUIRE_THAT(output[Tag<0>{}].ref(), Catch::WithinAbs(static_cast<f64>(elements), 1e-8));
    }

    AND_THEN("sum-max") {
        const auto shape = Shape4<i64>{1, 50, 1, 100};
        const auto elements = shape.n_elements();

        const auto buffer = std::make_unique<f32[]>(static_cast<size_t>(elements));
        std::fill_n(buffer.get(), elements, 1);
        buffer[234] = 12.f;
        auto input = noa::make_tuple(AccessorI64<f32, 4>(buffer.get(), shape.strides()));
        auto reduced = noa::make_tuple(AccessorValue<f64>(0.), AccessorValue<f32>(0.));
        auto output = noa::make_tuple(AccessorValue<f64>(0.), AccessorValue<f32>(0.));

        struct reduce_op_t {
            Tracked tracked{};

            void init(f32 input, f64& sum, f32& max) const {
                sum += static_cast<f64>(input);
                max = std::max(max, input);
            }
            void join(f64 current_sum, f32 current_max, f64& sum, f32& max) const {
                sum += current_sum;
                max = std::max(max, current_max);
            }
            void final(f64& sum, f32& max, Tuple<f64&, f32&>& output) {
                auto& [a, b] = output;
                a = sum + static_cast<f64>(tracked.count[0]);
                b = max + static_cast<f32>(tracked.count[1]);
            }
        } reduce_op;

        reduce_ewise<ReduceEwiseConfig<false, false, true>>(shape, reduce_op, input, reduced, output);
        REQUIRE_THAT(output[Tag<0>{}].ref(), Catch::WithinAbs(static_cast<f64>(elements + 12), 1e-8));
        REQUIRE_THAT(output[Tag<1>{}].ref(), Catch::WithinAbs(12., 1e-8));

        reduce_ewise<ReduceEwiseConfig<false, false, true>>(shape, std::move(reduce_op), input, reduced, output);
        REQUIRE_THAT(output[Tag<0>{}].ref(), Catch::WithinAbs(static_cast<f64>(elements + 11), 1e-8));
        REQUIRE_THAT(output[Tag<1>{}].ref(), Catch::WithinAbs(13., 1e-8));
    }
}
