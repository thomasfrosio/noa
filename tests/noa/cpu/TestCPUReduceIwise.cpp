#include <noa/cpu/ReduceIwise.hpp>

#include "Catch.hpp"

namespace {
    struct Tracked {
        std::array<int, 2> count{};
        Tracked() = default;
        Tracked(const Tracked& t) : count(t.count) { count[0] += 1; }
        Tracked(Tracked&& t)  noexcept : count(t.count) { count[1] += 1; }
    };
}

TEST_CASE("cpu::reduce_iwise") {
    using namespace noa::types;
    using noa::cpu::reduce_iwise;

    AND_THEN("simple 1d") {
        const auto shape = Shape1<i64>{100};
        const auto elements = shape.n_elements();

        const auto buffer = std::make_unique<f32[]>(static_cast<size_t>(elements));
        std::fill_n(buffer.get(), elements, 1);
        auto reduced = noa::make_tuple(AccessorValue<f64>(0.));
        auto output0 = noa::make_tuple(AccessorValue<Pair<i32, i32>>({0, 0}));
        auto output1 = noa::make_tuple(AccessorContiguousI64<Pair<i32, i32>, 1>(output0[Tag<0>{}].get()));

        using accessor_t = AccessorContiguousI64<const f32, 1>;
        struct reduce_op_t {
            accessor_t accessor;
            Tracked tracked{};

            void init(i64 i, f64& reduced) const {
                reduced += static_cast<f64>(accessor[i]);
            }
            void join(f64 to_reduce, f64& reduced) const {
                reduced += to_reduce;
            }
            void final(f64 reduced, Pair<i32, i32>& final) const {
                final.first = static_cast<i32>(reduced + tracked.count[0]);
                final.second = static_cast<i32>(reduced + tracked.count[1]);
            }
        } reduce_op{accessor_t(buffer.get())};

        reduce_iwise(shape, reduce_op, reduced, output0);
        REQUIRE(((output0[Tag<0>{}].ref().first == elements + 1) and
                 (output0[Tag<0>{}].ref().second == elements + 0)));

        reduce_iwise(shape, std::move(reduce_op), reduced, output1);
        REQUIRE(((output1[Tag<0>{}](0).first == elements + 0) and
                 (output1[Tag<0>{}](0).second == elements + 1)));
    }
}
