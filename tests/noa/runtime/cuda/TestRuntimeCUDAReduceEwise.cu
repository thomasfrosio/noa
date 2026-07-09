#include <noa/runtime/cuda/ReduceEwise.cuh>
#include <noa/runtime/cuda/Allocators.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

namespace {
    using namespace noa::types;

    struct SumVectorized {
        using enable_vectorization = bool;
        NOA_HD constexpr void operator()(auto input, auto& output) const {
            output += input;
        }
    };
    struct SumNoVectorized {
        NOA_HD constexpr void operator()(auto input, auto& output) const {
            output += input;
        }
    };

    struct SumMaxOp {
        constexpr void operator()(f32 input, f64& sum, f32& max) const {
            sum += static_cast<f64>(input);
            max = std::max(max, input);
        }
        static constexpr void join(f64 current_sum, f32 current_max, f64& sum, f32& max) {
            sum += current_sum;
            max = std::max(max, current_max);
        }
        static constexpr void post(f64& sum, f32& max, Tuple<f64&, i32&>& output) {
            auto& [a, b] = output;
            a = sum + 1;
            b = static_cast<i32>(max + 1);
        }
    };
}

TEST_CASE("runtime::cuda::reduce_ewise") {
    using namespace noa::types;
    using noa::cuda::reduce_ewise;
    using noa::cuda::ReduceEwiseConfig;
    using noa::cuda::Stream;
    using noa::cuda::Device;
    using noa::cuda::AllocatorManaged;

    Stream stream(Device::current());

    SECTION("simple sum, contiguous") {
        const auto shapes = noa::make_tuple(
            test::random_shape<isize, 1>(1),
            test::random_shape<isize, 2>(1),
            test::random_shape<isize, 2>(2),
            test::random_shape<isize, 3>(1),
            test::random_shape<isize, 3>(2),
            test::random_shape<isize, 3>(3),
            test::random_shape<isize, 4>(1),
            test::random_shape<isize, 4>(2),
            test::random_shape<isize, 4>(3),
            test::random_shape<isize, 5>(1),
            test::random_shape<isize, 5>(2),
            test::random_shape<isize, 5>(3),
            test::random_shape<isize, 6>(1),
            test::random_shape<isize, 6>(2),
            test::random_shape<isize, 6>(3),
            test::random_shape_batched<isize, 4>(3),
            test::random_shape_batched<isize, 5>(3),
            test::random_shape_batched<isize, 6>(3)
        );
        shapes.for_each([&]<typename T, usize N>(const Shape<T, N>& shape) {
            for (auto is_strided: std::array<bool, 2>{false, true}) {
                auto buffer_shape = shape;
                if (is_strided) {
                    if constexpr (N > 1)
                        for (usize i{}; i < N - 1; ++i)
                            buffer_shape[i] += 1;
                }
                auto strides = buffer_shape.strides();
                if (is_strided) {
                    strides *= 2;
                    buffer_shape[N - 1] *= 2;
                }

                INFO("shape=" << shape << "is_strided=" << is_strided);

                const auto b0 = AllocatorManaged::allocate<isize>(buffer_shape.n_elements(), stream);
                test::fill(b0.get(), buffer_shape.n_elements(), 1);
                test::arange(Span<isize, N>(b0.get(), shape, strides));

                const auto n_elements = shape.n_elements();
                const auto expected_sum = static_cast<isize>(noa::round(
                    (static_cast<f64>(n_elements) / 2) * static_cast<f64>(n_elements - 1)));

                const auto b1 = AllocatorManaged::allocate<isize>(1, stream);
                auto reduced = noa::make_tuple(noa::AccessorValue<isize>(0.));
                auto output = noa::make_tuple(noa::AccessorRestrictContiguous<isize, 1>(b1.get())); {
                    // no vectorization
                    auto input = noa::make_tuple(noa::Accessor<const isize, N>(b0.get(), strides));
                    reduce_ewise(shape, SumNoVectorized{}, input, reduced, output, stream);
                    stream.synchronize();
                    REQUIRE(b1[0] == expected_sum);
                } {
                    // vectorization
                    auto input = noa::make_tuple(noa::Accessor<const isize, N>(b0.get(), strides));
                    reduce_ewise(shape, SumVectorized{}, input, reduced, output, stream);
                    stream.synchronize();
                    REQUIRE(b1[0] == expected_sum);
                }
            }
        });
    }

    SECTION("sum-max") {
        const auto shapes = noa::make_tuple(
            test::random_shape<isize, 2>(1),
            test::random_shape<isize, 3>(3)
        );
        shapes.for_each([&]<typename T, usize N>(const Shape<T, N>& shape) {
            const auto n_elements = shape.n_elements();

            const auto b0 = AllocatorManaged::allocate<f32>(n_elements, stream);
            const auto b1 = AllocatorManaged::allocate<f64>(1, stream);
            const auto b2 = AllocatorManaged::allocate<i32>(1, stream);
            std::fill_n(b0.get(), n_elements, 1);
            b0[234] += 12.f;
            auto input = noa::make_tuple(noa::Accessor<f32, N>(b0.get(), shape.strides()));
            auto reduced = noa::make_tuple(noa::AccessorValue(0.), noa::AccessorValue(0.f));
            auto output = noa::make_tuple(
                noa::AccessorRestrictContiguous<f64, 1, i32>(b1.get()),
                noa::AccessorRestrictContiguous<i32, 1, i32>(b2.get())
            );

            using config = ReduceEwiseConfig<false, false, true>;
            reduce_ewise<config>(shape, SumMaxOp{}, input, reduced, output, stream);
            stream.synchronize();
            REQUIRE_THAT(b1[0], Catch::Matchers::WithinAbs(static_cast<f64>(n_elements + 12 + 1), 1e-8));
            REQUIRE(b2[0] == 13 + 1);
        });
    }
}
