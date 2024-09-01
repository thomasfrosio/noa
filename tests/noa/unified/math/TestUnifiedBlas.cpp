#include <noa/unified/Random.hpp>
#include <noa/unified/Blas.hpp>
#include <noa/unified/Reduce.hpp>
#include <noa/unified/Factory.hpp>
#include <noa/unified/Ewise.hpp>

#include <catch2/catch.hpp>
#include "Utils.hpp"

using namespace ::noa::types;

namespace {
    template<typename T>
    void naive_matmul_(const Array<T>& lhs, const Array<T>& rhs, const Array<T>& out) {
        const auto [mnk, secondmost_strides, are_column_major] = noa::indexing::extract_matmul_layout(
                lhs.strides(), lhs.shape(), rhs.strides(), rhs.shape(), out.strides(), out.shape(),
                false, false);

        const auto span_lhs = lhs.template span<T, 3>();
        const auto span_rhs = rhs.template span<T, 3>();
        const auto span_out = out.template span<T, 3>();

        for (i64 batch{}; batch < out.shape()[0]; ++batch) {
            for (i64 m{}; m < mnk[0]; ++m) {
                for (i64 n{}; n < mnk[1]; ++n) {
                    for (i64 k{}; k < mnk[2]; ++k) {
                        span_out.at(batch, m, n) +=
                                span_lhs.at(batch, m, k) *
                                span_rhs.at(batch, k, n);
                    }
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("unified::dot()", "[noa][unified]", f32, f64, c32, c64) {
    using real_t = noa::traits::value_type_t<TestType>;
    const i64 batches = test::Randomizer<i64>(1, 5).get();

    test::Randomizer<i64> randomizer(4096, 1048576);
    const auto shape = Shape4<i64>{batches, 1, 1, randomizer.get()};
    const auto reduced_shape = Shape4<i64>{batches, 1, 1, 1};

    std::vector<Device> devices{"cpu"};
    if (Device::is_any(Device::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device, Stream::DEFAULT);
        const auto options = ArrayOption(device, MemoryResource::MANAGED);
        INFO(device);

        const auto lhs = noa::random(noa::Uniform<TestType>{-5, 5}, shape, options);
        const auto rhs = noa::random(noa::Uniform<TestType>{-5, 5}, shape, options);
        const auto output = noa::empty<TestType>(reduced_shape, options);

        // Compute output:
        noa::dot(lhs, rhs, output);

        // Compute expected:
        const auto expected = noa::empty<TestType>(reduced_shape, options);
        noa::ewise(noa::wrap(lhs, rhs), lhs, noa::Multiply{});
        noa::sum(lhs, expected);

        REQUIRE(test::allclose_abs_safe(output, expected, std::is_same_v<real_t, f64> ? 1e-7 : 5e-3));
    }
}

TEMPLATE_TEST_CASE("unified::dot() vs matmul", "[noa][unified]", f32, f64) {
    using real_t = noa::traits::value_type_t<TestType>;

    test::Randomizer<i64> randomizer(4096, 1048576);
    const i64 n = randomizer.get();
    const auto lhs_shape = Shape4<i64>{1, 1, 1, n};
    const auto rhs_shape = Shape4<i64>{1, 1, n, 1};
    const auto out_shape = Shape4<i64>{1, 1, 1, 1};

    std::vector<Device> devices{"cpu"};
    if (Device::is_any(Device::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device, Stream::DEFAULT);
        const auto options = ArrayOption(device, MemoryResource::MANAGED);
        INFO(device);

        const auto lhs = noa::random(noa::Uniform<TestType>{-5, 5}, lhs_shape, options);
        const auto rhs = noa::random(noa::Uniform<TestType>{-5, 5}, rhs_shape, options);
        const auto out_dot = noa::dot(lhs, rhs);

        const auto output_matmul = noa::empty<TestType>(out_shape, options);
        noa::matmul(lhs, rhs, output_matmul);

        output_matmul.eval();
        REQUIRE_THAT(out_dot,
                     Catch::WithinRel(static_cast<f64>(output_matmul.first()),
                                      std::is_same_v<real_t, f64> ? 1e-7 : 5e-3));
    }
}

TEMPLATE_TEST_CASE("unified::matmul()", "[noa][unified]", f32, f64, c32, c64) {
    using real_t = noa::traits::value_type_t<TestType>;
    const i64 batches = test::Randomizer<i64>(1, 4).get();

    test::Randomizer<i64> randomizer(8, 256);
    const i64 m = randomizer.get(), n = randomizer.get(), k = randomizer.get();
    const auto lhs_shape = Shape4<i64>{batches, 1, m, k};
    const auto rhs_shape = Shape4<i64>{batches, 1, k, n};
    const auto out_shape = Shape4<i64>{batches, 1, m, n};
    INFO(lhs_shape);
    INFO(rhs_shape);
    INFO(out_shape);

    std::vector<Device> devices{"cpu"};
    if (Device::is_any(Device::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device, Stream::DEFAULT);
        const auto options = ArrayOption(device, MemoryResource::MANAGED);
        INFO(device);

        const auto lhs = noa::random(noa::Uniform<TestType>{-5, 5}, lhs_shape, options);
        const auto rhs = noa::random(noa::Uniform<TestType>{-5, 5}, rhs_shape, options);
        const auto output = noa::empty<TestType>(out_shape, options);

        // Compute output:
        noa::matmul(lhs, rhs, output);

        // Compute expected:
        const auto expected = noa::zeros<TestType>(out_shape);
        naive_matmul_(lhs, rhs, expected);

        REQUIRE(test::allclose_abs_safe(output, expected, std::is_same_v<real_t, f64> ? 1e-7 : 5e-3));
    }
}
