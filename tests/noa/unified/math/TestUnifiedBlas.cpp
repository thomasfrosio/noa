#include <noa/unified/math/Random.hpp>
#include <noa/unified/math/Blas.hpp>
#include <noa/unified/math/Reduce.hpp>
#include <noa/unified/memory/Factory.hpp>
#include <noa/unified/Ewise.hpp>

#include <catch2/catch.hpp>
#include "Helpers.h"

using namespace ::noa;

namespace {
    template<typename T>
    void naive_matmul_(const Array<T>& lhs, const Array<T>& rhs, const Array<T>& out) {
        const auto [mnk, secondmost_strides, are_column_major] = noa::indexing::extract_matmul_layout(
                lhs.strides(), lhs.shape(), rhs.strides(), rhs.shape(), out.strides(), out.shape(),
                false, false);

        const auto accessor_lhs = lhs.template accessor<T, 3>();
        const auto accessor_rhs = rhs.template accessor<T, 3>();
        const auto accessor_out = out.template accessor<T, 3>();

        for (i64 batch = 0; batch < out.shape()[0]; ++batch) {
            for (i64 m = 0; m < mnk[0]; ++m) {
                for (i64 n = 0; n < mnk[1]; ++n) {
                    for (i64 k = 0; k < mnk[2]; ++k) {
                        accessor_out(batch, m, n) +=
                                accessor_lhs(batch, m, k) *
                                accessor_rhs(batch, k, n);
                    }
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("unified::math::dot()", "[noa][unified]", f32, f64, c32, c64) {
    using real_t = noa::traits::value_type_t<TestType>;
    const i64 batches = test::Randomizer<i64>(1, 5).get();

    test::Randomizer<i64> randomizer(4096, 1048576);
    const auto shape = Shape4<i64>{batches, 1, 1, randomizer.get()};
    const auto reduced_shape = Shape4<i64>{batches, 1, 1, 1};

    std::vector<Device> devices{Device{"cpu"}};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device, StreamMode::DEFAULT);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        const auto lhs = noa::math::random<TestType>(noa::math::uniform_t{}, shape, -5, 5, options);
        const auto rhs = noa::math::random<TestType>(noa::math::uniform_t{}, shape, -5, 5, options);
        const auto output = noa::memory::empty<TestType>(reduced_shape, options);

        // Compute output:
        noa::math::dot(lhs, rhs, output);

        // Compute expected:
        const auto expected = noa::memory::empty<TestType>(reduced_shape, options);
        noa::ewise_binary(lhs, rhs, lhs, noa::multiply_t{});
        noa::math::sum(lhs, expected);

        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, output, expected, std::is_same_v<real_t, f64> ? 1e-7 : 5e-3));
    }
}

TEMPLATE_TEST_CASE("unified::math::dot() vs matmul", "[noa][unified]", f32, f64) {
    using real_t = traits::value_type_t<TestType>;

    test::Randomizer<size_t> randomizer(4096, 1048576);
    const size_t n = randomizer.get();
    const auto lhs_shape = Shape4<i64>{1, 1, 1, n};
    const auto rhs_shape = Shape4<i64>{1, 1, n, 1};
    const auto out_shape = Shape4<i64>{1, 1, 1, 1};

    std::vector<Device> devices{Device{"cpu"}};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device, StreamMode::DEFAULT);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        const auto lhs = noa::math::random<TestType>(noa::math::uniform_t{}, lhs_shape, -5, 5, options);
        const auto rhs = noa::math::random<TestType>(noa::math::uniform_t{}, rhs_shape, -5, 5, options);
        const auto out_dot = noa::math::dot(lhs, rhs);

        const auto output_matmul = noa::memory::empty<TestType>(out_shape, options);
        noa::math::matmul(lhs, rhs, output_matmul);

        output_matmul.eval();
        REQUIRE_THAT(out_dot,
                     Catch::WithinRel(static_cast<f64>(output_matmul(0, 0, 0, 0)),
                                      std::is_same_v<real_t, f64> ? 1e-7 : 5e-3));
    }
}

TEMPLATE_TEST_CASE("unified::math::matmul()", "[noa][unified]", f32) { //, f64, c32, c64
    using real_t = traits::value_type_t<TestType>;
    const i64 batches = test::Randomizer<i64>(1, 4).get();

    test::Randomizer<i64> randomizer(8, 256);
    const i64 m = randomizer.get(), n = randomizer.get(), k = randomizer.get();
    const auto lhs_shape = Shape4<i64>{batches, 1, m, k};
    const auto rhs_shape = Shape4<i64>{batches, 1, k, n};
    const auto out_shape = Shape4<i64>{batches, 1, m, n};
    INFO(lhs_shape);
    INFO(rhs_shape);
    INFO(out_shape);

    std::vector<Device> devices{Device{"cpu"}};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device, StreamMode::DEFAULT);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        const auto lhs = noa::math::random<TestType>(math::uniform_t{}, lhs_shape, real_t{-5}, real_t{5}, options);
        const auto rhs = noa::math::random<TestType>(math::uniform_t{}, rhs_shape, real_t{-5}, real_t{5}, options);
        const auto output = noa::memory::empty<TestType>(out_shape, options);

        // Compute output:
        noa::math::matmul(lhs, rhs, output);
        output.eval();

        // Compute expected:
        const auto expected = noa::memory::zeros<TestType>(out_shape);
        naive_matmul_(lhs, rhs, expected);

        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, output, expected, std::is_same_v<real_t, f64> ? 1e-7 : 5e-3));
    }
}
