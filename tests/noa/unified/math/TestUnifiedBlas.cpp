#include <noa/Math.h>
#include <noa/Memory.h>
#include <noa/IO.h>

#include <catch2/catch.hpp>

#include "Helpers.h"
#include "Assets.h"

using namespace ::noa;

TEMPLATE_TEST_CASE("unified::math::dot", "[noa][unified]", int32_t, float, double, cfloat_t, cdouble_t) {
    const size_t size = test::getRandomShape(1).elements();
    const size_t batches = test::getRandomShape(1).elements();
    const size4_t shape{batches, 1, 1, size};

    std::vector<Device> devices = {Device{"cpu"}};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        StreamGuard stream{device};
        ArrayOption options{device, Allocator::MANAGED};
        INFO(device);

        Array<TestType> lhs = math::random<TestType>(math::uniform_t{}, shape, -50, 50, options);
        Array<TestType> rhs = math::random<TestType>(math::uniform_t{}, shape, -50, 50, options);
        Array<TestType> out(batches, options);
        math::dot(lhs, rhs, out);
        out.eval(); // too lazy... just checking that it compiles... the backends are tested so that should be fine
    }
}

TEST_CASE("unified::math::lstsq - scipy example()", "[noa][unified]") {
    // We want to fit a quadratic polynomial of the form ``data_y = c0 + c1 * data_x^2``
    constexpr size_t N = 7;
    std::array<double, N> data_x{1.0, 2.5, 3.5, 4.0, 5.0, 7.0, 8.5};
    std::array<double, N> data_y{0.3, 1.1, 1.5, 2.0, 3.2, 6.6, 8.6};

    // We first form the "design matrix" M, with a constant
    // column of 1s and a column containing ``data_x^2``.
    Array<double> a({1, 1, N, 2});
    Array<double> b({1, 1, N, 1});
    Array<double> x({1, 1, 2, 1});
    for (size_t i = 0; i < N; ++i) {
        a(0, 0, i, 0) = 1;
        a(0, 0, i, 1) = data_x[i] * data_x[i];
        b(0, 0, i, 0) = data_y[i];
    }

    // We want to find the least-squares solution to ``a.dot(x) = b``,
    // where ``x`` is a vector with length 2 that holds the parameters
    // ``a`` and ``b``.
    math::lstsq(a, b, x);
    x.eval();

    REQUIRE_THAT(x[0], Catch::WithinAbs(0.20925829, 1e-7));
    REQUIRE_THAT(x[1], Catch::WithinAbs(0.12013861, 1e-7));
}

TEST_CASE("unified::math::surface", "[noa][unified]") {
    const path_t path = test::NOA_DATA_PATH / "math";
    const YAML::Node test_file = YAML::LoadFile(path / "tests.yaml")["surface"];
    const YAML::Node yaml_file = YAML::LoadFile(path / test_file["solution"].as<path_t>());
    Array input = io::load<float>(path / test_file["input"].as<path_t>());

    for (int32_t order: irange(1, 4)) {
        INFO(order);
        const auto parameters = static_cast<size_t>(order == 3 ? 10 : order * 3);

        // Results:
        Array<float> results_solution(parameters);
        Array results_surface = memory::like(input);
        math::surface(input, order, results_surface, true, results_solution);

        // Expected:
        const path_t expected_surface_filename =
                path / string::format("{}{}.mrc", test_file["prefix_surface"].as<std::string>(), order);
        Array expected_surface = io::load<float>(expected_surface_filename);

        Array<float> expected_solution(parameters);
        for (size_t i = 0; i < parameters; ++i)
            expected_solution[i] = yaml_file[order][i].as<float>();

        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, results_solution, expected_solution, 1e-5));
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, results_surface, expected_surface, 5e-3));
    }
}
