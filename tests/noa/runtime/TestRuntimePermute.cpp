#include <noa/runtime/Array.hpp>
#include <noa/runtime/Random.hpp>
#include <noa/runtime/Factory.hpp>

#include <noa/io/IO.hpp>

#include "Assets.hpp"
#include "Catch.hpp"
#include "Utils.hpp"

using namespace ::noa::types;
namespace fs = std::filesystem;

TEST_CASE("runtime::permute()", "[asset]") {
    const fs::path path_base = test::noa_data_path() / "runtime";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["transpose"]["tests"];
    const bool pad = GENERATE(false, true);
    INFO("pad=" << pad);

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        for (size_t nb{}; nb < tests.size(); ++nb) {
            INFO("test number = " << nb);

            const YAML::Node& test = tests[nb];
            const auto filename_input = path_base / test["input"].as<fs::path>();
            const auto filename_expected = path_base / test["expected"].as<fs::path>();
            const auto permutation = test["permutation"].as<Vec<i32, 4>>();
            const auto inplace = test["inplace"].as<bool>();

            const auto input = noa::read_image<f32, 4>(filename_input, {}, options).data;
            const auto expected = noa::read_image<f32, 4>(filename_expected, {}, options).data;

            if (inplace) {
                noa::permute_copy(input, input, permutation);
                REQUIRE(test::allclose_abs(expected, input, 1e-8));
            } else {
                const auto result = input.permute_copy(permutation);
                REQUIRE(test::allclose_abs(expected, result, 1e-8));

                const auto result_reordered = input.permute(permutation);
                REQUIRE(test::allclose_abs(expected, result_reordered, 1e-8));
            }
        }
    }
}

TEST_CASE("runtime::permute") {
    const auto shapes = noa::make_tuple(
        test::random_shape<isize, 1>(1, {.size_range = {5, 20}}),
        test::random_shape<isize, 2>(2, {.size_range = {5, 20}}),
        test::random_shape<isize, 3>(3, {.size_range = {5, 20}}),
        test::random_shape<isize, 4>(3, {.size_range = {5, 20}, .batch_range = {2, 6}}),
        test::random_shape<isize, 5>(3, {.size_range = {5, 20}, .batch_range = {2, 6}}),
        test::random_shape<isize, 6>(3, {.size_range = {5, 20}, .batch_range = {2, 6}})
    );

    std::vector<Device> devices{"cpu"};
    if (Device::is_any(Device::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        shapes.for_each([&]<usize N>(const Shape<isize, N>& shape) {
            auto permutation = Vec<i32, N>::arange();

            do {
                INFO(permutation);
                for (bool pad: {false, true}) {
                    auto padded_shape = shape;
                    if (pad)
                        padded_shape += 3;

                    auto data = noa::random<i32, N>(noa::Uniform{-500, 500}, padded_shape, options);
                    data = data.subregion(Slices<N>{.start = {}, .end = shape.vec});

                    const auto expected = noa::permute(data, permutation);
                    const auto result = noa::permute_copy(data, permutation);
                    REQUIRE(test::allclose_abs(expected, result, 1e-8));
                }
            } while (std::ranges::next_permutation(permutation).found);

            // Inplace.
            do {
                const auto permuted_axes = permutation.cmp_ne(Vec<i32, N>::arange());
                if (sum(permuted_axes.template as<i32>()) != 2)
                    continue;

                // Swapped axes should have the same size.
                auto valid_shape = shape;
                auto size = isize{};
                for (usize i{}; i < N; ++i)
                    if (permuted_axes[i])
                        size = valid_shape[i];
                for (usize i{}; i < N; ++i)
                    if (permuted_axes[i])
                        valid_shape[i] = size;

                INFO(permutation);
                for (bool pad: {false, true}) {
                    auto padded_shape = valid_shape;
                    if (pad)
                        padded_shape += 3;
                    auto data = noa::random<i32, N>(noa::Uniform{-500, 500}, padded_shape, options);
                    data = data.subregion(Slices<N>{.start = {}, .end = valid_shape.vec});

                    const auto expected = noa::permute(data, permutation).copy();
                    noa::permute_copy(data, data, permutation);
                    REQUIRE(test::allclose_abs(expected, data, 1e-8));
                }
            } while (std::ranges::next_permutation(permutation).found);
        });
    }
}

TEST_CASE("runtime::permute, broadcast") {
    constexpr std::array permutations{
        Vec{0, 1, 2, 3},
        Vec{0, 1, 3, 2},
        Vec{0, 3, 1, 2},
        Vec{0, 3, 2, 1},
        Vec{0, 2, 1, 3},
        Vec{0, 2, 3, 1}
    };
    constexpr auto shape = Shape4{1, 20, 50, 60};

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        INFO(device);
        auto stream = StreamGuard(device, Stream::DEFAULT);
        const auto options = ArrayOption(device, Allocator::MANAGED);

        for (const auto& permutation: permutations) {
            INFO(permutation);
            const auto permuted_shape = shape.permute(permutation);
            const Array<f32> result0(permuted_shape, options);
            const Array<f32> result1(permuted_shape, options);

            const Array<f32> data0 = noa::arange<f32, 4>({1, 1, 50, 60}, noa::Arange{0, 1}, options);
            const Array<f32> data1({1, 20, 50, 60}, options);
            noa::copy(data0, data1);

            noa::permute_copy(data0, result0, permutation);
            noa::permute_copy(data1, result1, permutation);
            REQUIRE(test::allclose_abs(result0, result1, 1e-8));
        }
    }
}
