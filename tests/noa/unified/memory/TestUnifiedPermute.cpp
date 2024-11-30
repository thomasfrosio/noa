#include <noa/unified/Array.hpp>
#include <noa/unified/Random.hpp>
#include <noa/unified/Factory.hpp>
#include <noa/unified/IO.hpp>

#include <catch2/catch.hpp>
#include "Assets.h"
#include "Utils.hpp"

using namespace ::noa::types;
using Path = std::filesystem::path;

TEST_CASE("unified::permute()", "[asset][noa][unified]") {
    const Path path_base = test::NOA_DATA_PATH / "memory";
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
            const auto filename_input = path_base / test["input"].as<Path>();
            const auto filename_expected = path_base / test["expected"].as<Path>();
            const auto permutation = test["permutation"].as<Vec4<i64>>();
            const auto inplace = test["inplace"].as<bool>();

            const auto input = noa::io::read_data<f32>(filename_input, {}, options);
            const auto expected = noa::io::read_data<f32>(filename_expected, {}, options);

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

TEMPLATE_TEST_CASE("unified::permute", "[noa][unified]", i32, f32, f64, c32) {
    constexpr std::array permutations{
        Vec4<i64>{0, 1, 2, 3},
        Vec4<i64>{0, 1, 3, 2},
        Vec4<i64>{0, 3, 1, 2},
        Vec4<i64>{0, 3, 2, 1},
        Vec4<i64>{0, 2, 1, 3},
        Vec4<i64>{0, 2, 3, 1}
    };
    const i64 ndim = GENERATE(2, 3);
    const bool pad = GENERATE(false, true);
    INFO("pad=" << pad);

    std::vector<Device> devices{"cpu"};
    if (Device::is_any(Device::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        const auto shape = test::random_shape_batched(ndim);
        auto padded_shape = shape;
        if (pad) {
            padded_shape[1] += 10;
            padded_shape[2] += 11;
            padded_shape[3] += 12;
        }
        Array<TestType> data = noa::random<TestType>(noa::Uniform{-5, 5}, padded_shape, options);
        data = data.subregion(
                noa::indexing::Ellipsis{},
                noa::indexing::Slice{0, shape[1]},
                noa::indexing::Slice{0, shape[2]},
                noa::indexing::Slice{0, shape[3]});

        for (const auto& permutation: permutations) {
            if (ndim == 2 and not (all(permutation == Vec4<i64>{0, 1, 2, 3}) or all(permutation == Vec4<i64>{0, 1, 3, 2})))
                return; // while this is technically OK, it doesn't make much sense to test these...

            const auto expected = noa::permute(data, permutation);
            const auto result = noa::permute_copy(data, permutation);
            REQUIRE(test::allclose_abs(expected, result, 1e-8));
        }
    }
}

TEST_CASE("unified::permute, broadcast", "[noa][unified]") {
    constexpr std::array permutations{
        Vec4<i64>{0, 1, 2, 3},
        Vec4<i64>{0, 1, 3, 2},
        Vec4<i64>{0, 3, 1, 2},
        Vec4<i64>{0, 3, 2, 1},
        Vec4<i64>{0, 2, 1, 3},
        Vec4<i64>{0, 2, 3, 1}
    };
    constexpr auto shape = Shape4<i64>{1, 20, 50, 60};

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        INFO(device);
        auto stream = StreamGuard(device, Stream::DEFAULT);
        const auto options = ArrayOption(device, Allocator::MANAGED);

        for (const auto& permutation: permutations) {
            INFO(permutation);
            const auto permuted_shape = shape.reorder(permutation);
            const Array<f32> result0(permuted_shape, options);
            const Array<f32> result1(permuted_shape, options);

            const Array<f32> data0 = noa::arange<f32>({1, 1, 50, 60}, noa::Arange{0, 1}, options);
            const Array<f32> data1({1, 20, 50, 60}, options);
            noa::copy(data0, data1);

            noa::permute_copy(data0, result0, permutation);
            noa::permute_copy(data1, result1, permutation);
            REQUIRE(test::allclose_abs(result0, result1, 1e-8));
        }
    }
}
