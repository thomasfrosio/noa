#include <noa/unified/Array.hpp>
#include <noa/unified/math/Random.hpp>
#include <noa/unified/memory/Copy.hpp>
#include <noa/unified/memory/Factory.hpp>
#include <noa/unified/memory/Permute.hpp>
#include <noa/unified/io/ImageFile.hpp>

#include <catch2/catch.hpp>
#include "Assets.h"
#include "Helpers.h"

using namespace ::noa;

TEST_CASE("unified::memory::permute()", "[asset][noa][unified]") {
    const Path path_base = test::NOA_DATA_PATH / "memory";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["transpose"]["tests"];
    const bool pad = GENERATE(false, true);
    INFO("pad=" << pad);

    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        for (size_t nb = 0; nb < tests.size(); ++nb) {
            INFO("test number = " << nb);

            const YAML::Node& test = tests[nb];
            const auto filename_input = path_base / test["input"].as<Path>();
            const auto filename_expected = path_base / test["expected"].as<Path>();
            const auto permutation = test["permutation"].as<Vec4<i64>>();
            const auto inplace = test["inplace"].as<bool>();

            const auto input = noa::io::load_data<f32>(filename_input, false, options);
            const auto expected = noa::io::load_data<f32>(filename_expected, false, options);

            if (inplace) {
                noa::memory::permute_copy(input, input, permutation);
                REQUIRE(test::Matcher(test::MATCH_ABS, expected, input, 1e-8));
            } else {
                const auto result = input.permute_copy(permutation);
                REQUIRE(test::Matcher(test::MATCH_ABS, expected, result, 1e-8));

                const auto result_reordered = input.permute(permutation);
                REQUIRE(test::Matcher(test::MATCH_ABS, expected, result_reordered, 1e-8));
            }
        }
    }
}

TEMPLATE_TEST_CASE("unified::memory::permute", "[noa][unified]", i32, f32, f64, c32) {
    const std::array permutations{Vec4<i64>{0, 1, 2, 3},
                                  Vec4<i64>{0, 1, 3, 2},
                                  Vec4<i64>{0, 3, 1, 2},
                                  Vec4<i64>{0, 3, 2, 1},
                                  Vec4<i64>{0, 2, 1, 3},
                                  Vec4<i64>{0, 2, 3, 1}};
    const i64 ndim = GENERATE(2, 3);
    const bool pad = GENERATE(false, true);
    INFO("pad=" << pad);

    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        const auto shape = test::get_random_shape4_batched(ndim);
        auto padded_shape = shape;
        if (pad) {
            padded_shape[1] += 10;
            padded_shape[2] += 11;
            padded_shape[3] += 12;
        }
        Array<TestType> data = noa::math::random<TestType>(noa::math::uniform_t{}, padded_shape, -5, 5, options);
        data = data.subregion(
                noa::indexing::Ellipsis{},
                noa::indexing::Slice{0, shape[1]},
                noa::indexing::Slice{0, shape[2]},
                noa::indexing::Slice{0, shape[3]});

        for (const auto& permutation: permutations) {
            if (ndim == 2 && !(all(permutation == Vec4<i64>{0, 1, 2, 3}) || all(permutation == Vec4<i64>{0, 1, 3, 2})))
                return; // while this is technically OK, it doesn't make much sense to test these...

            const auto expected = noa::memory::permute(data, permutation);
            const auto result = noa::memory::permute_copy(data, permutation);
            REQUIRE(test::Matcher(test::MATCH_ABS, expected, result, 1e-8));
        }
    }
}

TEST_CASE("unified::memory::permute, broadcast", "[noa][unified]") {
    const std::array permutations{Vec4<i64>{0, 1, 2, 3},
                                  Vec4<i64>{0, 1, 3, 2},
                                  Vec4<i64>{0, 3, 1, 2},
                                  Vec4<i64>{0, 3, 2, 1},
                                  Vec4<i64>{0, 2, 1, 3},
                                  Vec4<i64>{0, 2, 3, 1}};
    const auto shape = Shape4<i64>{1, 20, 50, 60};

    std::vector<Device> devices = {Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        INFO(device);
        auto stream = StreamGuard(device, StreamMode::DEFAULT);
        const auto options = ArrayOption(device, Allocator::MANAGED);

        for (const auto& permutation: permutations) {
            INFO(permutation);
            const auto permuted_shape = shape.reorder(permutation);
            const Array<f32> result0(permuted_shape, options);
            const Array<f32> result1(permuted_shape, options);

            const Array<f32> data0 = noa::memory::arange<f32>({1, 1, 50, 60}, 0, 1, options);
            const Array<f32> data1({1, 20, 50, 60}, options);
            noa::memory::copy(data0, data1);

            noa::memory::permute_copy(data0, result0, permutation);
            noa::memory::permute_copy(data1, result1, permutation);
            REQUIRE(test::Matcher(test::MATCH_ABS, result0, result1, 1e-8));
        }
    }
}
