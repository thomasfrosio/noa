#include <noa/unified/Array.hpp>
#include <noa/unified/Sort.hpp>
#include <noa/unified/io/ImageFile.hpp>

#include "Helpers.h"
#include "Assets.h"
#include <catch2/catch.hpp>

using namespace noa;

TEST_CASE("unified::sort()", "[assets][noa][unified]") {
    const auto path = test::NOA_DATA_PATH / "math";
    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["sort"];

    std::vector<Device> devices = {Device{}};
    if (Device::any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (size_t nb = 0; nb < tests["tests"].size(); ++nb) {
        INFO(nb);
        const YAML::Node test = tests["tests"][nb];

        const auto input_filename = path / test["input"].as<Path>();
        const auto output_filename = path / test["output"].as<Path>();
        const auto shape = test["shape"].as<Shape4<i64>>();
        const auto axis = test["axis"].as<i32>();
        const auto ascending = test["ascending"].as<bool>();

        Array input = noa::io::load_data<f32>(input_filename);
        Array expected = noa::io::load_data<f32>(output_filename);
        REQUIRE((noa::all(input.shape() == shape) && noa::all(expected.shape() == shape)));

        for (auto& device: devices) {
            const auto options = ArrayOption(device, Allocator::MANAGED);
            input = input.device().is_gpu() ? input.to(options) : input;
            expected = expected.device().is_gpu() ? expected.to(options) : expected;

            const auto result = input.to(options);
            sort(result, ascending, axis);
            REQUIRE(test::Matcher(test::MATCH_ABS, expected, result, 1e-7));
        }
    }
}
