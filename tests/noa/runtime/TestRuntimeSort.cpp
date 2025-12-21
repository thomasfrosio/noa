#include <noa/runtime/Array.hpp>
#include <noa/runtime/Sort.hpp>

#include <noa/io/IO.hpp>

#include "Assets.hpp"
#include "Catch.hpp"
#include "Utils.hpp"

using namespace noa::types;

TEST_CASE("runtime::sort", "[asset]") {
    const auto path = test::NOA_DATA_PATH / "math";
    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["sort"];

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (size_t nb{}; nb < tests["tests"].size(); ++nb) {
        INFO(nb);
        const YAML::Node test = tests["tests"][nb];

        const auto input_filename = path / test["input"].as<Path>();
        const auto output_filename = path / test["output"].as<Path>();
        const auto shape = test["shape"].as<Shape4>();
        const auto axis = test["axis"].as<i32>();
        const auto ascending = test["ascending"].as<bool>();

        Array input = noa::read_image<f32>(input_filename).data;
        Array expected = noa::read_image<f32>(output_filename).data;
        REQUIRE((input.shape() == shape and expected.shape() == shape));

        for (auto& device: devices) {
            const auto guard = DeviceGuard(device);
            const auto options = ArrayOption(device, Allocator::MANAGED);
            input = input.device().is_gpu() ? input.to(options) : input;
            expected = expected.device().is_gpu() ? expected.to(options) : expected;

            const auto result = input.to(options);
            noa::sort(result, {ascending, axis});
            REQUIRE(test::allclose_abs(expected, result, 1e-7));
        }
    }
}
