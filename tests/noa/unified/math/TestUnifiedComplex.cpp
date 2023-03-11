#include <noa/unified/math/Complex.hpp>
#include <noa/unified/math/Random.hpp>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("unified::math::decompose() and complex()", "[noa][unified]", f32, f64) {
    const auto pad = GENERATE(true, false);
    const auto subregion_shape = test::get_random_shape4_batched(3);
    auto shape = subregion_shape;
    if (pad) {
        shape[1] += 8;
        shape[2] += 9;
        shape[3] += 10;
    }

    std::vector<Device> devices = {Device{}};
    if (Device::any(DeviceType::GPU))
        devices.emplace_back("gpu");

    using complex_t = Complex<TestType>;
    auto data = Array<complex_t>(shape);
    test::Randomizer<complex_t> randomizer(-10, 10);
    test::randomize(data.get(), data.elements(), randomizer);

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);
        data = device == data.device() ? data : data.to(options);

        const auto subregion = data.subregion(
                noa::indexing::full_extent_t{},
                noa::indexing::slice_t{0, subregion_shape[1]},
                noa::indexing::slice_t{0, subregion_shape[2]},
                noa::indexing::slice_t{0, subregion_shape[3]});

        const auto [d_real, d_imag] = math::decompose(subregion.eval());
        const auto real = math::real(subregion.eval());
        const auto imag = math::imag(subregion.eval());
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, d_real, real, 1e-9));
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, d_imag, imag, 1e-9));

        const auto fused = math::complex(real, imag);
        fused.eval();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, subregion, fused, 1e-9));
    }
}
