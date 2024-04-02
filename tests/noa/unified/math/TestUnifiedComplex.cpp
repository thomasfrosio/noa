#include <noa/unified/Complex.hpp>
#include <noa/unified/Random.hpp>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa::types;

TEMPLATE_TEST_CASE("unified::decompose() and complex()", "[noa][unified]", f32, f64) {
    const auto pad = GENERATE(true, false);
    const auto subregion_shape = test::get_random_shape4_batched(3);
    auto shape = subregion_shape;
    if (pad) {
        shape[1] += 8;
        shape[2] += 9;
        shape[3] += 10;
    }

    std::vector<Device> devices{"cpu"};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    using complex_t = Complex<TestType>;
    auto data = Array<complex_t>(shape);
    test::Randomizer<complex_t> randomizer(-10, 10);
    test::randomize(data.get(), data.elements(), randomizer);

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, MemoryResource::MANAGED);
        INFO(device);
        data = device == data.device() ? data : data.to(options);

        using namespace noa::indexing;
        const auto subregion = data.subregion(
                FullExtent{},
                Slice{0, subregion_shape[1]},
                Slice{0, subregion_shape[2]},
                Slice{0, subregion_shape[3]});

        const auto [d_real, d_imag] = noa::decompose(subregion.eval());
        const auto real = noa::real(subregion.eval());
        const auto imag = noa::imag(subregion.eval());
        REQUIRE(test::allclose_abs_safe(d_real, real, 1e-9));
        REQUIRE(test::allclose_abs_safe(d_imag, imag, 1e-9));

        const auto fused = noa::complex(real, imag);
        fused.eval();
        REQUIRE(test::allclose_abs_safe(subregion, fused, 1e-9));
    }
}
