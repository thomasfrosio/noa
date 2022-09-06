#include <noa/Array.h>
#include <noa/FFT.h>
#include <noa/Memory.h>
#include <noa/Math.h>
#include <noa/unified/signal/fft/Standardize.h>

#include <catch2/catch.hpp>
#include "Helpers.h"

using namespace ::noa;

TEMPLATE_TEST_CASE("unified::signal::fft::standardize()", "[noa][unified]", float, double) {
    using real_t = TestType;

    const fft::Norm norm = GENERATE(fft::NORM_FORWARD, fft::NORM_BACKWARD, fft::NORM_ORTHO, fft::NORM_NONE);
    const uint ndim = GENERATE(2u, 3u);
    const size4_t shape = test::getRandomShape(ndim);
    INFO(shape);

    std::vector<Device> devices;// = {Device("cpu")};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        StreamGuard stream(device);
        ArrayOption options(device, Allocator::MANAGED);
        INFO(device);

        Array image = math::random<real_t>(math::normal_t{}, shape, 2.5, 5.2, options);
        Array image_fft = fft::r2c(image, norm);
        signal::fft::standardize<fft::H2H>(image_fft, image_fft, shape, norm);
        fft::c2r(image_fft, image, norm);

        if (norm == fft::NORM_NONE)
            image *= 1 / static_cast<real_t>(shape.elements());

        const real_t mean = math::mean(image);
        const real_t std = math::std(image);

        REQUIRE_THAT(mean, Catch::WithinAbs(0, 1e-6));
        REQUIRE_THAT(std, Catch::WithinAbs(1, 1e-5));
    }
}