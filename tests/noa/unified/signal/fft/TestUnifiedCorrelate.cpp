#include <noa/common/geometry/Euler.h>
#include <noa/unified/fft/Remap.h>
#include <noa/unified/fft/Transform.h>
#include <noa/unified/math/Random.h>
#include <noa/unified/memory/Factory.h>
#include <noa/unified/signal/fft/Correlate.h>
#include <noa/unified/io/ImageFile.h>

#include <catch2/catch.hpp>
#include "Helpers.h"

using namespace ::noa;

TEST_CASE("unified::signal::fft::autocorrelate", "[.]") {
    std::vector<Device> devices{Device("cpu")};
//    if (Device::any(Device::GPU))
//        devices.emplace_back("gpu");

    const auto shape = dim4_t{1, 1, 512, 512};

    for (auto& device: devices) {
        StreamGuard stream(device);
        ArrayOption options(device, Allocator::MANAGED);

        Array lhs = math::random<float>(math::uniform_t{}, shape, -50, 50, options);
        Array lhs_fft = fft::r2c(lhs);
        Array rhs_fft = lhs_fft.copy();
        Array xmap = memory::like(lhs);
        signal::fft::xmap<fft::H2F>(lhs_fft, rhs_fft, xmap);
        io::save(xmap, test::NOA_DATA_PATH / "signal" / "fft" / "xmap_autocorr.mrc");
    }
}
