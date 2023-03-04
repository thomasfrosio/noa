#include <noa/unified/Device.hpp>
#include <noa/gpu/Backend.hpp>

#include "catch2/catch.hpp"

TEST_CASE("unified::Device", "[noa][unified]") {
    THEN("parse") {
        const noa::Device a; // CPU
        REQUIRE(a.is_cpu());
        REQUIRE(a.id() == -1);

        const auto c = noa::Device("gpu", noa::Device::DeviceUnchecked{});
        REQUIRE(c.id() == 0);
        const auto d = noa::Device("gpu:0", noa::Device::DeviceUnchecked{});
        REQUIRE(d.id() == 0);
        const auto e = noa::Device("cuda", noa::Device::DeviceUnchecked{});
        REQUIRE(e.id() == 0);
        const auto f = noa::Device("cuda:1", noa::Device::DeviceUnchecked{});
        REQUIRE(f.id() == 1);
    }

    AND_THEN("validate") {
        [[maybe_unused]] const auto a = noa::Device("cpu");
        if (noa::Device::any(noa::DeviceType::GPU)) {
            [[maybe_unused]] const auto b = noa::Device("gpu");
            [[maybe_unused]] const auto c = noa::Device("gpu:0");

            if constexpr (noa::gpu::Backend::cuda()) {
                [[maybe_unused]] const auto d = noa::Device("cuda");
                [[maybe_unused]] const auto e = noa::Device("cuda:0");
            } else {
                REQUIRE_THROWS_AS(noa::Device("cuda"), noa::Exception);
                REQUIRE_THROWS_AS(noa::Device("cuda:0"), noa::Exception);
            }
        }
    }

    AND_THEN("current, guard") {
        const noa::Device a(noa::DeviceType::CPU);
        REQUIRE(a.id() == -1); // current device is the cpu by default

        constexpr auto GPU = noa::DeviceType::GPU;
        if (noa::Device::any(GPU)) {
            const noa::Device c(GPU);
            noa::Device::set_current(c);
            REQUIRE(c.id() == noa::Device::current(GPU).id());
            {
                const noa::DeviceGuard e(GPU, noa::Device::count(GPU) - 1);
                REQUIRE(e.id() == noa::Device::current(GPU).id());
            }
            REQUIRE(c.id() == noa::Device::current(GPU).id());
        }
    }
}
