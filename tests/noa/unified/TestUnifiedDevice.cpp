#include <noa/unified/Device.hpp>
#include <noa/gpu/Backend.hpp>

#include "catch2/catch.hpp"

TEST_CASE("unified::Device", "[noa][unified]") {
    using namespace noa::types;

    THEN("parse") {
        Device a; // CPU
        REQUIRE(a.is_cpu());
        REQUIRE(a.id() == -1);

        a = "cpu"; // implicit conversion to string literal.
        REQUIRE(a.is_cpu());

        if (Device::is_any(DeviceType::GPU)) {
            a = Device("gpu");
            REQUIRE(a.id() == 0);
            a = Device("gpu:0");
            REQUIRE(a.id() == 0);
            if (Device::count(DeviceType::GPU) > 1) {
                a = Device("gpu:1");
                REQUIRE(a.id() == 1);
            } else {
                REQUIRE_THROWS(Device("gpu:1"));
            }
        } else {
            REQUIRE_THROWS(Device("gpu"));
        }
    }

    AND_THEN("validate") {
        [[maybe_unused]] const auto a = Device("cpu");
        if (Device::is_any(DeviceType::GPU)) {
            [[maybe_unused]] const auto b = Device("gpu");
            [[maybe_unused]] const auto c = Device("gpu:0");

            if constexpr (noa::gpu::Backend::cuda()) {
                [[maybe_unused]] const auto d = Device("cuda");
                [[maybe_unused]] const auto e = Device("cuda:0");
            } else {
                REQUIRE_THROWS_AS(Device("cuda"), noa::Exception);
                REQUIRE_THROWS_AS(Device("cuda:0"), noa::Exception);
            }
        }
    }

    AND_THEN("current, guard") {
        const Device a(DeviceType::CPU);
        REQUIRE(a.id() == -1); // current device is the cpu by default

        constexpr auto GPU = DeviceType::GPU;
        if (Device::is_any(GPU)) {
            const Device c(GPU);
            Device::set_current(c);
            REQUIRE(c.id() == Device::current(GPU).id());
            {
                const DeviceGuard e(GPU, Device::count(GPU) - 1);
                REQUIRE(e.id() == Device::current(GPU).id());
            }
            REQUIRE(c.id() == Device::current(GPU).id());
        }
    }
}
