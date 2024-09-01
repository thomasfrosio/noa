#include <noa/unified/Device.hpp>
#include <noa/gpu/Backend.hpp>
#include <catch2/catch.hpp>


TEST_CASE("unified::Device", "[noa][unified]") {
    using namespace noa::types;

    THEN("parse") {
        Device a; // CPU by default
        REQUIRE(a.is_cpu());
        REQUIRE(a.id() == -1);

        a = "cpu"; // implicit conversion from string literal.
        REQUIRE(a.is_cpu());

        if (Device::is_any_gpu()) {
            a = Device("gpu");
            REQUIRE(a.id() == 0);
            a = Device("gpu:0");
            REQUIRE(a.id() == 0);
            if (Device::count_gpus() > 1) {
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
        if (Device::is_any_gpu()) {
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
        [[maybe_unused]] const Device a = Device::current(Device::CPU); // not very useful

        constexpr auto GPU = Device::GPU;
        if (Device::is_any_gpu()) {
            const auto c = Device(GPU);
            Device::set_current(c);
            REQUIRE(c.id() == Device::current_gpu().id());
            {
                const DeviceGuard e(GPU, Device::count_gpus() - 1);
                REQUIRE(e.id() == Device::current_gpu().id());
            }
            REQUIRE(c.id() == Device::current_gpu().id());
        }
    }
}
