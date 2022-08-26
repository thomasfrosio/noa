#include <noa/unified/Device.h>
#include <noa/gpu/Backend.h>

#include "catch2/catch.hpp"

TEST_CASE("unified::Device", "[noa][unified]") {
    using namespace ::noa;

    THEN("parse") {
        Device a; // CPU
        REQUIRE(a.cpu());
        REQUIRE(a.id() == -1);

        constexpr bool UNSAFE = true;
        Device c("gpu", UNSAFE);
        REQUIRE(c.id() == 0);
        Device d("gpu:0", UNSAFE);
        REQUIRE(d.id() == 0);
        Device e("cuda", UNSAFE);
        REQUIRE(e.id() == 0);
        Device f("cuda:1", UNSAFE);
        REQUIRE(f.id() == 1);
    }

    AND_THEN("validate") {
        Device a("cpu");
        if (Device::any(Device::GPU)) {
            Device b("gpu");
            Device c("gpu:0");

            if constexpr (gpu::Backend::cuda()) {
                Device d("cuda");
                Device e("cuda:0");
            } else {
                REQUIRE_THROWS_AS(Device("cuda"), noa::Exception);
                REQUIRE_THROWS_AS(Device("cuda:0"), noa::Exception);
            }
        }
    }

    AND_THEN("current, guard") {
        Device a(Device::CPU);
        REQUIRE(a.id() == -1); // current device is the cpu by default

        if (Device::any(Device::GPU)) {
            Device c(Device::GPU);
            Device::current(c);
            REQUIRE(c.id() == Device::current(Device::GPU).id());
            {
                DeviceGuard e("gpu", Device::count(Device::GPU) - 1);
                REQUIRE(e.id() == Device::current(Device::GPU).id());
            }
            REQUIRE(c.id() == Device::current(Device::GPU).id());
        }
    }
}
