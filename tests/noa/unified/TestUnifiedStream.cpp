#include <noa/unified/Stream.h>

#include "catch2/catch.hpp"

TEST_CASE("unified::Stream", "[noa][unified]") {
    using namespace ::noa;

    AND_THEN("parse") {
        Device cpu;
        Stream a{cpu};
        REQUIRE(a.device() == cpu);
        REQUIRE(a.device().cpu());
        cpu::Stream& cpu_a = a.cpu();

        Stream b = a;

        cpu::Stream& cpu_b = b.cpu();
        cpu_b.enqueue([]() { std::this_thread::sleep_for(std::chrono::seconds(2)); });
        REQUIRE(cpu_a.busy()); // cpu_a and cpu_b points to the same worker

        a = Stream{cpu};
        a = std::move(b);

        if (Device::any(Device::GPU)) {
            Device device{Device::GPU, 0};
            Stream c{device};
            REQUIRE(c.device().gpu());
            REQUIRE(c.device().id() == 0);
            [[maybe_unused]] gpu::Stream& gpu_c = c.gpu();
            c = a;
            REQUIRE_THROWS_AS(c.gpu(), noa::Exception);
            [[maybe_unused]] cpu::Stream& cpu_c = c.cpu();
        }
    }

    AND_THEN("current, guard") {
        if (!Device::any(Device::GPU))
            return;

        Device::current(Device{"gpu"});
        Stream& current_stream = Stream::current(Device::current(Device::GPU));
        REQUIRE(current_stream.device().gpu());
        REQUIRE(current_stream.device().id() == 0);
        Stream::current(current_stream); // does nothing since it is already the current stream

        Stream a{Device{}}; // new stream on the current device
        REQUIRE(a.device().cpu());
        Stream::current(a);

        Device gpu{Device::GPU, static_cast<int>(Device::count(Device::GPU) - 1)};
        Device::current(gpu);

        Stream c{gpu}; // new stream on the current device
        REQUIRE(c.device().gpu());
        REQUIRE(c.device().id() == gpu.id());
        Stream::current(c);
        REQUIRE(Stream::current(gpu).device() == gpu);
        {
            StreamGuard d{gpu}; // new stream on the current device
            REQUIRE(Stream::current(gpu).device() == gpu);
        }
        REQUIRE(Stream::current(gpu).device() == gpu);
    }
}
