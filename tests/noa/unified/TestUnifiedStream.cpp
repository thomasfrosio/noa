#include <noa/unified/Stream.h>

#include "catch2/catch.hpp"

TEST_CASE("Stream", "[noa][unified]") {
    using namespace ::noa;

    AND_THEN("parse") {
        Device dev{"cpu"};
        Stream a{dev};
        REQUIRE(a.device() == dev);
        REQUIRE(a.device().cpu());
        [[maybe_unused]] cpu::Stream& cpu_stream = a.cpu();

        if (Device::any(Device::GPU)) {
            Device device{Device::GPU, 0};
            Stream b{device};
            REQUIRE(b.device().gpu());
            REQUIRE(b.device().id() == 0);
            [[maybe_unused]] gpu::Stream& gpu_stream = b.gpu();
        }
    }

    AND_THEN("current, guard") {
        Device::current(Device("cpu"));
        Stream& current_stream = Stream::current();
        REQUIRE(current_stream.device().cpu()); // cpu is the current device
        Stream::current(current_stream); // do nothing since it is already the current stream

        Stream a; // new stream on the current device
        REQUIRE(a.device().cpu());
        Stream::current(a);
        REQUIRE(&a == &Stream::current());
        Stream b;
        Stream::current(b);
        REQUIRE(&b == &Stream::current());

        if (Device::any(Device::GPU)) {
            Device my_gpu{Device::GPU};
            Device::current(my_gpu);

            Stream c; // new stream on the current device
            REQUIRE(c.device().gpu());
            Stream::current(c);
            REQUIRE(&Stream::current() == &c);
            {
                StreamGuard d; // new stream on the current device
                REQUIRE(&Stream::current() == &d);
            }
            REQUIRE(&Stream::current() == &c);
        }
    }
}
