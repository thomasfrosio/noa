#include <noa/unified/Stream.hpp>
#include <catch2/catch.hpp>


TEST_CASE("unified::Stream", "[noa][unified]") {
    using namespace ::noa::types;

    AND_THEN("parse") {
        const Device cpu;
        Stream a(cpu);
        REQUIRE(a.device() == cpu);
        REQUIRE(a.device().is_cpu());
        noa::cpu::Stream& cpu_a = a.cpu();

        Stream b = a;

        noa::cpu::Stream& cpu_b = b.cpu();
        cpu_b.enqueue([]() { std::this_thread::sleep_for(std::chrono::seconds(2)); });
        REQUIRE(cpu_a.is_busy()); // cpu_a and cpu_b points to the same worker

        a = Stream(cpu);
        a = std::move(b);

        if (Device::is_any_gpu()) {
            const Device device(Device::GPU, 0);
            Stream c(device);
            REQUIRE(c.device().is_gpu());
            REQUIRE(c.device().id() == 0);
            [[maybe_unused]] const noa::gpu::Stream& gpu_c = c.gpu();
            c = a;
            REQUIRE_THROWS_AS(c.gpu(), noa::Exception);
            [[maybe_unused]] const noa::cpu::Stream& cpu_c = c.cpu();
        }
    }

    AND_THEN("current, guard") {
        if (!Device::is_any_gpu())
            return;

        Device::set_current(Device("gpu"));
        Stream& current_stream = Stream::current(Device::current_gpu());
        REQUIRE(current_stream.device().is_gpu());
        REQUIRE(current_stream.device().id() == 0);
        Stream::set_current(current_stream); // does nothing since it is already the current stream

        Stream a(Device{}); // new stream on the current device
        REQUIRE(a.device().is_cpu());
        Stream::set_current(a);

        const Device gpu(Device::GPU, Device::count_gpus() - 1);
        Device::set_current(gpu);

        Stream c(gpu); // new stream on the current device
        REQUIRE(c.device().is_gpu());
        REQUIRE(c.device().id() == gpu.id());
        Stream::set_current(c);
        REQUIRE(Stream::current(gpu).device() == gpu);
        {
            const StreamGuard d(gpu); // new stream on the current device
            REQUIRE(Stream::current(gpu).device() == gpu);
        }
        REQUIRE(Stream::current(gpu).device() == gpu);
    }

    // Thread limit.
    {
        const i64 old_limit = Session::thread_limit();
        Session::set_thread_limit(old_limit + 1);
        Stream f(Device{});
        REQUIRE(f.thread_limit() == old_limit + 1);
        REQUIRE(f.cpu().thread_limit() == old_limit + 1);
        f.set_thread_limit(old_limit + 2);
        REQUIRE(f.thread_limit() == old_limit + 2);
        REQUIRE(f.cpu().thread_limit() == old_limit + 2);
        Session::set_thread_limit(old_limit);
    }
}
