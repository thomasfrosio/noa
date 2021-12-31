#include <noa/cpu/Stream.h>
#include <catch2/catch.hpp>

TEST_CASE("cpu::Stream", "[noa][cpu]") {
    int flag = 0;
    auto task1 = []() { return 1; };
    auto task2 = [&flag](int v) {
        flag = v;
        return 2;
    };
    auto task3 = [](int& flag_, int v) { flag_ = v; };
    auto task4 = []() {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        throw std::runtime_error("oops");
    };
    auto task5 = [&flag]() {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        flag = 5;
    };

    SECTION("default stream") {
        using namespace ::noa;
        cpu::Stream stream;
        stream.enqueue(task1);
        stream.enqueue(task2, 3);
        stream.synchronize();
        REQUIRE(flag == 3);
        stream.enqueue(task3, std::ref(flag), 4);
        stream.synchronize();
        REQUIRE(flag == 4);
        REQUIRE_THROWS(stream.enqueue(task4));
        stream.enqueue(task5);
        REQUIRE(stream.hasCompleted());
        REQUIRE(flag == 5);
    }

    SECTION("async stream") {
        using namespace ::noa;
        cpu::Stream stream(cpu::Stream::SERIAL);
        stream.enqueue(task1);
        stream.enqueue(task2, 3);
        stream.synchronize();
        REQUIRE(flag == 3);
        stream.enqueue(task3, std::ref(flag), 4);
        stream.synchronize();
        REQUIRE(flag == 4);
        stream.enqueue(task4);
        REQUIRE_FALSE(stream.hasCompleted());
        REQUIRE_THROWS_AS(stream.synchronize(), std::exception);
        REQUIRE(stream.hasCompleted());
        stream.enqueue(task5);
        REQUIRE_FALSE(stream.hasCompleted());
        stream.synchronize();
        REQUIRE(stream.hasCompleted());
        REQUIRE(flag == 5);
    }
}
