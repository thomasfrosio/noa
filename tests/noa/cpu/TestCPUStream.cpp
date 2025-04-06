#include <noa/cpu/Stream.hpp>

#include "Catch.hpp"

namespace {
    struct Tracked {
        std::array<int, 2> count{};
        Tracked() = default;
        Tracked(const Tracked& t) : count(t.count) {
            count[0] += 1;
        }
        Tracked(Tracked&& t) noexcept: count(t.count) {
            count[1] += 1;
        }
        auto operator()() const { return count; }
    };
}

TEST_CASE("cpu::Stream") {
    using noa::cpu::Stream;

    int flag = 0;
    auto task1 = []() { return 1; };
    auto task2 = [&flag](int v) {
        flag = v;
        return 2;
    };
    auto task3 = [](int& flag_, int v) { flag_ = v; };
    auto task4 = []() {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        throw std::runtime_error("oops");
    };
    auto task5 = [&flag]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        flag = 5;
    };

    SECTION("default stream") {
        Stream stream(Stream::SYNC, 1);
        stream.enqueue(task1);
        stream.enqueue(task2, 3);
        stream.synchronize();
        REQUIRE_FALSE(stream.is_busy());
        REQUIRE(flag == 3);
        stream.enqueue(task3, std::ref(flag), 4);
        stream.synchronize();
        REQUIRE(flag == 4);
        REQUIRE_THROWS(stream.enqueue(task4));
        stream.enqueue(task5);
        REQUIRE_FALSE(stream.is_busy());
        REQUIRE(flag == 5);
    }

    SECTION("async stream") {
        {
            Stream stream(Stream::ASYNC, 1);
            stream.enqueue(task1);
            stream.enqueue(task2, 3);
            stream.synchronize();
            REQUIRE_FALSE(stream.is_busy());
            REQUIRE(flag == 3);
            stream.enqueue(task3, std::ref(flag), 4);
            stream.synchronize();
            REQUIRE(flag == 4);
            stream.enqueue(task4);
            REQUIRE(stream.is_busy());
            REQUIRE_THROWS_AS(stream.synchronize(), std::exception);
            REQUIRE_FALSE(stream.is_busy());
            stream.enqueue(task5);
            stream.enqueue(task2, 1);
            REQUIRE(stream.is_busy());
            stream.synchronize();
            REQUIRE(flag == 1);
            REQUIRE_FALSE(stream.is_busy());
            stream.enqueue(task5);
            stream.synchronize();
            REQUIRE(flag == 5);
        }

        int count{};
        auto task6 = [&count]() {
            ++count;
        };
        for (int i{}; i < 5; ++i) {
            {
                Stream async_stream(Stream::ASYNC, 1);
                for (int j{}; j < 50; ++j)
                    async_stream.enqueue(task6);
            }
            REQUIRE(count == (i + 1) * 50);
        }
        REQUIRE(count == 250);

        count = 0;
        for (int i{}; i < 5; ++i) {
            Stream async_stream(Stream::ASYNC, 1);
            for (int j{}; j < 100; ++j)
                async_stream.enqueue(task6);
            async_stream.synchronize();
            REQUIRE(count == (i + 1) * 100);
        }
        REQUIRE(count == 500);
    }

    SECTION("forwarding arguments") {
        // libc++ takes the function by value, so triggers an additional move
        // compared to libstdc++ which uses forwarding references...
        Tracked t{};
        auto f = std::function(std::move(t));
        t.count = f();

        // Stream should essentially have the same cost than moving something to std::function,
        // plus an additional copy/move when saved to the no_args lambda (which is required because
        // we transfer it to another thread, so we want to store by value).
        Tracked t0{};
        Tracked t1{};
        Tracked t2{};
        Stream async_stream(Stream::ASYNC, 1);
        async_stream.enqueue([&t1](auto& tracked) {
            t1.count = tracked.count;
        }, t0);
        async_stream.synchronize();
        REQUIRE((t1.count[0] == t.count[0] + 1 and t1.count[1] == t.count[1]));

        async_stream.enqueue([&t1](auto tracked) {
            t1.count = tracked.count;
        }, t0);
        async_stream.synchronize();
        REQUIRE((t1.count[0] == t.count[0] + 2 and t1.count[1] == t.count[1]));

        async_stream.enqueue([&t1](auto&& tracked) {
            t1.count = tracked.count;
        }, std::move(t0));
        async_stream.synchronize();
        REQUIRE((t1.count[0] == t.count[0] and t1.count[1] == t.count[1] + 1));

        async_stream.enqueue([&t1](auto tracked) {
            t1.count = tracked.count;
        }, std::move(t2));
        async_stream.synchronize();
        REQUIRE((t1.count[0] == t.count[0] and t1.count[1] == t.count[1] + 2));
    }
}
