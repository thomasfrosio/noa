#include <noa/cpu/Stream.hpp>
#include <catch2/catch.hpp>

namespace {
    struct Tracked {
        std::array<int, 2> count{};
        Tracked() = default;
        Tracked(const Tracked& t) : count(t.count) { count[0] += 1; }
        Tracked(Tracked&& t) noexcept: count(t.count) { count[1] += 1; }
    };
}

TEST_CASE("cpu::Stream", "[noa][cpu]") {
    using noa::cpu::Stream;
    using noa::cpu::StreamMode;

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
        Stream stream(StreamMode::DEFAULT, 1);
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
            Stream stream(StreamMode::ASYNC, 1);
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

        int count{0};
        auto task6 = [&count]() {
            ++count;
        };
        for (int i = 0; i < 5; ++i) {
            {
                Stream async_stream(StreamMode::ASYNC, 1);
                for (int j = 0; j < 50; ++j)
                    async_stream.enqueue(task6);
            }
            REQUIRE(count == (i + 1) * 50);
        }
        REQUIRE(count == 250);

        count = 0;
        for (int i = 0; i < 5; ++i) {
            Stream async_stream(StreamMode::ASYNC, 1);
            for (int j = 0; j < 100; ++j)
                async_stream.enqueue(task6);
            async_stream.synchronize();
            REQUIRE(count == (i + 1) * 100);
        }
        REQUIRE(count == 500);
    }

    SECTION("nested async stream") {
        {
            Stream async_stream(StreamMode::ASYNC, 1);
            async_stream.enqueue([=]() mutable {
                async_stream.enqueue([](){});
            });
            REQUIRE_THROWS_AS(async_stream.synchronize(), noa::Exception);
        }
    }

    SECTION("forwarding arguments") {
        Tracked t0{};
        Tracked t1{};
        Tracked t2{};
        Stream async_stream(StreamMode::ASYNC, 1);
        async_stream.enqueue([&t1](auto& tracked) {
            t1.count = tracked.count;
        }, t0);
        async_stream.synchronize();
        REQUIRE((t1.count[0] == 1 and t1.count[1] == 1));

        async_stream.enqueue([&t1](auto tracked) {
            t1.count = tracked.count;
        }, t0);
        async_stream.synchronize();
        REQUIRE((t1.count[0] == 2 and t1.count[1] == 1));

        async_stream.enqueue([&t1](auto&& tracked) {
            t1.count = tracked.count;
        }, std::move(t0));
        async_stream.synchronize();
        REQUIRE((t1.count[0] == 0 and t1.count[1] == 2));

        async_stream.enqueue([&t1](auto tracked) {
            t1.count = tracked.count;
        }, std::move(t2));
        async_stream.synchronize();
        REQUIRE((t1.count[0] == 0 and t1.count[1] == 3));
    }
}
