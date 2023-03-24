#include <noa/cpu/Stream.hpp>
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cpu::Stream", "[noa][cpu]") {
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
        cpu::Stream stream(cpu::StreamMode::DEFAULT);
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
            cpu::Stream stream(cpu::StreamMode::ASYNC);
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
                cpu::Stream async_stream(cpu::StreamMode::ASYNC);
                for (int j = 0; j < 50; ++j)
                    async_stream.enqueue(task6);
            }
            REQUIRE(count == (i + 1) * 50);
        }
        REQUIRE(count == 250);

        count = 0;
        for (int i = 0; i < 5; ++i) {
            cpu::Stream async_stream(cpu::StreamMode::ASYNC);
            for (int j = 0; j < 100; ++j)
                async_stream.enqueue(task6);
            async_stream.synchronize();
            REQUIRE(count == (i + 1) * 100);
        }
        REQUIRE(count == 500);
    }

    SECTION("nested async stream") {
        {
            cpu::Stream async_stream(cpu::StreamMode::ASYNC);
            async_stream.enqueue([=]() mutable {
                async_stream.enqueue([](){});
            });
            REQUIRE_THROWS_AS(async_stream.synchronize(), Exception);
        }
    }
}
