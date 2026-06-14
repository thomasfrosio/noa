#include <benchmark/benchmark.h>

#include <noa/unified/Reduce.hpp>
#include <noa/unified/Random.hpp>
#include <noa/unified/Array.hpp>
#include <noa/unified/Event.hpp>

namespace {
    using namespace noa::types;

    template<typename T>
    void bench000_reduce_f32(benchmark::State& state) {
        const auto shape = Shape<i64, 4>{3, 1, 8000, 8000};
        StreamGuard stream{Device{"gpu"}, Stream::DEFAULT};
        // stream.set_thread_limit(1);

        Array src = noa::random<f32>(noa::Uniform<f32>{-5, 5}, shape, {.device = stream.device()});
        f32 dst{};
        stream.synchronize();

        noa::Event start, end;
        for (auto _: state) {
            start.record(stream);
            noa::reduce_ewise(src.view(), T{}, dst, noa::ReduceSum{});
            end.record(stream);
            end.synchronize();

            state.SetIterationTime(Event::elapsed(start, end).count());
            ::benchmark::DoNotOptimize(&dst);
        }
        // fmt::println("sum={}", dst);
    }

    void bench000_reduce_kahan(benchmark::State& state) {
        const auto shape = Shape<i64, 4>{3, 1, 8000, 8000};
        StreamGuard stream{Device{"gpu"}, Stream::DEFAULT};
        // stream.set_thread_limit(1);

        Array src = noa::random<f32>(noa::Uniform<f32>{-5, 5}, shape, {.device = stream.device()});
        f32 dst{};
        stream.synchronize();

        noa::Event start, end;
        for (auto _: state) {
            start.record(stream);
            noa::reduce_ewise(src.view(), Vec<f64, 2>{}, dst, noa::ReduceSumKahan{});
            end.record(stream);
            end.synchronize();

            state.SetIterationTime(Event::elapsed(start, end).count());
            ::benchmark::DoNotOptimize(&dst);
        }
        // fmt::println("sum={}", dst);
    }

    void bench000_variance(benchmark::State& state) {
        const auto shape = Shape<i64, 4>{3, 1, 8000, 8000};
        StreamGuard stream{Device{"cpu"}, Stream::DEFAULT};
        stream.set_thread_limit(1);

        Array src = noa::random<f32>(noa::Uniform<f32>{-5, 5}, shape, {.device = stream.device()});
        Pair<f32, f32> dst{};
        stream.synchronize();

        noa::Event start, end;
        for (auto _: state) {
            start.record(stream);
            dst = noa::mean_variance(src.view());
            end.record(stream);
            end.synchronize();

            state.SetIterationTime(Event::elapsed(start, end).count());
            ::benchmark::DoNotOptimize(&dst);
        }
        // fmt::println("sum={}", dst);
    }
    void bench000_variance2(benchmark::State& state) {
        const auto shape = Shape<i64, 4>{3, 1, 8000, 8000};
        StreamGuard stream{Device{"cpu"}, Stream::DEFAULT};
        stream.set_thread_limit(1);

        Array src = noa::random<f32>(noa::Uniform<f32>{-5, 5}, shape, {.device = stream.device()});
        Pair<f32, f32> dst{};
        stream.synchronize();

        noa::Event start, end;
        for (auto _: state) {
            start.record(stream);
            dst = noa::mean_variance(src.view());
            end.record(stream);
            end.synchronize();

            state.SetIterationTime(Event::elapsed(start, end).count());
            ::benchmark::DoNotOptimize(&dst);
        }
        // fmt::println("sum={}", dst);
    }
    void bench000_variance3(benchmark::State& state) {
        const auto shape = Shape<i64, 4>{3, 1, 8000, 8000};
        StreamGuard stream{Device{"cpu"}, Stream::DEFAULT};
        stream.set_thread_limit(1);

        Array src = noa::random<f32>(noa::Uniform<f32>{-5, 5}, shape, {.device = stream.device()});
        Pair<f32, f32> dst{};
        stream.synchronize();

        noa::Event start, end;
        for (auto _: state) {
            start.record(stream);
            dst = noa::mean_variance(src.view(), {.accurate = true});
            end.record(stream);
            end.synchronize();

            state.SetIterationTime(Event::elapsed(start, end).count());
            ::benchmark::DoNotOptimize(&dst);
        }
        // fmt::println("sum={}", dst);
    }
}

// BENCHMARK_TEMPLATE(bench000_reduce_f32, f32)->Unit(benchmark::kMillisecond)->UseRealTime();
// BENCHMARK_TEMPLATE(bench000_reduce_f32, f64)->Unit(benchmark::kMillisecond)->UseRealTime();
// BENCHMARK(bench000_reduce_kahan)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(bench000_variance)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(bench000_variance2)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(bench000_variance3)->Unit(benchmark::kMillisecond)->UseRealTime();


/*
-----------------------------------------------------------------------------
Benchmark                                   Time             CPU   Iterations
-----------------------------------------------------------------------------
bench000_reduce_f32<f32>/real_time       1.12 ms         1.12 ms          654   GPU
bench000_reduce_f32<f64>/real_time       1.34 ms         1.34 ms          662   GPU
bench000_reduce_kahan/real_time          1.34 ms         1.34 ms          662   GPU

GPU:
-----------------------------------------------------------------------
Benchmark                             Time             CPU   Iterations
-----------------------------------------------------------------------
bench000_variance/real_time        2.04 ms         2.04 ms          332
bench000_variance2/real_time       1.35 ms         1.35 ms          662
bench000_variance3/real_time       1.43 ms         1.42 ms          661

CPU serial:
-----------------------------------------------------------------------
Benchmark                             Time             CPU   Iterations
-----------------------------------------------------------------------
bench000_variance/real_time         439 ms          439 ms            2
bench000_variance2/real_time        206 ms          206 ms            3
bench000_variance3/real_time        349 ms          348 ms            2

CPU parallel:
-----------------------------------------------------------------------
Benchmark                             Time             CPU   Iterations
-----------------------------------------------------------------------
bench000_variance/real_time        13.8 ms         13.8 ms           51
bench000_variance2/real_time       6.43 ms         6.40 ms          108
bench000_variance3/real_time       11.1 ms         10.9 ms           65

 */
