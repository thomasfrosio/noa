#include <noa/Runtime.hpp>
#include <noa/Signal.hpp>

#include <benchmark/benchmark.h>

namespace {
    using namespace noa::types;

    template<typename T, bool GPU>
    void median_filter_2d(benchmark::State& state) {
        const auto shape = Shape3{1, 512, 512};

        auto device = Device(GPU ? "gpu" : "cpu");
        auto stream = StreamGuard(device, Stream::DEFAULT);
        stream.set_thread_limit(1);

        const auto src = noa::random<T>(noa::Uniform(T{-5}, T{5}), shape, {
            .device = device,
            .allocator = Allocator::ASYNC,
        });
        const auto dst = noa::empty_like(src);

        for (int i = 0; i < 2; ++i) {
            noa::signal::median_filter_2d(src.view(), dst.view(), {.window_size = 5, .border = noa::Border::REFLECT});
        }

        stream.synchronize();

        for (auto _: state) {
            noa::Event start, end;
            start.record(stream);

            noa::signal::median_filter_2d(src.view(), dst.view(), {.window_size = 5, .border = noa::Border::REFLECT});

            end.record(stream);
            end.synchronize();

            auto ms = static_cast<f64>(noa::Event::elapsed(start, end).count());
            state.SetIterationTime(ms * 1e-3);

            ::benchmark::DoNotOptimize(dst.get());
        }
    }
}

BENCHMARK_TEMPLATE(median_filter_2d, float, true)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(median_filter_2d, double, true)->Unit(benchmark::kMillisecond)->UseManualTime();

BENCHMARK_TEMPLATE(median_filter_2d, float, false)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(median_filter_2d, double, false)->Unit(benchmark::kMillisecond)->UseManualTime();
