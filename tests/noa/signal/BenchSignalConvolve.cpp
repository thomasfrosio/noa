#include <noa/Runtime.hpp>
#include <noa/Signal.hpp>

#include <benchmark/benchmark.h>

namespace {
    using namespace noa::types;

    template<typename T, bool GPU>
    void convolve_2d(benchmark::State& state) {
        const auto shape = Shape3{5, 2048, 2048};

        auto device = Device(GPU ? "gpu" : "cpu");
        auto stream = StreamGuard(device, Stream::DEFAULT);
        stream.set_thread_limit(1);

        const auto src = noa::random<T>(noa::Uniform(T{-5}, T{5}), shape, {
            .device = device,
            .allocator = Allocator::ASYNC,
        });
        const auto dst = noa::empty_like(src);
        const auto kernel = noa::signal::window_gaussian<T>(9, 1., {.normalize = true})
            .to(src.options())
            .reshape(Shape2{3, 3});

        for (int i = 0; i < 2; ++i) {
            noa::signal::convolve_2d(src.view(), dst.view(), kernel.view(), {.border = noa::Border::REFLECT});
        }

        stream.synchronize();

        for (auto _: state) {
            noa::Event start, end;
            start.record(stream);

            noa::signal::convolve_2d(src.view(), dst.view(), kernel.view(), {.border = noa::Border::REFLECT});

            end.record(stream);
            end.synchronize();

            auto ms = static_cast<f64>(noa::Event::elapsed(start, end).count());
            state.SetIterationTime(ms * 1e-3);

            ::benchmark::DoNotOptimize(dst.get());
        }
    }
}

BENCHMARK_TEMPLATE(convolve_2d, float, true)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(convolve_2d, double, true)->Unit(benchmark::kMillisecond)->UseManualTime();

BENCHMARK_TEMPLATE(convolve_2d, float, false)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(convolve_2d, double, false)->Unit(benchmark::kMillisecond)->UseManualTime();
