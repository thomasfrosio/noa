#include <benchmark/benchmark.h>

#include <noa/Array.hpp>
#include <noa/unified/Event.hpp>
#include <noa/unified/Random.hpp>

using namespace ::noa::types;

namespace {
    constexpr Shape<i64, 4> shapes[]{
        {1, 1, 512, 512},
        {1, 1, 4096, 4096},
        {1, 256, 256, 256},
        {1, 512, 512, 512},
    };

    struct Kernel {
        const float* a;
        const float* b;
        const float* c;
        i32 f_size;
        float* output;

        NOA_DEVICE void operator()(i64 i) const {
            float f{};
            for (i32 w{}; w < f_size; ++w)
                f += __ldg(&c[w]);
            output[i] = __ldg(&a[i]) + __ldg(&b[i]) * f;
        }
    };

    struct Kernel2 {
        const float* a;
        const float* b;
        const float* c;
        i32 f_size;
        float* output;

        NOA_DEVICE void operator()(i64 i) const {
            float f{};
            for (i32 w{}; w < f_size; ++w)
                f += c[w];
            output[i] = a[i] + b[i] * f;
        }
    };

    template<typename T>
    void bench000_gpu_ewise(benchmark::State& state) {
        const auto shape = shapes[state.range(0)];
        auto stream = StreamGuard(Device{"gpu:1"}, Stream::DEFAULT);

        Array<T> a{};// = noa::random<T>(noa::Uniform<T>{-5, 5}, shape, {.device = stream.device()});
        Array<T> b{};// = noa::random<T>(noa::Uniform<T>{-5, 5}, shape, {.device = stream.device()});
        Array<T> c{};// = noa::random<T>(noa::Uniform<T>{-5, 5}, shape, {.device = stream.device()});
        Array dst = noa::like(a);
        stream.synchronize();

        Event start, end;
        for (auto _: state) {
            start.record(stream);
            // noa::ewise(noa::wrap(a, b), dst, noa::Plus{});
            noa::iwise(Shape{shape.n_elements()}, stream.device(), Kernel2{
                .a = a.data(),
                .b = b.data(),
                .c = c.data(),
                .f_size = 120,
                .output = dst.data(),
            });
            end.record(stream);
            end.synchronize();

            state.SetIterationTime(Event::elapsed(start, end).count());
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void bench000_gpu_ewise2(benchmark::State& state) {
        const auto shape = shapes[state.range(0)];
        auto stream = StreamGuard(Device{"gpu:1"}, Stream::DEFAULT);

        Array a = noa::random<T>(noa::Uniform<T>{-5, 5}, shape, {.device = stream.device()});
        Array b = noa::random<T>(noa::Uniform<T>{-5, 5}, shape, {.device = stream.device()});
        Array c = noa::random<T>(noa::Uniform<T>{-5, 5}, shape, {.device = stream.device()});
        Array dst = noa::like(a);
        stream.synchronize();

        Event start, end;
        for (auto _: state) {
            start.record(stream);
            noa::iwise(Shape{shape.n_elements()}, stream.device(), Kernel{
                .a = a.data(),
                .b = b.data(),
                .c = c.data(),
                .f_size = 120,
                .output = dst.data(),
            });
            end.record(stream);
            end.synchronize();

            state.SetIterationTime(Event::elapsed(start, end).count());
            ::benchmark::DoNotOptimize(dst.get());
        }
    }
}

BENCHMARK_TEMPLATE(bench000_gpu_ewise, f32)->DenseRange(0, 3)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(bench000_gpu_ewise2, f32)->DenseRange(0, 3)->Unit(benchmark::kMillisecond)->UseRealTime();
