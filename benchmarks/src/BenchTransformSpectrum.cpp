#include <benchmark/benchmark.h>

#include <noa/Array.hpp>
#include <noa/unified/Random.hpp>
#include <noa/unified/geometry/TransformSpectrum.hpp>

using namespace ::noa::types;
using Interp = noa::Interp;
using Remap = noa::Remap;

namespace {
    constexpr Shape<i64, 4> shapes[]{
        {1, 1, 512, 512},
        {1, 1, 2048, 2048},
        {1, 256, 256, 256},
        {1, 512, 512, 512},
    };

    constexpr Interp interps[] {
        Interp::NEAREST,
        Interp::LINEAR,
        Interp::CUBIC,
        Interp::LANCZOS4,
        Interp::LANCZOS6,
        Interp::LANCZOS8,
    };

    template<typename T>
    void bench001_transform_spectrum_2d(benchmark::State& state) {
        const auto shape = shapes[1];
        StreamGuard stream{Device{}, Stream::DEFAULT};
        stream.set_thread_limit(1);

        Array src = noa::random<T>(noa::Uniform<T>{-5, 5}, shape.rfft());
        Array dst = noa::like(src);
        auto rotation = noa::geometry::rotate(noa::deg2rad(45.f));

        for (auto _: state) {
            noa::geometry::transform_spectrum_2d<Remap::HC2HC>(
                src, dst, shape, rotation, {},
                {.interp = interps[state.range(0)], .fftfreq_cutoff = 0.5});

            ::benchmark::DoNotOptimize(dst.get());
        }
    }
}

BENCHMARK_TEMPLATE(bench001_transform_spectrum_2d, f32)
    // ->DenseRange(0, 1)
    ->DenseRange(0, 5)
    ->Unit(benchmark::kMillisecond)->UseRealTime();

/*
------------------------------------------------------------------------------------------
Benchmark                                                Time             CPU   Iterations
------------------------------------------------------------------------------------------
bench001_transform_spectrum_2d<f32>/0/real_time       17.1 ms         17.1 ms           38
bench001_transform_spectrum_2d<f32>/1/real_time       27.7 ms         27.7 ms           25
bench001_transform_spectrum_2d<f32>/2/real_time       76.9 ms         76.9 ms            9
bench001_transform_spectrum_2d<f32>/3/real_time        124 ms          124 ms            5
bench001_transform_spectrum_2d<f32>/4/real_time        177 ms          177 ms            4
bench001_transform_spectrum_2d<f32>/5/real_time        212 ms          212 ms            3
*/
