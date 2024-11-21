#include <benchmark/benchmark.h>

#include <noa/Array.hpp>
#include <noa/Geometry.hpp>
#include <noa/unified/Event.hpp>

using namespace ::noa::types;
namespace ng = ::noa::geometry;
using Interp = noa::Interp;
using Remap = noa::Remap;

namespace {
    constexpr Shape<i64, 4> shapes[]{
        {1, 1, 512, 512},
        {1, 1, 2048, 2048},
        {1, 1, 4096, 4096},
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
        const auto shape = shapes[0];
        const auto options = ArrayOption{.device = "cpu"};
        StreamGuard stream{options.device, Stream::DEFAULT};
        stream.set_thread_limit(1);

        Array src = noa::random<T>(noa::Uniform<T>{-5, 5}, shape.rfft(), options);
        Array dst = noa::like(src);
        auto rotation = ng::rotate(noa::deg2rad(45.f));

        for (auto _: state) {
            // noa::Event start, end;
            // start.record(stream);

            ng::transform_spectrum_2d<Remap::HC2HC>(
                src, dst, shape, rotation, {}, {
                    .interp = interps[state.range(0)],
                    .fftfreq_cutoff = 0.5,
                });

            // end.record(stream);
            // end.synchronize();
            // state.SetIterationTime(Event::elapsed(start, end).count());

            benchmark::DoNotOptimize(dst.get());
        }
    }
}

BENCHMARK_TEMPLATE(bench001_transform_spectrum_2d, f32)
    ->DenseRange(0, 5)
    ->Unit(benchmark::kMillisecond)->UseRealTime();
// BENCHMARK_TEMPLATE(bench001_transform_spectrum_2d, c32)
//     ->DenseRange(0, 5)
//     ->Unit(benchmark::kMillisecond)->UseRealTime();

/*
------------------------------------------------------------------------------------------
Benchmark                                                Time             CPU   Iterations
------------------------------------------------------------------------------------------
bench001_transform_spectrum_2d<f32>/0/real_time       13.9 ms         13.8 ms           47
bench001_transform_spectrum_2d<f32>/1/real_time       18.6 ms         18.6 ms           37
bench001_transform_spectrum_2d<f32>/2/real_time       51.4 ms         51.4 ms           13
bench001_transform_spectrum_2d<f32>/3/real_time        119 ms          119 ms            6
bench001_transform_spectrum_2d<f32>/4/real_time        166 ms          166 ms            4
bench001_transform_spectrum_2d<f32>/5/real_time        207 ms          207 ms            3

bench001_transform_spectrum_2d<f32>/0/real_time       13.7 ms         13.7 ms           50
bench001_transform_spectrum_2d<f32>/1/real_time       18.3 ms         18.3 ms           38
bench001_transform_spectrum_2d<f32>/2/real_time       83.3 ms         83.3 ms            8
bench001_transform_spectrum_2d<f32>/3/real_time        157 ms          156 ms            4
bench001_transform_spectrum_2d<f32>/4/real_time        214 ms          214 ms            3
bench001_transform_spectrum_2d<f32>/5/real_time        267 ms          267 ms            3
bench001_transform_spectrum_2d<c32>/0/real_time       17.4 ms         17.4 ms           38
bench001_transform_spectrum_2d<c32>/1/real_time       26.3 ms         26.3 ms           26
bench001_transform_spectrum_2d<c32>/2/real_time       95.2 ms         95.2 ms            7
bench001_transform_spectrum_2d<c32>/3/real_time        163 ms          163 ms            4
bench001_transform_spectrum_2d<c32>/4/real_time        223 ms          223 ms            3
bench001_transform_spectrum_2d<c32>/5/real_time        295 ms          295 ms            2
*/
