#include <benchmark/benchmark.h>

#include <noa/gpu/cuda/Event.hpp>
#include <noa/Array.hpp>
#include <noa/unified/Texture.hpp>
#include <noa/unified/Random.hpp>
#include <noa/unified/Event.hpp>
#include <noa/unified/geometry/Transform.hpp>

using namespace ::noa::types;
using Interp = noa::Interp;
using Border = noa::Border;
using Remap = noa::Remap;

namespace {
    constexpr Shape<i64, 4> shapes[]{
        {1, 1, 512, 512},
        {1, 1, 2048, 2048},
    };

    constexpr Interp interps[] {
        Interp::NEAREST_FAST,
        Interp::LINEAR_FAST,
        Interp::CUBIC_FAST,
        Interp::CUBIC_BSPLINE_FAST,
        Interp::LANCZOS4_FAST,
        Interp::LANCZOS6_FAST,
        Interp::LANCZOS8_FAST,
    };

    template<typename T>
    void bench001_transform_2d(benchmark::State& state) {
        const auto shape = shapes[1];

        Array src = noa::random<T>(noa::Uniform<T>{-5, 5}, shape.rfft(), {.device = "gpu:free"});
        Array dst = noa::like(src);
        auto rotation = noa::deg2rad(45.f);
        const auto center = shape.filter(2, 3).vec.as<f32>();
        const auto inverse_rotation_matrix =
            noa::geometry::translate(center) *
            noa::geometry::linear2affine(noa::geometry::rotate(-rotation)) *
            noa::geometry::translate(-center);

        Stream stream = Stream::current(src.device());

        for (auto _: state) {
            Event start, end;
            start.record(stream);

            noa::geometry::transform_2d(
                src.view(), dst.view(), inverse_rotation_matrix,
                {.interp = interps[state.range(0)], .border=Border::ZERO});

            end.record(stream);
            end.synchronize();
            state.SetIterationTime(Event::elapsed(start, end).count());
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void bench001_transform_2d_texture(benchmark::State& state) {
        const auto shape = shapes[1];

        Array src = noa::random<T>(noa::Uniform<T>{-5, 5}, shape.rfft(), {.device = "gpu"});
        Texture<T> tex(src, src.device(), interps[state.range(0)]);
        Array dst = noa::like(src);
        auto rotation = noa::deg2rad(45.f);
        const auto center = shape.filter(2, 3).vec.as<f32>();
        const auto inverse_rotation_matrix =
            noa::geometry::translate(center) *
            noa::geometry::linear2affine(noa::geometry::rotate(-rotation)) *
            noa::geometry::translate(-center);

        Stream stream = Stream::current(src.device());

        for (auto _: state) {
            Event start, end;
            start.record(stream);

            noa::geometry::transform_2d(tex, dst, inverse_rotation_matrix);

            end.record(stream);
            end.synchronize();
            state.SetIterationTime(Event::elapsed(start, end).count());
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void bench001_transform_2d_cpu(benchmark::State& state) {
        const auto shape = shapes[1];

        Array src = noa::random<T>(noa::Uniform<T>{-5, 5}, shape.rfft());
        Array dst = noa::like(src);
        auto rotation = noa::deg2rad(45.f);
        const auto center = shape.filter(2, 3).vec.as<f32>();
        const auto inverse_rotation_matrix =
            noa::geometry::translate(center) *
            noa::geometry::linear2affine(noa::geometry::rotate(-rotation)) *
            noa::geometry::translate(-center);

        auto guard = StreamGuard(Device{}, Stream::SYNC);
        guard.set_thread_limit(2);

        for (auto _: state) {
            noa::geometry::transform_2d(
                src.view(), dst.view(), inverse_rotation_matrix,
                {.interp = interps[state.range(0)], .border = Border::ZERO});

            ::benchmark::DoNotOptimize(dst.get());
        }
    }
}

BENCHMARK_TEMPLATE(bench001_transform_2d, f32)
    ->DenseRange(0, 6)
    ->Unit(benchmark::kMillisecond)->UseRealTime();
// BENCHMARK_TEMPLATE(bench001_transform_2d_texture, f32)
//     ->DenseRange(0, 6)
//     ->Unit(benchmark::kMillisecond)->UseRealTime();

/*
// Without _FAST
-----------------------------------------------------------------------------------------
Benchmark                                               Time             CPU   Iterations
-----------------------------------------------------------------------------------------
bench001_transform_2d<f32>/0/real_time              0.066 ms        0.066 ms        11084
bench001_transform_2d<f32>/1/real_time              0.075 ms        0.075 ms         9726
bench001_transform_2d<f32>/2/real_time              0.147 ms        0.147 ms         4601
bench001_transform_2d<f32>/3/real_time              0.146 ms        0.146 ms         4837
bench001_transform_2d<f32>/4/real_time              0.171 ms        0.171 ms         4118
bench001_transform_2d<f32>/5/real_time              0.258 ms        0.258 ms         2683
bench001_transform_2d<f32>/6/real_time              0.379 ms        0.379 ms         1841
bench001_transform_2d_texture<f32>/0/real_time      0.068 ms        0.068 ms        12262
bench001_transform_2d_texture<f32>/1/real_time      0.072 ms        0.072 ms         9608
bench001_transform_2d_texture<f32>/2/real_time      0.151 ms        0.151 ms         4697
bench001_transform_2d_texture<f32>/3/real_time      0.153 ms        0.153 ms         4478
bench001_transform_2d_texture<f32>/4/real_time      0.157 ms        0.157 ms         4527
bench001_transform_2d_texture<f32>/5/real_time      0.269 ms        0.269 ms         2526
bench001_transform_2d_texture<f32>/6/real_time      0.440 ms        0.440 ms         1578

// With _FAST:
-----------------------------------------------------------------------------------------
Benchmark                                               Time             CPU   Iterations
-----------------------------------------------------------------------------------------
bench001_transform_2d<f32>/0/real_time              0.068 ms        0.068 ms        10652
bench001_transform_2d<f32>/1/real_time              0.075 ms        0.075 ms         9758
bench001_transform_2d<f32>/2/real_time              0.147 ms        0.147 ms         4770
bench001_transform_2d<f32>/3/real_time              0.147 ms        0.147 ms         4692
bench001_transform_2d<f32>/4/real_time              0.169 ms        0.169 ms         4108
bench001_transform_2d<f32>/5/real_time              0.256 ms        0.256 ms         2736
bench001_transform_2d<f32>/6/real_time              0.376 ms        0.376 ms         1847
bench001_transform_2d_texture<f32>/0/real_time      0.066 ms        0.066 ms        10832
bench001_transform_2d_texture<f32>/1/real_time      0.065 ms        0.065 ms        10267   <- a tiny bit faster
bench001_transform_2d_texture<f32>/2/real_time      0.150 ms        0.150 ms         4731
bench001_transform_2d_texture<f32>/3/real_time      0.082 ms        0.082 ms         8500   <- ~1.8x faster
bench001_transform_2d_texture<f32>/4/real_time      0.156 ms        0.156 ms         4507
bench001_transform_2d_texture<f32>/5/real_time      0.273 ms        0.273 ms         2529
bench001_transform_2d_texture<f32>/6/real_time      0.440 ms        0.440 ms         1572

// So basically textures are currently not super useful. It would have been nice to have the lerp technique
// working for cubic and lanczos too (~2.2 faster than the default).

// Rerunning but using View instead of Array, so about ~40 microseconds of overhead.
---------------------------------------------------------------------------------
Benchmark                                       Time             CPU   Iterations
---------------------------------------------------------------------------------
bench001_transform_2d<f32>/0/real_time      0.027 ms        0.027 ms        25108
bench001_transform_2d<f32>/1/real_time      0.036 ms        0.036 ms        19094
bench001_transform_2d<f32>/2/real_time      0.106 ms        0.106 ms         6540
bench001_transform_2d<f32>/3/real_time      0.106 ms        0.106 ms         6681
bench001_transform_2d<f32>/4/real_time      0.128 ms        0.128 ms         5498
bench001_transform_2d<f32>/5/real_time      0.215 ms        0.215 ms         3220
bench001_transform_2d<f32>/6/real_time      0.335 ms        0.335 ms         2056

// This is on the CPU with 1 thread.
-------------------------------------------------------------------------------------
Benchmark                                           Time             CPU   Iterations
-------------------------------------------------------------------------------------
bench001_transform_2d_cpu<f32>/0/real_time       9.74 ms         9.74 ms           70
bench001_transform_2d_cpu<f32>/1/real_time       17.5 ms         17.5 ms           40   <- GPU is ~480x faster
bench001_transform_2d_cpu<f32>/2/real_time       66.0 ms         65.9 ms           10
bench001_transform_2d_cpu<f32>/3/real_time       38.1 ms         38.1 ms           18
bench001_transform_2d_cpu<f32>/4/real_time        115 ms          115 ms            6
bench001_transform_2d_cpu<f32>/5/real_time        149 ms          149 ms            5
bench001_transform_2d_cpu<f32>/6/real_time        188 ms          187 ms            4   <- GPU is ~560x faster

// Same but with 2 threads.
-------------------------------------------------------------------------------------
Benchmark                                           Time             CPU   Iterations
-------------------------------------------------------------------------------------
bench001_transform_2d_cpu<f32>/0/real_time       6.17 ms         6.17 ms          110
bench001_transform_2d_cpu<f32>/1/real_time       12.7 ms         12.7 ms           55   <- GPU is ~360x faster
bench001_transform_2d_cpu<f32>/2/real_time       31.4 ms         31.4 ms           22
bench001_transform_2d_cpu<f32>/3/real_time       24.4 ms         24.4 ms           29
bench001_transform_2d_cpu<f32>/4/real_time       56.0 ms         56.0 ms           12
bench001_transform_2d_cpu<f32>/5/real_time       76.1 ms         76.1 ms            9
bench001_transform_2d_cpu<f32>/6/real_time       92.0 ms         91.9 ms            7   <- GPU is ~270x faster
*/
