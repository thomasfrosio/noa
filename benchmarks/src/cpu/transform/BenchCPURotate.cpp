#include <benchmark/benchmark.h>

#include <noa/cpu/Stream.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/transform/Rotate.h>

#include "Helpers.h"

using namespace ::noa;

namespace {
    constexpr size3_t g_shapes[] = {
            {4096, 4096, 1},
            {256,  256,  256},
    };

    constexpr InterpMode g_interp[] = {
            INTERP_NEAREST,
            INTERP_LINEAR,
            INTERP_COSINE,
            INTERP_CUBIC,
            INTERP_CUBIC_BSPLINE,
    };

    constexpr BorderMode g_border[] = {
            BORDER_ZERO,
            BORDER_VALUE,
            BORDER_CLAMP,
            BORDER_PERIODIC,
            BORDER_MIRROR,
    };

    template<typename T>
    void CPU_transform_rotate2D(benchmark::State& state) {
        const size3_t shape = g_shapes[0];
        const InterpMode interp = g_interp[state.range(0)];
        const BorderMode border = g_border[state.range(1)];

        const size2_t shape_2d{shape.x, shape.y};
        cpu::memory::PtrHost<T> src(elements(shape));
        cpu::memory::PtrHost<T> dst(src.size());

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::transform::rotate2D(src.get(), shape.x, dst.get(), shape.x, shape_2d, 3.1415f,
                                     float2_t(shape_2d / 2), interp, border, 0.f, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T, bool PREFILTER>
    void CPU_transform_rotate2D_bspline(benchmark::State& state) {
        const size3_t shape = g_shapes[0];
        const InterpMode interp = INTERP_CUBIC_BSPLINE;
        const BorderMode border = BORDER_VALUE;

        const size2_t shape_2d{shape.x, shape.y};
        cpu::memory::PtrHost<T> src(elements(shape));
        cpu::memory::PtrHost<T> dst(src.size());

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::transform::rotate2D<PREFILTER>(src.get(), shape.x, dst.get(), shape.x, shape_2d, 3.1415f,
                                                float2_t(shape_2d / 2), interp, border, 0.f, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T, bool BATCH>
    void CPU_transform_rotate2D_batch(benchmark::State& state) {
        const size2_t shape_2d{8000, 8000};
        const auto batches = static_cast<size_t>(state.range(0));
        const InterpMode interp = INTERP_LINEAR;
        const BorderMode border = BORDER_ZERO;

        const size_t elements = noa::elements(shape_2d);
        cpu::memory::PtrHost<T> src(elements * batches);
        cpu::memory::PtrHost<T> dst(src.size());
        cpu::memory::PtrHost<float> rotations(batches);
        cpu::memory::PtrHost<float2_t> rotation_centers(batches);
        for (auto& e: rotation_centers)
            e = float2_t(shape_2d / 2);

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);
        test::randomize(rotations.get(), rotations.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            if constexpr (BATCH) {
                cpu::transform::rotate2D(src.get(), shape_2d, dst.get(), shape_2d, shape_2d,
                                         rotations.get(), rotation_centers.get(), batches, interp, border, 0.f, stream);
            } else {
                for (size_t i = 0; i < batches; ++i)
                    cpu::transform::rotate2D(src.get() + i * elements, shape_2d.x,
                                             dst.get() + i * elements, shape_2d.x, shape_2d,
                                             rotations[i], rotation_centers[i],
                                             interp, border, 0.f, stream);
            }
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_transform_rotate3D(benchmark::State& state) {
        const size3_t shape = g_shapes[1];
        const InterpMode interp = g_interp[state.range(0)];
        const BorderMode border = g_border[state.range(1)];
        const float33_t rotation = noa::transform::toMatrix(float3_t{3.1415, 0, 0});

        const size2_t shape_2d{shape.x, shape.y};
        cpu::memory::PtrHost<T> src(elements(shape));
        cpu::memory::PtrHost<T> dst(src.size());

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::transform::rotate3D(src.get(), shape_2d, dst.get(), shape_2d, shape, rotation,
                                     float3_t(shape / 2), interp, border, 0.f, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }
}

// The border has almost no effect. The interpolation method is the critical part and is as expected.
// Cubic is about x2 slower than linear. Cubic-B-spline (with prefilter) is about x3.5 slower than cubic.
// Multithreading is scaling relatively well...
BENCHMARK_TEMPLATE(CPU_transform_rotate2D, float)
        ->ArgsProduct({{0, 1, 2, 3, 4},
                       {1, 3}})
        ->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_TEMPLATE(CPU_transform_rotate3D, float)
        ->ArgsProduct({{0, 1, 2, 3, 4},
                       {1, 3}})
        ->Unit(benchmark::kMillisecond)->UseRealTime();

// The prefiltering is more expansive than the transformation itself. When possible, save and reuse the prefilter.
// Without prefiltering, cubic is about the same as cubic-b-spline.
BENCHMARK_TEMPLATE(CPU_transform_rotate2D_bspline, float, true)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_transform_rotate2D_bspline, float, false)->Unit(benchmark::kMillisecond)->UseRealTime();

// Passing the batch or calling the function batch times seems to be equivalent performance-wise.
BENCHMARK_TEMPLATE(CPU_transform_rotate2D_batch, float, true)->Arg(1)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_transform_rotate2D_batch, float, true)->Arg(41)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_transform_rotate2D_batch, float, false)->Arg(41)->Unit(benchmark::kMillisecond)->UseRealTime();


