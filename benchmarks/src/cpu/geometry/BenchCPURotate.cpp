#include <benchmark/benchmark.h>

#include <noa/cpu/Stream.hpp>
#include <noa/cpu/math/Random.hpp>
#include <noa/cpu/memory/PtrHost.hpp>
#include <noa/cpu/geometry/Rotate.h>

using namespace ::noa;

namespace {
    constexpr size4_t g_shapes[] = {
            {1, 1,   4096, 4096},
            {1, 256, 256,  256},
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
            BORDER_CLAMP,
            BORDER_PERIODIC,
            BORDER_MIRROR,
    };

    template<typename T>
    void CPU_transform_rotate2D(benchmark::State& state) {
        const size4_t shape = g_shapes[0];
        const size4_t stride = shape.strides();
        const size_t elements = shape.elements();

        const InterpMode interp = g_interp[state.range(0)];
        const BorderMode border = g_border[state.range(1)];

        cpu::memory::PtrHost<T> src{elements};
        cpu::memory::PtrHost<T> dst{elements};

        cpu::Stream stream{cpu::Stream::DEFAULT};
        cpu::math::randomize(math::uniform_t{}, src.share(), src.elements(), T{-5}, T{5}, stream);

        const float rotation = 3.1415f;
        const float2_t center = float2_t{shape.get() + 2} / 2;
        for (auto _: state) {
            cpu::geometry::rotate2D(src.share(), stride, shape,
                                    dst.share(), stride, shape,
                                    rotation, center, interp, border, 0.f, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }
}

// The border has little effect on performance. The interpolation method is the critical part and is as expected.
// Cubic is about x2 slower than linear. Cubic-B-spline (with prefilter) is about x3.5 slower than cubic.
// The prefiltering is more expansive than the transformation itself. When possible, save and reuse the prefilter.
// Multithreading is scaling relatively well...
BENCHMARK_TEMPLATE(CPU_transform_rotate2D, float)
        ->ArgsProduct({{0, 1, 2, 3, 4},
                       {0, 1}})
        ->Unit(benchmark::kMillisecond)->UseRealTime();
