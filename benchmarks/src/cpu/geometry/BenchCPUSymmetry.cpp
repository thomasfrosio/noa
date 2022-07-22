#include <benchmark/benchmark.h>

#include <noa/cpu/Stream.h>
#include <noa/cpu/math/Random.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/geometry/Symmetry.h>

using namespace ::noa;

namespace {
    constexpr InterpMode g_interp[] = {
            INTERP_NEAREST,
            INTERP_LINEAR,
            INTERP_COSINE,
            INTERP_CUBIC,
            INTERP_CUBIC_BSPLINE,
    };

    const char* g_symmetry[] = {
            "c1", "c2", "c8", "d4", "o", "i1",
    };

    template<typename T>
    void CPU_geometry_symmetrize3D(benchmark::State& state) {
        const size4_t shape{1, 256, 256, 256};
        const size4_t stride = shape.strides();
        const size_t elements = shape.elements();
        const float3_t center = float3_t{shape.get() + 1} / 2;

        const InterpMode interp = g_interp[state.range(0)];
        const geometry::Symmetry symmetry(g_symmetry[state.range(1)]);

        cpu::memory::PtrHost<T> src{elements};
        cpu::memory::PtrHost<T> dst{elements};

        cpu::Stream stream{cpu::Stream::DEFAULT};
        cpu::math::randomize(math::uniform_t{}, src.share(), src.elements(), T{-5}, T{5}, stream);

        for (auto _: state) {
            cpu::geometry::symmetrize3D(src.share(), stride, dst.share(), stride, shape,
                                        symmetry, center, interp, true, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_geometry_symmetrize3D_normalize(benchmark::State& state) {
        const size4_t shape{1, 256, 256, 256};
        const size4_t stride = shape.strides();
        const size_t elements = shape.elements();
        const float3_t center = float3_t{shape.get() + 1} / 2;

        const InterpMode interp = INTERP_LINEAR;
        const bool normalize = static_cast<bool>(state.range(0));

        cpu::memory::PtrHost<T> src{elements};
        cpu::memory::PtrHost<T> dst{elements};

        cpu::Stream stream{cpu::Stream::DEFAULT};
        cpu::math::randomize(math::uniform_t{}, src.share(), src.elements(), T{-5}, T{5}, stream);

        for (auto _: state) {
            cpu::geometry::symmetrize3D(src.share(), stride, dst.share(), stride, shape,
                                         geometry::Symmetry{"C4"}, center, interp, normalize, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }
}

BENCHMARK_TEMPLATE(CPU_geometry_symmetrize3D, float)
        ->ArgsProduct({{1, 3, 4},
                       {0, 1, 2, 3, 4, 5}})
        ->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_TEMPLATE(CPU_geometry_symmetrize3D_normalize, float)->Arg(0)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_geometry_symmetrize3D_normalize, float)->Arg(1)->Unit(benchmark::kMillisecond)->UseRealTime();
