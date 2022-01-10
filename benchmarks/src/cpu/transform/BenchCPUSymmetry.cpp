#include <benchmark/benchmark.h>

#include <noa/cpu/Stream.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/transform/Symmetry.h>

#include "Helpers.h"

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
    void CPU_transform_symmetrize3D(benchmark::State& state) {
        const size3_t shape{256, 256, 256};
        const InterpMode interp = g_interp[state.range(0)];
        const transform::Symmetry symmetry(g_symmetry[state.range(1)]);

        cpu::memory::PtrHost<T> src(elements(shape));
        cpu::memory::PtrHost<T> dst(src.size());

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::transform::symmetrize3D(src.get(), shape, dst.get(), shape, shape, 1,
                                         symmetry, float3_t(shape / 2), interp, true, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_transform_symmetrize3D_normalize(benchmark::State& state) {
        const size3_t shape{256, 256, 256};
        const InterpMode interp = INTERP_LINEAR;
        const bool normalize = static_cast<bool>(state.range(0));

        cpu::memory::PtrHost<T> src(elements(shape));
        cpu::memory::PtrHost<T> dst(src.size());

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::transform::symmetrize3D(src.get(), shape, dst.get(), shape, shape, 1,
                                         transform::Symmetry("C4"), float3_t(shape / 2), interp, normalize, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }
}

BENCHMARK_TEMPLATE(CPU_transform_symmetrize3D, float)
        ->ArgsProduct({{1, 3, 4},
                       {0, 1, 2, 3, 4, 5}})
        ->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_TEMPLATE(CPU_transform_symmetrize3D_normalize, float)->Arg(0)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_transform_symmetrize3D_normalize, float)->Arg(1)->Unit(benchmark::kMillisecond)->UseRealTime();


