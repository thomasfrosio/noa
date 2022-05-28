#include <benchmark/benchmark.h>

#include <noa/gpu/cuda/Event.h>
#include <noa/gpu/cuda/Stream.h>
#include <noa/gpu/cuda/math/Random.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/geometry/Symmetry.h>

using namespace ::noa;

namespace {
    constexpr InterpMode g_interp[] = {
            INTERP_LINEAR_FAST,
            INTERP_CUBIC,
            INTERP_CUBIC_BSPLINE_FAST,
    };

    const char* g_symmetry[] = {
            "c1", "c2", "c8", "d4", "o", "i1",
    };

    template<typename T>
    void CUDA_geometry_symmetrize3D(benchmark::State& state) {
        const size4_t shape{1, 256,256,256};
        const size4_t stride = shape.stride();
        const size_t elements = shape.elements();
        const float3_t center = float3_t{shape.get() + 1} / 2;

        const InterpMode interp = g_interp[state.range(0)];
        const geometry::Symmetry symmetry(g_symmetry[state.range(1)]);

        cuda::Stream stream{cuda::Stream::DEFAULT};
        cuda::memory::PtrDevice<T> src{elements, stream};
        cuda::math::randomize(math::uniform_t{}, src.share(), src.elements(), T{-5}, T{5}, stream);
        stream.synchronize();

        for (auto _: state) {
            cuda::Event start, end;
            start.record(stream);

            cuda::geometry::symmetrize3D(src.share(), stride, src.share(), stride, shape,
                                         symmetry, center, interp, true, stream);

            end.record(stream);
            end.synchronize();
            stream.clear();
            state.SetIterationTime(cuda::Event::elapsed(start, end));
            ::benchmark::DoNotOptimize(src.get());
        }
    }

    template<typename T>
    void CUDA_geometry_symmetrize3D_normalize(benchmark::State& state) {
        const size4_t shape{1, 256, 256, 256};
        const size4_t stride = shape.stride();
        const size_t elements = shape.elements();
        const float3_t center = float3_t{shape.get() + 1} / 2;

        const InterpMode interp = INTERP_LINEAR;
        const bool normalize = static_cast<bool>(state.range(0));

        cuda::Stream stream{cuda::Stream::DEFAULT};
        cuda::memory::PtrDevice<T> src{elements, stream};
        cuda::math::randomize(math::uniform_t{}, src.share(), src.elements(), T{-5}, T{5}, stream);

        for (auto _: state) {
            cuda::Event start, end;
            start.record(stream);

            cuda::geometry::symmetrize3D(src.share(), stride, src.share(), stride, shape,
                                         geometry::Symmetry{"C4"}, center, interp, normalize, stream);

            end.record(stream);
            end.synchronize();
            stream.clear();
            state.SetIterationTime(cuda::Event::elapsed(start, end));
            ::benchmark::DoNotOptimize(src.get());
        }
    }
}

BENCHMARK_TEMPLATE(CUDA_geometry_symmetrize3D, float)
        ->ArgsProduct({{0, 1, 2},
                       {0, 1, 2, 3, 4, 5}})
        ->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_TEMPLATE(CUDA_geometry_symmetrize3D_normalize, float)->Arg(0)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CUDA_geometry_symmetrize3D_normalize, float)->Arg(1)->Unit(benchmark::kMillisecond)->UseRealTime();
