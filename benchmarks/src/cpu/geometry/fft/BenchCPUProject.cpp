#include <benchmark/benchmark.h>

#include <noa/common/geometry/Euler.h>
#include <noa/common/io/MRCFile.h>
#include <noa/common/Functors.h>

#include <noa/cpu/Event.h>
#include <noa/cpu/Stream.h>
#include <noa/cpu/math/Complex.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/geometry/fft/Project.h>

#include "Helpers.h"

using namespace ::noa;

namespace {
    template<typename T>
    void CPU_project_insert(benchmark::State& state) {
        const size4_t grid_shape{1, 149, 149, 149};
        const size4_t grid_strides = grid_shape.fft().strides();

        const size4_t slice_shape{1000, 1, 149, 149};
        const size4_t slice_strides = slice_shape.fft().strides();

        cpu::memory::PtrHost<T> slice(slice_shape.fft().elements());
        cpu::memory::PtrHost<T> grid(grid_shape.fft().elements());

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(slice.get(), slice.elements(), randomizer);
        test::memset(grid.get(), grid.elements(), T{0});

        cpu::memory::PtrHost<float33_t> rotations(slice_shape[0]);
        for (size_t i = 0; i < slice_shape[0]; ++i)
            rotations[i] = geometry::euler2matrix(float3_t{math::deg2rad((float)i), 0, 0});

        cpu::Stream stream;
        stream.threads(16);

        for (auto _: state) {
            cpu::Event start, end;
            start.record(stream);

            cpu::geometry::fft::insert3D<fft::HC2HC, T>(slice.share(), slice_strides, slice_shape,
                                                        grid.share(), grid_strides, grid_shape,
                                                        nullptr,
                                                        rotations.share(),
                                                        0.5f, float3_t{1, 1, 1}, float2_t{0}, stream);

            end.record(stream);
            end.synchronize();
            state.SetIterationTime(cpu::Event::elapsed(start, end));
            stream.synchronize();

            ::benchmark::DoNotOptimize(grid.get());
        }
    }
}

BENCHMARK_TEMPLATE(CPU_project_insert, float)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_project_insert, double)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_project_insert, cfloat_t)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_project_insert, cdouble_t)->Unit(benchmark::kMillisecond)->UseRealTime();
