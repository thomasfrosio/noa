#include <benchmark/benchmark.h>

#include <noa/common/geometry/Euler.h>
#include <noa/common/io/MRCFile.h>
#include <noa/common/Functors.h>

#include <noa/cpu/Event.hpp>
#include <noa/cpu/Stream.hpp>
#include <noa/cpu/math/Complex.hpp>
#include <noa/cpu/memory/AllocatorHeap.hpp>
#include <noa/cpu/geometry/fft/Project.hpp>

#include "Helpers.h"

using namespace ::noa;

namespace {
    template<typename T>
    void CPU_project_insert(benchmark::State& state) {
        const size4_t grid_shape{1, 512, 512, 512};
        const size4_t grid_strides = grid_shape.fft().strides();

        const size4_t slice_shape{5, 1, 512, 512};
        const size4_t slice_strides = slice_shape.fft().strides();

        cpu::memory::PtrHost<T> slice(slice_shape.fft().elements());
        cpu::memory::PtrHost<T> grid(grid_shape.fft().elements());

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(slice.get(), slice.elements(), randomizer);
        test::memset(grid.get(), grid.elements(), T{0});

        cpu::memory::PtrHost<float33_t> rotations(slice_shape[0]);
        for (size_t i = 0; i < slice_shape[0]; ++i)
            rotations[i] = geometry::euler2matrix(float3_t{math::deg2rad((float)i), 0, 0});

        cpu::Stream stream(cpu::Stream::DEFAULT);
        stream.threads(1);

        for (auto _: state) {
            cpu::Event start, end;
            start.record(stream);

            cpu::geometry::fft::insert3D<fft::HC2HC, T>(slice.share(), slice_strides, slice_shape,
                                                        grid.share(), grid_strides, grid_shape,
                                                        float22_t{},
                                                        rotations.share(),
                                                        0.5f, dim4_t{}, float2_t{0}, stream);

            end.record(stream);
            end.synchronize();
            state.SetIterationTime(cpu::Event::elapsed(start, end));
            stream.synchronize();

            ::benchmark::DoNotOptimize(grid.get());
        }
    }

    template<typename T>
    void CPU_project_insert_thick(benchmark::State& state) {
        const size4_t grid_shape{1, 512, 512, 512};
        const size4_t grid_strides = grid_shape.fft().strides();

        const size4_t slice_shape{1, 1, 512, 512};
        const size4_t slice_strides = slice_shape.fft().strides();

        cpu::memory::PtrHost<T> slice(slice_shape.fft().elements());
        cpu::memory::PtrHost<T> grid(grid_shape.fft().elements());

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(slice.get(), slice.elements(), randomizer);
        test::memset(grid.get(), grid.elements(), T{0});

        cpu::memory::PtrHost<float33_t> rotations(slice_shape[0]);
        for (size_t i = 0; i < slice_shape[0]; ++i)
            rotations[i] = geometry::euler2matrix(float3_t{math::deg2rad((float)i), 0, 0});

        cpu::Stream stream(cpu::Stream::DEFAULT);
        stream.threads(1);

        for (auto _: state) {
            cpu::Event start, end;
            start.record(stream);

            cpu::geometry::fft::insert3D<fft::HC2HC, T>(slice.share(), slice_strides, slice_shape,
                                                        grid.share(), grid_strides, grid_shape,
                                                        float22_t{},
                                                        rotations.share(),
                                                        0.5f, dim4_t{}, float2_t{0}, 0.02f, stream);

            end.record(stream);
            end.synchronize();
            state.SetIterationTime(cpu::Event::elapsed(start, end));
            stream.synchronize();

            ::benchmark::DoNotOptimize(grid.get());
        }
    }

    template<typename T>
    void CPU_project_insert_extract_using_grid(benchmark::State& state) {
        const dim4_t input_slice_shape{41, 1, 512, 512};
        const dim4_t output_slice_shape{1, 1, 512, 512};
        const dim4_t grid_shape{1, 512, 512, 512};

        const dim4_t input_slice_strides = input_slice_shape.fft().strides();
        const dim4_t output_slice_strides = output_slice_shape.fft().strides();
        const dim4_t grid_strides = grid_shape.fft().strides();

        cpu::memory::PtrHost<T> input_slice(input_slice_shape.fft().elements());
        cpu::memory::PtrHost<T> output_slice(output_slice_shape.fft().elements());
        cpu::memory::PtrHost<T> grid(grid_shape.fft().elements());

        cpu::memory::PtrHost<float33_t> input_rotation_matrices(input_slice_shape[0]);
        float count = 0;
        for (uint i = 0; i < input_slice_shape[0]; ++i) {
            input_rotation_matrices[i] = geometry::euler2matrix(math::deg2rad(float3_t{count, 0, 0}), "YZX", false);
            count += 1;
        }
        const auto output_rotation_matrix = geometry::euler2matrix(math::deg2rad(float3_t{count, 0, 0}), "YZX", false);

        const float cutoff = 0.5f;
        const float slice_z_radius = 0.004f;

        cpu::Stream stream(cpu::Stream::DEFAULT);
        cpu::memory::set(input_slice.begin(), input_slice.end(), T{1});

        stream.threads(1);

        for (auto _: state) {
            cpu::memory::set(output_slice.begin(), output_slice.end(), T{0});
            cpu::memory::set(grid.begin(), grid.end(), T{0});

            cpu::Event start, end;
            start.record(stream);

            cpu::geometry::fft::insert3D<fft::HC2HC>(
                    input_slice.share(), input_slice_strides, input_slice_shape,
                    grid.share(), grid_strides, grid_shape,
                    float22_t{}, input_rotation_matrices.share(),
                    cutoff, {}, {}, slice_z_radius, stream);

            cpu::geometry::fft::extract3D<fft::HC2HC>(
                    grid.share(), grid_strides, grid_shape,
                    output_slice.share(), output_slice_strides, output_slice_shape,
                    float22_t{}, output_rotation_matrix,
                    cutoff, {}, {}, stream);

            end.record(stream);
            end.synchronize();
            state.SetIterationTime(cpu::Event::elapsed(start, end));
            stream.synchronize();

            ::benchmark::DoNotOptimize(output_slice.get());
        }
    }

    template<typename T>
    void CPU_project_insert_extract_combined(benchmark::State& state) {
        const dim4_t input_slice_shape{41, 1, 512, 512};
        const dim4_t output_slice_shape{1, 1, 512, 512};

        const dim4_t input_slice_strides = input_slice_shape.fft().strides();
        const dim4_t output_slice_strides = output_slice_shape.fft().strides();

        cpu::memory::PtrHost<T> input_slice(input_slice_shape.fft().elements());
        cpu::memory::PtrHost<T> output_slice(output_slice_shape.fft().elements());

        cpu::memory::PtrHost<float33_t> input_rotation_matrices(input_slice_shape[0]);
        float count = 0;
        for (uint i = 0; i < input_slice_shape[0]; ++i) {
            input_rotation_matrices[i] = geometry::euler2matrix(math::deg2rad(float3_t{count, 0, 0}), "YZX", false);
            count += 1;
        }
        const auto output_rotation_matrix = geometry::euler2matrix(math::deg2rad(float3_t{count, 0, 0}), "YZX", false);

        const float cutoff = 0.5f;
        const float slice_z_radius = 0.004f;

        cpu::Stream stream(cpu::Stream::DEFAULT);
        cpu::memory::set(input_slice.begin(), input_slice.end(), T{1});

        stream.threads(1);

        for (auto _: state) {
            cpu::memory::set(output_slice.begin(), output_slice.end(), T{0});

            cpu::Event start, end;
            start.record(stream);

            cpu::geometry::fft::extract3D<fft::HC2HC>(
                    input_slice.share(), input_slice_strides, input_slice_shape,
                    output_slice.share(), output_slice_strides, output_slice_shape,
                    float22_t{}, input_rotation_matrices.share(),
                    float22_t{}, output_rotation_matrix,
                    cutoff, {}, slice_z_radius, stream);

            end.record(stream);
            end.synchronize();
            state.SetIterationTime(cpu::Event::elapsed(start, end));
            stream.synchronize();

            ::benchmark::DoNotOptimize(output_slice.get());
        }
    }
}

BENCHMARK_TEMPLATE(CPU_project_insert, float)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_project_insert, double)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_project_insert, cfloat_t)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_project_insert, cdouble_t)->Unit(benchmark::kMillisecond)->UseRealTime();

// This is ~9times slower that the above version, which is completely understandable given that
// we have to loop through the grid. Multithreading scales quite well: 8 threads is ~6times faster
// for both versions. Also, the runtime depends on the thickness of the slices, as expected.
BENCHMARK_TEMPLATE(CPU_project_insert_thick, float)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_project_insert_thick, double)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_project_insert_thick, cfloat_t)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_project_insert_thick, cdouble_t)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_TEMPLATE(CPU_project_insert_extract_using_grid, float)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_project_insert_extract_using_grid, double)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_project_insert_extract_using_grid, cfloat_t)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_project_insert_extract_using_grid, cdouble_t)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_TEMPLATE(CPU_project_insert_extract_combined, float)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_project_insert_extract_combined, double)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_project_insert_extract_combined, cfloat_t)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_project_insert_extract_combined, cdouble_t)->Unit(benchmark::kMillisecond)->UseRealTime();
