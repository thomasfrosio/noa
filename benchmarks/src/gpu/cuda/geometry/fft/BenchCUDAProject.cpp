#include <benchmark/benchmark.h>

#include <noa/common/geometry/Euler.h>
#include <noa/common/io/MRCFile.h>

#include <noa/gpu/cuda/Event.h>
#include <noa/gpu/cuda/geometry/fft/Project.h>
#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrPinned.h>
#include <noa/gpu/cuda/memory/PtrTexture.h>
#include <noa/gpu/cuda/memory/PtrArray.h>
#include <noa/gpu/cuda/memory/Set.h>
#include <noa/gpu/cuda/Stream.h>

#include "Helpers.h"

using namespace ::noa;

namespace {
    template<typename T>
    void CUDA_geometry_insert3D(benchmark::State& state) {
        const size4_t slice_shape{10, 1, 512, 512};
        const size4_t slice_strides = slice_shape.fft().strides();

        const size4_t grid_shape{1, 512, 512, 512};
        const size4_t grid_strides = grid_shape.fft().strides();

        cuda::Stream stream{cuda::Stream::DEFAULT};
        cuda::memory::PtrDevice<T> slice{slice_shape.fft().elements(), stream};
        cuda::memory::PtrDevice<T> grid{grid_shape.fft().elements(), stream};

        cuda::memory::set(slice.share(), slice.elements(), T{1}, stream);
        cuda::memory::set(grid.share(), grid.elements(), T{0}, stream);
        stream.synchronize();

        std::vector<float> tilt_angles{0, 3, 6, 9, -3, -6, -9, 12, 15, 18, -12, -15, -18};
        cuda::memory::PtrPinned<float22_t> scaling_factors(tilt_angles.size());
        cuda::memory::PtrPinned<float33_t> rotations(tilt_angles.size());
        for (size_t i = 0; i < tilt_angles.size(); ++i) {
            scaling_factors[i] = float22_t{1};
            rotations[i] = geometry::euler2matrix(float3_t{0, math::deg2rad(tilt_angles[i]), 0});
        }

        const float cutoff = 0.5f;
        const float2_t ews_radius{0};

        for (auto _: state) {
            cuda::Event start, end;
            start.record(stream);

            cuda::geometry::fft::insert3D<fft::HC2HC>(slice.share(), slice_strides, slice_shape,
                                                      grid.share(), grid_strides, grid_shape,
                                                      nullptr, rotations.share(),
                                                      cutoff, float3_t{1, 1, 1}, ews_radius, stream);

            end.record(stream);
            end.synchronize();
            stream.clear();
            state.SetIterationTime(cuda::Event::elapsed(start, end));
            ::benchmark::DoNotOptimize(grid.get());
        }
    }

    template<typename T>
    void CUDA_geometry_extract3D(benchmark::State& state) {
        const size4_t slice_shape{1, 1, 512, 512};
        const size4_t slice_strides = slice_shape.fft().strides();

        const size4_t grid_shape{1, 512, 512, 512};
        const size4_t grid_strides = grid_shape.fft().strides();

        cuda::Stream stream{cuda::Stream::DEFAULT};
        cuda::memory::PtrDevice<T> slice{slice_shape.fft().elements(), stream};
        cuda::memory::PtrDevice<T> grid{grid_shape.fft().elements(), stream};

        cuda::memory::set(slice.share(), slice.elements(), T{1}, stream);
        cuda::memory::set(grid.share(), grid.elements(), T{0}, stream);
        stream.synchronize();

        std::vector<float> tilt_angles{0, 3, 6, 9, -3, -6, -9, 12, 15, 18, -12, -15, -18};
        cuda::memory::PtrPinned<float22_t> scaling_factors(tilt_angles.size());
        cuda::memory::PtrPinned<float33_t> rotations(tilt_angles.size());
        for (size_t i = 0; i < tilt_angles.size(); ++i) {
            scaling_factors[i] = float22_t{1};
            rotations[i] = geometry::euler2matrix(float3_t{0, math::deg2rad(tilt_angles[i]), 0});
        }

        cuda::memory::PtrArray<T> array(size3_t(grid_shape.fft().get(1)));
        cuda::memory::PtrTexture texture(array.get(), INTERP_LINEAR, BORDER_ZERO);
        cuda::memory::copy(grid.get(), grid_strides[2], array.get(), array.shape(), stream);
        stream.synchronize();

        const float cutoff = 0.5f;
        const float2_t ews_radius{0};

        for (auto _: state) {
            cuda::Event start, end;
            start.record(stream);

            // The copy is almost 2/3 of the runtime here.
            cuda::memory::copy(grid.get(), grid_strides[2], array.get(), array.shape(), stream);
            cuda::geometry::fft::extract3D<fft::HC2HC>(array.share(), texture.share(),
                                                       int3_t(array.shape()),
                                                       slice.share(), slice_strides, slice_shape,
                                                       nullptr, rotations.share(),
                                                       cutoff, float3_t{1, 1, 1}, ews_radius, stream);

            end.record(stream);
            end.synchronize();
            stream.synchronize();

            state.SetIterationTime(cuda::Event::elapsed(start, end));
            ::benchmark::DoNotOptimize(grid.get());

            cuda::memory::set(grid.share(), grid.elements(), T{0}, stream);
            stream.synchronize();
        }
    }

    template<typename T>
    void CUDA_geometry_extract3D_notexture(benchmark::State& state) {
        const size4_t slice_shape{1, 1, 512, 512};
        const size4_t slice_strides = slice_shape.fft().strides();

        const size4_t grid_shape{1, 512, 512, 512};
        const size4_t grid_strides = grid_shape.fft().strides();

        cuda::Stream stream(cuda::Stream::DEFAULT);
        cuda::memory::PtrManaged<T> slice(slice_shape.fft().elements(), stream);
        cuda::memory::PtrDevice<T> grid(grid_shape.fft().elements(), stream);

        cuda::memory::set(slice.share(), slice.elements(), T{1}, stream);
        cuda::memory::set(grid.share(), grid.elements(), T{0}, stream);
        stream.synchronize();

        std::vector<float> tilt_angles{0, 3, 6, 9, -3, -6, -9, 12, 15, 18, -12, -15, -18};
        cuda::memory::PtrPinned<float22_t> scaling_factors(tilt_angles.size());
        cuda::memory::PtrPinned<float33_t> rotations(tilt_angles.size());
        for (size_t i = 0; i < tilt_angles.size(); ++i) {
            scaling_factors[i] = float22_t{1};
            rotations[i] = geometry::euler2matrix(float3_t{0, math::deg2rad(tilt_angles[i]), 0});
        }

        const float cutoff = 0.5f;
        const float2_t ews_radius{0};

        for (auto _: state) {
            cuda::Event start, end;
            start.record(stream);

            cuda::geometry::fft::extract3D<fft::HC2HC>(grid.share(), grid_strides, grid_shape,
                                                       slice.share(), slice_strides, slice_shape,
                                                       nullptr, rotations.share(),
                                                       cutoff, float3_t{1, 1, 1}, ews_radius, true, stream);

            end.record(stream);
            end.synchronize();
            stream.synchronize();

            state.SetIterationTime(cuda::Event::elapsed(start, end));
            ::benchmark::DoNotOptimize(grid.get());
            ::benchmark::DoNotOptimize(slice.get());

            cuda::memory::set(grid.share(), grid.elements(), T{2}, stream);
            cuda::memory::set(slice.share(), slice.elements(), T{0}, stream);
            stream.synchronize();
        }
    }
}

BENCHMARK_TEMPLATE(CUDA_geometry_insert3D, float)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CUDA_geometry_insert3D, cfloat_t)->Unit(benchmark::kMillisecond)->UseRealTime();

// The texture doesn't seem worth it here. It is slightly faster without the copy but uses almost double
// the memory because of the CUDA array of the grid...
BENCHMARK_TEMPLATE(CUDA_geometry_extract3D, float)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CUDA_geometry_extract3D, cfloat_t)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CUDA_geometry_extract3D_notexture, float)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CUDA_geometry_extract3D_notexture, cfloat_t)->Unit(benchmark::kMillisecond)->UseRealTime();
