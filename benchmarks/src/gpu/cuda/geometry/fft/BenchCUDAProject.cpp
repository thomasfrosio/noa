#include <benchmark/benchmark.h>

#include <noa/common/geometry/Euler.h>
#include <noa/gpu/cuda/Event.h>
#include <noa/gpu/cuda/geometry/fft/Project.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrPinned.h>
#include <noa/gpu/cuda/memory/Set.h>
#include <noa/gpu/cuda/Stream.h>

using namespace ::noa;

namespace {
    template<typename T>
    void CUDA_geometry_insert3D(benchmark::State& state) {
        const size4_t slice_shape{1, 1, 256, 256};
        const size4_t slice_stride = slice_shape.strides();

        const size4_t grid_shape{1, 512, 512, 512};
        const size4_t grid_stride = grid_shape.strides();

        cuda::Stream stream{cuda::Stream::DEFAULT};
        cuda::memory::PtrDevice<T> slice{slice_shape.elements(), stream};
        cuda::memory::PtrDevice<T> grid{grid_shape.elements(), stream};

        cuda::memory::set(slice.share(), slice.elements(), T{1}, stream);
        cuda::memory::set(grid.share(), grid.elements(), T{0}, stream);
        stream.synchronize();

        std::vector<float> tilt_angles{0, 3, 6, 9, -3, -6, -9};
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

            cuda::geometry::fft::insert3D<fft::HC2HC>(slice.share(), slice_stride, slice_shape,
                                                      grid.share(), grid_stride, grid_shape,
                                                      scaling_factors.share(), rotations.share(),
                                                      0.5f, ews_radius, stream);

            end.record(stream);
            end.synchronize();
            stream.clear();
            state.SetIterationTime(cuda::Event::elapsed(start, end));
            ::benchmark::DoNotOptimize(grid.get());
        }
    }
}

BENCHMARK_TEMPLATE(CUDA_geometry_insert3D, float)->Unit(benchmark::kMillisecond)->UseRealTime();

