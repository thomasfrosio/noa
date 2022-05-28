#include <benchmark/benchmark.h>

#include <noa/gpu/cuda/Event.h>
#include <noa/gpu/cuda/Stream.h>
#include <noa/gpu/cuda/math/Random.h>
#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrArray.h>
#include <noa/gpu/cuda/memory/PtrTexture.h>
#include <noa/gpu/cuda/geometry/Rotate.h>

using namespace ::noa;

namespace {
    constexpr size4_t g_shapes[] = {
            {1, 1,   4096, 4096},
            {1, 256, 256,  256},
    };

    constexpr InterpMode g_interp[] = {
            INTERP_NEAREST,
            INTERP_LINEAR_FAST,
            INTERP_LINEAR,
            INTERP_CUBIC,
            INTERP_CUBIC_BSPLINE_FAST,
    };

    constexpr BorderMode g_border[] = {
            BORDER_ZERO,
            BORDER_CLAMP,
    };

    template<typename T>
    void CUDA_geometry_rotate2D(benchmark::State& state) {
        const size4_t shape = g_shapes[0];
        const size4_t stride = shape.stride();
        const size_t elements = shape.elements();

        const InterpMode interp = g_interp[state.range(0)];
        const BorderMode border = g_border[state.range(1)];

        cuda::Stream stream{cuda::Stream::DEFAULT};
        cuda::memory::PtrDevice<T> src{elements, stream};
        cuda::memory::PtrDevice<T> dst{elements, stream};

        cuda::math::randomize(math::uniform_t{}, src.share(), src.elements(), T{-5}, T{5}, stream);
        stream.synchronize();

        const float rotation = 3.1415f;
        const float2_t center = float2_t{shape.get() + 2} / 2;
        for (auto _: state) {
            cuda::Event start, end;
            start.record(stream);

            cuda::geometry::rotate2D(src.share(), stride, shape,
                                     dst.share(), stride, shape,
                                     rotation, center, interp, border, stream);

            end.record(stream);
            end.synchronize();
            stream.clear();
            state.SetIterationTime(cuda::Event::elapsed(start, end));
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CUDA_geometry_rotate2D_texture(benchmark::State& state) {
        const size4_t shape = g_shapes[0];
        const size4_t stride = shape.stride();
        const size_t elements = shape.elements();

        const InterpMode interp = INTERP_LINEAR_FAST;
        const BorderMode border = BORDER_ZERO;

        cuda::Stream stream{cuda::Stream::DEFAULT};
        cuda::memory::PtrDevice<T> src{elements, stream};
        cuda::memory::PtrDevice<T> dst{elements, stream};

        cuda::math::randomize(math::uniform_t{}, src.share(), src.elements(), T{-5}, T{5}, stream);
        stream.synchronize();

        const size3_t shape_3d{shape[1], shape[2], shape[3]};
        cuda::memory::PtrArray<T> array{shape_3d};
        cuda::memory::PtrTexture texture{array.get(), interp, border};
        cuda::memory::copy(src.share(), shape[3], array.share(), shape_3d, stream);

        const float rotation = 3.1415f;
        const float2_t center = float2_t{shape.get() + 2} / 2;
        const float23_t matrix{noa::geometry::translate(center) *
                               float33_t{noa::geometry::rotate(-rotation)} *
                               noa::geometry::translate(-center)};

        for (auto _: state) {
            cuda::Event start, end;
            start.record(stream);

            cuda::memory::copy(src.share(), stride[2], array.share(), shape_3d, stream);
            cuda::geometry::transform2D(texture.get(), size2_t{shape[2], shape[3]}, interp, border,
                                        dst.get(), stride, shape,
                                        matrix, stream);

            end.record(stream);
            end.synchronize();
            stream.clear();

            state.SetIterationTime(cuda::Event::elapsed(start, end));
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CUDA_geometry_rotate3D(benchmark::State& state) {
        const size4_t shape = g_shapes[1];
        const size4_t stride = shape.stride();
        const size_t elements = shape.elements();

        const InterpMode interp = g_interp[state.range(0)];
        const BorderMode border = g_border[state.range(1)];

        cuda::Stream stream{cuda::Stream::DEFAULT};
        cuda::memory::PtrDevice<T> src{elements, stream};

        cuda::math::randomize(math::uniform_t{}, src.share(), src.elements(), T{-5}, T{5}, stream);
        stream.synchronize();

        const float33_t rotm = geometry::euler2matrix(float3_t{1.34, 0.45, 0});
        const float3_t center = float3_t{shape.get() + 1} / 2;
        for (auto _: state) {
            cuda::Event start, end;
            start.record(stream);

            cuda::geometry::rotate3D(src.share(), stride, shape,
                                     src.share(), stride, shape,
                                     rotm, center, interp, border, stream);

            end.record(stream);
            end.synchronize();
            stream.clear();
            state.SetIterationTime(cuda::Event::elapsed(start, end));
            ::benchmark::DoNotOptimize(src.get());
        }
    }
}

// The border has no measurable effect on performance. The interpolation method is about the same, except for b-spline.
// Cubic-B-spline is about ~3x slower than the rest, but it is only because of the prefiltering. Without prefiltering,
// INTERP_CUBIC_BSPLINE_FAST is as fast as other interpolation methods (other than INTERP_CUBIC which is ~15% slower
// than everything else).
BENCHMARK_TEMPLATE(CUDA_geometry_rotate2D, float)
        ->ArgsProduct({{0, 1, 2, 3, 4},
                       {0, 1}})
        ->Unit(benchmark::kMillisecond)->UseRealTime();

// This is quite impressive when you think about it. The amount of computation
BENCHMARK_TEMPLATE(CUDA_geometry_rotate3D, float)
        ->ArgsProduct({{0, 1},
                       {0, 1}})
        ->Unit(benchmark::kMillisecond)->UseRealTime();

// Reusing the CUDA array and texture is ~6x to ~7x faster.
// Reusing the CUDA array and texture but copying the input to the CUDA array is still ~3x faster.
BENCHMARK_TEMPLATE(CUDA_geometry_rotate2D_texture, float)
        ->Unit(benchmark::kMillisecond)->UseRealTime();
