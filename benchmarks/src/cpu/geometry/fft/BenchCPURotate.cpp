#include <benchmark/benchmark.h>

#include "noa/common/geometry/Transform.h"
#include <noa/cpu/Stream.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/geometry/fft/Transform.h>

#include "Helpers.h"

using namespace ::noa;

namespace {
    template<typename T>
    void CPU_transform_fft_rotate2D(benchmark::State& state) {
        const size4_t shape{1, 1, 1024, 1024};
        const InterpMode interp = INTERP_LINEAR;

        cpu::memory::PtrHost<T> src(shape.fft().elements());
        cpu::memory::PtrHost<T> dst(shape.fft().elements());

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream(cpu::Stream::DEFAULT);
        stream.threads(4);
        for (auto _: state) {
            cpu::geometry::fft::transform2D<fft::HC2HC>(
                    src.share(), shape.fft().strides(), dst.share(), shape.fft().strides(), shape,
                    geometry::rotate(3.1415f), float2_t{}, 0.5f, interp, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }
//
//    template<typename T>
//    void CPU_transform_fft_rotate3D(benchmark::State& state) {
//        const size3_t shape{256,  256,  256};
//        const InterpMode interp = INTERP_LINEAR;
//
//        const size2_t pitch{shape.x / 2 + 1, shape.y};
//        cpu::memory::PtrHost<T> src(elementsFFT(shape));
//        cpu::memory::PtrHost<T> dst(src.size());
//
//        test::Randomizer<T> randomizer(-5, 5);
//        test::randomize(src.get(), src.elements(), randomizer);
//
//        cpu::Stream stream;
//        for (auto _: state) {
//            cpu::transform::fft::apply3D<fft::HC2HC>(
//                    src.get(), pitch, dst.get(), pitch, shape,
//                    transform::toMatrix(float3_t{3.1415, 0, 0}), {}, 0.5f, interp, stream);
//            ::benchmark::DoNotOptimize(dst.get());
//        }
//    }
}

BENCHMARK_TEMPLATE(CPU_transform_fft_rotate2D, float)->Unit(benchmark::kMillisecond)->UseRealTime();
//BENCHMARK_TEMPLATE(CPU_transform_fft_rotate3D, float)->Unit(benchmark::kMillisecond)->UseRealTime();
