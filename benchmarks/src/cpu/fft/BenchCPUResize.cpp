#include <benchmark/benchmark.h>

#include <noa/cpu/Stream.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/memory/Copy.h>
#include <noa/cpu/fft/Resize.h>

#include "Helpers.h"

using namespace ::noa;

namespace {
    template<typename T>
    void CPU_fft_resize_cropH2H(benchmark::State& state) {
        const size3_t src_shape = {512, 512, 512};
        const size3_t dst_shape = {256, 256, 256};
        const size3_t src_pitch = shapeFFT(src_shape);
        const size3_t dst_pitch = shapeFFT(dst_shape);

        cpu::memory::PtrHost<T> src(elementsFFT(src_shape));
        cpu::memory::PtrHost<T> dst(elementsFFT(dst_shape));

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::fft::resize<fft::H2H>(src.get(), src_pitch, src_shape, dst.get(), dst_pitch, dst_shape, 1, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_fft_resize_cropF2F(benchmark::State& state) {
        const size3_t src_shape = {512, 512, 512};
        const size3_t dst_shape = {256, 256, 256};

        cpu::memory::PtrHost<T> src(elements(src_shape));
        cpu::memory::PtrHost<T> dst(elements(dst_shape));

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::fft::resize<fft::F2F>(src.get(), src_shape, src_shape, dst.get(), dst_shape, dst_shape, 1, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_fft_resize_padH2H(benchmark::State& state) {
        const size3_t src_shape = {256, 256, 256};
        const size3_t dst_shape = {512, 512, 512};
        const size3_t src_pitch = shapeFFT(src_shape);
        const size3_t dst_pitch = shapeFFT(dst_shape);

        cpu::memory::PtrHost<T> src(elementsFFT(src_shape));
        cpu::memory::PtrHost<T> dst(elementsFFT(dst_shape));

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::fft::resize<fft::H2H>(src.get(), src_pitch, src_shape, dst.get(), dst_pitch, dst_shape, 1, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_fft_resize_padF2F(benchmark::State& state) {
        const size3_t src_shape = {256, 256, 256};
        const size3_t dst_shape = {512, 512, 512};

        cpu::memory::PtrHost<T> src(elements(src_shape));
        cpu::memory::PtrHost<T> dst(elements(dst_shape));

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::fft::resize<fft::F2F>(src.get(), src_shape, src_shape, dst.get(), dst_shape, dst_shape, 1, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }
}

BENCHMARK_TEMPLATE(CPU_fft_resize_cropH2H, half_t)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_fft_resize_cropH2H, float)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_fft_resize_cropH2H, double)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_fft_resize_cropH2H, cfloat_t)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_TEMPLATE(CPU_fft_resize_cropF2F, half_t)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_fft_resize_cropF2F, float)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_fft_resize_cropF2F, double)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_fft_resize_cropF2F, cfloat_t)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_TEMPLATE(CPU_fft_resize_padH2H, half_t)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_fft_resize_padH2H, float)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_fft_resize_padH2H, double)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_fft_resize_padH2H, cfloat_t)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_TEMPLATE(CPU_fft_resize_padF2F, half_t)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_fft_resize_padF2F, float)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_fft_resize_padF2F, double)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_fft_resize_padF2F, cfloat_t)->Unit(benchmark::kMillisecond)->UseRealTime();

namespace {
    template<typename T>
    void CPU_fft_resize_copyH2H_trigger(benchmark::State& state) {
        const size3_t shape = {256, 256, 256};

        cpu::memory::PtrHost<T> src(elements(shape));
        cpu::memory::PtrHost<T> dst(elements(shape));

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::fft::resize<fft::H2H>(src.get(), shapeFFT(shape), shape, dst.get(), shapeFFT(shape), shape, 1, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_fft_resize_copyH2H(benchmark::State& state) {
        const size3_t shape = {256, 256, 256};

        cpu::memory::PtrHost<T> src(elementsFFT(shape));
        cpu::memory::PtrHost<T> dst(elementsFFT(shape));

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::memory::copy(src.get(), shapeFFT(shape), dst.get(), shapeFFT(shape), shapeFFT(shape), 1, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_fft_resize_copyF2F_trigger(benchmark::State& state) {
        const size3_t shape = {256, 256, 256};

        cpu::memory::PtrHost<T> src(elements(shape));
        cpu::memory::PtrHost<T> dst(elements(shape));

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::fft::resize<fft::F2F>(src.get(), shape, shape, dst.get(), shape, shape, 1, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_fft_resize_copyF2F(benchmark::State& state) {
        const size3_t shape = {256, 256, 256};

        cpu::memory::PtrHost<T> src(elements(shape));
        cpu::memory::PtrHost<T> dst(elements(shape));

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::memory::copy(src.get(), shape, dst.get(), shape, shape, 1, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }
}

BENCHMARK_TEMPLATE(CPU_fft_resize_copyH2H_trigger, float)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_fft_resize_copyH2H, float)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_TEMPLATE(CPU_fft_resize_copyF2F_trigger, float)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_fft_resize_copyF2F, float)->Unit(benchmark::kMillisecond)->UseRealTime();
