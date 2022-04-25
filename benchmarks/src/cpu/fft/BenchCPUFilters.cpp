#include <benchmark/benchmark.h>

#include <noa/cpu/Stream.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/fft/Filters.h>

#include "Helpers.h"

using namespace ::noa;

namespace {
    template<typename T>
    void CPU_fft_filters_lowpass_inplace(benchmark::State& state) {
        const path_t path_base = benchmark::NOA_DATA_PATH / "fft";
        const YAML::Node benchmarks = YAML::LoadFile(path_base / "benchmarks.yaml")["lowpass"][state.range(0)];

        const auto shape = benchmarks["shape"].as<size4_t>();
        const auto threads = benchmarks["threads"].as<size_t>();
        const auto cutoff = benchmarks["cutoff"].as<float>();
        const auto width = benchmarks["width"].as<float>();

        const size4_t stride = shape.fft().stride();
        const size_t elements = shape.fft().elements();
        cpu::memory::PtrHost<T> input_result(elements);

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(input_result.get(), input_result.elements(), randomizer);

        cpu::Stream stream(cpu::Stream::DEFAULT);
        stream.threads(threads);

        for (auto _: state) {
            // Test on-the-fly, in-place.
            // half_t is much slower here.
            cpu::fft::lowpass<fft::H2H, T>(input_result.get(), stride,
                                           input_result.get(), stride,
                                           shape, cutoff, width, stream);
            ::benchmark::DoNotOptimize(input_result.get());
        }
    }

//    template<typename T>
//    void CPU_fft_filters_lowpass(benchmark::State& state) {
//        path_t path_base = benchmark::NOA_DATA_PATH / "fft";
//        YAML::Node benchmarks = YAML::LoadFile(path_base / "benchmarks.yaml")["lowpass"][state.range(0)];
//
//        auto threads = benchmarks["threads"].as<size_t>();
//        auto batches = benchmarks["batches"].as<size_t>();
//        auto shape = benchmarks["shape"].as<size3_t>();
//        auto cutoff = benchmarks["cutoff"].as<float>();
//        auto width = benchmarks["width"].as<float>();
//
//        size_t elements = noa::elementsFFT(shape);
//        size3_t pitch = noa::shapeFFT(shape);
//        cpu::memory::PtrHost<T> input_result(elements * batches);
//
//        test::Randomizer<T> randomizer(-5, 5);
//        test::randomize(input_result.get(), input_result.elements(), randomizer);
//
//        cpu::Stream stream;
//        stream.threads(threads);
//
//        for (auto _: state) {
//            // In this case, half_t is almost as fast as float since there's no half-precision multiplication.
//            cpu::fft::lowpass<fft::H2H, T>(nullptr, pitch,
//                                           input_result.get(), pitch,
//                                           shape, batches,
//                                           cutoff, width, stream);
//            ::benchmark::DoNotOptimize(input_result.get());
//        }
//    }
//
//    template<typename T, int TYPE>
//    void CPU_fft_filters_compare(benchmark::State& state) {
//        size_t threads = 1;
//        size_t batches = 1;
//        size3_t shape = {512, 512, 1};
//
//        size_t elements = noa::elementsFFT(shape);
//        size3_t pitch = noa::shapeFFT(shape);
//        cpu::memory::PtrHost<T> input_result(elements * batches);
//
//        test::Randomizer<T> randomizer(-5, 5);
//        test::randomize(input_result.get(), input_result.elements(), randomizer);
//
//        cpu::Stream stream;
//        stream.threads(threads);
//
//        for (auto _: state) {
//            // Test on-the-fly, in-place.
//            if constexpr (TYPE == 0) {
//                cpu::fft::lowpass<fft::H2H, T>(nullptr, pitch,
//                                               input_result.get(), pitch,
//                                               shape, batches,
//                                               0.4f, 0.1f, stream);
//            } else if constexpr (TYPE == 1) {
//                cpu::fft::highpass<fft::H2H, T>(nullptr, pitch,
//                                                input_result.get(), pitch,
//                                                shape, batches,
//                                                0.4f, 0.1f, stream);
//            } else {
//                cpu::fft::bandpass<fft::H2H, T>(nullptr, pitch,
//                                                input_result.get(), pitch,
//                                                shape, batches,
//                                                0.3f, 0.4f, 0.1f, 0.1f, stream);
//            }
//            ::benchmark::DoNotOptimize(input_result.get());
//        }
//    }
}

BENCHMARK_TEMPLATE(CPU_fft_filters_lowpass_inplace, half_t)->DenseRange(0, 5)->Unit(
        benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_fft_filters_lowpass_inplace, float)->DenseRange(0, 5)->Unit(
        benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_fft_filters_lowpass_inplace, chalf_t)->DenseRange(0, 5)->Unit(
        benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_fft_filters_lowpass_inplace, cfloat_t)->DenseRange(0, 5)->Unit(
        benchmark::kMillisecond)->UseRealTime();

//BENCHMARK_TEMPLATE(CPU_fft_filters_lowpass, half_t)->DenseRange(0, 5)->Unit(benchmark::kMillisecond)->UseRealTime();
//BENCHMARK_TEMPLATE(CPU_fft_filters_lowpass, float)->DenseRange(0, 5)->Unit(benchmark::kMillisecond)->UseRealTime();
//BENCHMARK_TEMPLATE(CPU_fft_filters_lowpass, double)->DenseRange(0, 5)->Unit(benchmark::kMillisecond)->UseRealTime();
//
//BENCHMARK_TEMPLATE(CPU_fft_filters_compare, float, 0)->Unit(benchmark::kMillisecond)->UseRealTime();
//BENCHMARK_TEMPLATE(CPU_fft_filters_compare, float, 1)->Unit(benchmark::kMillisecond)->UseRealTime();
//BENCHMARK_TEMPLATE(CPU_fft_filters_compare, float, 2)->Unit(benchmark::kMillisecond)->UseRealTime();
