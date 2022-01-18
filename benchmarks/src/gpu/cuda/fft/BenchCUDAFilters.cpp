#include <benchmark/benchmark.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/Stream.h>
#include <noa/gpu/cuda/Event.h>
#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/fft/Filters.h>

#include "Helpers.h"

using namespace ::noa;

namespace {
    template<typename T>
    void CUDA_fft_lowpass_inplace(benchmark::State& state) {
        path_t path_base = benchmark::PATH_NOA_DATA / "fft";
        YAML::Node benchmarks = YAML::LoadFile(path_base / "benchmarks.yaml")["lowpass"][state.range(0)];

        auto batches = benchmarks["batches"].as<size_t>();
        auto shape = benchmarks["shape"].as<size3_t>();
        auto cutoff = benchmarks["cutoff"].as<float>();
        auto width = benchmarks["width"].as<float>();

        size_t elements = noa::elementsFFT(shape);
        size3_t pitch = noa::shapeFFT(shape);
        cpu::memory::PtrHost<T> h_input_result(elements * batches);
        cuda::memory::PtrDevicePadded<T> d_input_result({pitch.x, pitch.y * pitch.z, batches});

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(h_input_result.get(), h_input_result.elements(), randomizer);

        cuda::Stream stream;
        cuda::memory::copy(h_input_result.get(), pitch.x, d_input_result.get(), d_input_result.pitch(), pitch, batches, stream);

        for (auto _: state) {
            cuda::Event start, end;
            start.record(stream);
            cuda::fft::lowpass<fft::H2H, T>(d_input_result.get(), {d_input_result.pitch(), pitch.y, pitch.z},
                                            d_input_result.get(), {d_input_result.pitch(), pitch.y, pitch.z},
                                            shape, batches,
                                            cutoff, width, stream);
            end.record(stream);
            end.synchronize();
            state.SetIterationTime(cuda::Event::elapsed(start, end));
            ::benchmark::DoNotOptimize(d_input_result.get());
        }
    }

    template<typename T>
    void CUDA_fft_lowpass_inplace_contiguous(benchmark::State& state) {
        path_t path_base = benchmark::PATH_NOA_DATA / "fft";
        YAML::Node benchmarks = YAML::LoadFile(path_base / "benchmarks.yaml")["lowpass"][state.range(0)];

        auto batches = benchmarks["batches"].as<size_t>();
        auto shape = benchmarks["shape"].as<size3_t>();
        auto cutoff = benchmarks["cutoff"].as<float>();
        auto width = benchmarks["width"].as<float>();

        size_t elements = noa::elementsFFT(shape);
        size3_t pitch = noa::shapeFFT(shape);
        cpu::memory::PtrHost<T> h_input_result(elements * batches);
        cuda::memory::PtrDevice<T> d_input_result(h_input_result.size());

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(h_input_result.get(), h_input_result.elements(), randomizer);

        cuda::Stream stream;
        cuda::memory::copy(h_input_result.get(), pitch.x, d_input_result.get(), pitch.x, pitch, batches, stream);

        for (auto _: state) {
            cuda::Event start, end;
            start.record(stream);
            cuda::fft::lowpass<fft::H2H, T>(d_input_result.get(), pitch,
                                            d_input_result.get(), pitch,
                                            shape, batches,
                                            cutoff, width, stream);
            end.record(stream);
            end.synchronize();
            state.SetIterationTime(cuda::Event::elapsed(start, end));
            ::benchmark::DoNotOptimize(d_input_result.get());
        }
    }
}

// half_t is about the same as float. chalf_t is the same as half_t/float
BENCHMARK_TEMPLATE(CUDA_fft_lowpass_inplace, half_t)->ArgsProduct({{0, 2, 3, 5}})->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CUDA_fft_lowpass_inplace, float)->ArgsProduct({{0, 2, 3, 5}})->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CUDA_fft_lowpass_inplace, chalf_t)->ArgsProduct({{0, 2, 3, 5}})->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CUDA_fft_lowpass_inplace, cfloat_t)->ArgsProduct({{0, 2, 3, 5}})->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_TEMPLATE(CUDA_fft_lowpass_inplace_contiguous, half_t)->ArgsProduct({{0, 2, 3, 5}})->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CUDA_fft_lowpass_inplace_contiguous, float)->ArgsProduct({{0, 2, 3, 5}})->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CUDA_fft_lowpass_inplace_contiguous, chalf_t)->ArgsProduct({{0, 2, 3, 5}})->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CUDA_fft_lowpass_inplace_contiguous, cfloat_t)->ArgsProduct({{0, 2, 3, 5}})->Unit(benchmark::kMillisecond)->UseRealTime();
