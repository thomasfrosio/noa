#include <benchmark/benchmark.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/Stream.h>
#include <noa/gpu/cuda/Event.h>
#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/fft/Filters.h>

#include "Helpers.h"

using namespace ::noa;

namespace {
    template<typename T>
    void CUDA_fft_lowpass_inplace(benchmark::State& state) {
        const path_t path_base = benchmark::NOA_DATA_PATH / "fft";
        const YAML::Node benchmarks = YAML::LoadFile(path_base / "benchmarks.yaml")["lowpass"][state.range(0)];

        const auto shape = benchmarks["shape"].as<size4_t>();
        const auto cutoff = benchmarks["cutoff"].as<float>();
        const auto width = benchmarks["width"].as<float>();

        const size4_t stride = shape.fft().stride();
        const size_t elements = shape.fft().elements();
        cpu::memory::PtrHost<T> h_input_result(elements);
        cuda::memory::PtrDevice<T> d_input_result(elements);

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(h_input_result.get(), h_input_result.elements(), randomizer);

        cuda::Stream stream(cuda::Stream::DEFAULT);
        cuda::memory::copy(h_input_result.get(), stride, d_input_result.get(), stride, shape.fft(), stream);

        for (auto _: state) {
            cuda::Event start, end;
            start.record(stream);
            cuda::fft::lowpass<fft::H2H, T>(d_input_result.get(), stride,
                                            d_input_result.get(), stride,
                                            shape, cutoff, width, stream);
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
