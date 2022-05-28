#include <benchmark/benchmark.h>

#include <noa/cpu/Stream.h>
#include <noa/cpu/math/Random.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/fft/Remap.h>

#include "Helpers.h"

using namespace ::noa;

namespace {
    template<typename T>
    void CPU_fft_remap(benchmark::State& state) {
        path_t path_base = benchmark::NOA_DATA_PATH / "fft";
        YAML::Node benchmarks = YAML::LoadFile(path_base / "benchmarks.yaml")["remap"][state.range(0)];

        using enum_t = std::underlying_type_t<fft::Remap>;
        const auto shape = benchmarks["shape"].as<size4_t>();
        const auto permutation = benchmarks["permutations"][state.range(1)].as<fft::Remap>();

        size4_t src_pitch = static_cast<enum_t>(permutation) & fft::SRC_FULL ? shape : shape.fft();
        size4_t dst_pitch = static_cast<enum_t>(permutation) & fft::DST_FULL ? shape : shape.fft();
        cpu::memory::PtrHost<T> src{src_pitch.elements()};
        cpu::memory::PtrHost<T> dst{dst_pitch.elements()};

        using real_t = traits::value_type_t<T>;
        cpu::Stream stream{cpu::Stream::DEFAULT};
        cpu::math::randomize(math::uniform_t{}, src.share(), src.elements(), real_t{-5}, real_t{5}, stream);

        for (auto _: state) {
            cpu::fft::remap(permutation, src.share(), src_pitch.stride(), dst.share(), dst_pitch.stride(), shape, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_fft_remap_h2hc(benchmark::State& state) {
        const size4_t shape = {1, 256, 256, 256};
        const size4_t stride = shape.fft().stride();

        cpu::memory::PtrHost<T> src{shape.fft().elements()};
        cpu::memory::PtrHost<T> dst{shape.fft().elements()};

        using real_t = traits::value_type_t<T>;
        cpu::Stream stream{cpu::Stream::DEFAULT};
        cpu::math::randomize(math::uniform_t{}, src.share(), src.elements(), real_t{-5}, real_t{5}, stream);

        for (auto _: state) {
            cpu::fft::remap(fft::H2HC, src.share(), stride, dst.share(), stride, shape, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_fft_remap_h2hc_inplace(benchmark::State& state) {
        const size4_t shape = {1, 256, 256, 256};
        const size4_t stride = shape.fft().stride();

        cpu::memory::PtrHost<T> src{shape.fft().elements()};

        using real_t = traits::value_type_t<T>;
        cpu::Stream stream{cpu::Stream::DEFAULT};
        cpu::math::randomize(math::uniform_t{}, src.share(), src.elements(), real_t{-5}, real_t{5}, stream);

        for (auto _: state) {
            cpu::fft::remap(fft::H2HC, src.share(), stride, src.share(), stride, shape, stream);
            ::benchmark::DoNotOptimize(src.get());
        }
    }

    void customArguments(benchmark::internal::Benchmark* b) {
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 10; ++j) // all permutations
                b->Args({i, j});
    }
}

// As expected, this is memory bound. The larger the type, the slower it gets.
BENCHMARK_TEMPLATE(CPU_fft_remap, float)->Apply(customArguments)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_fft_remap, cfloat_t)->Apply(customArguments)->Unit(benchmark::kMillisecond)->UseRealTime();

// In-place is ~30 to 50% faster
BENCHMARK_TEMPLATE(CPU_fft_remap_h2hc, half_t)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_fft_remap_h2hc_inplace, half_t)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_fft_remap_h2hc, float)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_fft_remap_h2hc_inplace, float)->Unit(benchmark::kMillisecond)->UseRealTime();
