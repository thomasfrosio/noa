#include <benchmark/benchmark.h>

#include <noa/cpu/Stream.h>
#include <noa/cpu/math/Random.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/memory/Copy.h>
#include <noa/cpu/fft/Resize.h>

#include "Helpers.h"

using namespace ::noa;

namespace {
    template<typename T>
    void CPU_fft_resize_cropH2H(benchmark::State& state) {
        const size4_t src_shape = {1, 512, 512, 512};
        const size4_t dst_shape = {1, 256, 256, 256};
        const size4_t src_stride = src_shape.fft().stride();
        const size4_t dst_stride = dst_shape.fft().stride();

        cpu::memory::PtrHost<T> src{src_stride[0]};
        cpu::memory::PtrHost<T> dst{dst_stride[0]};

        using real_t = traits::value_type_t<T>;
        cpu::Stream stream{cpu::Stream::DEFAULT};
        cpu::math::randomize(math::uniform_t{}, src.share(), src.elements(), real_t{-5}, real_t{5}, stream);

        for (auto _: state) {
            cpu::fft::resize<fft::H2H>(src.share(), src_stride, src_shape, dst.share(), dst_stride, dst_shape, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_fft_resize_cropF2F(benchmark::State& state) {
        const size4_t src_shape = {1, 512, 512, 512};
        const size4_t dst_shape = {1, 256, 256, 256};
        const size4_t src_stride = src_shape.stride();
        const size4_t dst_stride = dst_shape.stride();

        cpu::memory::PtrHost<T> src{src_stride[0]};
        cpu::memory::PtrHost<T> dst{dst_stride[0]};

        using real_t = traits::value_type_t<T>;
        cpu::Stream stream{cpu::Stream::DEFAULT};
        cpu::math::randomize(math::uniform_t{}, src.share(), src.elements(), real_t{-5}, real_t{5}, stream);

        for (auto _: state) {
            cpu::fft::resize<fft::F2F>(src.share(), src_stride, src_shape, dst.share(), dst_stride, dst_shape, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_fft_resize_padH2H(benchmark::State& state) {
        const size4_t src_shape = {1, 256, 256, 256};
        const size4_t dst_shape = {1, 512, 512, 512};
        const size4_t src_stride = src_shape.fft().stride();
        const size4_t dst_stride = dst_shape.fft().stride();

        cpu::memory::PtrHost<T> src{src_stride[0]};
        cpu::memory::PtrHost<T> dst{dst_stride[0]};

        using real_t = traits::value_type_t<T>;
        cpu::Stream stream{cpu::Stream::DEFAULT};
        cpu::math::randomize(math::uniform_t{}, src.share(), src.elements(), real_t{-5}, real_t{5}, stream);

        for (auto _: state) {
            cpu::fft::resize<fft::H2H>(src.share(), src_stride, src_shape, dst.share(), dst_stride, dst_shape, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_fft_resize_padF2F(benchmark::State& state) {
        const size4_t src_shape = {1, 256, 256, 256};
        const size4_t dst_shape = {1, 512, 512, 512};
        const size4_t src_stride = src_shape.stride();
        const size4_t dst_stride = dst_shape.stride();

        cpu::memory::PtrHost<T> src{src_stride[0]};
        cpu::memory::PtrHost<T> dst{dst_stride[0]};

        using real_t = traits::value_type_t<T>;
        cpu::Stream stream{cpu::Stream::DEFAULT};
        cpu::math::randomize(math::uniform_t{}, src.share(), src.elements(), real_t{-5}, real_t{5}, stream);

        for (auto _: state) {
            cpu::fft::resize<fft::F2F>(src.share(), src_stride, src_shape, dst.share(), dst_stride, dst_shape, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }
}

// Padding is ~3 to 5 times slower than cropping, which makes sense since the padding loops through the output
// which is two times larger than the input. Also, the implementation needs two passes, one through the output
// and one through the input. Otherwise, there's not much to say.
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
        const size4_t shape = {1, 256, 256, 256};
        const size4_t stride = shape.fft().stride();

        cpu::memory::PtrHost<T> src{stride[0]};
        cpu::memory::PtrHost<T> dst{stride[0]};

        using real_t = traits::value_type_t<T>;
        cpu::Stream stream{cpu::Stream::DEFAULT};
        cpu::math::randomize(math::uniform_t{}, src.share(), src.elements(), real_t{-5}, real_t{5}, stream);

        for (auto _: state) {
            cpu::fft::resize<fft::H2H>(src.share(), stride, shape, dst.share(), stride, shape, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_fft_resize_copyH2H(benchmark::State& state) {
        const size4_t shape = {1, 256, 256, 256};
        const size4_t stride = shape.fft().stride();

        cpu::memory::PtrHost<T> src{stride[0]};
        cpu::memory::PtrHost<T> dst{stride[0]};

        using real_t = traits::value_type_t<T>;
        cpu::Stream stream{cpu::Stream::DEFAULT};
        cpu::math::randomize(math::uniform_t{}, src.share(), src.elements(), real_t{-5}, real_t{5}, stream);

        for (auto _: state) {
            cpu::memory::copy(src.share(), stride, dst.share(), stride, shape.fft(), stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_fft_resize_copyF2F_trigger(benchmark::State& state) {
        const size4_t shape = {1, 256, 256, 256};
        const size4_t stride = shape.stride();

        cpu::memory::PtrHost<T> src{stride[0]};
        cpu::memory::PtrHost<T> dst{stride[0]};

        using real_t = traits::value_type_t<T>;
        cpu::Stream stream{cpu::Stream::DEFAULT};
        cpu::math::randomize(math::uniform_t{}, src.share(), src.elements(), real_t{-5}, real_t{5}, stream);

        for (auto _: state) {
            cpu::fft::resize<fft::F2F>(src.share(), stride, shape, dst.share(), stride, shape, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_fft_resize_copyF2F(benchmark::State& state) {
        const size4_t shape = {1, 256, 256, 256};
        const size4_t stride = shape.stride();

        cpu::memory::PtrHost<T> src{stride[0]};
        cpu::memory::PtrHost<T> dst{stride[0]};

        using real_t = traits::value_type_t<T>;
        cpu::Stream stream{cpu::Stream::DEFAULT};
        cpu::math::randomize(math::uniform_t{}, src.share(), src.elements(), real_t{-5}, real_t{5}, stream);

        for (auto _: state) {
            cpu::memory::copy(src.share(), stride, dst.share(), stride, shape, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }
}

// Nothing special to say. It is memory bound...
// Interestingly, if the copy is in-place with a contiguous array, the implementation calls std::copy() which
// detects that the copy is useless, and simply returns.
BENCHMARK_TEMPLATE(CPU_fft_resize_copyH2H_trigger, float)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_fft_resize_copyH2H, float)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_TEMPLATE(CPU_fft_resize_copyF2F_trigger, float)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_fft_resize_copyF2F, float)->Unit(benchmark::kMillisecond)->UseRealTime();
