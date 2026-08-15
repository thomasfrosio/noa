#include <benchmark/benchmark.h>

#include <noa/runtime/Event.hpp>
#include <noa/runtime/Array.hpp>
#include <noa/runtime/Random.hpp>
#include <noa/runtime/Factory.hpp>
#include <noa/fft/Transform.hpp>
#include <noa/xform/Transform.hpp>
#include <noa/xform/TransformSpectrum.hpp>
#include <noa/xform/Texture.hpp>
#include <noa/IO.hpp>

namespace nx = noa::xform;
using namespace noa::types;

// namespace {
//     template<typename Interpolator, typename Input, typename Output, typename Xform>
//     struct Transform2D {
//         Interpolator interpolator;
//         Input input;
//         Output output;
//         Xform xforms;
//         constexpr void operator()(isize b, isize y, isize x) const {
//             auto coordinates = Vec<f32, 2>::from_values(y, x);
//             coordinates = xforms[b].pop_back() * coordinates.push_back(1);
//             output(b, y, x) = interpolator.get(input[b], coordinates);
//         }
//     };
//
//     template<typename T, bool GPU>
//     void transform_2d_many_iwise(benchmark::State& state) {
//         const auto shape = noa::Shape3{5, 4096, 4096};
//
//         auto device = noa::Device(GPU ? "gpu" : "cpu");
//         auto stream = noa::StreamGuard(device, noa::Stream::ASYNC);
//         stream.set_thread_limit(10);
//
//         auto src = noa::random<T>(noa::Uniform(T{-5}, T{5}), shape, {.device = device, .allocator = noa::Allocator::ASYNC});
//         auto dst = noa::empty_like(src);
//
//         stream.synchronize();
//
//         const auto rotation = noa::deg2rad(45.);
//         const auto center = (shape.pop_front().vec / 2).as<noa::f64>();
//         const auto xform = (
//             nx::translate(center) *
//             nx::rotate<true>(rotation) *
//             nx::translate(-center)
//         ).as<noa::f32>();
//         auto xforms = noa::empty<noa::Mat<noa::f32, 3, 3>>(shape[0]);
//         for (auto& e: xforms.span_1d())
//             e = xform;
//         xforms = std::move(xforms).to(src.options());
//
//         for (auto _: state) {
//             noa::Event start, end;
//             start.record(stream);
//
//             using index_t = std::conditional_t<GPU, noa::i32, noa::isize>;
//             using interpolator_t = noa::xform::Interpolator<noa::xform::Interp::LINEAR, noa::Border::ZERO, f32, 2, index_t, false>;
//             auto interpolator = interpolator_t(shape.pop_front().as<index_t>());
//             auto op = Transform2D{
//                 .interpolator = interpolator,
//                 .input = src.template span_contiguous<const f32, 3, index_t>().accessor(),
//                 .output = dst.template span_contiguous<f32, 3, index_t>().accessor(),
//                 .xforms = xforms.span_1d().as_const().as_index<index_t>().accessor(),
//             };
//             noa::iwise(shape, device, op, src.view(), dst.view());
//
//             end.record(stream);
//             end.synchronize();
//             state.SetIterationTime(noa::Event::elapsed(start, end).count());
//             ::benchmark::DoNotOptimize(dst.get());
//         }
//     }
//
//     template<typename T, bool GPU>
//     void transform_2d_many(benchmark::State& state) {
//         const auto shape = noa::Shape3{5, 4096, 4096};
//         const auto interp = noa::xform::Interp::LINEAR_FAST;
//         const auto border = noa::Border::ZERO;
//
//         auto device = noa::Device(GPU ? "gpu" : "cpu");
//         auto stream = noa::StreamGuard(device, noa::Stream::ASYNC);
//         stream.set_thread_limit(10);
//
//         auto src = noa::random<T>(noa::Uniform(T{-5}, T{5}), shape, {.device = device, .allocator = noa::Allocator::ASYNC});
//         auto tex = noa::xform::Texture2D<f32, 3>(src, device, interp, {.border = border});
//         auto dst = noa::empty_like(src);
//
//         stream.synchronize();
//
//         std::shared_ptr<f32[]> a;
//         std::shared_ptr<const void> b = a;
//
//         const auto rotation = noa::deg2rad(45.);
//         const auto center = (shape.pop_front().vec / 2).as<noa::f64>();
//         const auto xform = (
//             nx::translate(center) *
//             nx::rotate<true>(rotation) *
//             nx::translate(-center)
//         ).as<noa::f32>();
//         auto xforms = noa::empty<noa::Mat<noa::f32, 3, 3>>(shape[0]);
//         for (auto& e: xforms.span_1d())
//             e = xform;
//         xforms = std::move(xforms).to(src.options());
//
//         for (auto _: state) {
//             noa::Event start, end;
//             start.record(stream);
//
//             nx::transform_2d(tex.view(), dst.view(), xforms.view(), {.interp = interp, .border = border});
//
//             end.record(stream);
//             end.synchronize();
//             state.SetIterationTime(noa::Event::elapsed(start, end).count());
//             ::benchmark::DoNotOptimize(dst.get());
//         }
//     }
// }

// No differences between manual iwise and transform_2d. Good!
// BENCHMARK_TEMPLATE(transform_2d_many, float, true)->Unit(benchmark::kMillisecond)->UseRealTime();
// BENCHMARK_TEMPLATE(transform_2d_many_iwise, float, true)->Unit(benchmark::kMillisecond)->UseRealTime();
// BENCHMARK_TEMPLATE(transform_2d_many, float, false)->Unit(benchmark::kMillisecond)->UseRealTime();
// BENCHMARK_TEMPLATE(transform_2d_many_iwise, float, false)->Unit(benchmark::kMillisecond)->UseRealTime();

//transform_2d_many<float, true>/real_time  1.69 ms // array
//transform_2d_many<float, true>/real_time  1.44 ms // GPU texture              - 15% faster
//transform_2d_many<float, true>/real_time  1.24 ms // GPU texture, LINEAR_FAST - 25% faster

namespace {
    constexpr nx::Interp g_interp[] = {
        nx::Interp::NEAREST,
        nx::Interp::LINEAR,
        nx::Interp::CUBIC,
        nx::Interp::CUBIC_BSPLINE,
        nx::Interp::LANCZOS4,
        nx::Interp::LANCZOS6,
    };

    // constexpr noa::Border g_border[] = {
    //     noa::Border::ZERO,
    //     noa::Border::PERIODIC,
    // };
    //
    // template<typename T, bool GPU>
    // void transform_2d(benchmark::State& state) {
    //     const auto shape = noa::Shape3{5, GPU ? 4096 : 256, GPU ? 4096 : 256};
    //     const auto interp = g_interp[state.range(0)];
    //     const auto border = g_border[state.range(1)];
    //
    //     auto device = noa::Device(GPU ? "gpu" : "cpu");
    //     auto stream = noa::StreamGuard(device, noa::Stream::ASYNC);
    //     stream.set_thread_limit(1);
    //
    //     auto src = noa::random<T>(noa::Uniform(T{-5}, T{5}), shape, {.device = device, .allocator = noa::Allocator::ASYNC});
    //     auto dst = noa::empty_like(src);
    //
    //     stream.synchronize();
    //
    //     const auto rotation = noa::deg2rad(45.);
    //     const auto center = (shape.pop_front().vec / 2).as<noa::f64>();
    //     const auto xform = (
    //         nx::translate(center) *
    //         nx::rotate<true>(rotation) *
    //         nx::translate(-center)
    //     ).as<noa::f32>();
    //     for (auto _: state) {
    //         noa::Event start, end;
    //         start.record(stream);
    //
    //         nx::transform_2d(src.view(), dst.view(), xform, {.interp = interp, .border = border});
    //
    //         end.record(stream);
    //         end.synchronize();
    //         state.SetIterationTime(noa::Event::elapsed(start, end).count());
    //         ::benchmark::DoNotOptimize(dst.get());
    //     }
    // }

    template<typename T, bool GPU>
    void transform_spectrum_2d(benchmark::State& state) {
        const auto shape = noa::Shape3{5, GPU ? 4096 : 256, GPU ? 4096 : 256};
        const auto interp = g_interp[state.range(0)];

        auto device = noa::Device(GPU ? "gpu" : "cpu");
        auto stream = noa::StreamGuard(device, noa::Stream::DEFAULT);
        stream.set_thread_limit(1);

        auto src = noa::random<T>(noa::Uniform(T{-5}, T{5}), shape, {
            .device = device,
            .allocator = noa::Allocator::ASYNC,
        });
        auto src_rfft = noa::fft::rfft2(src);
        auto dst_rfft = noa::empty_like(src_rfft);

        stream.synchronize();

        const auto rotation = noa::deg2rad(45.);
        const auto xform = nx::rotate(rotation).as<noa::f32>();

        stream.synchronize();
        for (int i = 0; i < 2; ++i) {
            nx::transform_spectrum_2d<"h2h">(
                src_rfft.view(), dst_rfft.view(), shape, xform, {},
                {.interp = interp, .fftfreq_cutoff = 0.45});
        }
        stream.synchronize();

        for (auto _: state) {
            noa::Event start, end;
            start.record(stream);

            nx::transform_spectrum_2d<"h2h">(
                src_rfft.view(), dst_rfft.view(), shape, xform, {}, {.interp = interp, .fftfreq_cutoff=0.45});

            end.record(stream);
            end.synchronize();

            auto ms = static_cast<f64>(noa::Event::elapsed(start, end).count());
            state.SetIterationTime(ms * 1e-3);

            ::benchmark::DoNotOptimize(dst_rfft.get());
        }
    }
}

BENCHMARK_TEMPLATE(transform_spectrum_2d, float, true)
    ->DenseRange(0, 5)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();
BENCHMARK_TEMPLATE(transform_spectrum_2d, float, false)
    ->DenseRange(0, 5)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

// 1:always-inbound
// 2:always-flip-per-index
// 3:optimistic-inbound-check
//                                           old        --1        1-1       ---
// transform_spectrum_2d<float, true>/0    1.07 ms   0.599 ms   0.600 ms   0.599 ms
// transform_spectrum_2d<float, true>/1    1.08 ms   0.646 ms    1.08 ms   0.623 ms
// transform_spectrum_2d<float, true>/2    3.56 ms    2.30 ms    1.76 ms    2.25 ms
// transform_spectrum_2d<float, true>/3    3.53 ms    2.27 ms    1.76 ms    2.24 ms
// transform_spectrum_2d<float, true>/4    4.66 ms    3.68 ms    2.38 ms    3.64 ms
// transform_spectrum_2d<float, true>/5    8.41 ms    6.61 ms    6.40 ms    6.56 ms
//
// transform_spectrum_2d<float, false>/0   2.35 ms   0.635 ms   0.647 ms   0.638 ms
// transform_spectrum_2d<float, false>/1   2.98 ms    1.83 ms    1.40 ms    1.30 ms
// transform_spectrum_2d<float, false>/2   5.57 ms    3.72 ms    4.68 ms    4.30 ms
// transform_spectrum_2d<float, false>/3   5.13 ms    2.82 ms    3.86 ms    3.50 ms
// transform_spectrum_2d<float, false>/4   8.84 ms    6.45 ms    7.24 ms    6.54 ms
// transform_spectrum_2d<float, false>/5   13.0 ms    8.95 ms    10.3 ms    8.12 ms

// BENCHMARK_TEMPLATE(transform_2d, float, true)
//     ->ArgsProduct({{0, 1, 2, 3, 4, 5}, {1}})
//     ->Unit(benchmark::kMillisecond)->UseRealTime();
// BENCHMARK_TEMPLATE(transform_2d, float, false)
//     ->ArgsProduct({{0, 1, 2, 3, 4, 5}, {1}})
//     ->Unit(benchmark::kMillisecond)->UseRealTime();

/*
5x4096x4096 - GPU                 old    new(nooob)  noexpli   no/iob  nooob-noexpli  no/iob-noexplicit
transform_2d<float, true>/0/0   1.49 ms    1.49 ms    1.50 ms    1.50 ms   1.49 ms    1.50 ms
transform_2d<float, true>/1/0   1.62 ms    1.63 ms    1.63 ms    1.62 ms   1.61 ms    1.61 ms
transform_2d<float, true>/2/0   3.94 ms    3.28 ms    3.36 ms    4.12 ms   3.31 ms    3.79 ms
transform_2d<float, true>/3/0   3.86 ms    3.23 ms    3.29 ms    4.04 ms   3.25 ms    3.79 ms
transform_2d<float, true>/4/0   10.6 ms    4.93 ms    5.02 ms    10.7 ms   4.98 ms    5.91 ms
transform_2d<float, true>/5/0   19.8 ms    14.5 ms    14.4 ms    19.3 ms   14.5 ms    19.2 ms
transform_2d<float, true>/0/1   1.50 ms    1.49 ms    1.51 ms    1.50 ms   1.50 ms    1.50 ms
transform_2d<float, true>/1/1   1.75 ms    1.64 ms    1.65 ms    1.40 ms   1.64 ms    1.39 ms
transform_2d<float, true>/2/1   3.86 ms    3.24 ms    2.88 ms    2.64 ms   2.87 ms    2.62 ms
transform_2d<float, true>/3/1   3.85 ms    3.24 ms    2.88 ms    2.63 ms   2.87 ms    2.64 ms
transform_2d<float, true>/4/1   5.51 ms    4.72 ms    4.68 ms    4.04 ms   4.66 ms    4.05 ms
transform_2d<float, true>/5/1   15.9 ms    7.78 ms    7.78 ms    6.12 ms   7.76 ms    6.15 ms

// TODO benchmark PERIODIC
GPU:
    - ZERO: best is check inbound, no-oob, no explicit.
    - CLAMP: best is no-inbound, no-oob, no explicit.

CPU:
    - ZERO: best is check inbound, no-oob, no explicit.
    - CLAMP: best is check inbound, check or no-oob is same, no explicit.

5x256x256 - CPU                       old       new      noexpli    noo/iob     nooob  nooob-noexpli  no/iob-noexplicit
transform_2d<float, false>/0/0      10.8 ms    1.78 ms    1.75 ms    1.78 ms   1.77 ms   1.77 ms     1.78 ms
transform_2d<float, false>/1/0      2.67 ms    7.66 ms    6.81 ms    2.57 ms   2.44 ms   2.57 ms     2.62 ms
transform_2d<float, false>/2/0      32.6 ms    17.8 ms    15.3 ms    29.9 ms   14.8 ms   12.1 ms     23.4 ms
transform_2d<float, false>/3/0      20.8 ms    14.7 ms    12.6 ms    27.6 ms   12.4 ms   10.4 ms     12.0 ms
transform_2d<float, false>/4/0      32.4 ms    24.2 ms    22.7 ms    39.5 ms   25.1 ms   20.6 ms     25.2 ms
transform_2d<float, false>/5/0      52.5 ms    29.3 ms    29.2 ms    32.5 ms   27.8 ms   27.8 ms     32.6 ms
transform_2d<float, false>/0/1      1.79 ms    1.81 ms    1.79 ms    1.77 ms   1.82 ms   1.76 ms     1.78 ms
transform_2d<float, false>/1/1      2.68 ms    2.48 ms    2.59 ms    2.56 ms   2.44 ms   2.60 ms     2.69 ms
transform_2d<float, false>/2/1      14.5 ms    15.8 ms    13.2 ms    38.7 ms   15.7 ms   12.9 ms     14.0 ms
transform_2d<float, false>/3/1      12.1 ms    13.6 ms    9.21 ms    36.2 ms   13.3 ms   11.6 ms     12.0 ms
transform_2d<float, false>/4/1      24.9 ms    25.9 ms    20.3 ms    48.9 ms   26.0 ms   20.1 ms     24.2 ms
transform_2d<float, false>/5/1      29.1 ms    27.0 ms    27.0 ms    29.3 ms   27.3 ms   27.3 ms     29.5 ms

 */

