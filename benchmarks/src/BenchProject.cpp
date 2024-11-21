#include <benchmark/benchmark.h>

#include <noa/Array.hpp>
#include <noa/unified/Event.hpp>
#include <noa/unified/Random.hpp>
#include <noa/unified/geometry/DrawShape.hpp>
#include <noa/unified/geometry/Project.hpp>
#include <noa/core/geometry/Euler.hpp>
#include <noa/core/utils/Zip.hpp>

using namespace ::noa::types;
namespace ng = noa::geometry;

namespace {
    void bench000_forward_project(benchmark::State& state) {
        StreamGuard stream{Device{"cpu"}, Stream::DEFAULT};
        stream.set_thread_limit(4);

        constexpr size_t n_images = 5;
        auto shifts = Array<Vec<f64, 2>>::from_values(
            Vec{23., 32.},
            Vec{33., -42.},
            Vec{-52., -22.},
            Vec{13., -18.},
            Vec{37., 17.}
        );

        constexpr i64 volume_depth = 80;
        constexpr auto volume_shape = Shape<i64, 3>{volume_depth, 256, 256};
        constexpr auto center = (volume_shape.vec / 2).as<f64>();
        const auto tilts = Array<f64>::from_values(-60, -30, 0, 30, 60);

        auto backward_projection_matrices = noa::empty<Mat<f64, 2, 4>>(n_images);
        auto forward_projection_matrices = noa::empty<Mat<f64, 3, 4>>(n_images);
        i64 projection_window_size{};

        for (auto&& [backward_matrix, forward_matrix, shift, tilt]: noa::zip(
                 backward_projection_matrices.span_1d(),
                 forward_projection_matrices.span_1d(),
                 shifts.span_1d(),
                 tilts.span_1d())) {
            auto matrix =
                    ng::translate((center.pop_front() + shift).push_front(0)) *
                    ng::linear2affine(ng::rotate_y(noa::deg2rad(tilt))) *
                    ng::translate(-center);

            backward_matrix = matrix.filter_rows(1, 2);
            forward_matrix = matrix.inverse().pop_back();

            projection_window_size = noa::max(
                projection_window_size,
                ng::forward_projection_window_size(volume_shape, forward_matrix)
            );
        }

        auto options = ArrayOption{.device = "cpu"};
        if (options.device != Device()) {
            backward_projection_matrices = std::move(backward_projection_matrices).to(options);
            forward_projection_matrices = std::move(forward_projection_matrices).to(options);
        }

        const auto input_volume = noa::empty<f32>(volume_shape.push_front(1), options);
        ng::draw_shape({}, input_volume, ng::Sphere{.center = center, .radius = 32., .smoothness = 0.});
        auto projected_images = noa::zeros<f32>({n_images, 1, 256, 256}, options);

        stream.synchronize();

        for (auto _: state) {
            // Event start, end;
            // start.record(stream);

            ng::forward_project_3d(
                input_volume, projected_images, forward_projection_matrices,
                projection_window_size, {.add_to_output = true}
            );

            // end.record(stream);
            // end.synchronize();
            // state.SetIterationTime(Event::elapsed(start, end).count());
            ::benchmark::DoNotOptimize(projected_images.get());
            ::benchmark::DoNotOptimize(input_volume.get());
        }
    }

    void bench001_backward_forward_project(benchmark::State& state) {
        constexpr size_t n_images = 5;
        const auto options = ArrayOption{.device = "cpu"};
        StreamGuard stream{options.device, Stream::DEFAULT};
        stream.set_thread_limit(4);

        const auto images = noa::empty<f32>({n_images, 1, 256, 256}, options);
        ng::draw_shape({}, images, ng::Sphere{.center = Vec{128., 128.}, .radius = 32., .smoothness = 5.});

        constexpr i64 volume_depth = 80;
        constexpr auto volume_shape = Shape<i64, 3>{volume_depth, 256, 256};
        constexpr auto center = (volume_shape.vec / 2).as<f64>();

        const auto tilts = std::array{-60., -30., 0., 30., 60.};
        auto backward_projection_matrices = noa::empty<Mat<f64, 2, 4>>(n_images);
        for (auto&& [backward_matrix, tilt]: noa::zip(backward_projection_matrices.span_1d(), tilts)) {
            backward_matrix = (
                ng::translate((center.pop_front()).push_front(0)) *
                ng::linear2affine(ng::rotate_y(noa::deg2rad(tilt))) *
                ng::translate(-center)
            ).filter_rows(1, 2);
        }

        auto forward_projection_matrix = (
            ng::translate((center.pop_front()).push_front(0)) *
            ng::linear2affine(ng::euler2matrix(noa::deg2rad(Vec{0., 30., 0.}), {.axes = "zyx", .intrinsic = false})) *
            ng::translate(-center)
        ).inverse().pop_back();
        i64 projection_window_size = ng::forward_projection_window_size(volume_shape, forward_projection_matrix);

        auto volume = noa::empty<f32>(volume_shape.push_front(1), options);
        auto projected_image = noa::zeros<f32>({1, 1, 256, 256}, options);
        if (options.device != Device())
            backward_projection_matrices = std::move(backward_projection_matrices).to(options);

        stream.synchronize();

        for (auto _: state) {
            // Event start, end;
            // start.record(stream);

            ng::backward_project_3d(images, volume, backward_projection_matrices);
            ng::forward_project_3d(
                volume, projected_image, forward_projection_matrix,
                projection_window_size, {.add_to_output = true}
            );
            // ng::backward_and_forward_project_3d(
            //     images, projected_image, volume_shape,
            //     backward_projection_matrices, forward_projection_matrix,
            //     projection_window_size, {.add_to_output = true}
            // );

            // end.record(stream);
            // end.synchronize();
            // state.SetIterationTime(Event::elapsed(start, end).count());
            ::benchmark::DoNotOptimize(volume.get());
            ::benchmark::DoNotOptimize(projected_image.get());
        }
    }
}

BENCHMARK(bench000_forward_project)->Unit(benchmark::kMillisecond)->UseRealTime();
// BENCHMARK(bench001_backward_forward_project)->Unit(benchmark::kMillisecond)->UseRealTime();
/*
2024-11-07T22:18:22+01:00
Running /home/thomas/Projects/noa/cmake-build-release-cpu/benchmarks/noa_benchmarks
Run on (32 X 4761.23 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x16)
  L1 Instruction 32 KiB (x16)
  L2 Unified 512 KiB (x16)
  L3 Unified 16384 KiB (x4)
Load Average: 1.73, 2.12, 2.00
***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
-----------------------------------------------------------------------------
Benchmark                                   Time             CPU   Iterations
-----------------------------------------------------------------------------
bench000_forward_project/real_time       1256 ms         1256 ms            1       -> 1 thread
bench000_forward_project/real_time        335 ms          299 ms            2       -> 4 threads
bench000_forward_project/real_time      0.188 ms        0.188 ms         3666       -> GPU

// normal
bench001_backward_forward_project/real_time        535 ms          535 ms            1      -> 1 thread
bench001_backward_forward_project/real_time        132 ms          130 ms            5      -> 4 threads
bench001_backward_forward_project/real_time       2.96 ms         2.96 ms          233      -> GPU

// fused
bench001_backward_forward_project/real_time        518 ms          518 ms            1      -> 1 thread
bench001_backward_forward_project/real_time        141 ms          128 ms            5      -> 4 threads
bench001_backward_forward_project/real_time      0.066 ms        0.066 ms         9639      -> GPU
*/
