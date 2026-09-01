#include <noa/runtime/Factory.hpp>
#include <noa/runtime/Random.hpp>
#include <noa/runtime/Reduce.hpp>

#include <noa/fft/Transform.hpp>
#include <noa/fft/Remap.hpp>
#include <noa/signal/Standardize.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

using namespace ::noa::types;
using Norm = noa::fft::Norm;

TEST_CASE("signal::standardize_ifft(), rfft") {
    constexpr auto shape = Shape2{128, 128};

    Norm norm = GENERATE(Norm::FORWARD, Norm::BACKWARD, Norm::ORTHO, Norm::NONE);

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        StreamGuard stream(device);
        ArrayOption options(device, Allocator::MANAGED);
        INFO(device);

        const auto input = noa::random(noa::Normal{2.4f, 4.1f}, shape, options);
        const auto input_fft = noa::fft::rfft2(input, {.norm=norm});

        const auto input_fft_centered = noa::empty_like(input_fft);
        noa::fft::remap_2d("H2HC", input_fft, input_fft_centered, shape);

        noa::signal::standardize_ifft2<"HC2HC">(input_fft_centered, input_fft_centered, shape, {.norm = norm});

        noa::fft::remap_2d("HC2H", input_fft_centered, input_fft, shape);
        noa::fft::irfft2(input_fft, input, {.norm=norm});
        if (norm == Norm::NONE)
            noa::ewise(noa::wrap(input, 1 / static_cast<f32>(shape.n_elements())), input, noa::Multiply{});

        const auto mean = noa::mean(input);
        const auto std = noa::stddev(input, {.ddof = 0});
        REQUIRE_THAT(mean, Catch::Matchers::WithinAbs(0, 1e-6));
        REQUIRE_THAT(std, Catch::Matchers::WithinAbs(1, 1e-5));
    }
}

TEST_CASE("signal::standardize_ifft(), fft") {
    constexpr auto shape = Shape2{128, 128};

    Norm norm = GENERATE(Norm::FORWARD, Norm::BACKWARD, Norm::ORTHO, Norm::NONE);

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        StreamGuard stream(device);
        ArrayOption options(device, Allocator::MANAGED);
        INFO(device);

        const auto input = noa::random(noa::Normal{2.4f, 4.1f}, shape, options);
        const auto input_fft = noa::fft::rfft2(input, {.norm=norm});

        const auto input_full_centered = noa::empty<c32>(shape, options);
        noa::fft::remap_2d("H2FC", input_fft, input_full_centered, shape);

        noa::signal::standardize_ifft2<"FC2FC">(input_full_centered, input_full_centered, shape, {.norm = norm});

        noa::fft::remap_2d("FC2H", input_full_centered, input_fft, shape);
        noa::fft::irfft2(input_fft, input, {.norm=norm});
        if (norm == Norm::NONE)
            noa::ewise(noa::wrap(input, 1 / static_cast<f32>(shape.n_elements())), input, noa::Multiply{});

        const auto mean = noa::mean(input);
        const auto std = noa::stddev(input, {.ddof = 0});
        REQUIRE_THAT(mean, Catch::Matchers::WithinAbs(0, 1e-6));
        REQUIRE_THAT(std, Catch::Matchers::WithinAbs(1, 1e-5));
    }
}

TEMPLATE_TEST_CASE("signal::standardize_ifft()", "", f64) {
    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        StreamGuard stream(device, Stream::DEFAULT);
        ArrayOption options(device, Allocator::MANAGED);
        INFO(device);

        constexpr auto shape_options = test::RandomShapeOptions{.size_range = {64, 128}, .batch_range = {1, 4}};
        const auto shapes = noa::make_tuple(
            Pair{test::random_shape_batched<isize, 1>(1, shape_options), usize{1}},
            Pair{test::random_shape_batched<isize, 2>(2, shape_options), usize{2}},
            Pair{test::random_shape_batched<isize, 3>(1, shape_options), usize{1}},
            Pair{test::random_shape_batched<isize, 3>(2, shape_options), usize{2}},
            Pair{test::random_shape_batched<isize, 3>(3, shape_options), usize{3}},
            Pair{test::random_shape_batched<isize, 6>(1, shape_options), usize{1}},
            Pair{test::random_shape_batched<isize, 6>(2, shape_options), usize{2}},
            Pair{test::random_shape_batched<isize, 6>(3, shape_options), usize{3}}
        );
        shapes.for_each([&]<usize N>(const Pair<Shape<isize, N>, usize>& pair) {
            const auto& [shape, rank] = pair;

            for (i32 i{}; Norm norm: {Norm::FORWARD, Norm::BACKWARD, Norm::ORTHO, Norm::NONE}) {
                Array image = noa::random<TestType>(noa::Normal{2.5, 5.2}, shape, options);
                Array image_fft = noa::fft::r2c(image, {.rank = rank, .norm = norm});

                if (i < 2) {
                    auto image_fft_c = noa::fft::remap("h2fc", image_fft, shape, {.rank = rank});
                    noa::signal::standardize_ifft<"fc">(image_fft_c, image_fft_c, shape, {.rank = rank, .norm = norm});
                    noa::fft::remap("fc2h", image_fft_c, image_fft, shape, {.rank = rank});
                } else {
                    noa::signal::standardize_ifft<"h">(image_fft, image_fft, shape, {.rank = rank, .norm = norm});
                }
                i++;

                noa::fft::c2r(image_fft, image, {.rank = rank, .norm = norm});

                auto shape_ranked = shape.template as_nd<4>().ranked(rank);
                if (norm == Norm::NONE) {
                    const auto n_elements_per_batch = static_cast<TestType>(shape_ranked.pop_front().n_elements());
                    noa::ewise(noa::wrap(image, 1 / n_elements_per_batch), image, noa::Multiply{});
                }

                auto image_ranked = image.reshape(shape_ranked);
                INFO(shape);
                INFO(rank);
                for (i64 batch: noa::irange(shape_ranked[0])) {
                    const auto [mean, stddev] = noa::mean_stddev(image_ranked.subregion(batch));
                    REQUIRE_THAT(mean, Catch::Matchers::WithinAbs(0, 1e-6));
                    REQUIRE_THAT(stddev, Catch::Matchers::WithinAbs(1, 1e-4));
                    // For f32, the stddev in FC mode, the error can be slightly above 1e-4 error for bigger shapes.
                }
            }
        });
    }
}
