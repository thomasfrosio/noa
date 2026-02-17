#include <noa/Runtime.hpp>
#include <noa/IO.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

using namespace noa::types;
namespace nx = noa::xform;
namespace ns = noa::signal;
namespace nf = noa::fft;
namespace nt = noa::traits;

namespace {
    struct Histogram {
        SpanContiguous<const f32, 2, i32> inputs; // (n,w)
        SpanContiguous<i32, 2, i32> histograms; // (n,b)

        static constexpr void init(nt::compute_handle auto& handle) {
            // Zero-initialize the per-block histogram if it exists.
            const auto& block = handle.block();
            block.template zeroed_scratch<i32>();
            block.synchronize();
        }

        constexpr void operator()(nt::compute_handle auto& handle, i32 b, i32 i) const {
            // Compute the bin of the current value.
            // For simplicity, assume values are between [0,1].
            const auto n_bins = histograms.shape()[1];
            const auto value_scaled = inputs(b, i) * static_cast<f32>(n_bins);
            auto bin = static_cast<i32>(noa::round(value_scaled));
            bin = noa::clamp(bin, 0, n_bins - 1);

            // Increment the bin count.
            // If the block has its own histogram, increment it
            // instead of incrementing the global histogram.
            const auto& grid = handle.grid();
            const auto& block = handle.block();
            if (block.has_scratch()) {
                auto scratch = block.template scratch<i32>();
                grid.atomic_add(1, scratch, bin);
            } else {
                grid.atomic_add(1, histograms[b], bin);
            }
        }

        constexpr void deinit(nt::compute_handle auto& handle, i32 b) const {
            const auto& block = handle.block();
            const auto& thread = handle.thread();
            if (not block.has_scratch())
                return;

            // If the block has its own histogram, add it to the global histogram.
            block.synchronize();
            const auto& grid = handle.grid();
            auto scratch = block.template scratch<i32>();
            for (i32 i = thread.lid(); i < scratch.n_elements(); i += block.size())
                grid.atomic_add(scratch[i], histograms, b, i);
        }
    };
}

TEST_CASE("runtime: histogram", "[asset]") {
    // const auto path = test::NOA_DATA_PATH / "runtime";
    // const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["reduce_to_stats"];
    //
    // const YAML::Node& input = tests["input"];
    // const auto shape = input["shape"].as<Shape4>();
    // const auto input_filename = path / input["path"].as<Path>();
    // const auto output_filename = path / tests["all"]["output_path"].as<Path>();
    //
    // const YAML::Node expected = YAML::LoadFile(output_filename);
    // const auto expected_max = expected["max"].as<f64>();
    // const auto expected_min = expected["min"].as<f64>();
    // const auto expected_median = expected["median"].as<f64>();
    // const auto expected_mean = expected["mean"].as<f64>();
    // const auto expected_norm = expected["norm"].as<f64>();
    // const auto expected_std = expected["std"].as<f64>();
    // const auto expected_sum = expected["sum"].as<f64>();
    // const auto expected_var = expected["var"].as<f64>();
    //
    // auto data = noa::read_image<f64>(input_filename).data;
    // REQUIRE(data.shape() == shape);
    //
    // std::vector<Device> devices{"cpu"};
    // if (Device::is_any_gpu())
    //     devices.emplace_back("gpu");
    //
    // for (auto& device: devices) {
    //     const auto stream = StreamGuard(device);
    //     const auto options = ArrayOption(device, "managed");
    //     INFO(device);
    //     data = device.is_cpu() ? data : data.to(options);
    //
    //     const auto min = noa::min(data);
    //     const auto max = noa::max(data);
    //     const auto min_max = noa::min_max(data);
    //     const auto median = noa::median(data);
    //     const auto sum = noa::sum(data);
    //     const auto mean = noa::mean(data);
    //     const auto norm = noa::l2_norm(data);
    //     const auto var = noa::variance(data);
    //     const auto std = noa::stddev(data);
    //     const auto mean_var = noa::mean_variance(data);
    //     const auto mean_std = noa::mean_stddev(data);
    //
    //     REQUIRE(noa::allclose(min, expected_min));
    //     REQUIRE(noa::allclose(max, expected_max));
    //     REQUIRE(noa::allclose(min_max.first, expected_min));
    //     REQUIRE(noa::allclose(min_max.second, expected_max));
    //     REQUIRE(noa::allclose(median, expected_median));
    //     REQUIRE(noa::allclose(sum, expected_sum));
    //     REQUIRE(noa::allclose(mean, expected_mean));
    //     REQUIRE(noa::allclose(norm, expected_norm));
    //     REQUIRE(noa::allclose(var, expected_var));
    //     REQUIRE(noa::allclose(std, expected_std));
    //     REQUIRE(noa::allclose(mean_var.first, expected_mean));
    //     REQUIRE(noa::allclose(mean_var.second, expected_var));
    //     REQUIRE(noa::allclose(mean_std.first, expected_mean));
    //     REQUIRE(noa::allclose(mean_std.second, expected_std));
    // }
}

TEST_CASE("runtime: histogram, cpu vs gpu") {
    if (not Device::is_any_gpu())
        return;

    const auto inputs_cpu = noa::random<f32>(noa::Normal(0.f, 1.f), {5, 1, 1, 100 * 100});
    noa::normalize_per_batch(inputs_cpu, inputs_cpu, {.mode = noa::Norm::MIN_MAX});

    const auto inputs_gpu = inputs_cpu.to({.device = "gpu", .allocator = "managed"});

    constexpr isize HISTOGRAM_SIZE = 128;
    const auto histograms_cpu = Array<i32>({5, 1, 1, HISTOGRAM_SIZE});
    const auto histograms_gpu = Array<i32>({5, 1, 1, HISTOGRAM_SIZE}, inputs_gpu.options());

    const auto shape = inputs_cpu.shape().filter(0, 3).as<i32>();
    const auto reduce_width = ReduceAxes{.width = true};

    // 1. Same implementation.
    noa::fill(histograms_cpu, 0);
    noa::reduce_axes_iwise(shape, inputs_cpu.device(), {}, reduce_width, Histogram{
        .inputs = inputs_cpu.span_contiguous<const f32, 2, i32>(),
        .histograms = histograms_cpu.span_contiguous<i32, 2, i32>(),
    });
    noa::fill(histograms_gpu, 0);
    noa::reduce_axes_iwise(shape, inputs_gpu.device(), {}, reduce_width, Histogram{
        .inputs = inputs_gpu.span_contiguous<const f32, 2, i32>(),
        .histograms = histograms_gpu.span_contiguous<i32, 2, i32>(),
    });
    REQUIRE(test::allclose_abs(histograms_cpu, histograms_gpu));

    // 2. Use scratch implementation for GPU.
    noa::fill(histograms_gpu, 0);
    constexpr auto OPTIONS = noa::ReduceIwiseOptions{
        .generate_cpu = false,
        .gpu_block_shape = {1, HISTOGRAM_SIZE * 4}, // 1d block
        .gpu_optimize_block_shape = false, // enforce the block shape
        .gpu_number_of_indices_per_threads = {1, 4}, // increase the value of the per-block histogram by working on it more
        .gpu_scratch_size = HISTOGRAM_SIZE * sizeof(i32), // per block histogram
    };
    noa::reduce_axes_iwise<OPTIONS>(shape, inputs_gpu.device(), {}, reduce_width, Histogram{
        .inputs = inputs_gpu.span_contiguous<const f32, 2, i32>(),
        .histograms = histograms_gpu.span_contiguous<i32, 2, i32>(),
    });
    REQUIRE(test::allclose_abs(histograms_cpu, histograms_gpu));
}
