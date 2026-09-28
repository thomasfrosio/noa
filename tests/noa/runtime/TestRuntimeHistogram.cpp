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

TEST_CASE("runtime: histogram, cpu vs gpu") {
    if (not Device::is_any_gpu())
        return;

    const auto inputs_cpu = noa::random<f32, 2>(noa::Normal(0.f, 1.f), {5, 100 * 100});
    noa::normalize_per_batch(inputs_cpu, inputs_cpu, {.mode = noa::Norm::MIN_MAX});

    const auto inputs_gpu = inputs_cpu.to({.device = "gpu", .allocator = "managed"});

    constexpr isize HISTOGRAM_SIZE = 128;
    const auto histograms_cpu = Array<i32, 2>({5, HISTOGRAM_SIZE});
    const auto histograms_gpu = Array<i32, 2>({5, HISTOGRAM_SIZE}, inputs_gpu.options());

    const auto shape = inputs_cpu.shape().as<i32>();
    const auto reduce_width = ReduceAxes<2>{}.reduce_axis(1);

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
