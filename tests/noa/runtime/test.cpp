#include <noa/Runtime.hpp>
#include <noa/Xform.hpp>
#include <noa/Signal.hpp>
#include <noa/FFT.hpp>
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
        Span<const f32, 2, i32> inputs; // (n,w)
        Span<i32, 2, i32> histograms; // (n,b)

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

    // struct MyConvolution {
    //     Span<f32, 3, i32> input_images; // (n,h,w)
    //     Span<f32, 3, i32> output_images; // (n,h,w)
    //     Span<f32, 2, i32> kernel;
    //
    //     static constexpr i32 KERNEL_SIZE = 11;
    //     static constexpr i32 PADDING = KERNEL_SIZE - 1;
    //     static constexpr i32 HALO = PADDING / 2;
    //
    //     static constexpr i32 GPU_BLOCK_SIZE = 16;
    //     static constexpr i32 GPU_SCRATCH_SIZE = GPU_BLOCK_SIZE + PADDING;
    //
    //     constexpr void operator()(nt::compute_handle_cpu auto&, i32 b, i32 i, i32 j) {
    //         const auto image_shape = input_images.shape().filter(1, 2);
    //
    //         f32 value{};
    //         for (i32 wk{}; wk < kernel.shape()[0]; ++wk) {
    //             for (i32 wl{}; wl < kernel.shape()[1]; ++wl) {
    //                 const i32 ik = i - HALO + wk;
    //                 const i32 il = j - HALO + wl;
    //                 if (noa::is_inbound(image_shape, ik, il))
    //                     value += input_images(b, ik, il) * kernel(wk, wl);
    //             }
    //         }
    //         output_images(b, i, j) = value;
    //     }
    //
    //     constexpr void operator()(nt::compute_handle_gpu auto& handle, i32 b, i32 i, i32 j) {
    //         const auto image_shape = input_images.shape().filter(1, 2);
    //         const auto block = handle.block();
    //         const auto scratch = SpanContiguous(
    //             reinterpret_cast<f32*>(block.scratch().data()),
    //             Shape{GPU_SCRATCH_SIZE, GPU_SCRATCH_SIZE}
    //         );
    //
    //         // Write to the scratch the input values that are inside the block's convolution window.
    //         const auto tid = handle.thread().template id<2>();
    //         for (i32 ly = tid[0], gy = i; ly < GPU_SCRATCH_SIZE; ly += GPU_BLOCK_SIZE, gy += GPU_BLOCK_SIZE) {
    //             for (i32 lx = tid[1], gx = j; lx < GPU_SCRATCH_SIZE; lx += GPU_BLOCK_SIZE, gx += GPU_BLOCK_SIZE) {
    //                 const i32 iy = gy - HALO;
    //                 const i32 ix = gx - HALO;
    //
    //                 f32 value{};
    //                 if (noa::is_inbound(image_shape, iy, ix))
    //                     value = input_images(b, iy, ix);
    //                 scratch(ly, lx) = value;
    //             }
    //         }
    //         block.synchronize();
    //
    //         // Convolve at location ij.
    //         if (i < image_shape[0] and j < image_shape[1]) {
    //             f32 result{};
    //             for (i32 y = 0; y < KERNEL_SIZE; ++y)
    //                 for (i32 x = 0; x < KERNEL_SIZE; ++x)
    //                     result += scratch(tid[0] + y, tid[1] + x) * kernel(y, x);
    //             output_images(b, i, j) = result;
    //         }
    //     }
    // };
}

TEST_CASE("runtime: tests") {
    auto input = Array<f32>({1, 1, 512, 512}, {.device = "gpu"});
    auto input_rfft = Array<c32>({1, 1, 512, 257}, {.device = "gpu"});
    nf::r2c(input, input_rfft, {.record_and_share_workspace = true});
    nf::c2r(input_rfft, input, {.record_and_share_workspace = true});
    nf::allocate_workspace(input.device(), Allocator::ASYNC);
    nf::r2c(input, input_rfft);
}

// TEST_CASE("runtime: tests") {
    // if (not Device::is_any_gpu())
    //     return;
    //
    // const auto inputs_cpu = noa::random<f32>(noa::Normal(0.f, 1.f), {5, 1, 1, 100 * 100});
    // noa::normalize_per_batch(inputs_cpu, inputs_cpu, {.mode = noa::Norm::MIN_MAX});
    //
    // const auto inputs_gpu = inputs_cpu.to({.device = "gpu", .allocator = "managed"});
    //
    // constexpr isize HISTOGRAM_SIZE = 128;
    // const auto histograms_cpu = Array<i32>({5, 1, 1, HISTOGRAM_SIZE});
    // const auto histograms_gpu = Array<i32>({5, 1, 1, HISTOGRAM_SIZE}, inputs_gpu.options());
    //
    // const auto shape = inputs_cpu.shape().filter(0, 3).as<i32>();
    // const auto reduce_width = ReduceAxes{.width = true};
    //
    // // 1. Same implementation.
    // noa::fill(histograms_cpu, 0);
    // noa::reduce_axes_iwise(shape, inputs_cpu.device(), {}, reduce_width, Histogram{
    //     .inputs = inputs_cpu.span<const f32, 2, i32>(),
    //     .histograms = histograms_cpu.span<i32, 2, i32>(),
    // });
    // noa::fill(histograms_gpu, 0);
    // noa::reduce_axes_iwise(shape, inputs_gpu.device(), {}, reduce_width, Histogram{
    //     .inputs = inputs_gpu.span<const f32, 2, i32>(),
    //     .histograms = histograms_gpu.span<i32, 2, i32>(),
    // });
    // REQUIRE(test::allclose_abs(histograms_cpu, histograms_gpu));
    //
    // // 2. Use scratch implementation for GPU.
    // noa::fill(histograms_gpu, 0);
    // constexpr auto OPTIONS = noa::ReduceIwiseOptions{
    //     .generate_cpu = false,
    //     .gpu_block_shape = {1, HISTOGRAM_SIZE * 4}, // 1d block
    //     .gpu_optimize_block_shape = false, // enforce the block shape
    //     .gpu_number_of_indices_per_threads = {1, 4}, // increase the value of the per-block histogram by working on it more
    //     .gpu_scratch_size = HISTOGRAM_SIZE * sizeof(i32), // per block histogram
    // };
    // noa::reduce_axes_iwise<OPTIONS>(shape, inputs_gpu.device(), {}, reduce_width, Histogram{
    //     .inputs = inputs_gpu.span<const f32, 2, i32>(),
    //     .histograms = histograms_gpu.span<i32, 2, i32>(),
    // });
    // REQUIRE(test::allclose_abs(histograms_cpu, histograms_gpu));

    //
    // {
    //     auto shape = Shape{5, 100, 100};
    //     Span<f32, 3, i32> input_images;
    //     Span<f32, 3, i32> output_images;
    //     Span<f32, 2, i32> kernel;
    //
    //     constexpr auto OPTIONS = noa::IwiseOptions{
    //         .gpu_block_size = MyConvolution::GPU_BLOCK_SIZE,
    //         .gpu_scratch_size = MyConvolution::GPU_SCRATCH_SIZE,
    //     };
    //     shape[1] = noa::next_multiple_of(shape[1], MyConvolution::GPU_BLOCK_SIZE);
    //     shape[2] = noa::next_multiple_of(shape[2], MyConvolution::GPU_BLOCK_SIZE);
    //     noa::iwise<OPTIONS>(shape, Device{}, MyConvolution{
    //         .input_images = input_images,
    //         .output_images = output_images,
    //         .kernel = kernel,
    //     });
    // }
// }

// namespace {
//     using namespace noa::types;
//     namespace nf = noa::fft;
//     namespace nx = noa::xform;
//     namespace ns = noa::signal;
//
//     auto filter_ts(Array<f32>& ts) {
//         f64 tilt_angle = -60.;
//         auto tilt_angles = std::vector<f64>{};
//         for (isize i{}; i < ts.shape()[0]; ++i) {
//             tilt_angles.push_back(tilt_angle);
//             tilt_angle += 3;
//         }
//
//         usize c{};
//         auto tilt_angles_filtered = std::vector<f64>{};
//         for (usize i{}; i < tilt_angles.size(); ++i) {
//             if (tilt_angles[i] < 40. or tilt_angles[i] > 58.)
//                 continue;
//             ts.subregion(i).to(ts.subregion(c));
//             tilt_angles_filtered.push_back(tilt_angles[i]);
//             ++c;
//         }
//
//         ts = ts.subregion(Slice{0, c});
//         return tilt_angles_filtered;
//     }
// }
// TEST_CASE("test tomo") {
//     auto tomogram = noa::read_image<f32>("/Users/cix56657/Datasets/10304/aretomo3/tilt1_ali_Vol.mrc").data;
//     tomogram = tomogram.permute({1, 0, 2, 3});
//     //aretomo3/tilt1_ali_Vol.mrc
//
//     auto output_image = Array<f32>(tomogram.shape().filter(2, 3).push_front<2>(1));
//     auto center_3d = (output_image.shape().vec.filter(1, 2, 3) / 2).as<f64>();
//     auto center_2d = (output_image.shape().vec.filter(2, 3) / 2).as<f64>();
//     auto projection_matrix = (
//         nx::translate(+center_3d) *
//         nx::rotate_y<true>(noa::deg2rad(60.)) *
//         nx::translate(-center_2d.push_front(0))
//     );
//
//     auto window_size = nx::forward_projection_window_size(tomogram.shape().pop_front(), projection_matrix);
//     nx::forward_project_3d(tomogram, output_image, projection_matrix, window_size);
//
//     noa::write_image(output_image, "/Users/cix56657/Datasets/10304/aretomo3/projected_image.mrc");
// }
//
// TEST_CASE("test EWS") {
//     const auto ts_path = std::filesystem::path("/Users/cix56657/Datasets/10304/etomo/tilt1_ali.mrc");
//     auto [ts, header] = noa::io::read_image<f32>(ts_path);
//     auto tilt_angles = filter_ts(ts);
//     fmt::println("tilt_angles={}", tilt_angles);
//
//     noa::normalize_per_batch(ts, ts, {.mode = noa::Norm::MEAN_STD});
//
//     const auto n_images = ts.shape()[0];
//     const auto shape = ts.shape().filter(2, 3);
//     const auto options = ArrayOption{.device = "cpu", .allocator = "managed"};
//     auto original_center = (shape.vec / 2).as<f64>();
//
//     nx::draw(ts, ts, nx::Rectangle{.center=original_center, .radius = original_center - 100, .smoothness = 100.}.draw<f32>());
//
//     // Rotations.
//     auto insert_inv_rotations = Array<Mat<f64, 3, 3>>(n_images, options);
//     for (auto&& [tilt, rotation]: noa::zip(tilt_angles, insert_inv_rotations.span_1d()))
//         rotation = nx::rotate_y(noa::deg2rad(-tilt)).as<f64>();
//     auto extract_fwd_rotation = nx::rotate_y(noa::deg2rad(60.)).as<f64>(); //Mat<f32, 3, 3>::eye(1);
//     noa::write_image(ts.subregion(ts.shape()[0] - 1), "/Users/cix56657/Datasets/10304/etomo/output_target.mrc");
//
//
//     // Get the central slices ready
//     auto shape_padded = Shape2::from_value(max(shape) * 2);
//     auto padding = shape_padded - shape;
//     auto ts_padded = noa::resize(ts, {}, padding.vec.push_front<2>(0));
//     auto ts_padded_rfft = nf::r2c(ts_padded);
//     ns::phase_shift_2d<"h">(ts_padded_rfft, ts_padded_rfft, ts_padded.shape(), -original_center);
//     ns::bandpass<"h">(ts_padded_rfft, ts_padded_rfft, ts_padded.shape(), {0.03, 0.03, 0.45, 0.05});
//
//     //
//     constexpr auto INSERT_SINC_OSCILLATIONS = 8;
//     const f64 virtual_volume_size = static_cast<f64>(shape_padded[0]) / 1;
//     const f64 insert_fftfreq_sinc = 1 / virtual_volume_size;
//     const f64 insert_fftfreq_blackman = INSERT_SINC_OSCILLATIONS * insert_fftfreq_sinc;
//
//     constexpr auto EXTRACT_SINC_OSCILLATIONS = 8;
//     const f64 sample_thickness_nm = 100;
//     const f64 thickness_estimate_pixels = sample_thickness_nm / (mean(header.spacing.pop_front()) * 1e-1);
//     const f64 extract_fftfreq_z_sinc = 1 / thickness_estimate_pixels;
//     const f64 extract_fftfreq_z_blackman = EXTRACT_SINC_OSCILLATIONS * extract_fftfreq_z_sinc;
//     fmt::println("thickness_estimate_pixels={}", thickness_estimate_pixels);
//
//     auto output_padded_shape = ts_padded.shape().set<0>(1);
//     auto output_slice_padded_rfft = like(ts_padded_rfft.subregion(0));
//     auto output_slice_padded_weight = like<f32>(ts_padded_rfft.subregion(0));
//     noa::fill(output_slice_padded_weight, 0);
//
//     auto ts_padded_weights = like<f32>(ts_padded_rfft);
//     noa::fill(ts_padded_weights, 1);
//
//     nx::insert_and_extract_central_slices_3d<"h">(
//         ts_padded_rfft, ts_padded_weights, ts_padded.shape(),
//         output_slice_padded_rfft, output_slice_padded_weight, output_padded_shape,
//         {}, insert_inv_rotations,
//         {}, extract_fwd_rotation,
//         {
//             .interp = nx::Interp::LINEAR,
//             .input_windowed_sinc = {insert_fftfreq_sinc, insert_fftfreq_blackman},
//             .w_windowed_sinc = {extract_fftfreq_z_sinc, extract_fftfreq_z_blackman},
//             .add_to_output = false,
//             .correct_weights = false,
//             .ews_radius = {},
//         }
//     );
//     noa::ewise(output_slice_padded_weight, output_slice_padded_rfft, [](f32 w, c32& o) {
//         o /= noa::max(abs(w), 1.f);
//     });
//
//     ns::phase_shift_2d<"h">(output_slice_padded_rfft, output_slice_padded_rfft, output_padded_shape, original_center);
//     auto output_image_padded = nf::c2r(output_slice_padded_rfft, output_padded_shape);
//     auto output_image = noa::resize(output_image_padded, {}, -padding.vec.push_front<2>(0));
//
//     noa::write_image(output_image_padded, "/Users/cix56657/Datasets/10304/etomo/output_image_padded_ews0.mrc");
//     noa::write_image(output_image, "/Users/cix56657/Datasets/10304/etomo/output_image_ews0.mrc");
//     noa::write_image(output_slice_padded_weight, "/Users/cix56657/Datasets/10304/etomo/output_weight_ews0.mrc");
// }
