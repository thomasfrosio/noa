// OLD - UNUSED

#include <noa/Array.hpp>
#include <noa/FFT.hpp>
#include <noa/Math.hpp>
#include <noa/Memory.hpp>
#include <noa/Geometry.hpp>
#include <noa/Signal.hpp>
#include <noa/IO.hpp>
#include <noa/Utils.h>

#include <iostream>

#include <catch2/catch.hpp>

#include "Helpers.h"

namespace {
    using namespace ::noa;
    const path_t g_output_dir = "/home/thomas/Projects/noa-data/assets/signal/fft/alignment2";

    class Projector {
    public:
        Projector(size4_t volume_shape, size4_t slice_shape, ArrayOption options)
                : m_volume_fft(memory::zeros<cfloat_t>(volume_shape.fft(), options)),
                  m_volume_weights_fft(memory::zeros<float>(volume_shape.fft(), options)),
                  m_weights_ones_fft(memory::ones<float>(slice_shape.fft(), options)),
                  m_weights_extract_fft(memory::empty<float>(slice_shape.fft(), options)),
                  m_volume_shape(volume_shape),
                  m_slice_shape(slice_shape) {}

        void backward(const Array<cfloat_t>& slice_fft, float33_t rotation,
                      float2_t extra_shift = {0, 0}) {
            const float2_t slice_center = float2_t(m_slice_shape.get(2)) / 2;

            signal::fft::shift2D<fft::H2H>(slice_fft, slice_fft, m_slice_shape, -slice_center + extra_shift);
            geometry::fft::insert3D<fft::H2HC>(slice_fft, m_slice_shape,
                                               m_volume_fft, m_volume_shape,
                                               float22_t{}, rotation, 0.5f);

            geometry::fft::insert3D<fft::H2HC>(m_weights_ones_fft, m_slice_shape,
                                               m_volume_weights_fft, m_volume_shape,
                                               float22_t{}, rotation, 0.5f);
        }

        void forward(Array<cfloat_t>& slice_fft, float33_t rotation,
                     float2_t extra_shift = {0, 0}) {
            const Array<float33_t> rot_(&rotation, 1);
            const float2_t slice_center = float2_t(m_slice_shape.get(2)) / 2;

            geometry::fft::extract3D<fft::HC2H>(m_volume_fft, m_volume_shape,
                                                slice_fft, m_slice_shape,
                                                float22_t{}, rotation, 0.5f);
            signal::fft::shift2D<fft::H2H>(slice_fft, slice_fft, m_slice_shape, slice_center + extra_shift);

            geometry::fft::extract3D<fft::HC2H>(m_volume_weights_fft, m_volume_shape,
                                                m_weights_extract_fft, m_slice_shape,
                                                float22_t{}, rotation, 0.5f);
            math::ewise(slice_fft, m_weights_extract_fft, 1e-3f, slice_fft,
                        math::divide_epsilon_t{});
        }

    private:
        Array<cfloat_t> m_volume_fft;
        Array<float> m_volume_weights_fft;
        Array<float> m_weights_ones_fft;
        Array<float> m_weights_extract_fft;
        size4_t m_volume_shape;
        size4_t m_slice_shape;
    };

    class StackGeometry {
        struct View {
            float3_t angles;
            float2_t shift;
            int order;
        };

    private:
        std::vector<int> m_order;
        std::vector<float3_t> m_angles;
        std::vector<float2_t> m_shifts;
    };

    std::vector<float2_t> alignShiftProjectionMatching_(const path_t filename,
                                                        const Array<float33_t>& rotations,
                                                        const std::vector<int>& index_order) {
        ArrayOption options{Device("gpu")};

        io::ImageFile file(filename, io::READ);
        const size4_t original_tilt_series_shape = file.shape();
        const size4_t original_shape{1, 1, original_tilt_series_shape[2], original_tilt_series_shape[3]};
        const size4_t slice_shape{1, 1, 2048, 2048};
        const size4_t volume_shape{1, 256, slice_shape[2], slice_shape[3]};
        const float3_t sampling_factor;

        // Preprocess the tilt-series.
        Timer timer0;
        timer0.start();
        Array tilt_series_fft = memory::empty<cfloat_t>(original_tilt_series_shape.fft(), options);
        Array tilt_series = fft::alias(tilt_series_fft, original_tilt_series_shape);
        file.read(tilt_series);
        {
            fft::r2c(tilt_series, tilt_series_fft);
            signal::fft::highpass<fft::H2H>(tilt_series_fft, tilt_series_fft, original_tilt_series_shape, 0.05f, 0.05f);
            signal::fft::standardize<fft::H2H>(tilt_series_fft, tilt_series_fft, original_tilt_series_shape);
            fft::c2r(tilt_series_fft, tilt_series);
            const float2_t shape(original_shape.get(2));
            const float2_t center = shape / 2;
            signal::rectangle(tilt_series, tilt_series, center, center - 60, 60);
        }
        tilt_series.eval();
        fmt::print("Loading and preprocessing the tilt-series: {}ms\n", timer0.elapsed());

        // Backward project the reference.
        timer0.start();
        Array reference_pad_fft = memory::empty<cfloat_t>(slice_shape.fft(), options);
        Array reference_pad = fft::alias(reference_pad_fft, slice_shape);
        memory::resize(tilt_series.subregion(index_order[0]), reference_pad);
        fft::r2c(reference_pad, reference_pad_fft);
        Projector projector(volume_shape, slice_shape, options);
        projector.backward(reference_pad_fft, rotations[0]);

        std::vector<float2_t> output_shifts;
        output_shifts.emplace_back(0);

        // Prepare reference and target arrays.
        Array target_pad_fft = memory::empty<cfloat_t>(slice_shape.fft(), options);
        Array target_pad = fft::alias(target_pad_fft, slice_shape);
        Array xmap = memory::empty<float>(slice_shape, options);
        tilt_series.eval();
        fmt::print("Initialization took: {}ms\n", timer0.elapsed());

        timer0.start();
        for (size_t i = 1; i < index_order.size(); ++i) {
            Timer timer1;
            timer1.start();

            // Get the target:
            memory::resize(tilt_series.subregion(index_order[i]), target_pad);
            fft::r2c(target_pad, target_pad_fft);

            // Get the reference by forward projecting at the target rotation:
            projector.forward(reference_pad_fft, rotations[i]);

            // Find and apply shift:
            signal::fft::xmap<fft::H2F>(reference_pad_fft, target_pad_fft, xmap,
                                        signal::CONVENTIONAL_CORRELATION, fft::NORM_DEFAULT, reference_pad_fft);
            const auto [peak, _] = signal::fft::xpeak2D<fft::F2F>(xmap);
            const float2_t slice_center = float2_t(slice_shape.get(2)) / 2;
            const float2_t shift = peak - slice_center;
            output_shifts.emplace_back(shift);

            // Backward project the shifted target:
            projector.backward(target_pad_fft, rotations[i], shift);

            tilt_series.eval();
            fmt::print("Iteration took: {}ms, shift: {}\n", timer1.elapsed(), shift);
        }
        tilt_series.eval();
        fmt::print("Alignment took: {}ms\n", timer0.elapsed());
        return output_shifts;
    }

    std::vector<float2_t> alignShiftCosineStretching_(const path_t filename,
                                                      std::vector<float> tilt_angles,
                                                      float rotation_angle,
                                                      const std::vector<int>& index_order) {
        ArrayOption options{Device("gpu")};

        io::ImageFile file(filename, io::READ);
        const size4_t tilt_series_shape = file.shape();
        const size4_t slice_shape{1, 1, tilt_series_shape[2], tilt_series_shape[3]};
        const float2_t slice_center = float2_t(slice_shape.get(2)) / 2;

        // Preprocess the tilt-series.
        Timer timer0;
        timer0.start();
        Array tilt_series_fft = memory::empty<cfloat_t>(tilt_series_shape.fft(), options);
        Array tilt_series = fft::alias(tilt_series_fft, tilt_series_shape);
        file.read(tilt_series);
        {
            fft::r2c(tilt_series, tilt_series_fft);
            signal::fft::highpass<fft::H2H>(tilt_series_fft, tilt_series_fft, tilt_series_shape, 0.05f, 0.05f);
            signal::fft::standardize<fft::H2H>(tilt_series_fft, tilt_series_fft, tilt_series_shape);
            fft::c2r(tilt_series_fft, tilt_series);
            const float2_t shape(slice_shape.get(2));
            const float2_t center = shape / 2;
            signal::rectangle(tilt_series, tilt_series, center, center - 60, 60);
        }
        tilt_series.eval();
        fmt::print("Loading and preprocessing the tilt-series: {}ms\n", timer0.elapsed());

        // Prepare the textures.
        timer0.start();
        Array<cfloat_t> reference_fft(slice_shape, options);
        Array<float> reference = fft::alias(reference_fft, slice_shape);

        Array<cfloat_t> target_fft(slice_shape, options);
        Array<float> target = fft::alias(target_fft, slice_shape);

        Array<float> buffer = memory::like(reference);
        Array xmap = memory::empty<float>(slice_shape, options);

        std::vector<float2_t> output_shifts;
        output_shifts.emplace_back(0);

        rotation_angle = math::deg2rad(rotation_angle);
        timer0.start();
        for (size_t i = 1; i < index_order.size(); ++i) {
            Timer timer1;
            timer1.start();

            // Get the reference and target.
            tilt_series.subregion(0).to(buffer);
            tilt_series.subregion(0).to(target);

            // Stretch the reference.
            const float tilt_angle = math::deg2rad(tilt_angles[i]);
            const float33_t cos_stretch{
                    geometry::translate(slice_center) *
                    float33_t{geometry::rotate(rotation_angle)} *
                    float33_t{geometry::scale(float2_t{1, math::cos(tilt_angle)})} *
                    float33_t{geometry::rotate(-rotation_angle)} *
                    geometry::translate(-slice_center)
            };
            geometry::transform2D(buffer, reference, math::inverse(cos_stretch), INTERP_LINEAR_FAST);

            // Find and apply shift:
            fft::r2c(reference, reference_fft);
            fft::r2c(target, target_fft);
            signal::fft::xmap<fft::H2F>(reference_fft, target_fft, xmap);
            const auto [peak, _] = signal::fft::xpeak2D<fft::F2F>(xmap);
            const float2_t shift = peak - slice_center;
            output_shifts.emplace_back(shift);

            tilt_series.eval();
            fmt::print("Iteration took: {}ms, shift: {}\n", timer1.elapsed(), shift);
        }
        tilt_series.eval();
        fmt::print("Alignment took: {}ms\n", timer0.elapsed());
        return output_shifts;
    }
}

TEST_CASE("unified::signal::fft, Find shifts with fwd projection", "[.]") {
    using namespace ::noa;

    const float rotation_angle = 176;
    const path_t input_path = "/home/thomas/Projects/data/ribo/tilt1/tilt1_bin6.mrc";
    const size_t slice_count = io::ImageFile(input_path, io::READ).shape()[0];

    // Prepare the rotations:
//    std::vector<int> index_order = {20, 21, 19, 22, 18, 23, 17, 24, 16, 25, 15, 26, 14, 27, 13, 28, 12, 29, 11,
//                                    30, 10, 31, 9, 32, 8, 33, 7, 34, 6, 35, 5, 36, 4, 37, 3, 38, 2, 39, 1, 40, 0};
//    std::vector<float> tlt_angles = {0, 3, -3, 6, -6, 9, -9, 12, -12, 15, -15, 18, -18, 21, -21, 24, -24, 27, -27,
//                                     30, -30, 33, -33, 36, -36, 39, -39, 42, -42, 45, -45, 48, -48, 51, -51,
//                                     54, -54, 57, -57, 60, -60};
    std::vector<int> index_order = {20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                    19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    std::vector<float> tlt_angles = {0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60,
                                     -3, -6, -9, -12, -15, -18, -21, -24, -27, -30, -33, -36, -39, -42, -45, -48, -51,
                                     -54, -57, -60};
    const Array<float33_t> rotations(slice_count);
    for (size_t i = 0; i < slice_count; ++i)
        rotations.get()[i] = geometry::euler2matrix(math::deg2rad(float3_t{rotation_angle, tlt_angles[i], 0}));

    // Get shifts:
    std::vector<float2_t> shifts = alignShiftProjectionMatching_(input_path, rotations, index_order);

    Array tilt_series = io::load<float>(input_path);
    Array<float> means({tilt_series.shape()[0], 1, 1, 1});
    math::mean(tilt_series, means);
    math::ewise(tilt_series, means, tilt_series, math::minus_t{});

    Array output = memory::like(tilt_series);
    for (size_t i = 0; i < slice_count; ++i)
        geometry::transform2D(tilt_series.subregion(index_order[i]),
                              output.subregion(index_order[i]),
                              geometry::translate(shifts[i]));
    io::save(output, g_output_dir / "shift_aligned.mrc");
}


//TEST_CASE("unified::signal::fft, Find shift with fwd projection", "[.]") {
//// Insert the 0deg image and extract at 3deg tilt,
//// then try to find shift and in-plane rotation.
//
//    using namespace ::noa;
//    Timer timer;
//    auto log_plot_functor = [](cfloat_t v) { return math::log(math::abs(v) + 1); };
//
//    const float rotation_angle = 176;
//    const path_t input_path = "/home/thomas/Projects/data/ribo/tilt1/tilt1_bin6.mrc";
//    const Array<float33_t> rotations(2);
//    rotations.get()[0] = geometry::euler2matrix(math::deg2rad(float3_t{rotation_angle, 0, 0}));
//    rotations.get()[1] = geometry::euler2matrix(math::deg2rad(float3_t{rotation_angle, 3, 0}));
//
//    io::ImageFile file(input_path, io::READ);
//    const size4_t stack_shape = file.shape();
//    size4_t original_slice_shape{1, 1, stack_shape[2], stack_shape[3]};
//
//    const Device device("cpu");
//    const ArrayOption options(device, Allocator::DEFAULT_ASYNC);
//
//    // Extract and preprocess the data:
//    Array slice_0deg = memory::empty<float>(original_slice_shape, options);
//    file.readSlice(slice_0deg, 20);
//    Array slice_0deg_fft = fft::r2c(slice_0deg);
//    signal::fft::highpass<fft::H2H>(slice_0deg_fft, slice_0deg_fft, original_slice_shape, 0.05f, 0.05f);
//    signal::fft::standardize<fft::H2H>(slice_0deg_fft, slice_0deg_fft, original_slice_shape);
//    fft::c2r(slice_0deg_fft, slice_0deg);
//    taper_(slice_0deg, 60);
//    io::save(slice_0deg, g_output_dir / "slice_0deg.mrc");
//
//    // Zero pad:
//    Array slice_0deg_pad = resize_(slice_0deg, 2048, 2048);
//    io::save(slice_0deg_pad, g_output_dir / "slice_0deg_pad.mrc");
//
//    const size4_t slice_shape = slice_0deg_pad.shape();
//    slice_0deg_fft = fft::r2c(slice_0deg_pad);
//
//    // Backward project the 0deg:
//    const size4_t volume_shape{1, 256, slice_shape[2], slice_shape[3]};
//    Array volume_fft = memory::zeros<cfloat_t>(volume_shape.fft(), options);
//    Array volume_weights_fft = memory::zeros<float>(volume_shape.fft(), options);
//    backwardProject_(volume_fft, volume_weights_fft, slice_0deg_fft,
//                     volume_shape, slice_shape,
//                     rotations.get()[0]);
//
//    // Forward project the 3deg:
//    Array slice_3deg_projected_fft = forwardProject_(volume_fft, volume_weights_fft,
//                                                     volume_shape, slice_shape,
//                                                     rotations.get()[1]);
//
//    // Prepare reference for registration:
//    Array slice_3deg_projected = fft::c2r(slice_3deg_projected_fft, slice_shape);
//    slice_3deg_projected = resize_(slice_3deg_projected, original_slice_shape[2], original_slice_shape[3]);
//    taper_(slice_3deg_projected, 60);
//    io::save(slice_3deg_projected, g_output_dir / "slice_3deg_projected.mrc");
//
//    Array slice_3deg_projected_fft_abs = memory::empty<float>(slice_shape.fft(), options);
//    math::ewise(slice_3deg_projected_fft, slice_3deg_projected_fft_abs, log_plot_functor);
//    io::save(slice_3deg_projected_fft_abs, g_output_dir / "slice_3deg_projected_fft_abs.mrc");
//
//    // Prepare target for registration:
//    Array slice_3deg = memory::empty<float>(original_slice_shape, options);
//    file.readSlice(slice_3deg, 21);
//    Array slice_3deg_fft = fft::r2c(slice_3deg);
//    signal::fft::highpass<fft::H2H>(slice_3deg_fft, slice_3deg_fft, original_slice_shape, 0.05f, 0.05f);
//    signal::fft::standardize<fft::H2H>(slice_3deg_fft, slice_3deg_fft, original_slice_shape);
//    fft::c2r(slice_3deg_fft, slice_3deg);
//    taper_(slice_3deg, 60);
//    io::save(slice_3deg, g_output_dir / "slice_3deg.mrc");
//
//    // Phase correlate
//    auto[xmap, shift] = findShift_(slice_3deg_projected, slice_3deg, original_slice_shape);
//    io::save(xmap, g_output_dir / "xmap_3deg.mrc");
//    fmt::print("shift 3deg: {}\n", shift);
//
//    // Translate:
//    Array slice_3deg_shifted = memory::like(slice_3deg);
//    geometry::shift2D(slice_3deg, slice_3deg_shifted, shift);
//    taper_(slice_3deg_shifted, 60);
//    io::save(slice_3deg_shifted, g_output_dir / "slice_3deg_shifted.mrc");
//}


//    {
//        Array image_fft = slice_0deg_fft.copy();
//        Array image_fft_output = memory::zeros<cfloat_t>(slice_shape.fft(), options);
//
//        signal::fft::shift2D<fft::H2HC>(image_fft, image_fft_output, slice_shape, -slice_center);
//        float22_t rot = geometry::rotate(math::deg2rad(-rotation_angle));
//        geometry::fft::transform2D<fft::HC2H>(image_fft_output, image_fft, slice_shape,
//                                              rot, float2_t{});
//        signal::fft::shift2D<fft::H2H>(image_fft, image_fft, slice_shape, slice_center);
//        Array image_output = fft::c2r(image_fft, slice_shape);
//        io::save(image_output, g_output_dir / "test_rotation.mrc");
//    }

//    Array volume = fft::c2r(volume_fft, volume_shape);
//    geometry::fft::griddingCorrection(volume, volume, true);
//    geometry::fft::griddingCorrection(volume, volume, false);
//    fft::r2c(volume, volume_fft);
//    fft::remap(fft::H2HC, volume_fft, volume_fft, volume_shape);
