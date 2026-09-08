#include <noa/fft/Remap.hpp>
#include <noa/IO.hpp>
#include <noa/base/Zip.hpp>
#include <noa/runtime/Reduce.hpp>
#include <noa/signal/CTF.hpp>
#include <noa/xform/Draw.hpp>
#include <noa/xform/PolarTransformSpectrum.hpp>
#include <noa/xform/RotationalAverage.hpp>

#include "Assets.hpp"
#include "Utils.hpp"
#include "Catch.hpp"

using namespace noa::types;
namespace ns = noa::signal;
namespace nx = noa::xform;

TEST_CASE("xform::spectrum2polar") {
    auto devices = std::vector<Device>{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    constexpr auto spectrum_shape = Shape<isize, 2>{256, 256};
    constexpr auto spectrum_1d_shape = Shape<isize, 1>{256};
    constexpr auto polar_shape = Shape<isize, 2>{256, 129};
    const auto ctf = ns::CTFIsotropic<f64>::Parameters{
        .pixel_size = 3.,
        .defocus = 2.,
        .voltage = 300.,
        .amplitude = 0.7,
        .cs = 2.7,
        .phase_shift = 1.570796327,
        .bfactor = 0.,
        .scale = 1.,
    }.to_ctf();

    for (auto device: devices) {
        const auto options = ArrayOption{.device = device, .allocator = Allocator::MANAGED};
        INFO(device);

        const auto spectrum = noa::Array<f32, 2>(spectrum_shape.rfft(), options);
        ns::ctf_isotropic<"h">(spectrum, spectrum_shape, ctf, {
            .fftfreq_range = noa::Linspace{.start = 0., .stop = 0.4, .endpoint = true},
        });

        const auto polar = noa::Array<f32, 2>(polar_shape, options);
        nx::spectrum2polar<"h2fc">(spectrum, spectrum_shape, polar, {
            .spectrum_fftfreq = noa::Linspace{.start = 0., .stop = 0.4, .endpoint = true},
            .rho_range = noa::Linspace{.start = 0.1, .stop = 0.35, .endpoint = true},
            .interp = nx::Interp::CUBIC,
        });

        const auto polar_1d = noa::Array<f32, 1>(polar_shape[1], options);
        noa::reduce_axes_ewise(polar, f64{0}, polar_1d, noa::ReduceMean{static_cast<f64>(polar_shape[0])});

        auto spectrum_1d = noa::Array<f32, 1>(spectrum_1d_shape.rfft(), options);
        ns::ctf_isotropic<"h">(spectrum_1d, spectrum_1d_shape, ctf, {
            .fftfreq_range = noa::Linspace{.start = 0.1, .stop = 0.35, .endpoint = true},
        });

        REQUIRE(test::allclose_abs(polar_1d, spectrum_1d, 5e-4));
    }
}

TEST_CASE("xform::rotational_average") {
    auto devices = std::vector<Device>{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    const isize size = GENERATE(64, 65);
    INFO(size);

    // Handpicked frequencies that fall exactly onto a shell.
    const auto frequency_range = noa::Linspace{
        .start = size == 64 ? 0.125 : 0.2,
        .stop = size == 64 ? 0.3125 : 0.4,
    };
    const i64 frequency_range_n_shells = size == 64 ? 13 : 14;
    const auto frequency_range_start_index = size == 64 ? 8 : 13;
    const auto subregion_within_full_range = noa::Subregion(
        Ellipsis{},
        Slice{
            .start = frequency_range_start_index,
            .end = frequency_range_start_index + frequency_range_n_shells
        }
    );

    const isize n_batch = 3;
    const auto shape_2d = Shape{size, size};
    const auto shape = Shape{n_batch, size, size};
    const auto rotational_average_size = noa::min(shape_2d) / 2 + 1;
    const auto polar_shape = Shape3{n_batch, 256, rotational_average_size};
    const auto rotational_average_shape = Shape2{n_batch, rotational_average_size};
    const auto center = (shape_2d.vec / 2).as<f64>();

    for (auto device: devices) {
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        const auto input = noa::zeros<f32>(shape, options);
        nx::draw_2d({}, input, nx::Sphere{
            .center=center,
            .radius=0.,
            .smoothness=noa::min(center),
        }.get());

        // Rotational average using polar transformation.
        const auto input_rfft = noa::fft::remap_2d("FC2HC", input, shape);
        const auto polar = noa::zeros<f32>(polar_shape, options);
        nx::spectrum2polar<"HC2FC">(input_rfft, shape, polar);
        const auto polar_reduced = noa::zeros<f32, 2>(rotational_average_shape, options);
        noa::mean(polar, polar_reduced.reshape(Shape3{n_batch, 1, -1}));

        // Rotational average.
        const auto output = noa::empty<f32>(rotational_average_shape, options);
        const auto weight = noa::empty<f32>(rotational_average_shape, options);
        nx::rotational_average<"HC2H">(input_rfft, shape, output, weight);

        // fmt::print("{:.3f}\n", fmt::join(polar_reduced.eval().span_1d(), ","));
        // fmt::print("{:.3f}\n", fmt::join(output.eval().span_1d(), ","));
        // fmt::print("{:.3f}\n", fmt::join(weight.eval().span_1d(), ","));

        REQUIRE(test::allclose_abs(polar_reduced, output, 1e-3));

        // Rotational average within a range.
        // Use the same number of shells, so it can be compared with the full range.
        const auto output_range = noa::empty<f32, 2>({n_batch, frequency_range_n_shells}, options);
        const auto weight_range = noa::empty_like(output_range);
        nx::rotational_average<"HC2H">(
            input_rfft, shape, output_range, weight_range, {.output_fftfreq = frequency_range});

        const auto output_cropped = output.subregion(subregion_within_full_range);
        const auto weight_cropped = weight.subregion(subregion_within_full_range);

        // fmt::print("{:.3f}\n", fmt::join(output_range.eval().span_1d(), ","));
        // fmt::print("{:.3f}\n", fmt::join(weight_range.eval().span_1d(), ","));

        // We don't expect the first and last shell to be the same as with the full range because of the lerp
        // that happens when collecting the signal from every frequency. Here, the cutoff excludes everything
        // that is not in the range, but with the full range and then cropping, we still have the contribution
        // from the frequencies between the first and first-1 shells, and last and last+1 shells.
        REQUIRE(test::allclose_abs(
            output_cropped.subregion(Ellipsis{}, Slice{1, -1}),
            output_range.subregion(Ellipsis{}, Slice{1, -1}),
            1e-3));
        REQUIRE(test::allclose_abs(
            weight_cropped.subregion(Ellipsis{}, Slice{1, -1}),
            weight_range.subregion(Ellipsis{}, Slice{1, -1}),
            1e-3));
    }
}

TEST_CASE("xform::equiphase_average vs rotational_average") {
    // Test that with an isotropic ctf it gives the same results as the classic rotational average.

    auto devices = std::vector<Device>{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    // Needs to be the same size otherwise some signal close to the end
    // will not be present when rescaling the anisotropic field.
    const auto shape = Shape2{512, 512};
    const auto n_shells = noa::min(shape) / 2 + 1;

    using CTFIsotropic64 = ns::CTFIsotropic<f64>;
    using CTFAnisotropic64 = ns::CTFAnisotropic<f64>;
    const auto ctf_iso = CTFIsotropic64::Parameters{
        .pixel_size = 3.,
        .defocus = 2.,
        .voltage = 300.,
        .amplitude = 0.7,
        .cs = 2.7,
        .phase_shift = 1.570796327,
        .bfactor = 0.,
        .scale = 1.,
    }.to_ctf();

    // First test that if no anisotropy, it generates exactly the same rotational average.
    for (auto device: devices) {
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        const auto simulated_ctf = noa::empty<f32>(shape.rfft(), options);
        const auto rotational_averages = noa::empty<f32, 2>({2, n_shells}, options);
        const auto result = rotational_averages.subregion(0).as_1d();
        const auto expected = rotational_averages.subregion(1).as_1d();

        ns::ctf_isotropic_2d<"H2H">(simulated_ctf, shape, ctf_iso);

        nx::rotational_average<"H2H">(simulated_ctf, shape, result);
        nx::equiphase_average<"H2H">(simulated_ctf, shape, CTFAnisotropic64::from_isotropic_ctf(ctf_iso), expected);

        REQUIRE(test::allclose_abs(result, expected, 1e-4));
    }

    auto ctfs = std::array{
        CTFAnisotropic64(ctf_iso, 0.0, 0.),
        CTFAnisotropic64(ctf_iso, 0.1, 0.),
        CTFAnisotropic64(ctf_iso, 0.2, 0.43),
        CTFAnisotropic64(ctf_iso, 0.3, 2.97),
        CTFAnisotropic64(ctf_iso, 0.4, 0.52),
        CTFAnisotropic64(ctf_iso, 0.5, 1.24),
    };

    for (auto device: devices) {
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        const auto batched_shape = shape.push_front(std::ssize(ctfs));
        const auto simulated_ctf = noa::empty<f32>(batched_shape.rfft(), options);
        const auto rotational_averages = noa::empty<f32, 2>({std::ssize(ctfs), n_shells}, options);
        const auto ctfs_array = Array(ctfs.data(), std::ssize(ctfs)).to(options);

        ns::ctf_anisotropic_2d<"H2H">(simulated_ctf, batched_shape, ctfs_array);
        nx::equiphase_average<"H2H">(simulated_ctf, batched_shape, ctfs_array, rotational_averages);

        const auto expected = rotational_averages.subregion(0).as_1d();
        for (auto i: noa::irange(usize{1}, ctfs.size())) {
            REQUIRE(test::allclose_abs(expected, rotational_averages.subregion(i).as_1d(), 1e-3));
        }
    }
}

namespace {
    template<typename Real>
    void save_vector_to_text(Span<Real, 2> x, const Path& filename) {
        noa::check(x.contiguity()[1] == true);

        std::string format;
        for (auto i: noa::irange(x.shape()[0]))
            format += fmt::format("{}\n", fmt::join(x.subregion(i).span(), ","));
        noa::write_text(format, filename);
    }
}

TEST_CASE("xform::equiphase_rescale_and_average_1d") {
    // Simulate two 1d CTFs with different spacing and defocus, then equiphase rescale, and check that they match.
    // const auto directory = test::NOA_DATA_PATH / "xform";

    using CTFIsotropic64 = ns::CTFIsotropic<f64>;
    constexpr auto defocus = std::array{2.15, 2.90, 2.4};
    constexpr auto spacing = std::array{1.80, 2.40, 2.};
    constexpr auto phase_shifts = std::array{0.45, 0.75, 0.55};
    auto input_ctfs = Array<CTFIsotropic64, 1>(3);
    for (auto&& [d, s, p, ctf]: noa::zip(defocus, spacing, phase_shifts, input_ctfs.span())) {
        ctf = CTFIsotropic64::Parameters{
            .pixel_size = s,
            .defocus = d,
            .voltage = 300.,
            .amplitude = 0.1,
            .cs = 2.7,
            .phase_shift = p,
            .bfactor = 0.,
            .scale = 1.,
        }.to_ctf();
    }

    constexpr isize size = 2048;
    constexpr auto logical_shape = Shape2{3, size};
    constexpr auto shape_rfft = logical_shape.rfft();
    const auto input = noa::Array<f64, 2>(shape_rfft);
    ns::ctf_isotropic<"h2h">(input, logical_shape, input_ctfs);
    // save_vector_to_text(input.view(), directory / "test_input_average.txt");

    // Target CTF.
    auto target_ctf = CTFIsotropic64::Parameters{
        .pixel_size = 2.1,
        .defocus = 2.45,
        .voltage = 300.,
        .amplitude = 0.1,
        .cs = 2.7,
        .phase_shift = 0.5,
        .bfactor = 0.,
        .scale = 1.,
    }.to_ctf();

    const auto target = noa::Array<f64, 1>(shape_rfft[1]);
    ns::ctf_isotropic<"h2h">(target, Shape{logical_shape[1]}, target_ctf, {.fftfreq_range = {0.1, 0.4}});

    // Output CTF.
    const auto output = noa::empty<f64>(shape_rfft.set<0>(1));
    nx::equiphase_rescale_and_average_1d(
        input, {.start = 0., .stop = 0.5}, input_ctfs,
        output, {.start = 0.1, .stop = 0.4}, target_ctf // stop at 0.4 to remove missing signal from different fields
    );
    // save_vector_to_text(target.view(), directory / "test_expected_average.txt");
    // save_vector_to_text(output.view(), directory / "test_result_average.txt");

    REQUIRE(test::allclose_abs(target, output.as_1d(), 5e-2));
}

// TODO Add better tests...
TEST_CASE("xform::equiphase_rescale_1d") {
    using CTFIsotropic64 = ns::CTFIsotropic<f64>;
    constexpr auto defocus = std::array{2.15, 2.90, 2.4};
    constexpr auto spacing = std::array{1.80, 2.40, 2.};
    constexpr auto phase_shifts = std::array{0.45, 0.75, 0.55};
    auto input_ctfs = Array<CTFIsotropic64, 1>(3);
    for (auto&& [d, s, p, ctf]: noa::zip(defocus, spacing, phase_shifts, input_ctfs.span())) {
        ctf = CTFIsotropic64::Parameters{
            .pixel_size = s,
            .defocus = d,
            .voltage = 300.,
            .amplitude = 0.1,
            .cs = 2.7,
            .phase_shift = p,
            .bfactor = 0.,
            .scale = 1.,
        }.to_ctf();
    }

    constexpr i64 size = 2048;
    constexpr auto logical_shape = Shape2{3, size};
    constexpr auto shape_rfft = logical_shape.rfft();
    const auto input = noa::Array<f64, 2>(shape_rfft);
    ns::ctf_isotropic<"h2h">(input, logical_shape, input_ctfs);
    // save_vector_to_text(input.view(), directory / "test_input_average.txt");

    // Target CTF.
    auto target_ctf = CTFIsotropic64::Parameters{
        .pixel_size = 2.1,
        .defocus = 2.45,
        .voltage = 300.,
        .amplitude = 0.1,
        .cs = 2.7,
        .phase_shift = 0.5,
        .bfactor = 0.,
        .scale = 1.,
    }.to_ctf();

    const auto target = noa::Array<f64, 1>(shape_rfft[1]);
    ns::ctf_isotropic<"h2h">(target, Shape{logical_shape[1]}, target_ctf, {.fftfreq_range = {0.1, 0.4}});

    // Output CTF.
    const auto output = noa::empty<f64, 2>(shape_rfft);
    nx::equiphase_rescale_1d(
        input, {.start = 0., .stop = 0.5}, input_ctfs,
        output, {.start = 0.1, .stop = 0.4}, target_ctf
    );

    // const auto directory = test::NOA_DATA_PATH / "xform";
    // save_vector_to_text(target.view(), directory / "test_expected_average.txt");
    // save_vector_to_text(output.view(), directory / "test_result_average.txt");

    REQUIRE(test::allclose_abs(noa::broadcast(target.reshape(target.shape().push_front(1)), output.shape()), output, 5e-2));
}

// TEST_CASE("xform::rotational_average_anisotropic, test", "[.]") {
//     const auto directory = test::NOA_DATA_PATH / "xform";
//     const auto shape = Shape4{1, 1, 1024, 1024};
//     const auto n_shells = noa::min(shape.filter(2, 3)) / 2 + 1;
//
//     using CTFAnisotropic64 = ns::CTFAnisotropic<f64>;
//     const auto ctf_aniso = CTFAnisotropic64({1.8, 2.2}, {1., 0.35, 1.563}, 300, 0., 2.7, 1.570796327, 0, 1);
//
//     const auto ctf = noa::empty<f32>(shape);
//     ns::ctf_anisotropic<"FC2FC">(ctf, shape, ctf_aniso);
//     noa::write_image(ctf, directory / "test_simulated_ctf.mrc");
//
//     const auto rotational_average = noa::empty<f32>(n_shells);
//     nx::rotational_average_anisotropic<"FC2H">(
//         ctf, shape, ctf_aniso, rotational_average);
//     noa::write_text(
//         fmt::format("{}", fmt::join(rotational_average.span_1d(), ",")),
//         directory / "test_rotational_average.txt");
// }


// namespace {
//     using namespace noa::types;
//     namespace ns = noa::signal;
//     namespace ni = noa::indexing;
//
//     struct ReduceAnisotropic {
//         SpanContiguous<const f32, 3> polar;
//         ns::CTFIsotropic<f64> isotropic_ctf; // target
//         ns::CTFAnisotropic<f64> anisotropic_ctf; // actual
//
//         f64 phi_start;
//         f64 phi_step;
//
//         f64 rho_start;
//         f64 rho_step;
//         f64 rho_range;
//
//         constexpr void init(i64 batch, i64 row, i64 col, f64& r0, f64& r1) const {
//             auto phi = static_cast<f64>(row) * phi_step + phi_start; // radians
//             auto rho = static_cast<f64>(col) * rho_step + rho_start; // fftfreq
//
//             // Get the target phase.
//             auto phase = isotropic_ctf.phase_at(rho);
//
//             // Get the corresponding fftfreq within the astigmatic field.
//             auto ctf = ns::CTFIsotropic(anisotropic_ctf);
//             ctf.set_defocus(anisotropic_ctf.defocus_at(phi));
//             auto fftfreq = ctf.fftfreq_at(phase);
//
//             // Scale back to unnormalized frequency.
//             const auto width = polar.shape().width();
//             const auto frequency = static_cast<f64>(width - 1) * (fftfreq - rho_start) / rho_range;
//
//             // Lerp the polar array at this frequency.
//             const auto floored = noa::floor(frequency);
//             const auto fraction = frequency - floored;
//             const auto index = static_cast<i64>(floored);
//
//             f32 v0{}, w0{}, v1{}, w1{};
//             if (index >= 0 and index < width) {
//                 v0 = polar(batch, row, index);
//                 w0 = 1;
//             }
//             if (index + 1 >= 0 and index + 1 < width) {
//                 v1 = polar(batch, row, index + 1);
//                 w1 = 1;
//             }
//             r0 += v0 * (1 - fraction) + v1 * fraction;
//             r1 += w0 * (1 - fraction) + w1 * fraction;
//         }
//
//         static constexpr void join(f64 r0, f64 r1, f64& j0, f64& j1) {
//             j0 += r0;
//             j1 += r1;
//         }
//
//         using remove_default_final = bool;
//         static constexpr void final(f64 j0, f64 j1, f32& f) {
//             f = j1 > 1 ? static_cast<f32>(j0 / j1) : 0.f;
//         }
//     };
// }
//
// TEST_CASE("xform::spectrum2polar, anisotropic") {
//     std::vector<Device> devices{"cpu"};
//     if (Device::is_any_gpu())
//         devices.emplace_back("gpu");
//
//
//     constexpr auto spectrum_shape = Shape<i64, 4>{1, 1, 512, 512};
//     constexpr auto polar_shape = Shape<i64, 4>{1, 1, 512, 257};
//     const auto ctf = ns::CTFAnisotropic<f64>::Parameters{
//         .pixel_size = {2., 2.},
//         .defocus = {3., 0., 0.},
//         .voltage = 300.,
//         .amplitude = 0.7,
//         .cs = 2.7,
//         .phase_shift = 0,
//         .bfactor = 0.,
//         .scale = 1.,
//     }.to_ctf();
//
//     for (auto device: devices) {
//         const auto options = ArrayOption{device, Allocator::MANAGED};
//         Stream::current(device).set_thread_limit(1);
//         INFO(device);
//
//         const auto input_range = noa::Linspace{0., 0.5, true};
//         const auto rho_range = noa::Linspace{0.1, 0.4, true};
//         const auto phi_range = noa::Linspace{0., noa::Constant<f64>::PI, true};
//
//         const auto spectrum = noa::Array<f32>(spectrum_shape.rfft(), options);
//         ns::ctf_anisotropic<"h2h">(spectrum, spectrum_shape, ctf, {
//             .fftfreq_range = input_range,
//             .ctf_squared = true,
//         });
//
//         const auto polar = noa::Array<f32>(polar_shape, options);
//
//         auto spectrum2 = spectrum.copy();
//         noa::cubic_bspline_prefilter(spectrum2, spectrum2);
//         nx::spectrum2polar<"h2fc">(spectrum2, spectrum_shape, polar, {
//             .spectrum_fftfreq = input_range,
//             .rho_range = rho_range,
//             .phi_range = phi_range,
//             .interp = nx::Interp::CUBIC_BSPLINE,
//         });
//
//         noa::write_image(polar, "/Users/cix56657/Tmp/test_polar.mrc");
//
//         const auto rotational_average = noa::Array<f32>(polar.shape().set<2>(1), options);
//         noa::reduce_axes_iwise(polar.shape().filter(0, 2, 3), device, noa::wrap(f64{}, f64{}), rotational_average, ReduceAnisotropic{
//             .polar = polar.span().filter(0, 2, 3).as_contiguous(),
//             .isotropic_ctf = ns::CTFIsotropic(ctf),
//             .anisotropic_ctf = ctf,
//             .phi_start = phi_range.start,
//             .phi_step = phi_range.for_size(polar_shape.height()).step,
//             .rho_start = rho_range.start,
//             .rho_step = rho_range.for_size(polar_shape.width()).step,
//             .rho_range = rho_range.stop - rho_range.start, // assumes endpoint=true
//         });
//
//         noa::write_image(rotational_average, "/Users/cix56657/Tmp/test_rotational_average.mrc");
//
//         const auto rotational_average2 = noa::Array<f32>(polar.shape().set<2>(1), options);
//         nx::rotational_average_anisotropic<"h2h">(spectrum, spectrum_shape, ctf, rotational_average2, {}, {
//             .input_fftfreq = input_range,
//             .output_fftfreq = rho_range
//         });
//
//         noa::write_image(rotational_average2, "/Users/cix56657/Tmp/test_rotational_average2.mrc");
//     }
// }
