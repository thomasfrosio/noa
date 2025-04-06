#include <noa/core/io/TextFile.hpp>
#include <noa/core/utils/Zip.hpp>
#include <noa/unified/IO.hpp>

#include <noa/unified/fft/Remap.hpp>
#include <noa/unified/geometry/DrawShape.hpp>
#include <noa/unified/geometry/PolarTransformSpectrum.hpp>
#include <noa/unified/geometry/RotationalAverage.hpp>
#include <noa/unified/Reduce.hpp>
#include <noa/unified/signal/CTF.hpp>

#include "Assets.hpp"
#include "Utils.hpp"
#include "Catch.hpp"

using namespace noa::types;

TEST_CASE("unified::geometry::fft::rotational_average") {
    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    const i64 size = GENERATE(64, 65);
    INFO(size);

    // Handpicked frequencies that fall exactly onto a shell.
    const auto frequency_range = noa::Linspace{
        size == 64 ? 0.125 : 0.2,
        size == 64 ? 0.3125 : 0.4,
    };
    const i64 frequency_range_n_shells = size == 64 ? 13 : 14;
    const auto frequency_range_start_index = size == 64 ? 8 : 13;
    const auto subregion_within_full_range = noa::indexing::make_subregion<4>(
        noa::indexing::Ellipsis{},
        noa::indexing::Slice{
            frequency_range_start_index,
            frequency_range_start_index + frequency_range_n_shells
        }
    );

    const i64 n_batch = 3;
    const auto shape = Shape4<i64>{n_batch, 1, size, size};
    const auto rotational_average_size = noa::min(shape.filter(2, 3)) / 2 + 1;
    const auto polar_shape = Shape4<i64>{n_batch, 1, 256, rotational_average_size};
    const auto rotational_average_shape = Shape4<i64>{n_batch, 1, 1, rotational_average_size};
    const auto center = (shape.vec.filter(2, 3) / 2).as<f64>();

    for (auto device: devices) {
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        const auto input = noa::zeros<f32>(shape, options);
        noa::geometry::draw_shape({}, input, noa::geometry::Sphere{
            .center=center,
            .radius=0.,
            .smoothness=noa::min(center),
        });

        // Rotational average using polar transformation.
        const auto input_rfft = noa::fft::remap(noa::Remap::FC2HC, input, shape);
        const auto polar = noa::zeros<f32>(polar_shape, options);
        noa::geometry::spectrum2polar<"HC2FC">(input_rfft, shape, polar);
        const auto polar_reduced = noa::zeros<f32>(rotational_average_shape, options);
        noa::mean(polar, polar_reduced);

        // Rotational average.
        const auto output = noa::empty<f32>(rotational_average_shape, options);
        const auto weight = noa::empty<f32>(rotational_average_shape, options);
        noa::geometry::rotational_average<noa::Remap::HC2H>(input_rfft, shape, output, weight);

        // fmt::print("{:.3f}\n", fmt::join(polar_reduced.eval().span_1d(), ","));
        // fmt::print("{:.3f}\n", fmt::join(output.eval().span_1d(), ","));
        // fmt::print("{:.3f}\n", fmt::join(weight.eval().span_1d(), ","));

        REQUIRE(test::allclose_abs(polar_reduced, output, 1e-3));

        // Rotational average within a range.
        // Use the same number of shells, so it can be compared with the full range.
        const auto output_range = noa::empty<f32>({n_batch, 1, 1, frequency_range_n_shells}, options);
        const auto weight_range = noa::like(output_range);
        noa::geometry::rotational_average<"HC2H">(
            input_rfft, shape, output_range, weight_range, {.output_fftfreq = frequency_range});

        const auto output_cropped = output.subregion(subregion_within_full_range);
        const auto weight_cropped = weight.subregion(subregion_within_full_range);

        // fmt::print("{:.3f}\n", fmt::join(output_range.eval().span_1d(), ","));
        // fmt::print("{:.3f}\n", fmt::join(weight_range.eval().span_1d(), ","));

        // We don't expect the first and last shell to be the same as with the full range because of the lerp
        // that happens when collecting the signal from every frequency. Here, the cutoff excludes everything
        // that is not in the range, but with the full range and then cropping, we still have the contribution
        // from the frequencies between the first and first-1 shells, and last and last+1 shells.
        using namespace noa::indexing;
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

TEST_CASE("unified::geometry::fft::rotational_average_anisotropic, vs isotropic") {
    // Test that with an isotropic ctf it gives the same results as the classic rotational average.

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    const auto shape = Shape4<i64>{1, 1, 1024, 1024}; //test::get_random_shape4_batched(2);
    const auto n_shells = noa::min(shape.filter(2, 3)) / 2 + 1;

    using CTFIsotropic64 = noa::signal::CTFIsotropic<f64>;
    using CTFAnisotropic64 = noa::signal::CTFAnisotropic<f64>;
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
        const auto rotational_averages = noa::empty<f32>({2, 1, 1, n_shells}, options);
        const auto result = rotational_averages.subregion(0);
        const auto expected = rotational_averages.subregion(1);

        noa::signal::ctf_isotropic<"H2H">(simulated_ctf, shape, ctf_iso);

        noa::geometry::rotational_average<"H2H">(simulated_ctf, shape, result);
        noa::geometry::rotational_average_anisotropic<"H2H">(
            simulated_ctf, shape, CTFAnisotropic64::from_isotropic_ctf(ctf_iso), expected);

        REQUIRE(test::allclose_abs(result, expected, 1e-4));
    }

    std::array ctfs{
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

        const auto batched_shape = shape.set<0>(ctfs.size());
        const auto simulated_ctf = noa::empty<f32>(batched_shape.rfft(), options);
        const auto rotational_averages = noa::empty<f32>({ctfs.size(), 1, 1, n_shells}, options);
        const auto ctfs_array = View(ctfs.data(), ctfs.size()).to(options);

        noa::signal::ctf_anisotropic<"H2H">(simulated_ctf, batched_shape, ctfs_array);
        noa::geometry::rotational_average_anisotropic<"H2H">(
            simulated_ctf, batched_shape, ctfs_array, rotational_averages);

        const auto expected = rotational_averages.subregion(0);
        for (auto i: noa::irange(size_t{1}, ctfs.size())) {
            REQUIRE(test::allclose_abs(expected, rotational_averages.subregion(i), 1e-3));
        }
    }
}

namespace {
    template<typename Real>
    void save_vector_to_text(View<Real> x, const Path& filename) {
        noa::check(noa::indexing::is_contiguous_vector_batched_strided(x));

        std::string format;
        for (auto i: noa::irange(x.shape().batch()))
            format += fmt::format("{}\n", fmt::join(x.subregion(i).span_1d(), ","));
        noa::write_text(format, filename);
    }
}

// TODO Add better tests...
TEST_CASE("unified::geometry::fft::fuse_rotational_averages") {
    // Simulate two 1d CTFs with different spacing and defocus, then fuse to another CTF, and check that they match.
    const auto directory = test::NOA_DATA_PATH / "geometry";

    using CTFIsotropic64 = noa::signal::CTFIsotropic<f64>;
    constexpr auto defocus = std::array{2.15, 2.90, 2.4};
    constexpr auto spacing = std::array{1.80, 2.40, 2.};
    constexpr auto phase_shifts = std::array{0.45, 0.75, 0.55};
    auto input_ctfs = Array<CTFIsotropic64>(3);
    for (auto&& [d, s, p, ctf]: noa::zip(defocus, spacing, phase_shifts, input_ctfs.span_1d_contiguous())) {
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
    const auto input = noa::Array<f64>({3, 1, 1, size / 2 + 1});
    noa::signal::ctf_isotropic<"h2h">(input, input.shape().set<3>(size), input_ctfs);
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

    const auto target = noa::Array<f64>({1, 1, 1, size / 2 + 1});
    noa::signal::ctf_isotropic<"h2h">(target, target.shape().set<3>(size), target_ctf, {.fftfreq_range = {0.1, 0.4}});

    // Output CTF.
    const auto output = noa::like(target);
    noa::geometry::fuse_rotational_averages(
        input, {.start = 0., .stop = 0.5}, input_ctfs,
        output, {.start = 0.1, .stop = 0.4}, target_ctf
    );
    // save_vector_to_text(target.view(), directory / "test_expected_average.txt");
    // save_vector_to_text(output.view(), directory / "test_result_average.txt");

    REQUIRE(test::allclose_abs(target, output, 5e-2));
}

TEST_CASE("unified::geometry::fft::rotational_average_anisotropic, test", "[.]") {
    const auto directory = test::NOA_DATA_PATH / "geometry";
    const auto shape = Shape4<i64>{1, 1, 1024, 1024};
    const auto n_shells = noa::min(shape.filter(2, 3)) / 2 + 1;

    using CTFAnisotropic64 = noa::signal::CTFAnisotropic<f64>;
    const auto ctf_aniso = CTFAnisotropic64({1.8, 2.2}, {1., 0.35, 1.563}, 300, 0., 2.7, 1.570796327, 0, 1);

    const auto ctf = noa::empty<f32>(shape);
    noa::signal::ctf_anisotropic<"FC2FC">(ctf, shape, ctf_aniso);
    noa::write(ctf, directory / "test_simulated_ctf.mrc");

    const auto rotational_average = noa::empty<f32>(n_shells);
    noa::geometry::rotational_average_anisotropic<"FC2H">(
        ctf, shape, ctf_aniso, rotational_average);
    noa::write_text(
        fmt::format("{}", fmt::join(rotational_average.span_1d(), ",")),
        directory / "test_rotational_average.txt");
}
