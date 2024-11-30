#include <noa/unified/signal/CTF.hpp>
#include <noa/unified/Factory.hpp>
#include <noa/unified/Random.hpp>
#include <noa/unified/fft/Remap.hpp>
#include <noa/unified/IO.hpp>
#include <noa/unified/Ewise.hpp>

#include <catch2/catch.hpp>

#include "Utils.hpp"
#include "Assets.h"

using namespace noa::types;
using Remap = noa::Remap;

TEST_CASE("unified::signal::ctf_isotropic, assets", "[noa][unified][assets]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector<Device> devices{"cpu"};
    if (not COMPUTE_ASSETS and Device::is_any_gpu())
        devices.emplace_back("gpu");

    const Path path_base = test::NOA_DATA_PATH / "signal" / "fft";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["ctf_isotropic"];
    using CTFIsotropic64 = noa::signal::CTFIsotropic<f64>;

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const auto& test = tests[nb];
        const auto filename_asset = path_base / test["output"].as<Path>();
        const auto shape = test["shape"].as<Shape4<i64>>();
        const auto pixel_size = test["pixel_size"].as<f64>();
        const auto defocus = test["defocus"].as<f64>();
        const auto voltage = test["voltage"].as<f64>();
        const auto amplitude = test["amplitude"].as<f64>();
        const auto cs = test["cs"].as<f64>();
        const auto phase_shift = test["phase_shift"].as<f64>();
        const auto bfactor = test["bfactor"].as<f64>();

        const auto ctf = CTFIsotropic64(pixel_size, defocus, voltage, amplitude, cs, phase_shift, bfactor, 1);

        static_assert(noa::traits::spectrum_types<float, float>);

        if constexpr (COMPUTE_ASSETS) {
            const auto input = noa::empty<f32>(shape.rfft());
            noa::signal::ctf_isotropic<Remap::H2H>({}, input, shape, ctf);
            noa::io::write(input, filename_asset);
        } else {
            for (auto device: devices) {
                INFO(device);
                const auto options = ArrayOption(device, Allocator::MANAGED);
                const auto result_fft = noa::empty<f32>(shape, options);
                const auto result_rfft = noa::empty<f32>(shape.rfft(), options);
                const auto expected_rfft = noa::io::read_data<f32>(filename_asset, {}, options);

                // Check against asset.
                noa::signal::ctf_isotropic<Remap::H2H>({}, result_rfft, shape, ctf);
                REQUIRE(test::allclose_abs(result_rfft, expected_rfft, 1e-4));

                // Remap
                noa::signal::ctf_isotropic<Remap::HC2HC>({}, result_rfft, shape, ctf);
                auto expected_hc = noa::fft::remap(Remap::H2HC, expected_rfft, shape);
                REQUIRE(test::allclose_abs(result_rfft, expected_hc.release(), 1e-4));

                noa::signal::ctf_isotropic<Remap::F2F>({}, result_fft, shape, ctf);
                auto expected_f = noa::fft::remap(Remap::H2F, expected_rfft, shape);
                REQUIRE(test::allclose_abs(result_fft, expected_f.release(), 1e-4));

                noa::signal::ctf_isotropic<Remap::FC2FC>({}, result_fft, shape, ctf);
                auto expected_fc = noa::fft::remap(Remap::H2FC, expected_rfft, shape);
                REQUIRE(test::allclose_abs(result_fft, expected_fc.release(), 1e-4));

                // Remap-Multiply.
                auto input_rfft = noa::random(noa::Normal<f32>{-5, 5}, shape.rfft(), options);
                noa::signal::ctf_isotropic<Remap::HC2HC>({}, result_rfft, shape, ctf);
                auto input_rfft_centered = noa::fft::remap(Remap::H2HC, input_rfft, shape);
                noa::ewise(noa::wrap(input_rfft_centered, result_rfft), input_rfft_centered, noa::Multiply{});
                noa::signal::ctf_isotropic<Remap::H2HC>(input_rfft.copy(), input_rfft, shape, ctf);
                REQUIRE(test::allclose_abs(input_rfft.release(), input_rfft_centered.release(), 1e-4));

                auto input_fft = noa::random(noa::Normal<f32>{-5, 5}, shape, options);
                noa::signal::ctf_isotropic<Remap::FC2FC>({}, result_fft, shape, ctf);
                auto input_fft_centered = noa::fft::remap(Remap::F2FC, input_fft, shape);
                noa::ewise(noa::wrap(input_fft_centered, result_fft), input_fft_centered, noa::Multiply{});
                noa::signal::ctf_isotropic<Remap::F2FC>(input_fft.copy(), input_fft, shape, ctf);
                REQUIRE(test::allclose_abs(input_fft.release(), input_fft_centered.release(), 1e-4));
            }
        }
    }
}

TEST_CASE("unified::signal::ctf_isotropic, default range", "[noa][unified]") {
    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    const i64 ndim = GENERATE(1, 2, 3);
    using CTFIsotropic64 = noa::signal::CTFIsotropic<f64>;
    const auto ctf = CTFIsotropic64::Parameters{
        .pixel_size = 2.1,
        .defocus = 2.5,
        .voltage = 300.,
        .amplitude = 0.1,
        .cs = 2.7,
        .phase_shift = 0.,
        .bfactor = 10.,
        .scale = 1.,
    }.to_ctf();

    for (auto device: devices) {
        INFO(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        const auto shape = test::random_shape_batched(ndim);

        const auto input = noa::ones<f32>(shape, options);
        const auto output = noa::like(input);
        const auto expected_0 = noa::like(input);
        const auto expected_1 = noa::like(input);

        noa::signal::ctf_isotropic<Remap::FC2FC>(input, output, shape, ctf);
        noa::signal::ctf_isotropic<Remap::FC2FC>({}, expected_0, shape, ctf);
        noa::signal::ctf_isotropic<Remap::FC2FC>(expected_1, shape, ctf);
        REQUIRE(test::allclose_abs(output, expected_0, 1e-4));
        REQUIRE(test::allclose_abs(output, expected_1, 1e-4));
    }
}

TEST_CASE("unified::signal::ctf_isotropic, range", "[noa][unified]") {
    // Generate the full range, and truncate to subregion. Then generate the truncated range directly
    // with the same number of elements as in the subregion and check these ranges are equal.

    using CTFIsotropic64 = noa::signal::CTFIsotropic<f64>;
    const auto ctf = CTFIsotropic64::Parameters{
        .pixel_size = 2.1,
        .defocus = 2.67,
        .voltage = 300.,
        .amplitude = 0.07,
        .cs = 2.7,
        .phase_shift = 0.,
        .bfactor = 0.,
        .scale = 1.,
    }.to_ctf();

    constexpr auto resolution_range = Vec{40., 10.};
    const bool endpoint = GENERATE(true, false);

    const auto trimmed_range = [endpoint](
        const Vec2<f64>& fitting_range, // angstrom
        const Vec2<f64>& spacing, // angstrom/pixel
        i64 logical_size
    ) -> std::tuple<i64, noa::indexing::Slice, Vec2<f64>> {
        const auto logical_size_f = static_cast<f64>(logical_size);
        auto frequency_cutoff = noa::round(spacing / fitting_range * logical_size_f);
        const auto index_cutoff = frequency_cutoff.as<i64>();

        if (endpoint)
            frequency_cutoff[1] -= 1;
        const auto normalized_frequency_cutoff = frequency_cutoff / logical_size_f;
        const auto actual_fitting_range = spacing / normalized_frequency_cutoff;

        // For the new logical size, compute the even size.
        const auto new_size = index_cutoff[1] - index_cutoff[0];
        const auto new_logical_size = (new_size - 1) * 2;

        return {new_logical_size, noa::indexing::Slice{index_cutoff[0], index_cutoff[1]}, actual_fitting_range};
    };

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto device: devices) {
        INFO(device);
        const auto options = ArrayOption{.device = device, .allocator = Allocator::MANAGED};

        // Generate full range.
        const auto shape = Shape4<i64>{1, 1, 1, 512};
        const auto output = noa::empty<f32>(shape.rfft(), options);
        noa::signal::ctf_isotropic<Remap::H2H>({}, output, shape, ctf, {.ctf_squared=true});

        // Get the truncated range and truncate the full range.
        const auto [trimmed_size, trimmed_slice, trimmed_resolution_range] =
            trimmed_range(resolution_range, Vec<f64, 2>::from_value(ctf.pixel_size()), shape.n_elements());
        const auto output_truncated = output.subregion(noa::indexing::Ellipsis{}, trimmed_slice);

        // Generate the truncated
        const auto output_range = noa::empty<f32>(trimmed_size / 2 + 1, options);
        noa::signal::ctf_isotropic<Remap::H2H>(
            output_range, {1, 1, 1, trimmed_size}, ctf, {
                .fftfreq_range = (ctf.pixel_size() / trimmed_resolution_range).as<f64>(),
                .fftfreq_range_endpoint = endpoint,
                .ctf_squared = true,
            });
        REQUIRE(test::allclose_abs(output_truncated, output_range, 1e-4));
    }
}

TEST_CASE("unified::signal::ctf_anisotropic, assets", "[noa][unified][assets]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector<Device> devices{"cpu"};
    if (not COMPUTE_ASSETS and Device::is_any_gpu())
        devices.emplace_back("gpu");

    const Path path_base = test::NOA_DATA_PATH / "signal" / "fft";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["ctf_anisotropic"];
    using CTFAnisotropic64 = noa::signal::CTFAnisotropic<f64>;
    using DefocusAstigmatic64 = noa::signal::DefocusAstigmatic<f64>;

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const auto& test = tests[nb];
        const auto filename_asset = path_base / test["output"].as<Path>();
        const auto shape = test["shape"].as<Shape4<i64>>();
        const auto pixel_size = test["pixel_size"].as<Vec2<f64>>();
        const auto defocus = test["defocus"].as<Vec3<f64>>();
        const auto voltage = test["voltage"].as<f64>();
        const auto amplitude = test["amplitude"].as<f64>();
        const auto cs = test["cs"].as<f64>();
        const auto phase_shift = test["phase_shift"].as<f64>();
        const auto bfactor = test["bfactor"].as<f64>();

        const auto defocus_astigmatic = DefocusAstigmatic64{defocus[0], defocus[1], noa::deg2rad(defocus[2])};
        const auto ctf = CTFAnisotropic64(pixel_size, defocus_astigmatic, voltage, amplitude, cs, phase_shift, bfactor, 1);

        if constexpr (COMPUTE_ASSETS) {
            const auto input = noa::empty<f32>(shape);
            noa::signal::ctf_anisotropic<Remap::FC2FC>({}, input, shape, ctf);
            noa::io::write(input, filename_asset);
        } else {
            for (auto device: devices) {
                INFO(device);
                const auto options = ArrayOption(device, Allocator::MANAGED);
                const auto result_fft = noa::empty<f32>(shape, options);
                const auto result_rfft = noa::empty<f32>(shape.rfft(), options);
                const auto expected_fft = noa::io::read_data<f32>(filename_asset, {}, options);

                // Check against asset.
                noa::signal::ctf_anisotropic<Remap::FC2FC>(result_fft, shape, ctf);
                REQUIRE(test::allclose_abs(result_fft, expected_fft, 1e-4));

                // Remap
                noa::signal::ctf_anisotropic<Remap::F2F>(result_fft, shape, ctf);
                auto expected_f = noa::fft::remap(Remap::FC2F, expected_fft, shape);
                REQUIRE(test::allclose_abs(result_fft, expected_f.release(), 1e-4));

                // If it is astigmatic and angle is not a multiple of pi/2, the astigmatic field breaks
                // the remapping after Nyquist for even size. This is documented in noa::fft::remap().
                if (defocus_astigmatic.angle == 0) {
                    noa::signal::ctf_anisotropic<Remap::H2H>(result_rfft, shape, ctf);
                    auto expected_r = noa::fft::remap(Remap::FC2H, expected_fft, shape);
                    REQUIRE(test::allclose_abs(result_rfft, expected_r.release(), 1e-4));

                    noa::signal::ctf_anisotropic<Remap::HC2HC>(result_rfft, shape, ctf);
                    auto expected_hc = noa::fft::remap(Remap::FC2HC, expected_fft, shape);
                    REQUIRE(test::allclose_abs(result_rfft, expected_hc.release(), 1e-4));
                }

                // Remap-Multiply.
                auto input_rfft = noa::random(noa::Normal<f32>{-5, 5}, shape.rfft(), options);
                noa::signal::ctf_anisotropic<Remap::HC2HC>({}, result_rfft, shape, ctf);
                auto input_rfft_centered = noa::fft::remap(Remap::H2HC, input_rfft, shape);
                noa::ewise(noa::wrap(input_rfft_centered, result_rfft), input_rfft_centered, noa::Multiply{});
                noa::signal::ctf_anisotropic<Remap::H2HC>(input_rfft.copy(), input_rfft, shape, ctf);
                REQUIRE(test::allclose_abs(input_rfft.release(), input_rfft_centered.release(), 1e-4));

                auto input_fft = noa::random(noa::Normal<f32>{-5, 5}, shape, options);
                noa::signal::ctf_anisotropic<Remap::FC2FC>({}, result_fft, shape, ctf);
                auto input_fft_centered = noa::fft::remap(Remap::F2FC, input_fft, shape);
                noa::ewise(noa::wrap(input_fft_centered, result_fft), input_fft_centered, noa::Multiply{});
                noa::signal::ctf_anisotropic<Remap::F2FC>(input_fft.copy(), input_fft, shape, ctf);
                REQUIRE(test::allclose_abs(input_fft.release(), input_fft_centered.release(), 1e-4));
            }
        }
    }
}

TEST_CASE("unified::signal::ctf_anisotropic, default range, vs isotropic", "[noa][unified]") {
    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    using CTFIsotropic64 = noa::signal::CTFIsotropic<f64>;
    using CTFAnisotropic64 = noa::signal::CTFAnisotropic<f64>;
    const auto ctf_iso = CTFIsotropic64::Parameters{
        .pixel_size = 2.1,
        .defocus = 2.5,
        .voltage = 300.,
        .amplitude = 0.1,
        .cs = 2.7,
        .phase_shift = 0.,
        .bfactor = 10.,
        .scale = 1.,
    }.to_ctf();
    const auto ctf_aniso = CTFAnisotropic64(ctf_iso);

    for (auto device: devices) {
        INFO(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        const auto shape = test::random_shape_batched(2);

        {
            const auto input = noa::ones<f32>(shape, options);
            const auto output = noa::like(input);
            const auto expected_0 = noa::like(input);
            const auto expected_1 = noa::like(input);

            noa::signal::ctf_anisotropic<Remap::FC2FC>(input, output, shape, ctf_aniso);
            noa::signal::ctf_anisotropic<Remap::FC2FC>({}, expected_0, shape, ctf_aniso);
            noa::signal::ctf_anisotropic<Remap::FC2FC>(expected_1, shape, ctf_aniso);
            REQUIRE(test::allclose_abs(output, expected_0, 1e-4));
            REQUIRE(test::allclose_abs(output, expected_1, 1e-4));
        }

        {
            const auto isotropic = noa::ones<f32>(shape.rfft(), options);
            const auto anisotropic = noa::like(isotropic);
            noa::signal::ctf_isotropic<Remap::H2H>(isotropic, shape, ctf_iso);
            noa::signal::ctf_anisotropic<Remap::H2H>(anisotropic, shape, ctf_aniso);
            REQUIRE(test::allclose_abs(isotropic, anisotropic, 1e-4));
        }
    }
}
