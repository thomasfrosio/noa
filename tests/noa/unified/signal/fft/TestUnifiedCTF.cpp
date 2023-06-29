#include <noa/unified/signal/fft/CTF.hpp>
#include <noa/unified/memory/Factory.hpp>
#include <noa/unified/math/Random.hpp>
#include <noa/unified/fft/Remap.hpp>
#include <noa/unified/io/ImageFile.hpp>
#include <noa/unified/Ewise.hpp>

#include <catch2/catch.hpp>

#include "Helpers.h"
#include "Assets.h"

using namespace noa;

TEST_CASE("unified::signal::fft::ctf_isotropic, assets", "[noa][unified][assets]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector<Device> devices{Device("cpu")};
    if (!COMPUTE_ASSETS && Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const Path path_base = test::NOA_DATA_PATH / "signal" / "fft";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["ctf_isotropic"];
    using CTFIsotropic64 = noa::signal::fft::CTFIsotropic<f64>;

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

        const auto ctf = CTFIsotropic64(pixel_size, defocus, voltage, amplitude, cs, phase_shift, bfactor);

        if constexpr (COMPUTE_ASSETS) {
            const auto input = noa::memory::empty<f32>(shape.rfft());
            noa::signal::fft::ctf_isotropic<fft::H2H>({}, input, shape, ctf);
            noa::io::save(input, filename_asset);
        } else {
            for (auto device: devices) {
                INFO(device);
                const auto options = ArrayOption(device, Allocator::MANAGED);
                const auto result_fft = noa::memory::empty<f32>(shape, options);
                const auto result_rfft = noa::memory::empty<f32>(shape.rfft(), options);
                const auto expected_rfft = noa::io::load_data<f32>(filename_asset, false, options);

                // Check against asset.
                noa::signal::fft::ctf_isotropic<fft::H2H>({}, result_rfft, shape, ctf);
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, result_rfft, expected_rfft, 1e-6));

                // Remap
                noa::signal::fft::ctf_isotropic<fft::HC2HC>({}, result_rfft, shape, ctf);
                auto expected_hc = noa::fft::remap(fft::H2HC, expected_rfft, shape);
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, result_rfft, expected_hc.release(), 1e-6));

                noa::signal::fft::ctf_isotropic<fft::F2F>({}, result_fft, shape, ctf);
                auto expected_f = noa::fft::remap(fft::H2F, expected_rfft, shape);
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, result_fft, expected_f.release(), 1e-6));

                noa::signal::fft::ctf_isotropic<fft::FC2FC>({}, result_fft, shape, ctf);
                auto expected_fc = noa::fft::remap(fft::H2FC, expected_rfft, shape);
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, result_fft, expected_fc.release(), 1e-6));

                // Remap-Multiply.
                auto input_rfft = noa::math::random<f32>(noa::math::normal_t{}, shape.rfft(), -5, 5, options);
                noa::signal::fft::ctf_isotropic<fft::HC2HC>({}, result_rfft, shape, ctf);
                auto input_rfft_centered = noa::fft::remap(fft::H2HC, input_rfft, shape);
                noa::ewise_binary(input_rfft_centered, result_rfft, input_rfft_centered, noa::multiply_t{});
                noa::signal::fft::ctf_isotropic<fft::H2HC>(input_rfft.copy(), input_rfft, shape, ctf);
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, input_rfft.release(), input_rfft_centered.release(), 1e-6));

                auto input_fft = noa::math::random<f32>(noa::math::normal_t{}, shape, -5, 5, options);
                noa::signal::fft::ctf_isotropic<fft::FC2FC>({}, result_fft, shape, ctf);
                auto input_fft_centered = noa::fft::remap(fft::F2FC, input_fft, shape);
                noa::ewise_binary(input_fft_centered, result_fft, input_fft_centered, noa::multiply_t{});
                noa::signal::fft::ctf_isotropic<fft::F2FC>(input_fft.copy(), input_fft, shape, ctf);
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, input_fft.release(), input_fft_centered.release(), 1e-6));
            }
        }
    }
}

TEST_CASE("unified::signal::fft::ctf_isotropic, default range", "[noa][unified]") {
    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const i64 ndim = GENERATE(1, 2, 3);
    using CTFIsotropic64 = noa::signal::fft::CTFIsotropic<f64>;
    const auto ctf = CTFIsotropic64(2.1, 2.5, 300, 0.1, 2.7, 0, 10);

    for (auto device: devices) {
        INFO(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        const auto shape = test::get_random_shape4_batched(ndim);

        const auto input = noa::memory::ones<f32>(shape, options);
        const auto output = noa::memory::like(input);
        const auto expected_0 = noa::memory::like(input);
        const auto expected_1 = noa::memory::like(input);

        noa::signal::fft::ctf_isotropic<fft::FC2FC>(input, output, shape, ctf);
        noa::signal::fft::ctf_isotropic<fft::FC2FC>({}, expected_0, shape, ctf);
        noa::signal::fft::ctf_isotropic<fft::FC2FC>(expected_1, shape, ctf);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, output, expected_0, 1e-6));
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, output, expected_1, 1e-6));
    }
}

TEST_CASE("unified::signal::fft::ctf_isotropic, range", "[noa][unified]") {
    // Generate the full range, and truncate to subregion. Then generate the truncated range directly
    // with the same number of elements as in the subregion and check these ranges are equal.

    using CTFIsotropic64 = noa::signal::fft::CTFIsotropic<f64>;
    const auto ctf = CTFIsotropic64(2.1, 2.67, 300, 0.07, 2.7, 0, 0);
    const auto resolution_range = Vec2<f64>{40, 10};
    const bool endpoint = GENERATE(true, false);

    const auto trimmed_range = [endpoint](
            const Vec2<f64>& fitting_range, // angstrom
            const Vec2<f64>& spacing, // angstrom/pixel
            i64 logical_size
    ) -> std::tuple<i64, noa::indexing::Slice, Vec2<f64>> {
        const auto logical_size_f = static_cast<f64>(logical_size);
        auto frequency_cutoff = noa::math::round(spacing / fitting_range * logical_size_f);
        const auto index_cutoff = frequency_cutoff.as<i64>();

        if (endpoint)
            frequency_cutoff[1] -= 1;
        const auto normalized_frequency_cutoff = frequency_cutoff / logical_size_f;
        const auto actual_fitting_range = spacing / normalized_frequency_cutoff;

        // For the new logical size, simply compute the even size.
        const auto new_size = index_cutoff[1] - index_cutoff[0];
        const auto new_logical_size = (new_size - 1) * 2;

        return {new_logical_size, noa::indexing::Slice{index_cutoff[0], index_cutoff[1]}, actual_fitting_range};
    };

    // Generate full range.
    const auto shape = Shape4<i64>{1, 1, 1, 512};
    const auto output = noa::memory::empty<f32>(shape.rfft());
    noa::signal::fft::ctf_isotropic<fft::H2H>({}, output, shape, ctf, false, true);

    // Get the truncated range and truncate the full range.
    const auto [trimmed_size, trimmed_slice, trimmed_resolution_range] =
            trimmed_range(resolution_range, Vec2<f64>(ctf.pixel_size()), shape.elements());
    const auto output_truncated = output.subregion(noa::indexing::Ellipsis{}, trimmed_slice);

    // Generate the truncated
    const auto output_range = noa::memory::empty<f32>(trimmed_size / 2 + 1);
    noa::signal::fft::ctf_isotropic<fft::H2H>(
            output_range, {1, 1, 1, trimmed_size}, ctf, false, true,
            (ctf.pixel_size() / trimmed_resolution_range).as<f32>(), endpoint);

    REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, output_truncated, output_range, 1e-6));
}

// TODO rotation average
// TODO astig is same as iso
