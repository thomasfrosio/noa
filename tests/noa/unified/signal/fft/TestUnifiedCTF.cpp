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

// TODO rotation average
// TODO astig is same as iso
