#include <noa/FFT.hpp>
#include <noa/Geometry.hpp>
#include <noa/IO.hpp>
#include <noa/Math.hpp>
#include <noa/Memory.hpp>
#include <noa/Signal.hpp>

#include <catch2/catch.hpp>

#include "Helpers.h"
#include "Assets.h"

using namespace noa;

TEST_CASE("unified::geometry::fft::rotational_average", "[noa][unified]") {
    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const i64 size = GENERATE(64, 65);
    INFO(size);

    // Handpicked frequencies that fall exactly onto a shell.
    const Vec2<f32> frequency_range{
            size == 64 ? 0.125 : 0.2,
            size == 64 ? 0.3125 : 0.4
    };
    const i64 frequency_range_n_shells = size == 64 ? 13 : 14;
    const auto frequency_range_start_index = size == 64 ? 8 : 13;
    const auto subregion_within_full_range = noa::indexing::Subregion(
            noa::indexing::Ellipsis{},
            noa::indexing::Slice{frequency_range_start_index, frequency_range_start_index + frequency_range_n_shells});

    const i64 batches = 3;
    const auto shape = Shape4<i64>{batches, 1, size, size};
    const auto rotational_average_size = noa::math::min(shape.filter(2, 3)) / 2 + 1;
    const auto polar_shape = Shape4<i64>{batches, 1, 256, rotational_average_size};
    const auto rotational_average_shape = Shape4<i64>{batches, 1, 1, rotational_average_size};
    const auto center = (shape.vec().filter(2, 3) / 2).as<f32>();

    for (auto device: devices) {
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        const auto input = noa::memory::zeros<f32>(shape, options);
        noa::geometry::fft::sphere<fft::FC2FC>({}, input, center, 0.f, noa::math::min(center));

        // Rotational average using polar transformation.
        const auto input_rfft = noa::fft::remap(fft::FC2HC, input, shape);
        const auto polar = noa::memory::zeros<f32>(polar_shape, options);
        noa::geometry::fft::cartesian2polar<fft::HC2FC>(input_rfft, shape, polar);
        const auto polar_reduced = noa::memory::zeros<f32>(rotational_average_shape, options);
        noa::math::mean(polar, polar_reduced);

        // Rotational average.
        const auto output = noa::memory::empty<f32>(rotational_average_shape, options);
        const auto weight = noa::memory::empty<f32>(rotational_average_shape, options);
        noa::geometry::fft::rotational_average<fft::HC2H>(input_rfft, shape, output, weight);

//        fmt::print("{:.3f}\n", fmt::join(polar_reduced.eval().span(), ","));
//        fmt::print("{:.3f}\n", fmt::join(output.eval().span(), ","));
//        fmt::print("{:.3f}\n", fmt::join(weight.eval().span(), ","));

        REQUIRE(test::Matcher(test::MATCH_ABS, polar_reduced, output, 1e-3));

        // Rotational average within a range.
        // Use the same number of shells, so it can be compared with the full range.
        const auto output_range = noa::memory::empty<f32>({batches, 1, 1, frequency_range_n_shells}, options);
        const auto weight_range = noa::memory::like(output_range);
        noa::geometry::fft::rotational_average<fft::HC2H>(
                input_rfft, shape, output_range, weight_range, frequency_range);

        const auto output_cropped = output.subregion(subregion_within_full_range);
        const auto weight_cropped = weight.subregion(subregion_within_full_range);

//        fmt::print("{:.3f}\n", fmt::join(output_range.eval().span(), ","));
//        fmt::print("{:.3f}\n", fmt::join(weight_range.eval().span(), ","));

        // We don't expect the first and last shell to be the same as with the full range. Because of the lerp
        // that happens when collecting the signal from every frequency. Here, the cutoff excludes everything
        // that is not in the range, but with the full range and then cropping, we still have the contribution
        // from the frequencies between the first and first-1 shells, and last and last+1 shells.
        using namespace noa::indexing;
        REQUIRE(test::Matcher(test::MATCH_ABS,
                              output_cropped.subregion(Ellipsis{}, Slice{1, -1}),
                              output_range.subregion(Ellipsis{}, Slice{1, -1}),
                              1e-3));
        REQUIRE(test::Matcher(test::MATCH_ABS,
                              weight_cropped.subregion(Ellipsis{}, Slice{1, -1}),
                              weight_range.subregion(Ellipsis{}, Slice{1, -1}),
                              1e-3));
    }
}

TEST_CASE("unified::geometry::fft::rotational_average_anisotropic, vs isotropic", "[noa][unified]") {
    // Test that with an isotropic ctf it gives the same results as the classic rotational average.

    const auto directory = test::NOA_DATA_PATH / "geometry" / "fft";

    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const auto shape = Shape4<i64>{1, 1, 1024, 1024}; //test::get_random_shape4_batched(2);
    const auto n_shells = noa::math::min(shape.filter(2, 3)) / 2 + 1;

    using CTFIsotropic64 = noa::signal::fft::CTFIsotropic<f64>;
    using CTFAnisotropic64 = noa::signal::fft::CTFAnisotropic<f64>;
    const auto ctf_iso = CTFIsotropic64(3, 2, 300, 0.7, 2.7, 1.570796327, 0);

    // First test that if no anisotropy, it generates exactly the same rotational average.
    for (auto device: devices) {
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        const auto simulated_ctf = noa::memory::empty<f32>(shape.rfft(), options);
        const auto rotational_averages = noa::memory::empty<f32>({2, 1, 1, n_shells}, options);
        const auto result = rotational_averages.subregion(0);
        const auto expected = rotational_averages.subregion(1);

        noa::signal::fft::ctf_isotropic<fft::H2H>(simulated_ctf, shape, ctf_iso);

        noa::geometry::fft::rotational_average<fft::H2H>(simulated_ctf, shape, result);
        noa::geometry::fft::rotational_average_anisotropic<fft::H2H>(
                simulated_ctf, shape, CTFAnisotropic64(ctf_iso), expected);

        REQUIRE(test::Matcher(test::MATCH_ABS, result, expected, 1e-4));
    }

    std::array<CTFAnisotropic64, 6> ctfs{
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
        const auto simulated_ctf = noa::memory::empty<f32>(batched_shape.rfft(), options);
        const auto rotational_averages = noa::memory::empty<f32>({ctfs.size(), 1, 1, n_shells}, options);
        const auto ctfs_view = View(ctfs.data(), ctfs.size()).to(options);

        noa::signal::fft::ctf_anisotropic<fft::H2H>(simulated_ctf, batched_shape, ctfs_view);
        noa::geometry::fft::rotational_average_anisotropic<fft::H2H>(
                simulated_ctf, batched_shape, ctfs_view, rotational_averages);

        const auto expected = rotational_averages.subregion(0);
        for (auto i: noa::irange(size_t{1}, ctfs.size())) {
            REQUIRE(test::Matcher(test::MATCH_ABS, expected, rotational_averages.subregion(i), 1e-3));
        }
    }
}

TEST_CASE("unified::geometry::fft::rotational_average_anisotropic, test", "[.]") {
    // Test that with an isotropic ctf it gives the same results as the classic rotational average.

    const auto directory = test::NOA_DATA_PATH / "geometry" / "fft";
    const auto shape = Shape4<i64>{1, 1, 1024, 1024};
    const auto n_shells = noa::math::min(shape.filter(2, 3)) / 2 + 1;

    using CTFAnisotropic64 = noa::signal::fft::CTFAnisotropic<f64>;
    const auto ctf_aniso = CTFAnisotropic64({1.8, 2.2}, {1., 0.35, 1.563}, 300, 0., 2.7, 1.570796327, 0);

    const auto ctf = noa::memory::empty<f32>(shape);
    noa::signal::fft::ctf_anisotropic<fft::FC2FC>(ctf, shape, ctf_aniso);
    noa::io::save(ctf, directory / "test_simulated.mrc");

    const auto rotational_average = noa::memory::empty<f32>(n_shells);
    noa::geometry::fft::rotational_average_anisotropic<fft::FC2H>(
            ctf, shape, ctf_aniso, rotational_average);
    noa::io::save_text(fmt::format("{}", fmt::join(rotational_average.span(), ",")),
                       directory /"test_rotational_average.txt");
}
