#include <noa/unified/geometry/fft/Polar.hpp>
#include <noa/unified/geometry/fft/Shape.hpp>
#include <noa/unified/geometry/Polar.hpp>
#include <noa/unified/geometry/Shape.hpp>
#include <noa/unified/math/Reduce.hpp>
#include <noa/unified/memory/Factory.hpp>
#include <noa/unified/fft/Remap.hpp>
#include <noa/unified/io/ImageFile.hpp>

#include <catch2/catch.hpp>

#include "Helpers.h"
#include "Assets.h"

using namespace noa;

//TEST_CASE("unified::geometry::rotational_average, assets", "[noa][unified]") {
//    constexpr bool COMPUTE_ASSETS = false;
//    std::vector<Device> devices{Device("cpu")};
//    if (!COMPUTE_ASSETS && Device::is_any(DeviceType::GPU))
//        devices.emplace_back("gpu");
//
//    const Path path_base = test::NOA_DATA_PATH / "geometry";
//    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["polar"];
//
//    for (size_t nb = 0; nb < tests.size(); ++nb) {
//        INFO("test number = " << nb);
//
//        const auto shape = Shape4<i64>{1};
//        const auto input = noa::memory::zeros<f32>(shape);
//        noa::geometry::fft::rotational_average<fft::H2H>(input, input, shape);
//
//    }
//}

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
        const auto output = noa::memory::zeros<f32>(rotational_average_shape, options);
        const auto weight = noa::memory::zeros<f32>(rotational_average_shape, options);
        noa::geometry::fft::rotational_average<fft::HC2H>(input_rfft, shape, output, weight);

//        fmt::print("{:.3f}\n", fmt::join(polar_reduced.eval().span(), ","));
//        fmt::print("{:.3f}\n", fmt::join(output.eval().span(), ","));
//        fmt::print("{:.3f}\n", fmt::join(weight.eval().span(), ","));

        REQUIRE(test::Matcher(test::MATCH_ABS, polar_reduced, output, 1e-3));

        // Rotational average within a range.
        // Use the same number of shells, so it can be compared with the full range.
        const auto output_range = noa::memory::zeros<f32>({batches, 1, 1, frequency_range_n_shells}, options);
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
                              5e-5));
        REQUIRE(test::Matcher(test::MATCH_ABS,
                              weight_cropped.subregion(Ellipsis{}, Slice{1, -1}),
                              weight_range.subregion(Ellipsis{}, Slice{1, -1}),
                              5e-5));
    }
}
