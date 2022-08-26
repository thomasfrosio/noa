#include <noa/common/io/MRCFile.h>
#include "noa/common/geometry/Transform.h"
#include <noa/unified/math/Ewise.h>
#include <noa/unified/math/Random.h>
#include <noa/unified/math/Reduce.h>
#include <noa/unified/geometry/Transform.h>
#include <noa/unified/geometry/Prefilter.h>
#include <noa/unified/geometry/Shift.h>
#include <noa/unified/signal/Shape.h>

#include <catch2/catch.hpp>
#include "Helpers.h"

using namespace ::noa;

TEMPLATE_TEST_CASE("unified::geometry, Rotate subregion", "[.]", float) {
    const path_t directory = test::NOA_DATA_PATH / "signal" / "fft";

    std::vector<Device> devices = {Device{"cpu"}};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        StreamGuard stream{device};
        ArrayOption options{device, Allocator::MANAGED};
        INFO(device);

        io::MRCFile file{directory / "tilt1_slice21.mrc", io::READ};
        const size4_t shape = file.shape();

        // Select a subregion to save:
        const size4_t subregion_shape{1,1, shape[2], shape[3]};
        const float2_t subregion_shape2d{subregion_shape.get() + 2};
        const float2_t center = float2_t{shape.get() + 2} / 2;
        const float rotation = math::toRad(-9.1f);
        const float33_t matrix{
                geometry::translate(subregion_shape2d / 2) *
                float33_t{geometry::rotate(rotation)} *
                geometry::translate(-center)
        };

        Array<float> src{shape, options};
        file.readAll(src.get());

        // Standardize the input:
        float mean = math::mean(src);
        float stddev = math::std(src);
        math::ewise(src, mean, src, math::minus_t{});
        math::ewise(src, stddev, src, math::divide_t{});

        // Extract the output subregion:
        Array<float> dst{subregion_shape, options};
        geometry::shift2D(src, dst, -center + subregion_shape2d / 2);
        signal::sphere(dst, dst, center, math::min(subregion_shape2d) / 2 - 300, 300);
//        Array<float> tmp = dst.subregion(0,0, indexing::slice_t{400,5400}, indexing::slice_t{400,3600}).copy();
        mean = math::mean(dst);
        stddev = math::std(dst);
        math::ewise(dst, mean, dst, math::minus_t{});
        math::ewise(dst, stddev, dst, math::divide_t{});

        file.open(directory / string::format("tilt1_slice21_subregion_{}.mrc", device), io::WRITE);
        file.shape(dst.shape());
        dst.eval();
        file.writeAll(dst.get());
        file.close();

        geometry::transform2D(src, dst, math::inverse(matrix));
        dst += math::random<float>(math::uniform_t{}, dst.shape(), -0.1, 0.1, options);
        signal::sphere(dst, dst, center, math::min(subregion_shape2d) / 2 - 300, 300);
//        tmp = dst.subregion(0,0, indexing::slice_t{400,5400}, indexing::slice_t{400,3600}).copy();
        mean = math::mean(dst);
        stddev = math::std(dst);
        math::ewise(dst, mean, dst, math::minus_t{});
        math::ewise(dst, stddev, dst, math::divide_t{});
        file.open(directory / string::format("tilt1_slice21_subregion_rotated_{}.mrc", device), io::WRITE);
        file.shape(dst.shape());
        dst.eval();
        file.writeAll(dst.get());
        file.close();
    }
}
