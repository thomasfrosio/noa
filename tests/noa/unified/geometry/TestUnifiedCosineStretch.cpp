#include "noa/common/geometry/Transform.h"
#include <noa/unified/io/ImageFile.h>
#include <noa/unified/math/Ewise.h>
#include <noa/unified/math/Reduce.h>
#include <noa/unified/geometry/Transform.h>
#include <noa/unified/geometry/Prefilter.h>
#include <noa/unified/signal/Shape.h>

#include <catch2/catch.hpp>
#include "Helpers.h"

using namespace ::noa;

static constexpr float TILT_ANGLES[] = {
        -60.000, -57.000, -54.000, -51.000, -48.000, -45.000, -42.000, -39.000, -36.000, -33.000, -30.000, -27.000,
        -24.000, -21.000, -18.000, -15.000, -12.000, -9.000, -6.000, -3.000, 0.000, 3.000, 6.000, 9.000, 12.000, 15.000,
        18.000, 21.000, 24.000, 27.000, 30.000, 33.000, 36.000, 39.000, 42.000, 45.000, 48.000, 51.000, 54.000, 57.000,
        60.000
};

TEMPLATE_TEST_CASE("unified::geometry, cosine stretch", "[.]", float) {
    const path_t directory = test::NOA_DATA_PATH / "signal" / "fft";

    std::vector<Device> devices = {Device{"gpu"}};
    for (auto& device: devices) {
        StreamGuard stream{device};
        ArrayOption options{device, Allocator::MANAGED};
        INFO(device);

        io::ImageFile file{directory / "tilt1_slice21.mrc", io::READ};
        const size4_t shape = file.shape();
        Array<float> src{shape, options};
        file.read(src);

        const float2_t center{shape[2] / 2, shape[3] / 2};

        const float mean = math::mean(src);
        const float stddev = math::std(src);
        math::ewise(src, mean, src, math::minus_t{});
        math::ewise(src, stddev, src, math::divide_t{});
        signal::rectangle(src, src, center, center - 50, 45);

        file.open(directory / string::format("tilt1_slice21_mean0_{}.mrc", device), io::WRITE);
        file.shape(shape);
        file.write(src);

        geometry::bspline::prefilter(src, src);

        Array<float> dst{shape, options};
        const float tilt_axis_angle = math::deg2rad(-5.f);
        const size_t tilts = 41;

        file.open(directory / string::format("tilt1_slice21_cos_stretch_{}.mrc", device), io::WRITE);
        file.shape(size4_t{tilts, 1, shape[2], shape[3]});

        for (size_t i = 0; i < tilts; ++i) {
            const float tilt_angle = math::deg2rad(TILT_ANGLES[i]);
            const float33_t matrix{
                    geometry::translate(center) *
                    float33_t{geometry::rotate(tilt_axis_angle)} *
                    float33_t{geometry::scale(float2_t{1, math::cos(tilt_angle)})} *
                    float33_t{geometry::rotate(-tilt_axis_angle)} *
                    geometry::translate(-center)
            };
            geometry::transform2D(src, dst, math::inverse(matrix),
                                  INTERP_CUBIC_BSPLINE_FAST, BORDER_ZERO, 0.f, false);
//            file.writeSlice(dst, i, i + 1);
        }
    }
}
