#include <noa/common/io/ImageFile.h>
#include <noa/cpu/math/Ewise.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/memory/Arange.h>
#include <noa/cpu/memory/Set.h>
#include <noa/cpu/geometry/Polar.h>
#include <noa/cpu/signal/Shape.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cpu::geometry::cartesian2polar()", "[noa][cpu][geometry]") {
    const size4_t shape{1,1,256,256};
    const size4_t stride = shape.stride();
    const size_t elements = shape.elements();

    const float2_t center{128,128};
    const float radius = 10;
    const float taper = 118;

    cpu::Stream stream;
    cpu::memory::PtrHost<float> cartesian{elements};
    cpu::signal::sphere<false, float>(nullptr, {}, cartesian.share(), stride, shape, center, radius, taper, stream);

    stream.synchronize();
    io::ImageFile file{test::NOA_DATA_PATH / "geometry" / "test-polar-img.mrc", io::WRITE};
    file.shape(shape);
    file.writeAll(cartesian.get());

    const size4_t polar_shape{1, 1, 512, 128};
    const size4_t polar_stride = polar_shape.stride();
    const size_t polar_elements = polar_shape.elements();
    cpu::memory::PtrHost<float> polar{polar_elements};
    cpu::geometry::cartesian2polar(cartesian.share(), stride, shape,
                                   polar.share(), polar_stride, polar_shape,
                                   center, {0, radius+taper+10}, {0, math::Constants<float>::PI2},
                                   false, INTERP_LINEAR, stream);

    stream.synchronize();
    file.open(test::NOA_DATA_PATH / "geometry" / "test-polar-fwd.mrc", io::WRITE);
    file.shape(polar_shape);
    file.writeAll(polar.get());
}

TEST_CASE("cpu::geometry::polar2cartesian()", "[noa][cpu][geometry]") {
    const size4_t cartesian_shape{1,1,256,256};
    const size4_t cartesian_stride = cartesian_shape.stride();
    const size_t cartesian_elements = cartesian_shape.elements();

    const size4_t polar_shape{1, 1, 512, 128};
    const size4_t polar_stride = polar_shape.stride();
    const size_t polar_elements = polar_shape.elements();

    const float2_t center{100,140};
    const float radius = 10;
    const float taper = 50;

    cpu::Stream stream;
    cpu::memory::PtrHost<float> polar{polar_elements};
    cpu::memory::arange(polar.share(), polar_stride, polar_shape, 0.f, 1.f, stream);
    cpu::math::ewise(polar.share(), polar_stride, polar.share(), polar_stride, polar_shape,
                     [=](float x) { return noa::math::abs(x - static_cast<float>(polar_elements / 2)); }, stream);

    stream.synchronize();
    io::ImageFile file{test::NOA_DATA_PATH / "geometry" / "test-polar-img.mrc", io::WRITE};
    file.shape(polar_shape);
    file.writeAll(polar.get());

    cpu::memory::PtrHost<float> cartesian{cartesian_elements};
    cpu::geometry::polar2cartesian(polar.share(), polar_stride, polar_shape,
                                   cartesian.share(), cartesian_stride, cartesian_shape,
                                   center, {0, radius + taper}, {0, math::Constants<float>::PI2},
                                   false, INTERP_LINEAR, stream);

    stream.synchronize();
    file.open(test::NOA_DATA_PATH / "geometry" / "test-polar-0.mrc", io::WRITE);
    file.shape(cartesian_shape);
    file.writeAll(cartesian.get());

    cpu::memory::set(polar.share(), polar_stride, polar_shape, 0.f, stream);
    cpu::geometry::cartesian2polar(cartesian.share(), cartesian_stride, cartesian_shape,
                                   polar.share(), polar_stride, polar_shape,
                                   center, {1, radius + taper}, {0, math::Constants<float>::PI},
                                   false, INTERP_LINEAR, stream);

    stream.synchronize();
    file.open(test::NOA_DATA_PATH / "geometry" / "test-polar-1.mrc", io::WRITE);
    file.shape(polar_shape);
    file.writeAll(polar.get());
}