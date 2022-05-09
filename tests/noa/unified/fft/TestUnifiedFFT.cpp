#include <noa/unified/FFT.h>
#include <noa/unified/math/Random.h>
#include <noa/unified/memory/Factory.h>

#include <catch2/catch.hpp>
#include "Helpers.h"
using namespace ::noa;

TEMPLATE_TEST_CASE("unified::fft::r2c|c2r", "[noa][unified]", float, double) {
    using real_t = TestType;
    using complex_t = Complex<real_t>;

    const size4_t shape = test::getRandomShapeBatched(3);

    std::vector<Device> devices{Device{"cpu"}};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        StreamGuard stream{device};
        ArrayOption options{device, Allocator::MANAGED};
        INFO(device);

        Array image_fft = math::random<complex_t>(math::uniform_t{}, shape.fft(), -50, 50, options);
        Array image = image_fft.template as<real_t>();
        image = Array{image.share(), shape, image.stride(), options};

        Array result = image.copy();

        fft::r2c(image, image_fft);
        fft::c2r(image_fft, image);

        image = image.copy();

        result.eval();
        INFO(shape);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, image.get(), result.get(), shape.elements(), 5e-3));
    }
}

TEMPLATE_TEST_CASE("unified::fft::remap", "[noa][unified]", float, double, cfloat_t, cdouble_t) {
    const size4_t shape = test::getRandomShapeBatched(3);

    std::vector<Device> devices{Device{"cpu"}};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        StreamGuard stream{device};
        ArrayOption options{device, Allocator::MANAGED};
        INFO(device);

        Array in = memory::linspace<TestType>(shape.fft(), 0, 1000, true, options);
        Array out = memory::empty<TestType>(shape, options);

        Array tmp0 = memory::like(in);
        Array tmp1 = memory::like(out);

        fft::remap(fft::H2HC, in, tmp0, shape);
        fft::remap(fft::HC2FC, tmp0, tmp1, shape);
        fft::remap(fft::FC2F, tmp1, out, shape);
        fft::remap(fft::F2H, out, tmp0, shape);

        tmp0.eval();
        INFO(shape);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, in.get(), tmp0.get(), in.shape().elements(), 1e-6));
    }
}

TEMPLATE_TEST_CASE("unified::fft::resize", "[noa][unified]", float, double, cfloat_t, cdouble_t) {
    const size4_t shape = test::getRandomShapeBatched(3);
    const size4_t shape_padded{shape[0], shape[1] + 6, shape[2] + 10, shape[3] + 12};
    INFO("input" << shape << ", output:" << shape_padded);

    std::vector<Device> devices{Device{"cpu"}};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        StreamGuard stream{device};
        ArrayOption options{device, Allocator::MANAGED};
        INFO(device);

        AND_THEN("half") {
            Array in = memory::linspace<TestType>(shape.fft(), 0, 1000, true, options);
            Array out = memory::empty<TestType>(shape_padded.fft(), options);
            Array tmp0 = memory::like(in);

            fft::resize<fft::H2H>(in, shape, out, shape_padded);
            fft::resize<fft::H2H>(out, shape_padded, tmp0, shape);

            tmp0.eval();
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, in.get(), tmp0.get(), in.shape().elements(), 1e-6));
        }

        AND_THEN("full") {
            Array in = memory::linspace<TestType>(shape, 0, 1000, true, options);
            Array out = memory::empty<TestType>(shape_padded, options);
            Array tmp0 = memory::like(in);

            fft::resize<fft::F2F>(in, shape, out, shape_padded);
            fft::resize<fft::F2F>(out, shape_padded, tmp0, shape);

            tmp0.eval();
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, in.get(), tmp0.get(), in.shape().elements(), 1e-6));
        }
    }
}
