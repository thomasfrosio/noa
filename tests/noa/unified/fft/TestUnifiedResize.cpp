#include <noa/IO.h>
#include <noa/FFT.h>
#include <noa/Math.h>

#include <catch2/catch.hpp>
#include "Helpers.h"
using namespace ::noa;

TEMPLATE_TEST_CASE("unified::fft::resize remap", "[noa][unified]", float, double, cfloat_t, cdouble_t) {
    std::vector<Device> devices{Device("cpu")};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    const dim4_t input_shape = test::getRandomShapeBatched(4);
    dim4_t output_shape = input_shape;

    test::Randomizer<dim_t> randomizer(0, 20);
    output_shape[1] += randomizer.get();
    output_shape[2] += randomizer.get();
    output_shape[3] += randomizer.get();

    INFO(input_shape);
    INFO(output_shape);

    for (auto& device: devices) {
        StreamGuard stream(device);
        ArrayOption options(device, Allocator::MANAGED);
        INFO(device);

        Array a0 = math::random<TestType>(math::uniform_t{}, input_shape.fft(), -50, 50, options);
        Array a1 = fft::resize<fft::H2H>(a0, input_shape, output_shape);
        Array a2 = fft::remap(fft::H2HC, a1, output_shape);
        Array a3 = fft::resize<fft::HC2HC>(a2, output_shape, input_shape);
        Array a4 = fft::remap(fft::HC2H, a3, input_shape);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, a0, a4, 5e-3));

        a0 = math::random<TestType>(math::uniform_t{}, input_shape, -50, 50, options);
        a1 = fft::resize<fft::F2F>(a0, input_shape, output_shape);
        a2 = fft::remap(fft::F2FC, a1, output_shape);
        a3 = fft::resize<fft::FC2FC>(a2, output_shape, input_shape);
        a4 = fft::remap(fft::FC2F, a3, input_shape);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, a0, a4, 5e-3));
    }
}

TEST_CASE("unified::fft::resize, crop", "[.]") {
    using real_t = float;
    using complex_t = Complex<real_t>;

    const Device cpu{};
    StreamGuard stream(cpu);
    ArrayOption options{cpu, Allocator::DEFAULT};

    Array src = io::load<real_t>(test::NOA_DATA_PATH / "signal" / "fft" / "tilt1_slice21.mrc", false, options);
    Array src_fft = fft::r2c(src.release());

    const size_t reduce_factor = 6;
    const size4_t shape = src.shape();
    const size4_t shape_bin{shape[0], shape[1], shape[2] / reduce_factor, shape[3] / reduce_factor};

    Array<complex_t> dst_fft(shape_bin.fft(), options);
    fft::resize<fft::H2H>(src_fft.release(), shape, dst_fft, shape_bin);

    Array dst = fft::c2r(dst_fft.release(), shape_bin);
    io::save(dst, test::NOA_DATA_PATH / "signal" / "fft" / "tilt1_slice21_cropped2.mrc");
}

TEST_CASE("unified::fft::resize, crop entire stack", "[.]") {
    using real_t = float;
    using complex_t = Complex<real_t>;

    const Device cpu{};
    StreamGuard stream(cpu);
    ArrayOption options{cpu, Allocator::DEFAULT};

    const size_t reduce_factor = 6;

    io::ImageFile file(test::NOA_DATA_PATH / "tilt1.mrc", io::READ);
    const size4_t shape = file.shape();
    float3_t pixel_size = file.pixelSize();
    pixel_size[1] *= reduce_factor;
    pixel_size[2] *= reduce_factor;

    Array src = file.read<real_t>(options);
    Array src_fft = fft::r2c(src.release());

    const size4_t shape_binned{shape[0], shape[1], shape[2] / reduce_factor, shape[3] / reduce_factor};
    Array<complex_t> dst_fft{shape_binned.fft(), options};
    fft::resize<fft::H2H>(src_fft.release(), shape, dst_fft, shape_binned);

    Array dst = fft::c2r(dst_fft.release(), shape_binned);

    file.open(test::NOA_DATA_PATH / "tilt1_cropped.mrc", io::WRITE);
    file.pixelSize(pixel_size);
    file.write(dst);
}
