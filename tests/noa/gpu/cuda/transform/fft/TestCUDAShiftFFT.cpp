#include <noa/common/io/ImageFile.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/transform/fft/Shift.h>

#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/transform/fft/Shift.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cuda::transform::fft::shift2D(), assets", "[assets][noa][cuda][transform]") {
    io::ImageFile file;
    path_t path_base = test::PATH_TEST_DATA / "transform" / "fft";
    YAML::Node param = YAML::LoadFile(path_base / "param.yaml")["shift"]["2D"];

    const auto shape = param["shape"].as<size2_t>();
    const auto shift = param["shift"].as<float2_t>();
    const auto path_output = path_base / param["output"].as<path_t>(); // these are non-redundant non-centered
    const auto path_input = path_base / param["input"].as<path_t>();
    size_t half = shape.x / 2 + 1;

    cpu::memory::PtrHost<cfloat_t> input(elementsFFT(shape));
    file.open(path_input, io::READ);
    file.readAll(input.get(), false);

    cpu::memory::PtrHost<cfloat_t> expected(input.elements());
    file.open(path_output, io::READ);
    file.readAll(expected.get(), false);

    cuda::Stream stream(cuda::Stream::SERIAL);
    AND_THEN("in-place") {
        cuda::memory::PtrDevice<cfloat_t> d_input(input.elements());
        cuda::memory::copy(input.get(), d_input.get(), input.elements(), stream);
        cuda::transform::fft::shift2D<fft::H2H>(d_input.get(), half, d_input.get(), half, shape,
                                                shift, 1, stream); // in-place if no remap is OK
        cuda::memory::copy(d_input.get(), input.get(), input.elements(), stream);

        stream.synchronize();
        test::Matcher matcher(test::MATCH_ABS, input.get(), expected.get(), input.elements(), 5e-5);
        REQUIRE(matcher);
    }

    AND_THEN("out of place") {
        cuda::memory::PtrDevicePadded<cfloat_t> d_input({half, shape.y, 1});
        cuda::memory::copy(input.get(), half, d_input.get(), d_input.pitch(), d_input.shape(), stream);
        cuda::transform::fft::shift2D<fft::H2H>(d_input.get(), d_input.pitch(), d_input.get(), d_input.pitch(), shape,
                                                shift, 1, stream); // in-place if no remap is OK
        cuda::memory::copy(d_input.get(), d_input.pitch(), input.get(), half, d_input.shape(), stream);

        stream.synchronize();
        test::Matcher matcher(test::MATCH_ABS, input.get(), expected.get(), input.elements(), 5e-5);
        REQUIRE(matcher);
    }
}

TEST_CASE("cuda::transform::fft::shift3D(), assets", "[assets][noa][cuda][transform]") {
    io::ImageFile file;
    path_t path_base = test::PATH_TEST_DATA / "transform" / "fft";
    YAML::Node param = YAML::LoadFile(path_base / "param.yaml")["shift"]["3D"];

    const auto shape = param["shape"].as<size3_t>();
    const auto shift = param["shift"].as<float3_t>();
    const auto path_output = path_base / param["output"].as<path_t>(); // these are non-redundant non-centered
    const auto path_input = path_base / param["input"].as<path_t>();
    size_t half = shape.x / 2 + 1;

    cpu::memory::PtrHost<cfloat_t> input(elementsFFT(shape));
    file.open(path_input, io::READ);
    file.readAll(input.get(), false);

    cpu::memory::PtrHost<cfloat_t> expected(input.elements());
    file.open(path_output, io::READ);
    file.readAll(expected.get(), false);

    cuda::Stream stream(cuda::Stream::SERIAL);
    AND_THEN("in-place") {
        cuda::memory::PtrDevice<cfloat_t> d_input(input.elements());
        cuda::memory::copy(input.get(), d_input.get(), input.elements(), stream);
        cuda::transform::fft::shift3D<fft::H2H>(d_input.get(), half, d_input.get(), half, shape,
                                                shift, 1, stream); // in-place if no remap is OK
        cuda::memory::copy(d_input.get(), input.get(), input.elements(), stream);

        stream.synchronize();
        test::Matcher matcher(test::MATCH_ABS, input.get(), expected.get(), input.elements(), 1e-4);
        REQUIRE(matcher);
    }

    AND_THEN("out of place") {
        cuda::memory::PtrDevicePadded<cfloat_t> d_input(shapeFFT(shape));
        cuda::memory::copy(input.get(), half, d_input.get(), d_input.pitch(), d_input.shape(), stream);
        cuda::transform::fft::shift3D<fft::H2H>(d_input.get(), d_input.pitch(), d_input.get(), d_input.pitch(), shape,
                                                shift, 1, stream); // in-place if no remap is OK
        cuda::memory::copy(d_input.get(), d_input.pitch(), input.get(), half, d_input.shape(), stream);

        stream.synchronize();
        test::Matcher matcher(test::MATCH_ABS, input.get(), expected.get(), input.elements(), 1e-4);
        REQUIRE(matcher);
    }
}

TEMPLATE_TEST_CASE("cuda::transform::fft::shift(2|3)D()", "[noa][cuda][transform]", cfloat_t, cdouble_t) {
    const size_t ndim = GENERATE(as<size_t>(), 2, 3);
    const size3_t shape = test::getRandomShape(ndim);
    const float3_t shift = {31.5, -15.2, -21.1};

    // Get inputs ready:
    test::Randomizer<TestType> randomizer(-2., 2.);
    cpu::memory::PtrHost<TestType> h_input(elementsFFT(shape));
    test::randomize(h_input.get(), h_input.elements(), randomizer);
    size_t half = shape.x / 2 + 1;

    cuda::Stream stream(cuda::Stream::SERIAL);
    cuda::memory::PtrDevicePadded<TestType> d_input(shapeFFT(shape));
    cuda::memory::copy(h_input.get(), half, d_input.get(), d_input.pitch(), d_input.shape(), stream);

    // Get outputs ready:
    cpu::memory::PtrHost<TestType> h_output(h_input.elements());
    cpu::memory::PtrHost<TestType> h_output_cuda(h_input.elements());
    cuda::memory::PtrDevicePadded<TestType> d_output(d_input.shape());

    // Phase shift:
    const fft::Remap remap = GENERATE(fft::H2H, fft::H2HC, fft::HC2HC, fft::HC2H);
    if (ndim == 2) {
        const size2_t shape_2d = {shape.x, shape.y};
        const float2_t shift_2d = {shift.x, shift.y};
        switch (remap) {
            case fft::H2H: {
                cpu::transform::fft::shift2D<fft::H2H>(
                        h_input.get(), h_output.get(), shape_2d, shift_2d, 1);
                cuda::transform::fft::shift2D<fft::H2H>(
                        d_input.get(), d_input.pitch(), d_output.get(), d_output.pitch(), shape_2d,
                        shift_2d, 1, stream);
                break;
            }
            case fft::H2HC: {
                cpu::transform::fft::shift2D<fft::H2HC>(
                        h_input.get(), h_output.get(), shape_2d, shift_2d, 1);
                cuda::transform::fft::shift2D<fft::H2HC>(
                        d_input.get(), d_input.pitch(), d_output.get(), d_output.pitch(), shape_2d,
                        shift_2d, 1, stream);
                break;
            }
            case fft::HC2H: {
                cpu::transform::fft::shift2D<fft::HC2H>(
                        h_input.get(), h_output.get(), shape_2d, shift_2d, 1);
                cuda::transform::fft::shift2D<fft::HC2H>(
                        d_input.get(), d_input.pitch(), d_output.get(), d_output.pitch(), shape_2d,
                        shift_2d, 1, stream);
                break;
            }
            case fft::HC2HC: {
                cpu::transform::fft::shift2D<fft::HC2HC>(
                        h_input.get(), h_output.get(), shape_2d, shift_2d, 1);
                cuda::transform::fft::shift2D<fft::HC2HC>(
                        d_input.get(), d_input.pitch(), d_output.get(), d_output.pitch(), shape_2d,
                        shift_2d, 1, stream);
                break;
            }
            default:
                REQUIRE(false);
        }
    } else {
        switch (remap) {
            case fft::H2H: {
                cpu::transform::fft::shift3D<fft::H2H>(
                        h_input.get(), h_output.get(), shape, shift, 1);
                cuda::transform::fft::shift3D<fft::H2H>(
                        d_input.get(), d_input.pitch(), d_output.get(), d_output.pitch(), shape, shift, 1, stream);
                break;
            }
            case fft::H2HC: {
                cpu::transform::fft::shift3D<fft::H2HC>(
                        h_input.get(), h_output.get(), shape, shift, 1);
                cuda::transform::fft::shift3D<fft::H2HC>(
                        d_input.get(), d_input.pitch(), d_output.get(), d_output.pitch(), shape, shift, 1, stream);
                break;
            }
            case fft::HC2H: {
                cpu::transform::fft::shift3D<fft::HC2H>(
                        h_input.get(), h_output.get(), shape, shift, 1);
                cuda::transform::fft::shift3D<fft::HC2H>(
                        d_input.get(), d_input.pitch(), d_output.get(), d_output.pitch(), shape, shift, 1, stream);
                break;
            }
            case fft::HC2HC: {
                cpu::transform::fft::shift3D<fft::HC2HC>(
                        h_input.get(), h_output.get(), shape, shift, 1);
                cuda::transform::fft::shift3D<fft::HC2HC>(
                        d_input.get(), d_input.pitch(), d_output.get(), d_output.pitch(), shape, shift, 1, stream);
                break;
            }
            default:
                REQUIRE(false);
        }
    }
    cuda::memory::copy(d_output.get(), d_output.pitch(), h_output_cuda.get(), half, d_output.shape(), stream);
    stream.synchronize();

    test::Matcher matcher(test::MATCH_ABS, h_output.get(), h_output_cuda.get(), h_output.elements(), 5e-5);
    REQUIRE(matcher);
}
