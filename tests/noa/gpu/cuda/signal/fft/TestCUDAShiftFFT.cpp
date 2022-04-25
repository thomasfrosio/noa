#include <noa/common/io/ImageFile.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/signal/fft/Shift.h>

#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/PtrManaged.h>
#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/signal/fft/Shift.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cuda::signal::fft::shift2D(), assets", "[assets][noa][cuda][signal]") {
    io::ImageFile file;
    const path_t path_base = test::NOA_DATA_PATH / "signal" / "fft";
    const YAML::Node params = YAML::LoadFile(path_base / "tests.yaml")["shift"]["2D"];
    cuda::Stream stream;

    for (size_t i = 0; i < params.size(); i++) {
        const YAML::Node& param = params[i];
        const auto shape = param["shape"].as<size4_t>();
        const auto shift = param["shift"].as<float2_t>();
        const auto cutoff = param["cutoff"].as<float>();
        const auto path_output = path_base / param["output"].as<path_t>(); // these are non-redundant non-centered
        const auto path_input = path_base / param["input"].as<path_t>();
        const size4_t stride = shape.fft().stride();
        const size_t elements = stride[0] * shape[0];

        cuda::memory::PtrManaged<cfloat_t> input(elements, stream);
        cuda::memory::PtrManaged<cfloat_t> expected(elements, stream);

        if (path_input.filename().empty()) {
            cuda::signal::fft::shift2D<fft::H2H, cfloat_t>(
                    nullptr, {}, input.share(), stride, shape, shift, cutoff, stream);

            file.open(path_output, io::READ);
            file.readAll(expected.get(), false);

            stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), input.get(), elements, 1e-4f));

        } else {
            file.open(path_input, io::READ);
            file.readAll(input.get(), false);
            cuda::signal::fft::shift2D<fft::H2H>(
                    input.share(), stride, input.share(), stride, shape, shift, cutoff, stream);

            file.open(path_output, io::READ);
            file.readAll(expected.get(), false);

            stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), input.get(), elements, 1e-4f));
        }
    }
}

TEST_CASE("cuda::signal::fft::shift3D(), assets", "[assets][noa][cuda][signal]") {
    io::ImageFile file;
    const path_t path_base = test::NOA_DATA_PATH / "signal" / "fft";
    const YAML::Node params = YAML::LoadFile(path_base / "tests.yaml")["shift"]["3D"];
    cuda::Stream stream;

    for (size_t i = 0; i < params.size(); i++) {
        const YAML::Node& param = params[i];
        const auto shape = param["shape"].as<size4_t>();
        const auto shift = param["shift"].as<float3_t>();
        const auto cutoff = param["cutoff"].as<float>();
        const auto path_output = path_base / param["output"].as<path_t>(); // these are non-redundant non-centered
        const auto path_input = path_base / param["input"].as<path_t>();
        const size4_t stride = shape.fft().stride();
        const size_t elements = stride[0] * shape[0];

        cuda::memory::PtrManaged<cfloat_t> input(elements);
        cuda::memory::PtrManaged<cfloat_t> expected(elements);

        if (path_input.filename().empty()) {
            cuda::signal::fft::shift3D<fft::H2H, cfloat_t>(
                    nullptr, {}, input.share(), stride, shape, shift, cutoff, stream);

            file.open(path_output, io::READ);
            file.readAll(expected.get(), false);

            stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), input.get(), elements, 1e-4f));

        } else {
            file.open(path_input, io::READ);
            file.readAll(input.get(), false);
            cuda::signal::fft::shift3D<fft::H2H>(
                    input.share(), stride, input.share(), stride, shape, shift, cutoff, stream);

            file.open(path_output, io::READ);
            file.readAll(expected.get(), false);

            stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), input.get(), elements, 1e-4f));
        }
    }
}

TEMPLATE_TEST_CASE("cuda::transform::fft::shift(2|3)D()", "[noa][cuda][transform]", cfloat_t, cdouble_t) {
    const uint ndim = GENERATE(2u, 3u);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t shape_fft = shape.fft();
    const size4_t stride = shape_fft.stride();
    const size_t elements = stride[0] * shape[0];
    const float3_t shift = {31.5, -15.2, -21.1};
    const float cutoff = test::Randomizer<float>(0.2, 0.5).get();

    // Get inputs ready:
    test::Randomizer<TestType> randomizer(-2., 2.);
    cpu::memory::PtrHost<TestType> h_input(elements);
    test::randomize(h_input.get(), h_input.elements(), randomizer);

    cuda::Stream gpu_stream;
    cpu::Stream cpu_stream;
    cuda::memory::PtrDevicePadded<TestType> d_input(shape_fft);
    cuda::memory::copy(h_input.share(), stride, d_input.share(), d_input.stride(), d_input.shape(), gpu_stream);

    // Get outputs ready:
    cpu::memory::PtrHost<TestType> h_output(elements);
    cpu::memory::PtrHost<TestType> h_output_cuda(elements);
    cuda::memory::PtrDevicePadded<TestType> d_output(shape_fft);

    // Phase shift:
    const fft::Remap remap = GENERATE(fft::H2H, fft::H2HC, fft::HC2HC, fft::HC2H);
    if (ndim == 2) {
        const float2_t shift_2d{shift.get()};
        switch (remap) {
            case fft::H2H: {
                cpu::signal::fft::shift2D<fft::H2H>(
                        h_input.share(), stride, h_output.share(), stride, shape, shift_2d, cutoff, cpu_stream);
                cuda::signal::fft::shift2D<fft::H2H>(
                        d_input.share(), d_input.stride(), d_output.share(), d_output.stride(), shape,
                        shift_2d, cutoff, gpu_stream);
                break;
            }
            case fft::H2HC: {
                cpu::signal::fft::shift2D<fft::H2HC>(
                        h_input.share(), stride, h_output.share(), stride, shape, shift_2d, cutoff, cpu_stream);
                cuda::signal::fft::shift2D<fft::H2HC>(
                        d_input.share(), d_input.stride(), d_output.share(), d_output.stride(), shape,
                        shift_2d, cutoff, gpu_stream);
                break;
            }
            case fft::HC2H: {
                cpu::signal::fft::shift2D<fft::HC2H>(
                        h_input.share(), stride, h_output.share(), stride, shape, shift_2d, cutoff, cpu_stream);
                cuda::signal::fft::shift2D<fft::HC2H>(
                        d_input.share(), d_input.stride(), d_output.share(), d_output.stride(), shape,
                        shift_2d, cutoff, gpu_stream);
                break;
            }
            case fft::HC2HC: {
                cpu::signal::fft::shift2D<fft::HC2HC>(
                        h_input.share(), stride, h_output.share(), stride, shape, shift_2d, cutoff, cpu_stream);
                cuda::signal::fft::shift2D<fft::HC2HC>(
                        d_input.share(), d_input.stride(), d_output.share(), d_output.stride(), shape,
                        shift_2d, cutoff, gpu_stream);
                break;
            }
            default:
                REQUIRE(false);
        }
    } else {
        switch (remap) {
            case fft::H2H: {
                cpu::signal::fft::shift3D<fft::H2H>(
                        h_input.share(), stride, h_output.share(), stride, shape, shift, cutoff, cpu_stream);
                cuda::signal::fft::shift3D<fft::H2H>(
                        d_input.share(), d_input.stride(), d_output.share(), d_output.stride(), shape,
                        shift, cutoff, gpu_stream);
                break;
            }
            case fft::H2HC: {
                cpu::signal::fft::shift3D<fft::H2HC>(
                        h_input.share(), stride, h_output.share(), stride, shape, shift, cutoff, cpu_stream);
                cuda::signal::fft::shift3D<fft::H2HC>(
                        d_input.share(), d_input.stride(), d_output.share(), d_output.stride(), shape,
                        shift, cutoff, gpu_stream);
                break;
            }
            case fft::HC2H: {
                cpu::signal::fft::shift3D<fft::HC2H>(
                        h_input.share(), stride, h_output.share(), stride, shape, shift, cutoff, cpu_stream);
                cuda::signal::fft::shift3D<fft::HC2H>(
                        d_input.share(), d_input.stride(), d_output.share(), d_output.stride(), shape,
                        shift, cutoff, gpu_stream);
                break;
            }
            case fft::HC2HC: {
                cpu::signal::fft::shift3D<fft::HC2HC>(
                        h_input.share(), stride, h_output.share(), stride, shape, shift, cutoff, cpu_stream);
                cuda::signal::fft::shift3D<fft::HC2HC>(
                        d_input.share(), d_input.stride(), d_output.share(), d_output.stride(), shape,
                        shift, cutoff, gpu_stream);
                break;
            }
            default:
                REQUIRE(false);
        }
    }
    cuda::memory::copy(d_output.share(), d_output.stride(), h_output_cuda.share(), stride, d_output.shape(), gpu_stream);
    gpu_stream.synchronize();
    cpu_stream.synchronize();

    test::Matcher matcher(test::MATCH_ABS, h_output.get(), h_output_cuda.get(), h_output.elements(), 8e-5);
    REQUIRE(matcher);
}
