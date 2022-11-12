#include <noa/common/io/MRCFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/memory/Set.h>
#include <noa/cpu/geometry/Rotate.h>

#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/geometry/Transform.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cuda::geometry::rotate2D()", "[assets][noa][cuda][geometry]") {
    const path_t path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["rotate2D"];
    const auto input_filename = path_base / param["input"].as<path_t>();
    const auto rotate = math::deg2rad(param["rotate"].as<float>());
    const auto center = param["center"].as<float2_t>();
    const float33_t rotation_matrix =
            geometry::translate(center) *
            float33_t(geometry::rotate(-rotate)) *
            geometry::translate(-center);

    io::MRCFile file;
    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        const auto expected_filename = path_base / test["expected"].as<path_t>();
        const auto interp = test["interp"].as<InterpMode>();
        const auto border = test["border"].as<BorderMode>();

        // Some BorderMode, or BorderMode-InterpMode combination, are not supported on the CUDA implementations.
        if (border == BORDER_VALUE || border == BORDER_REFLECT)
            continue;
        else if (border == BORDER_MIRROR || border == BORDER_PERIODIC)
            if (interp != INTERP_LINEAR_FAST && interp != INTERP_NEAREST)
                continue;

        // Get input.
        file.open(input_filename, io::READ);
        const size4_t shape = file.shape();
        const size4_t stride = shape.strides();
        const size_t elements = shape.elements();
        cpu::memory::PtrHost<float> input(elements);
        file.readAll(input.get());

        // Get expected.
        cpu::memory::PtrHost<float> expected(elements);
        file.open(expected_filename, io::READ);
        file.readAll(expected.get());

        cuda::Stream stream;
        cpu::memory::PtrHost<float> output(elements);
        cuda::memory::PtrDevicePadded<float> d_input(shape);
        cuda::memory::copy(input.share(), stride, d_input.share(), d_input.strides(), shape, stream);
        cuda::geometry::transform2D(d_input.share(), d_input.strides(), shape,
                                    d_input.share(), d_input.strides(), shape,
                                    rotation_matrix, interp, border, true, stream);
        cuda::memory::copy(d_input.share(), d_input.strides(), output.share(), stride, shape, stream);
        stream.synchronize();

        if (interp != INTERP_NEAREST) {
            REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), output.get(), elements, 1e-4f));
        } else {
            const float diff = test::getDifference(expected.get(), output.get(), elements);
            REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
        }
    }
}

TEMPLATE_TEST_CASE("cuda::geometry::rotate2D() -- accurate modes", "[noa][cuda][geometry]", float, cfloat_t) {
    // INTERP_NEAREST isn't tests because it's very possible to have different results.
    // Textures have only 1 bytes of precision on the fraction used for the interpolation, so it is possible
    // to select the "wrong" element simply due to this loss of precision...
    const InterpMode interp = GENERATE(INTERP_LINEAR, INTERP_COSINE, INTERP_CUBIC, INTERP_CUBIC_BSPLINE);
    const BorderMode border = GENERATE(BORDER_ZERO, BORDER_CLAMP);
    INFO(interp);
    INFO(border);

    const TestType value = test::Randomizer<TestType>(-3., 3.).get();
    const float rotation = math::deg2rad(test::Randomizer<float>(-360., 360.).get());
    const size4_t shape = test::getRandomShapeBatched(2u);
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();
    float2_t center{shape.get() + 2};
    center /= test::Randomizer<float>(1, 4).get();
    const float33_t rotation_matrix =
            geometry::translate(center) *
            float33_t(geometry::rotate(-rotation)) *
            geometry::translate(-center);

    // Get input.
    cpu::memory::PtrHost<TestType> input(elements);
    test::Randomizer<TestType> randomizer(-2., 2.);
    test::randomize(input.get(), elements, randomizer);

    cuda::Stream gpu_stream;
    cpu::Stream cpu_stream;
    cuda::memory::PtrDevicePadded<TestType> d_input(shape);
    cuda::memory::copy(input.share(), stride, d_input.share(), d_input.strides(), shape, gpu_stream);
    cuda::geometry::transform2D(d_input.share(), d_input.strides(), shape,
                                d_input.share(), d_input.strides(), shape,
                                rotation_matrix, interp, border, true, gpu_stream);

    cpu::memory::PtrHost<TestType> output(elements);
    cpu::memory::PtrHost<TestType> output_cuda(elements);
    cuda::memory::copy(d_input.share(), d_input.strides(), output_cuda.share(), stride, shape, gpu_stream);
    cpu::geometry::rotate2D(input.share(), stride, shape, output.share(), stride, shape,
                            rotation, center, interp, border, value, true, cpu_stream);
    gpu_stream.synchronize();
    cpu_stream.synchronize();

    REQUIRE(test::Matcher(test::MATCH_ABS, output.get(), output_cuda.get(), elements, 5e-4f));
}

TEMPLATE_TEST_CASE("cuda::geometry::rotate2D() -- fast modes", "[noa][cuda][geometry]", float, cfloat_t) {
    const InterpMode interp = GENERATE(INTERP_LINEAR_FAST, INTERP_COSINE_FAST, INTERP_CUBIC_BSPLINE_FAST);
    const BorderMode border = GENERATE(BORDER_ZERO, BORDER_CLAMP, BORDER_MIRROR, BORDER_PERIODIC);
    if ((border == BORDER_MIRROR || border == BORDER_PERIODIC) && interp != INTERP_LINEAR_FAST)
        return;

    INFO(interp);
    INFO(border);

    const TestType value = test::Randomizer<TestType>(-3., 3.).get();
    const float rotation = math::deg2rad(test::Randomizer<float>(-360., 360.).get());
    const size4_t shape = test::getRandomShapeBatched(2u);
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();
    float2_t center{shape.get() + 2};
    center /= test::Randomizer<float>(1, 4).get();
    const float33_t rotation_matrix =
            geometry::translate(center) *
            float33_t(geometry::rotate(-rotation)) *
            geometry::translate(-center);

    // Get input.
    cpu::memory::PtrHost<TestType> input(elements);
    test::Randomizer<TestType> randomizer(-2., 2.);
    test::randomize(input.get(), elements, randomizer);

    cuda::Stream gpu_stream;
    cpu::Stream cpu_stream;
    cuda::memory::PtrDevicePadded<TestType> d_input(shape);
    cuda::memory::copy(input.share(), stride, d_input.share(), d_input.strides(), shape, gpu_stream);
    cuda::geometry::transform2D(d_input.share(), d_input.strides(), shape,
                                d_input.share(), d_input.strides(), shape,
                                rotation_matrix, interp, border, true, gpu_stream);

    cpu::memory::PtrHost<TestType> output(elements);
    cpu::memory::PtrHost<TestType> output_cuda(elements);
    cuda::memory::copy(d_input.share(), d_input.strides(), output_cuda.share(), stride, shape, gpu_stream);
    cpu::geometry::rotate2D(input.share(), stride, shape, output.share(), stride, shape,
                            rotation, center, interp, border, value, true, cpu_stream);
    gpu_stream.synchronize();
    cpu_stream.synchronize();

    float min_max_error = 0.05f; // for linear and cosine, it is usually around 0.01-0.03
    if (interp == INTERP_CUBIC_BSPLINE_FAST)
        min_max_error = 0.08f; // usually around 0.03-0.06
    REQUIRE(test::Matcher(test::MATCH_ABS, output.get(), output_cuda.get(), elements, min_max_error));
}

TEST_CASE("cuda::geometry::rotate3D()", "[assets][noa][cuda][geometry]") {
    const path_t path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["rotate3D"];
    const auto input_filename = path_base / param["input"].as<path_t>();
    const auto euler = math::deg2rad(param["euler"].as<float3_t>());
    const auto center = param["center"].as<float3_t>();
    const float44_t rotation_matrix =
            geometry::translate(center) *
            float44_t(geometry::euler2matrix(euler).transpose()) *
            geometry::translate(-center);

    io::MRCFile file;
    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        const auto expected_filename = path_base / test["expected"].as<path_t>();
        const auto interp = test["interp"].as<InterpMode>();
        const auto border = test["border"].as<BorderMode>();

        // Some BorderMode, or BorderMode-InterpMode combination, are not supported on the CUDA implementations.
        if (border == BORDER_VALUE || border == BORDER_REFLECT)
            continue;
        else if (border == BORDER_MIRROR || border == BORDER_PERIODIC)
            if (interp != INTERP_LINEAR_FAST && interp != INTERP_NEAREST)
                continue;

        // Get input.
        file.open(input_filename, io::READ);
        const size4_t shape = file.shape();
        const size4_t stride = shape.strides();
        const size_t elements = shape.elements();
        cpu::memory::PtrHost<float> input(elements);
        file.readAll(input.get());

        // Get expected.
        cpu::memory::PtrHost<float> expected(elements);
        file.open(expected_filename, io::READ);
        file.readAll(expected.get());

        cuda::Stream stream;
        cpu::memory::PtrHost<float> output(elements);
        cuda::memory::PtrDevicePadded<float> d_input(shape);
        cuda::memory::copy(input.share(), stride, d_input.share(), d_input.strides(), shape, stream);
        cuda::geometry::transform3D(d_input.share(), d_input.strides(), shape,
                                    d_input.share(), d_input.strides(), shape,
                                    rotation_matrix, interp, border, true, stream);
        cuda::memory::copy(d_input.share(), d_input.strides(), output.share(), stride, shape, stream);
        stream.synchronize();

        if (interp != INTERP_NEAREST) {
            REQUIRE(test::Matcher(test::MATCH_ABS, output.get(), expected.get(), elements, 1e-4f));
        } else {
            const float diff = test::getDifference(expected.get(), output.get(), elements);
            REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
        }
    }
}

TEMPLATE_TEST_CASE("cuda::geometry::rotate3D() -- accurate modes", "[noa][cuda][geometry]", float, cfloat_t) {
    // INTERP_NEAREST isn't there because it's very possible to have different results there.
    // Textures have only 1 bytes of precision on the fraction used for the interpolation, so it is possible
    // to select the "wrong" element simply because of this loss of precision...
    InterpMode interp = GENERATE(INTERP_LINEAR, INTERP_COSINE, INTERP_CUBIC, INTERP_CUBIC_BSPLINE);
    BorderMode border = GENERATE(BORDER_ZERO, BORDER_CLAMP);
    INFO(interp);
    INFO(border);

    const TestType value = test::Randomizer<TestType>(-3., 3.).get();
    float3_t eulers = {test::Randomizer<float>(-360., 360.).get(),
                       test::Randomizer<float>(-360., 360.).get(),
                       test::Randomizer<float>(-360., 360.).get()};
    eulers = math::deg2rad(eulers);
    const float33_t matrix = geometry::euler2matrix(eulers).transpose();

    const size4_t shape = test::getRandomShapeBatched(3u);
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();
    float3_t center{shape.get() + 1};
    center /= test::Randomizer<float>(1, 4).get();
    const float44_t rotation_matrix =
            geometry::translate(center) *
            float44_t(matrix) *
            geometry::translate(-center);

    // Get input.
    cpu::memory::PtrHost<TestType> input(elements);
    test::Randomizer<TestType> randomizer(-2., 2.);
    test::randomize(input.get(), elements, randomizer);

    cuda::Stream gpu_stream;
    cpu::Stream cpu_stream;
    cuda::memory::PtrDevicePadded<TestType> d_input(shape);
    cuda::memory::copy(input.share(), stride, d_input.share(), d_input.strides(), shape, gpu_stream);
    cuda::geometry::transform3D(d_input.share(), d_input.strides(), shape,
                                d_input.share(), d_input.strides(), shape,
                                rotation_matrix, interp, border, true, gpu_stream);

    cpu::memory::PtrHost<TestType> output(elements);
    cpu::memory::PtrHost<TestType> output_cuda(elements);
    cuda::memory::copy(d_input.share(), d_input.strides(), output_cuda.share(), stride, shape, gpu_stream);
    cpu::geometry::rotate3D(input.share(), stride, shape, output.share(), stride, shape,
                            matrix, center, interp, border, value, true, cpu_stream);
    gpu_stream.synchronize();
    cpu_stream.synchronize();

    REQUIRE(test::Matcher(test::MATCH_ABS, output.get(), output_cuda.get(), elements, 5e-4f));
}

TEMPLATE_TEST_CASE("cuda::geometry::rotate3D() -- fast modes", "[noa][cuda][geometry]", float, cfloat_t) {
    InterpMode interp = GENERATE(INTERP_LINEAR_FAST, INTERP_COSINE_FAST, INTERP_CUBIC_BSPLINE_FAST);
    BorderMode border = GENERATE(BORDER_ZERO, BORDER_CLAMP, BORDER_MIRROR, BORDER_PERIODIC);
    if ((border == BORDER_MIRROR || border == BORDER_PERIODIC) && interp != INTERP_LINEAR_FAST)
        return;

    INFO(interp);
    INFO(border);

    const TestType value = test::Randomizer<TestType>(-3., 3.).get();
    float3_t eulers = {test::Randomizer<float>(-360., 360.).get(),
                       test::Randomizer<float>(-360., 360.).get(),
                       test::Randomizer<float>(-360., 360.).get()};
    eulers = math::deg2rad(eulers);
    const float33_t matrix = geometry::euler2matrix(eulers).transpose();
    const size4_t shape = test::getRandomShapeBatched(3u);
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();
    float3_t center{shape.get() + 1};
    center /= test::Randomizer<float>(1, 4).get();
    const float44_t rotation_matrix =
            geometry::translate(center) *
            float44_t(matrix) *
            geometry::translate(-center);

    // Get input.
    cpu::memory::PtrHost<TestType> input(elements);
    test::Randomizer<TestType> randomizer(-2., 2.);
    test::randomize(input.get(), elements, randomizer);

    cuda::Stream gpu_stream;
    cpu::Stream cpu_stream;
    cuda::memory::PtrDevicePadded<TestType> d_input(shape);
    cuda::memory::copy(input.share(), stride, d_input.share(), d_input.strides(), shape, gpu_stream);
    cuda::geometry::transform3D(d_input.share(), d_input.strides(), shape,
                                d_input.share(), d_input.strides(), shape,
                                rotation_matrix, interp, border, true, gpu_stream);

    cpu::memory::PtrHost<TestType> output(elements);
    cpu::memory::PtrHost<TestType> output_cuda(elements);
    cuda::memory::copy(d_input.share(), d_input.strides(), output_cuda.share(), stride, shape, gpu_stream);
    cpu::geometry::rotate3D(input.share(), stride, shape, output.share(), stride, shape,
                            matrix, center, interp, border, value, true, cpu_stream);
    gpu_stream.synchronize();
    cpu_stream.synchronize();

    float min_max_error = 0.05f; // for linear and cosine, it is usually around 0.01-0.03
    if (interp == INTERP_CUBIC_BSPLINE_FAST)
        min_max_error = 0.2f; // usually around 0.09
    REQUIRE(test::Matcher(test::MATCH_ABS, output.get(), output_cuda.get(), elements, min_max_error));
}
