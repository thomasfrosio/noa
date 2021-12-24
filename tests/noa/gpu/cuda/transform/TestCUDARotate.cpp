#include <noa/common/io/ImageFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/memory/Set.h>
#include <noa/cpu/transform/Rotate.h>

#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/transform/Rotate.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cuda::transform::rotate2D()", "[assets][noa][cuda][transform]") {
    path_t path_base = test::PATH_NOA_DATA / "transform";
    YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["rotate2D"];
    auto input_filename = path_base / param["input"].as<path_t>();
    auto rotate = math::toRad(param["rotate"].as<float>());
    auto center = param["center"].as<float2_t>();

    io::ImageFile file;
    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        auto expected_filename = path_base / test["expected"].as<path_t>();
        auto interp = test["interp"].as<InterpMode>();
        auto border = test["border"].as<BorderMode>();

        // Some BorderMode, or BorderMode-InterpMode combination, are not supported on the CUDA implementations.
        if (border == BORDER_VALUE || border == BORDER_REFLECT)
            continue;
        else if (border == BORDER_MIRROR || border == BORDER_PERIODIC)
            if (interp != INTERP_LINEAR_FAST && interp != INTERP_NEAREST)
                continue;

        // Get input.
        file.open(input_filename, io::READ);
        size3_t shape = file.shape();
        size_t elements = noa::elements(shape);
        cpu::memory::PtrHost<float> input(elements);
        file.readAll(input.get());

        // Get expected.
        cpu::memory::PtrHost<float> expected(elements);
        file.open(expected_filename, io::READ);
        file.readAll(expected.get());

        cuda::Stream stream;
        cpu::memory::PtrHost<float> output(elements);
        cuda::memory::PtrDevicePadded<float> d_input(shape);
        cuda::memory::copy(input.get(), shape.x, d_input.get(), d_input.pitch(), shape, stream);
        cuda::transform::rotate2D(d_input.get(), d_input.pitch(), d_input.get(), d_input.pitch(), // inplace
                                  size2_t(shape.x, shape.y), rotate, center, interp, border, stream);
        cuda::memory::copy(d_input.get(), d_input.pitch(), output.get(), shape.x, shape, stream);
        stream.synchronize();

        if (interp != INTERP_NEAREST) {
            REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), output.get(), elements, 1e-4f));
        } else {
            float diff = test::getDifference(expected.get(), output.get(), elements);
            REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
        }
    }
}

TEMPLATE_TEST_CASE("cuda::transform::rotate2D() -- accurate modes", "[noa][cuda][transform]", float, cfloat_t) {
    // INTERP_NEAREST isn't there because it's very possible to have different results there.
    // Textures have only 1 bytes of precision on the fraction used for the interpolation, so it is possible
    // to select the "wrong" element simply because of this loss of precision...
    InterpMode interp = GENERATE(INTERP_LINEAR, INTERP_COSINE, INTERP_CUBIC, INTERP_CUBIC_BSPLINE);
    BorderMode border = GENERATE(BORDER_ZERO, BORDER_CLAMP);
    INFO(interp);
    INFO(border);

    TestType value = test::Randomizer<TestType>(-3., 3.).get();
    float rotation = math::toRad(test::Randomizer<float>(-360., 360.).get());
    size3_t shape = test::getRandomShape(2U);
    size2_t shape_2d(shape.x, shape.y);
    size_t elements = noa::elements(shape);
    float2_t rotation_center(shape.x, shape.y);
    rotation_center /= test::Randomizer<float>(1, 4).get();

    // Get input.
    cpu::memory::PtrHost<TestType> input(elements);
    test::Randomizer<TestType> randomizer(-2., 2.);
    test::randomize(input.get(), elements, randomizer);

    cuda::Stream stream;
    cuda::memory::PtrDevicePadded<TestType> d_input(shape);
    cuda::memory::copy(input.get(), shape.x, d_input.get(), d_input.pitch(), shape, stream);
    cuda::transform::rotate2D(d_input.get(), d_input.pitch(), d_input.get(), d_input.pitch(),
                              shape_2d, rotation, rotation_center, interp, border, stream);

    cpu::memory::PtrHost<TestType> output(elements);
    cpu::memory::PtrHost<TestType> output_cuda(elements);
    cuda::memory::copy(d_input.get(), d_input.pitch(), output_cuda.get(), shape.x, shape, stream);
    cpu::transform::rotate2D(input.get(), output.get(),
                             shape_2d, rotation, rotation_center, interp, border, value);
    stream.synchronize();

    REQUIRE(test::Matcher(test::MATCH_ABS, output.get(), output_cuda.get(), elements, 5e-4f));
}

TEMPLATE_TEST_CASE("cuda::transform::rotate2D() -- fast modes", "[noa][cuda][transform]", float, cfloat_t) {
    InterpMode interp = GENERATE(INTERP_LINEAR_FAST, INTERP_COSINE_FAST, INTERP_CUBIC_BSPLINE_FAST);
    BorderMode border = GENERATE(BORDER_ZERO, BORDER_CLAMP, BORDER_MIRROR, BORDER_PERIODIC);
    if((border == BORDER_MIRROR || border == BORDER_PERIODIC) && interp != INTERP_LINEAR_FAST)
        return;

    INFO(interp);
    INFO(border);

    InterpMode interp_cpu{};
    if (interp == INTERP_LINEAR_FAST)
        interp_cpu = INTERP_LINEAR;
    else if (interp == INTERP_COSINE_FAST)
        interp_cpu = INTERP_COSINE;
    else if (interp == INTERP_CUBIC_BSPLINE_FAST)
        interp_cpu = INTERP_CUBIC_BSPLINE;

    TestType value = test::Randomizer<TestType>(-3., 3.).get();
    float rotation = math::toRad(test::Randomizer<float>(-360., 360.).get());
    size3_t shape = test::getRandomShape(2U);
    size2_t shape_2d(shape.x, shape.y);
    size_t elements = noa::elements(shape);
    float2_t rotation_center(shape.x, shape.y);
    rotation_center /= test::Randomizer<float>(1, 4).get();

    // Get input.
    cpu::memory::PtrHost<TestType> input(elements);
    test::Randomizer<TestType> randomizer(-2., 2.);
    test::randomize(input.get(), elements, randomizer);

    cuda::Stream stream;
    cuda::memory::PtrDevicePadded<TestType> d_input(shape);
    cuda::memory::copy(input.get(), shape.x, d_input.get(), d_input.pitch(), shape, stream);
    cuda::transform::rotate2D(d_input.get(), d_input.pitch(), d_input.get(), d_input.pitch(),
                              shape_2d, rotation, rotation_center, interp, border, stream);

    cpu::memory::PtrHost<TestType> output(elements);
    cpu::memory::PtrHost<TestType> output_cuda(elements);
    cuda::memory::copy(d_input.get(), d_input.pitch(), output_cuda.get(), shape.x, shape, stream);
    cpu::transform::rotate2D(input.get(), output.get(),
                             shape_2d, rotation, rotation_center, interp_cpu, border, value);
    stream.synchronize();

    float min_max_error = 0.05f; // for linear and cosine, it is usually around 0.01-0.03
    if (interp ==INTERP_CUBIC_BSPLINE_FAST)
        min_max_error = 0.08f; // usually around 0.03-0.06
    REQUIRE(test::Matcher(test::MATCH_ABS, output.get(), output_cuda.get(), elements, min_max_error));
}

TEST_CASE("cuda::transform::rotate3D()", "[assets][noa][cuda][transform]") {
    path_t path_base = test::PATH_NOA_DATA / "transform";
    YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["rotate3D"];
    auto input_filename = path_base / param["input"].as<path_t>();
    auto euler = math::toRad(param["euler"].as<float3_t>());
    auto center = param["center"].as<float3_t>();

    float33_t matrix(transform::toMatrix<true>(euler));
    io::ImageFile file;
    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        auto expected_filename = path_base / test["expected"].as<path_t>();
        auto interp = test["interp"].as<InterpMode>();
        auto border = test["border"].as<BorderMode>();

        // Some BorderMode, or BorderMode-InterpMode combination, are not supported on the CUDA implementations.
        if (border == BORDER_VALUE || border == BORDER_REFLECT)
            continue;
        else if (border == BORDER_MIRROR || border == BORDER_PERIODIC)
            if (interp != INTERP_LINEAR_FAST && interp != INTERP_NEAREST)
                continue;

        // Get input.
        file.open(input_filename, io::READ);
        size3_t shape = file.shape();
        size_t elements = noa::elements(shape);
        cpu::memory::PtrHost<float> input(elements);
        file.readAll(input.get());

        // Get expected.
        cpu::memory::PtrHost<float> expected(elements);
        file.open(expected_filename, io::READ);
        file.readAll(expected.get());

        cuda::Stream stream;
        cpu::memory::PtrHost<float> output(elements);
        cuda::memory::PtrDevicePadded<float> d_input(shape);
        cuda::memory::copy(input.get(), shape.x, d_input.get(), d_input.pitch(), shape, stream);
        cuda::transform::rotate3D(d_input.get(), d_input.pitch(), d_input.get(), d_input.pitch(),
                                  shape, matrix, center, interp, border, stream);
        cuda::memory::copy(d_input.get(), d_input.pitch(), output.get(), shape.x, shape, stream);
        stream.synchronize();

        if (interp != INTERP_NEAREST) {
            REQUIRE(test::Matcher(test::MATCH_ABS, output.get(), expected.get(), elements, 1e-4f));
        } else {
            float diff = test::getDifference(expected.get(), output.get(), elements);
            REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
        }
    }
}

TEMPLATE_TEST_CASE("cuda::transform::rotate3D() -- accurate modes", "[noa][cuda][transform]", float, cfloat_t) {
    // INTERP_NEAREST isn't there because it's very possible to have different results there.
    // Textures have only 1 bytes of precision on the fraction used for the interpolation, so it is possible
    // to select the "wrong" element simply because of this loss of precision...
    InterpMode interp = GENERATE(INTERP_LINEAR, INTERP_COSINE, INTERP_CUBIC, INTERP_CUBIC_BSPLINE);
    BorderMode border = GENERATE(BORDER_ZERO, BORDER_CLAMP);
    INFO(interp);
    INFO(border);

    TestType value = test::Randomizer<TestType>(-3., 3.).get();
    test::Randomizer<float> angle_randomizer(-360., 360.);
    float3_t eulers(math::toRad(angle_randomizer.get()),
                    math::toRad(angle_randomizer.get()),
                    math::toRad(angle_randomizer.get()));
    size3_t shape = test::getRandomShape(3U);
    size_t elements = noa::elements(shape);
    float3_t rotation_center(shape);
    rotation_center /= test::Randomizer<float>(1, 4).get();

    // Get input.
    cpu::memory::PtrHost<TestType> input(elements);
    test::Randomizer<TestType> randomizer(-2., 2.);
    test::randomize(input.get(), elements, randomizer);

    cuda::Stream stream;
    cuda::memory::PtrDevicePadded<TestType> d_input(shape);
    cuda::memory::copy(input.get(), shape.x, d_input.get(), d_input.pitch(), shape, stream);
    cuda::transform::rotate3D(d_input.get(), d_input.pitch(), d_input.get(), d_input.pitch(),
                              shape, transform::toMatrix<true>(eulers), rotation_center, interp, border, stream);

    cpu::memory::PtrHost<TestType> output(elements);
    cpu::memory::PtrHost<TestType> output_cuda(elements);
    cuda::memory::copy(d_input.get(), d_input.pitch(), output_cuda.get(), shape.x, shape, stream);
    cpu::transform::rotate3D(input.get(), output.get(),
                             shape, transform::toMatrix<true>(eulers), rotation_center, interp, border, value);
    stream.synchronize();

    REQUIRE(test::Matcher(test::MATCH_ABS, output.get(), output_cuda.get(), elements, 5e-4f));
}


TEMPLATE_TEST_CASE("cuda::transform::rotate3D() -- fast modes", "[noa][cuda][transform]", float, cfloat_t) {
    InterpMode interp = GENERATE(INTERP_LINEAR_FAST, INTERP_COSINE_FAST, INTERP_CUBIC_BSPLINE_FAST);
    BorderMode border = GENERATE(BORDER_ZERO, BORDER_CLAMP, BORDER_MIRROR, BORDER_PERIODIC);
    if((border == BORDER_MIRROR || border == BORDER_PERIODIC) && interp != INTERP_LINEAR_FAST)
        return;

    INFO(interp);
    INFO(border);

    InterpMode interp_cpu{};
    if (interp == INTERP_LINEAR_FAST)
        interp_cpu = INTERP_LINEAR;
    else if (interp == INTERP_COSINE_FAST)
        interp_cpu = INTERP_COSINE;
    else if (interp == INTERP_CUBIC_BSPLINE_FAST)
        interp_cpu = INTERP_CUBIC_BSPLINE;

    TestType value = test::Randomizer<TestType>(-3., 3.).get();
    test::Randomizer<float> angle_randomizer(-360., 360.);
    float3_t eulers(math::toRad(angle_randomizer.get()),
                    math::toRad(angle_randomizer.get()),
                    math::toRad(angle_randomizer.get()));
    size3_t shape = test::getRandomShape(3U);
    size_t elements = noa::elements(shape);
    float3_t rotation_center(shape);
    rotation_center /= test::Randomizer<float>(1, 4).get();

    // Get input.
    cpu::memory::PtrHost<TestType> input(elements);
    test::Randomizer<TestType> randomizer(-2., 2.);
    test::randomize(input.get(), elements, randomizer);

    cuda::Stream stream;
    cuda::memory::PtrDevicePadded<TestType> d_input(shape);
    cuda::memory::copy(input.get(), shape.x, d_input.get(), d_input.pitch(), shape, stream);
    cuda::transform::rotate3D(d_input.get(), d_input.pitch(), d_input.get(), d_input.pitch(),
                              shape, transform::toMatrix<true>(eulers), rotation_center, interp, border, stream);

    cpu::memory::PtrHost<TestType> output(elements);
    cpu::memory::PtrHost<TestType> output_cuda(elements);
    cuda::memory::copy(d_input.get(), d_input.pitch(), output_cuda.get(), shape.x, shape, stream);
    cpu::transform::rotate3D(input.get(), output.get(),
                             shape, transform::toMatrix<true>(eulers), rotation_center, interp_cpu, border, value);
    stream.synchronize();

    float min_max_error = 0.05f; // for linear and cosine, it is usually around 0.01-0.03
    if (interp ==INTERP_CUBIC_BSPLINE_FAST)
        min_max_error = 0.2f; // usually around 0.09
    REQUIRE(test::Matcher(test::MATCH_ABS, output.get(), output_cuda.get(), elements, min_max_error));
}
