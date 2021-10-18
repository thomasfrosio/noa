#include <noa/common/files/ImageFile.h>
#include <noa/common/io/ImageFile.h>

#include <noa/common/transform/Euler.h>
#include <noa/common/transform/Geometry.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/filter/Rectangle.h>
#include <noa/cpu/math/Arithmetics.h>
#include <noa/cpu/math/Reductions.h>
#include <noa/cpu/transform/Apply.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cpu::transform::apply2D() - symmetry", "[noa][cpu][transform]") {
    YAML::Node param = YAML::Load(test::PATH_TEST_DATA / "transform" / "param.yaml");
    const int tests = param["apply_symmetry"]["tests"].as<int>();
    for (int i = 0; i < tests; ++i) {
        INFO("test number = " << i);

        // Parameters:
        auto current = param["apply_symmetry"][i];
        auto filename_input = current["input"].as<path_t>();
        auto filename_expected = current["expected"].as<path_t>();
        auto shape = current["shape"].as<size2_t>();
        auto shift = current["shift"].as<float2_t>();
        float22_t matrix = transform::rotate(current["angle"].as<float>());
        transform::Symmetry symmetry(current["symmetry"].as<std::string>());
        auto center = current["center"].as<float2_t>();
        auto interp = current["interp"].as<InterpMode>();

        // Prepare data:
        size_t elements = getElements(shape);
        cpu::memory::PtrHost<float> input(elements);
        cpu::memory::PtrHost<float> output(elements);
        cpu::memory::PtrHost<float> expected(elements);

        //
        constexpr bool GENERATE_ASSETS = true;
        if constexpr(GENERATE_ASSETS) {
            cpu::filter::rectangle(input.get(), {shape.x, shape.y, 1}, {}, {128, 64, 1}, 5);
            cpu::memory::PtrHost<float> tmp(elements);
            cpu::filter::rectangle(tmp.get(), {shape.x, shape.y, 1}, {128, 64, 0}, {32, 32, 1}, 3);
            cpu::math::addArray(input.get(), tmp.get(), input.get(), elements, 1);
            ImageFile::save(filename_input, input.get(), {shape.x, shape.y, 1});

            cpu::transform::apply2D(input.get(), output.get(), shape, shift, matrix, symmetry, center, interp);
            ImageFile::save(filename_expected, output.get(), {shape.x, shape.y, 1});
        } else {
            io::ImageFile file(filename_input, io::READ);
            file.readAll(input.get());
            file.open(filename_expected, io::READ);
            file.readAll(expected.get());

            cpu::transform::apply2D(input.get(), output.get(), shape, shift, matrix, symmetry, center, interp);
            if (interp == INTERP_LINEAR) {
                cpu::math::subtractArray(expected.get(), output.get(), output.get(), elements, 1);
                float min, max, mean;
                cpu::math::minMaxSumMean<float>(output.get(), &min, &max, nullptr, &mean, elements, 1);
                REQUIRE(math::abs(min) < 5e-4f);
                REQUIRE(math::abs(max) < 5e-4f);
                REQUIRE(math::abs(mean) < 1e-6f);
            } else {
                float diff = test::getDifference(expected.get(), output.get(), elements);
                REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-6));
            }
        }
    }
}
