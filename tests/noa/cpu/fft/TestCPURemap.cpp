#include <noa/common/io/ImageFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/fft/Remap.h>

#include "Helpers.h"
#include "Assets.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("cpu::fft::fc2f(), f2fc()", "[noa][cpu][fft]", float, double, cfloat_t, cdouble_t) {
    test::IntRandomizer<size_t> randomizer(1, 128);
    test::RealRandomizer<TestType> randomizer_data(1., 128.);

    uint ndim = GENERATE(1U, 2U, 3U);

    AND_THEN("fc > f > fc") {
        size3_t shape = test::getRandomShape(ndim);
        size_t elements = getElements(shape);
        cpu::memory::PtrHost<TestType> full_centered_in(elements);
        cpu::memory::PtrHost<TestType> full_centered_out(elements);
        cpu::memory::PtrHost<TestType> full(elements);

        test::initDataRandom(full_centered_in.get(), full_centered_in.elements(), randomizer_data);
        test::initDataZero(full_centered_out.get(), full_centered_out.elements());
        test::initDataZero(full.get(), full.elements());

        cpu::fft::remap(full_centered_in.get(), full.get(), shape, 1, fft::FC2F);
        cpu::fft::remap(full.get(), full_centered_out.get(), shape, 1, fft::F2FC);
        TestType diff = test::getDifference(full_centered_in.get(), full_centered_out.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-13));
    }

    AND_THEN("f > fc > f") {
        size3_t shape = test::getRandomShape(ndim);
        size_t elements = getElements(shape);
        cpu::memory::PtrHost<TestType> full_in(elements);
        cpu::memory::PtrHost<TestType> full_out(elements);
        cpu::memory::PtrHost<TestType> full_centered(elements);

        test::initDataRandom(full_in.get(), full_in.elements(), randomizer_data);
        test::initDataZero(full_out.get(), full_out.elements());
        test::initDataZero(full_centered.get(), full_centered.elements());

        cpu::fft::remap(full_in.get(), full_centered.get(), shape, 1, fft::F2FC);
        cpu::fft::remap(full_centered.get(), full_out.get(), shape, 1, fft::FC2F);
        TestType diff = test::getDifference(full_in.get(), full_out.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-13));
    }
}

TEST_CASE("cpu::fft::fc2f(), f2fc() -- vs numpy", "[assets][noa][cpu][fft]") {
    fs::path path = test::PATH_TEST_DATA / "fft";
    YAML::Node tests = YAML::LoadFile(path / "param.yaml")["remap"];
    io::ImageFile file;

    AND_THEN("2D") {
        file.open(path / tests["2D"]["input"].as<path_t>(), io::READ);
        size3_t shape = file.shape();
        size_t elements = getElements(shape);
        cpu::memory::PtrHost<float> array(elements);
        file.readAll(array.get());

        cpu::memory::PtrHost<float> array_reordered_expected(elements);
        cpu::memory::PtrHost<float> array_reordered_results(elements);

        // fftshift
        file.open(path / tests["2D"]["fftshift"].as<path_t>(), io::READ);
        file.readAll(array_reordered_expected.get());

        cpu::fft::remap(array.get(), array_reordered_results.get(), shape, 1, fft::F2FC);
        float diff = test::getDifference(array_reordered_expected.get(), array_reordered_results.get(), elements);
        REQUIRE_THAT(diff, Catch::WithinAbs(0., 1e-13));

        // ifftshift
        test::initDataZero(array_reordered_expected.get(), elements);
        file.open(path / tests["2D"]["ifftshift"].as<path_t>(), io::READ);
        file.readAll(array_reordered_expected.get());

        cpu::fft::remap(array.get(), array_reordered_results.get(), shape, 1, fft::FC2F);
        diff = test::getDifference(array_reordered_expected.get(), array_reordered_results.get(), elements);
        REQUIRE_THAT(diff, Catch::WithinAbs(0., 1e-13));
    }

    AND_THEN("3D") {
        file.open(path / tests["3D"]["input"].as<path_t>(), io::READ);
        size3_t shape = file.shape();
        size_t elements = getElements(shape);
        cpu::memory::PtrHost<float> array(elements);
        file.readAll(array.get());

        cpu::memory::PtrHost<float> array_reordered_expected(elements);
        cpu::memory::PtrHost<float> array_reordered_results(elements);

        // fftshift
        file.open(path / tests["3D"]["fftshift"].as<path_t>(), io::READ);
        file.readAll(array_reordered_expected.get());

        cpu::fft::remap(array.get(), array_reordered_results.get(), shape, 1, fft::F2FC);
        float diff = test::getDifference(array_reordered_expected.get(), array_reordered_results.get(), elements);
        REQUIRE_THAT(diff, Catch::WithinAbs(0., 1e-13));

        // ifftshift
        test::initDataZero(array_reordered_expected.get(), elements);
        file.open(path / tests["3D"]["ifftshift"].as<path_t>(), io::READ);
        file.readAll(array_reordered_expected.get());

        cpu::fft::remap(array.get(), array_reordered_results.get(), shape, 1, fft::FC2F);
        diff = test::getDifference(array_reordered_expected.get(), array_reordered_results.get(), elements);
        REQUIRE_THAT(diff, Catch::WithinAbs(0., 1e-13));
    }
}

TEMPLATE_TEST_CASE("cpu::fft::hc2h(), h2hc()", "[noa][cpu][fft]", float, double, cfloat_t, cdouble_t) {
    test::IntRandomizer<size_t> randomizer(1, 128);
    test::RealRandomizer<TestType> randomizer_data(1., 128.);
    uint ndim = GENERATE(1U, 2U, 3U);

    AND_THEN("hc > h > hc") {
        size3_t shape = test::getRandomShape(ndim);
        size_t elements = getElementsFFT(shape);
        cpu::memory::PtrHost<TestType> half_centered_in(elements);
        cpu::memory::PtrHost<TestType> half_centered_out(elements);
        cpu::memory::PtrHost<TestType> half(elements);

        test::initDataRandom(half_centered_in.get(), half_centered_in.elements(), randomizer_data);
        test::initDataZero(half.get(), half.elements());
        test::initDataZero(half_centered_out.get(), half_centered_out.elements());

        cpu::fft::remap(half_centered_in.get(), half.get(), shape, 1, fft::HC2H);
        cpu::fft::remap(half.get(), half_centered_out.get(), shape, 1, fft::H2HC);
        TestType diff = test::getDifference(half_centered_in.get(), half_centered_out.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-13));
    }

    AND_THEN("h > hc > h") {
        size3_t shape = test::getRandomShape(ndim);
        size_t elements = getElementsFFT(shape);
        cpu::memory::PtrHost<TestType> half_in(elements);
        cpu::memory::PtrHost<TestType> half_out(elements);
        cpu::memory::PtrHost<TestType> half_centered(elements);

        test::initDataRandom(half_in.get(), half_in.elements(), randomizer_data);
        test::initDataZero(half_centered.get(), half_centered.elements());
        test::initDataZero(half_out.get(), half_out.elements());

        cpu::fft::remap(half_in.get(), half_centered.get(), shape, 1, fft::H2HC);
        cpu::fft::remap(half_centered.get(), half_out.get(), shape, 1, fft::HC2H);
        TestType diff = test::getDifference(half_in.get(), half_out.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-13));
    }
}

TEMPLATE_TEST_CASE("cpu::fft::h2f(), f2h()", "[noa][cpu][fft]", float, double, cfloat_t, cdouble_t) {
    test::RealRandomizer<TestType> randomizer_data(1., 128.);

    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = test::getRandomShape(ndim);
    size_t elements = getElements(shape);
    size_t elements_fft = getElementsFFT(shape);

    AND_THEN("h > f > h") {
        cpu::memory::PtrHost<TestType> half_in(elements_fft);
        cpu::memory::PtrHost<TestType> half_out(elements_fft);
        cpu::memory::PtrHost<TestType> full(elements);

        test::initDataRandom(half_in.get(), half_in.elements(), randomizer_data);
        test::initDataZero(half_out.get(), half_out.elements());

        cpu::fft::remap(half_in.get(), full.get(), shape, 1, fft::H2F);
        cpu::fft::remap(full.get(), half_out.get(), shape, 1, fft::F2H);

        TestType diff = test::getAverageDifference(half_in.get(), half_out.get(), elements_fft);
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
    }
}

TEMPLATE_TEST_CASE("cpu::fft::fc2h()", "[noa][cpu][fft]", float, double, cfloat_t, cdouble_t) {
    test::RealRandomizer<TestType> randomizer_data(1., 128.);

    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = test::getRandomShape(ndim);
    size_t elements = getElements(shape);
    size_t elements_fft = getElementsFFT(shape);

    AND_THEN("h > f > fc > h") { // h2fc is not added so we have to fftshift separately.
        AND_THEN("complex") {
            cpu::memory::PtrHost<TestType> half_in(elements_fft);
            cpu::memory::PtrHost<TestType> half_out(elements_fft);
            cpu::memory::PtrHost<TestType> full(elements);
            cpu::memory::PtrHost<TestType> full_centered(elements);

            test::initDataRandom(half_in.get(), half_in.elements(), randomizer_data);
            test::initDataZero(half_out.get(), half_out.elements());

            cpu::fft::remap(half_in.get(), full.get(), shape, 1, fft::H2F);
            cpu::fft::remap(full.get(), full_centered.get(), shape, 1, fft::F2FC);
            cpu::fft::remap(full_centered.get(), half_out.get(), shape, 1, fft::FC2H);

            TestType diff = test::getAverageDifference(half_in.get(), half_out.get(), elements_fft);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }

        AND_THEN("real") {
            cpu::memory::PtrHost<TestType> half_in(elements_fft);
            cpu::memory::PtrHost<TestType> half_out(elements_fft);
            cpu::memory::PtrHost<TestType> full(elements);
            cpu::memory::PtrHost<TestType> full_centered(elements);

            test::initDataRandom(half_in.get(), half_in.elements(), randomizer_data);
            test::initDataZero(half_out.get(), half_out.elements());

            cpu::fft::remap(half_in.get(), full.get(), shape, 1, fft::H2F);
            cpu::fft::remap(full.get(), full_centered.get(), shape, 1, fft::F2FC);
            cpu::fft::remap(full_centered.get(), half_out.get(), shape, 1, fft::FC2H);

            TestType diff = test::getAverageDifference(half_in.get(), half_out.get(), elements_fft);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }
    }
}
