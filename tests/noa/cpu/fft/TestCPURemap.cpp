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
        size_t size = elements(shape);
        cpu::memory::PtrHost<TestType> full_centered_in(size);
        cpu::memory::PtrHost<TestType> full_centered_out(size);
        cpu::memory::PtrHost<TestType> full(size);

        test::initDataRandom(full_centered_in.get(), full_centered_in.size(), randomizer_data);
        test::initDataZero(full_centered_out.get(), full_centered_out.size());
        test::initDataZero(full.get(), full.size());

        cpu::fft::remap(fft::FC2F, full_centered_in.get(), full.get(), shape, 1);
        cpu::fft::remap(fft::F2FC, full.get(), full_centered_out.get(), shape, 1);
        TestType diff = test::getDifference(full_centered_in.get(), full_centered_out.get(), size);
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-13));
    }

    AND_THEN("f > fc > f") {
        size3_t shape = test::getRandomShape(ndim);
        size_t size = elements(shape);
        cpu::memory::PtrHost<TestType> full_in(size);
        cpu::memory::PtrHost<TestType> full_out(size);
        cpu::memory::PtrHost<TestType> full_centered(size);

        test::initDataRandom(full_in.get(), full_in.size(), randomizer_data);
        test::initDataZero(full_out.get(), full_out.size());
        test::initDataZero(full_centered.get(), full_centered.size());

        cpu::fft::remap(fft::F2FC, full_in.get(), full_centered.get(), shape, 1);
        cpu::fft::remap(fft::FC2F, full_centered.get(), full_out.get(), shape, 1);
        TestType diff = test::getDifference(full_in.get(), full_out.get(), size);
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
        size_t size = elements(shape);
        cpu::memory::PtrHost<float> array(size);
        file.readAll(array.get());

        cpu::memory::PtrHost<float> array_reordered_expected(size);
        cpu::memory::PtrHost<float> array_reordered_results(size);

        // fftshift
        file.open(path / tests["2D"]["fftshift"].as<path_t>(), io::READ);
        file.readAll(array_reordered_expected.get());

        cpu::fft::remap(fft::F2FC, array.get(), array_reordered_results.get(), shape, 1);
        float diff = test::getDifference(array_reordered_expected.get(), array_reordered_results.get(), size);
        REQUIRE_THAT(diff, Catch::WithinAbs(0., 1e-13));

        // ifftshift
        test::initDataZero(array_reordered_expected.get(), size);
        file.open(path / tests["2D"]["ifftshift"].as<path_t>(), io::READ);
        file.readAll(array_reordered_expected.get());

        cpu::fft::remap(fft::FC2F, array.get(), array_reordered_results.get(), shape, 1);
        diff = test::getDifference(array_reordered_expected.get(), array_reordered_results.get(), size);
        REQUIRE_THAT(diff, Catch::WithinAbs(0., 1e-13));
    }

    AND_THEN("3D") {
        file.open(path / tests["3D"]["input"].as<path_t>(), io::READ);
        size3_t shape = file.shape();
        size_t size = elements(shape);
        cpu::memory::PtrHost<float> array(size);
        file.readAll(array.get());

        cpu::memory::PtrHost<float> array_reordered_expected(size);
        cpu::memory::PtrHost<float> array_reordered_results(size);

        // fftshift
        file.open(path / tests["3D"]["fftshift"].as<path_t>(), io::READ);
        file.readAll(array_reordered_expected.get());

        cpu::fft::remap(fft::F2FC, array.get(), array_reordered_results.get(), shape, 1);
        float diff = test::getDifference(array_reordered_expected.get(), array_reordered_results.get(), size);
        REQUIRE_THAT(diff, Catch::WithinAbs(0., 1e-13));

        // ifftshift
        test::initDataZero(array_reordered_expected.get(), size);
        file.open(path / tests["3D"]["ifftshift"].as<path_t>(), io::READ);
        file.readAll(array_reordered_expected.get());

        cpu::fft::remap(fft::FC2F, array.get(), array_reordered_results.get(), shape, 1);
        diff = test::getDifference(array_reordered_expected.get(), array_reordered_results.get(), size);
        REQUIRE_THAT(diff, Catch::WithinAbs(0., 1e-13));
    }
}

TEMPLATE_TEST_CASE("cpu::fft::hc2h(), h2hc()", "[noa][cpu][fft]", float, double, cfloat_t, cdouble_t) {
    test::IntRandomizer<size_t> randomizer(1, 128);
    test::RealRandomizer<TestType> randomizer_data(-128., 128.);
    uint ndim = GENERATE(1U, 2U, 3U);

    AND_THEN("hc > h > hc") {
        size3_t shape = test::getRandomShape(ndim);
        size_t size = elementsFFT(shape);
        cpu::memory::PtrHost<TestType> half_centered_in(size);
        cpu::memory::PtrHost<TestType> half_centered_out(size);
        cpu::memory::PtrHost<TestType> half(size);

        test::initDataRandom(half_centered_in.get(), half_centered_in.size(), randomizer_data);
        test::initDataZero(half.get(), half.size());
        test::initDataZero(half_centered_out.get(), half_centered_out.size());

        cpu::fft::remap(fft::HC2H, half_centered_in.get(), half.get(), shape, 1);
        cpu::fft::remap(fft::H2HC, half.get(), half_centered_out.get(), shape, 1);
        TestType diff = test::getDifference(half_centered_in.get(), half_centered_out.get(), size);
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-13));
    }

    AND_THEN("h > hc > h") {
        size3_t shape = test::getRandomShape(ndim);
        size_t size = elementsFFT(shape);
        cpu::memory::PtrHost<TestType> half_in(size);
        cpu::memory::PtrHost<TestType> half_out(size);
        cpu::memory::PtrHost<TestType> half_centered(size);

        test::initDataRandom(half_in.get(), half_in.size(), randomizer_data);
        test::initDataZero(half_centered.get(), half_centered.size());
        test::initDataZero(half_out.get(), half_out.size());

        cpu::fft::remap(fft::H2HC, half_in.get(), half_centered.get(), shape, 1);
        cpu::fft::remap(fft::HC2H, half_centered.get(), half_out.get(), shape, 1);
        TestType diff = test::getDifference(half_in.get(), half_out.get(), size);
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-13));
    }

    AND_THEN("in-place") {
        size_t batches = 2;
        size3_t shape = test::getRandomShape(3, true);
        INFO(shape);
        size_t elements = elementsFFT(shape);
        cpu::memory::PtrHost<TestType> half_in(elements * batches);
        cpu::memory::PtrHost<TestType> half_out(elements * batches);
        cpu::memory::PtrHost<TestType> half_centered(elements * batches);

        test::initDataRandom(half_in.get(), half_in.elements(), randomizer_data);

        cpu::fft::remap(fft::H2HC, half_in.get(), half_out.get(), shape, batches);
        cpu::fft::remap(fft::H2HC, half_in.get(), half_in.get(), shape, batches);
        TestType diff = test::getDifference(half_in.get(), half_out.get(), half_in.size());
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-13));
    }
}

TEMPLATE_TEST_CASE("cpu::fft::h2f(), f2h()", "[noa][cpu][fft]", float, double, cfloat_t, cdouble_t) {
    test::RealRandomizer<TestType> randomizer_data(1., 128.);

    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = test::getRandomShape(ndim);
    size_t size = elements(shape);
    size_t size_fft = elementsFFT(shape);

    AND_THEN("h > f > h") {
        cpu::memory::PtrHost<TestType> half_in(size_fft);
        cpu::memory::PtrHost<TestType> half_out(size_fft);
        cpu::memory::PtrHost<TestType> full(size);

        test::initDataRandom(half_in.get(), half_in.size(), randomizer_data);
        test::initDataZero(half_out.get(), half_out.size());

        cpu::fft::remap(fft::H2F, half_in.get(), full.get(), shape, 1);
        cpu::fft::remap(fft::F2H, full.get(), half_out.get(), shape, 1);

        TestType diff = test::getAverageDifference(half_in.get(), half_out.get(), size_fft);
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
    }
}

TEMPLATE_TEST_CASE("cpu::fft::hc2f(), f2hc()", "[noa][cpu][fft]", float) { // double, cfloat_t, cdouble_t
    test::RealRandomizer<TestType> randomizer_data(-128., 128.);

    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = test::getRandomShape(ndim);
    size_t elements = noa::elements(shape);
    size_t elements_fft = noa::elementsFFT(shape);

    cpu::memory::PtrHost<TestType> half(elements_fft);
    cpu::memory::PtrHost<TestType> half_centered(elements_fft);
    cpu::memory::PtrHost<TestType> half_2(elements_fft);
    cpu::memory::PtrHost<TestType> full(elements);
    cpu::memory::PtrHost<TestType> full_2(elements);

    AND_THEN("hc > f") {
        test::initDataRandom(half.get(), half.elements(), randomizer_data);

        cpu::fft::remap(fft::H2HC, half.get(), half_centered.get(), shape, 1);
        cpu::fft::remap(fft::HC2F, half_centered.get(), full.get(), shape, 1);
        cpu::fft::remap(fft::H2F, half.get(), full_2.get(), shape, 1);

        TestType diff = test::getAverageDifference(full.get(), full_2.get(), full_2.size());
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
    }

    AND_THEN("f > hc") {
        test::initDataRandom(full.get(), full.elements(), randomizer_data);

        cpu::fft::remap(fft::F2H, full.get(), half.get(), shape, 1);
        cpu::fft::remap(fft::H2HC, half.get(), half_centered.get(), shape, 1);
        cpu::fft::remap(fft::F2HC, full.get(), half_2.get(), shape, 1);

        TestType diff = test::getAverageDifference(half_centered.get(), half_2.get(), half.size());
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
    }
}

TEMPLATE_TEST_CASE("cpu::fft::fc2h()", "[noa][cpu][fft]", float, double, cfloat_t, cdouble_t) {
    test::RealRandomizer<TestType> randomizer_data(1., 128.);

    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = test::getRandomShape(ndim);
    size_t elements = noa::elements(shape);
    size_t elements_fft = noa::elementsFFT(shape);

    AND_THEN("h > f > fc > h") { // h2fc is not added so we have to fftshift separately.
        AND_THEN("complex") {
            cpu::memory::PtrHost<TestType> half_in(elements_fft);
            cpu::memory::PtrHost<TestType> half_out(elements_fft);
            cpu::memory::PtrHost<TestType> full(elements);
            cpu::memory::PtrHost<TestType> full_centered(elements);

            test::initDataRandom(half_in.get(), half_in.elements(), randomizer_data);
            test::initDataZero(half_out.get(), half_out.elements());

            cpu::fft::remap(fft::H2F, half_in.get(), full.get(), shape, 1);
            cpu::fft::remap(fft::F2FC, full.get(), full_centered.get(), shape, 1);
            cpu::fft::remap(fft::FC2H, full_centered.get(), half_out.get(), shape, 1);

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

            cpu::fft::remap(fft::H2F, half_in.get(), full.get(), shape, 1);
            cpu::fft::remap(fft::F2FC, full.get(), full_centered.get(), shape, 1);
            cpu::fft::remap(fft::FC2H, full_centered.get(), half_out.get(), shape, 1);

            TestType diff = test::getAverageDifference(half_in.get(), half_out.get(), elements_fft);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }
    }
}