#include <noa/cpu/fourier/Plan.h>
#include <noa/cpu/fourier/Remap.h>

#include <noa/cpu/PtrHost.h>
#include <noa/util/files/MRCFile.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

TEMPLATE_TEST_CASE("Fourier: FC2F <-> F2FC", "[noa][cpu][fourier]", float, double, cfloat_t, cdouble_t) {
    Test::IntRandomizer<size_t> randomizer(1, 128);
    Test::RealRandomizer<TestType> randomizer_data(1., 128.);

    uint ndim = GENERATE(1U, 2U, 3U);

    AND_THEN("fc > f > fc") {
        size3_t shape = Test::getRandomShape(ndim);
        size_t elements = getElements(shape);
        PtrHost<TestType> full_centered_in(elements);
        PtrHost<TestType> full_centered_out(elements);
        PtrHost<TestType> full(elements);

        Test::initDataRandom(full_centered_in.get(), full_centered_in.elements(), randomizer_data);
        Test::initDataZero(full_centered_out.get(), full_centered_out.elements());
        Test::initDataZero(full.get(), full.elements());

        Fourier::FC2F(full_centered_in.get(), full.get(), shape);
        Fourier::F2FC(full.get(), full_centered_out.get(), shape);
        TestType diff = Test::getDifference(full_centered_in.get(), full_centered_out.get(), elements);
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-13));
    }

    AND_THEN("f > fc > f") {
        size3_t shape = Test::getRandomShape(ndim);
        size_t elements = getElements(shape);
        PtrHost<TestType> full_in(elements);
        PtrHost<TestType> full_out(elements);
        PtrHost<TestType> full_centered(elements);

        Test::initDataRandom(full_in.get(), full_in.elements(), randomizer_data);
        Test::initDataZero(full_out.get(), full_out.elements());
        Test::initDataZero(full_centered.get(), full_centered.elements());

        Fourier::F2FC(full_in.get(), full_centered.get(), shape);
        Fourier::FC2F(full_centered.get(), full_out.get(), shape);
        TestType diff = Test::getDifference(full_in.get(), full_out.get(), elements);
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-13));
    }

    AND_THEN("compare with numpy (i)fftshift") {
        fs::path path = NOA_TESTS_DATA;
        path = path / "src" / "fourier";

        AND_THEN("2D") {
            MRCFile file_array(path / "array_2D.mrc", IO::READ);
            MRCFile file_array_reorder;
            size3_t shape = file_array.getShape();
            size_t elements = getElements(shape);
            PtrHost<float> array(elements);
            PtrHost<float> array_reordered_expected(elements);
            PtrHost<float> array_reordered_results(elements);
            file_array.readAll(array.get());

            // fftshift
            file_array_reorder.open(path / "array_fftshift_2D.mrc", IO::READ);
            file_array_reorder.readAll(array_reordered_expected.get());

            Fourier::F2FC(array.get(), array_reordered_results.get(), shape);
            float diff = Test::getDifference(array_reordered_expected.get(), array_reordered_results.get(), elements);
            REQUIRE_THAT(diff, Catch::WithinAbs(0., 1e-13));

            // ifftshift
            Test::initDataZero(array_reordered_expected.get(), elements);
            file_array_reorder.open(path / "array_ifftshift_2D.mrc", IO::READ);
            file_array_reorder.readAll(array_reordered_expected.get());

            Fourier::FC2F(array.get(), array_reordered_results.get(), shape);
            diff = Test::getDifference(array_reordered_expected.get(), array_reordered_results.get(), elements);
            REQUIRE_THAT(diff, Catch::WithinAbs(0., 1e-13));
        }

        AND_THEN("3D") {
            MRCFile file_array(path / "array_3D.mrc", IO::READ);
            MRCFile file_array_reorder;
            size3_t shape = file_array.getShape();
            size_t elements = getElements(shape);
            PtrHost<float> array(elements);
            PtrHost<float> array_reordered_expected(elements);
            PtrHost<float> array_reordered_results(elements);
            file_array.readAll(array.get());

            // fftshift
            file_array_reorder.open(path / "array_fftshift_3D.mrc", IO::READ);
            file_array_reorder.readAll(array_reordered_expected.get());

            Fourier::F2FC(array.get(), array_reordered_results.get(), shape);
            float diff = Test::getDifference(array_reordered_expected.get(), array_reordered_results.get(), elements);
            REQUIRE_THAT(diff, Catch::WithinAbs(0., 1e-13));

            // ifftshift
            Test::initDataZero(array_reordered_expected.get(), elements);
            file_array_reorder.open(path / "array_ifftshift_3D.mrc", IO::READ);
            file_array_reorder.readAll(array_reordered_expected.get());

            Fourier::FC2F(array.get(), array_reordered_results.get(), shape);
            diff = Test::getDifference(array_reordered_expected.get(), array_reordered_results.get(), elements);
            REQUIRE_THAT(diff, Catch::WithinAbs(0., 1e-13));
        }
    }
}

TEMPLATE_TEST_CASE("Fourier: HC2H <-> H2HC", "[noa][cpu][fourier]", float, double, cfloat_t, cdouble_t) {
    Test::IntRandomizer<size_t> randomizer(1, 128);
    Test::RealRandomizer<TestType> randomizer_data(1., 128.);
    uint ndim = GENERATE(1U, 2U, 3U);

    AND_THEN("hc > h > hc") {
        size3_t shape = Test::getRandomShape(ndim);
        size_t elements = getElementsFFT(shape);
        PtrHost<TestType> half_centered_in(elements);
        PtrHost<TestType> half_centered_out(elements);
        PtrHost<TestType> half(elements);

        Test::initDataRandom(half_centered_in.get(), half_centered_in.elements(), randomizer_data);
        Test::initDataZero(half.get(), half.elements());
        Test::initDataZero(half_centered_out.get(), half_centered_out.elements());

        Fourier::HC2H(half_centered_in.get(), half.get(), shape);
        Fourier::H2HC(half.get(), half_centered_out.get(), shape);
        TestType diff = Test::getDifference(half_centered_in.get(), half_centered_out.get(), elements);
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-13));
    }

    AND_THEN("h > hc > h") {
        size3_t shape = Test::getRandomShape(ndim);
        size_t elements = getElementsFFT(shape);
        PtrHost<TestType> half_in(elements);
        PtrHost<TestType> half_out(elements);
        PtrHost<TestType> half_centered(elements);

        Test::initDataRandom(half_in.get(), half_in.elements(), randomizer_data);
        Test::initDataZero(half_centered.get(), half_centered.elements());
        Test::initDataZero(half_out.get(), half_out.elements());

        Fourier::H2HC(half_in.get(), half_centered.get(), shape);
        Fourier::HC2H(half_centered.get(), half_out.get(), shape);
        TestType diff = Test::getDifference(half_in.get(), half_out.get(), elements);
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-13));
    }
}

TEMPLATE_TEST_CASE("Fourier: H2F <-> F2H", "[noa][cpu][fourier]", float, double, cfloat_t, cdouble_t) {
    Test::RealRandomizer<TestType> randomizer_data(1., 128.);

    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = Test::getRandomShape(ndim);
    size_t elements = getElements(shape);
    size_t elements_fft = getElementsFFT(shape);

    AND_THEN("h > f > h") {
        PtrHost<TestType> half_in(elements_fft);
        PtrHost<TestType> half_out(elements_fft);
        PtrHost<TestType> full(elements);

        Test::initDataRandom(half_in.get(), half_in.elements(), randomizer_data);
        Test::initDataZero(half_out.get(), half_out.elements());

        Fourier::H2F(half_in.get(), full.get(), shape);
        Fourier::F2H(full.get(), half_out.get(), shape);

        TestType diff = Test::getAverageDifference(half_in.get(), half_out.get(), elements_fft);
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-14));
    }
}

TEMPLATE_TEST_CASE("Fourier: FC2H", "[noa][cpu][fourier]", float, double, cfloat_t, cdouble_t) {
    Test::RealRandomizer<TestType> randomizer_data(1., 128.);

    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = Test::getRandomShape(ndim);
    size_t elements = getElements(shape);
    size_t elements_fft = getElementsFFT(shape);

    AND_THEN("h > f > fc > h") { // h2fc is not added so we have to fftshift separately.
        AND_THEN("complex") {
            PtrHost<TestType> half_in(elements_fft);
            PtrHost<TestType> half_out(elements_fft);
            PtrHost<TestType> full(elements);
            PtrHost<TestType> full_centered(elements);

            Test::initDataRandom(half_in.get(), half_in.elements(), randomizer_data);
            Test::initDataZero(half_out.get(), half_out.elements());

            Fourier::H2F(half_in.get(), full.get(), shape);
            Fourier::F2FC(full.get(), full_centered.get(), shape);
            Fourier::FC2H(full_centered.get(), half_out.get(), shape);

            TestType diff = Test::getAverageDifference(half_in.get(), half_out.get(), elements_fft);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-14));
        }

        AND_THEN("real") {
            PtrHost<TestType> half_in(elements_fft);
            PtrHost<TestType> half_out(elements_fft);
            PtrHost<TestType> full(elements);
            PtrHost<TestType> full_centered(elements);

            Test::initDataRandom(half_in.get(), half_in.elements(), randomizer_data);
            Test::initDataZero(half_out.get(), half_out.elements());

            Fourier::H2F(half_in.get(), full.get(), shape);
            Fourier::F2FC(full.get(), full_centered.get(), shape);
            Fourier::FC2H(full_centered.get(), half_out.get(), shape);

            TestType diff = Test::getAverageDifference(half_in.get(), half_out.get(), elements_fft);
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), 1e-14));
        }
    }
}
