#include <noa/cpu/fourier/Plan.h>
#include <noa/cpu/fourier/Remap.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/io/files/MRCFile.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("Fourier: FC2F <-> F2FC", "[noa][cpu][fourier]", float, double, cfloat_t, cdouble_t) {
    test::IntRandomizer<size_t> randomizer(1, 128);
    test::RealRandomizer<TestType> randomizer_data(1., 128.);

    uint ndim = GENERATE(1U, 2U, 3U);

    AND_THEN("fc > f > fc") {
        size3_t shape = test::getRandomShape(ndim);
        size_t elements = getElements(shape);
        memory::PtrHost<TestType> full_centered_in(elements);
        memory::PtrHost<TestType> full_centered_out(elements);
        memory::PtrHost<TestType> full(elements);

        test::initDataRandom(full_centered_in.get(), full_centered_in.elements(), randomizer_data);
        test::initDataZero(full_centered_out.get(), full_centered_out.elements());
        test::initDataZero(full.get(), full.elements());

        fourier::FC2F(full_centered_in.get(), full.get(), shape);
        fourier::F2FC(full.get(), full_centered_out.get(), shape);
        TestType diff = test::getDifference(full_centered_in.get(), full_centered_out.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-13));
    }

    AND_THEN("f > fc > f") {
        size3_t shape = test::getRandomShape(ndim);
        size_t elements = getElements(shape);
        memory::PtrHost<TestType> full_in(elements);
        memory::PtrHost<TestType> full_out(elements);
        memory::PtrHost<TestType> full_centered(elements);

        test::initDataRandom(full_in.get(), full_in.elements(), randomizer_data);
        test::initDataZero(full_out.get(), full_out.elements());
        test::initDataZero(full_centered.get(), full_centered.elements());

        fourier::F2FC(full_in.get(), full_centered.get(), shape);
        fourier::FC2F(full_centered.get(), full_out.get(), shape);
        TestType diff = test::getDifference(full_in.get(), full_out.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-13));
    }

    AND_THEN("compare with numpy (i)fftshift") {
        fs::path path = test::PATH_TEST_DATA / "fourier";

        AND_THEN("2D") {
            MRCFile file_array(path / "tmp_array_2D.mrc", io::READ);
            MRCFile file_array_reorder;
            size3_t shape = file_array.getShape();
            size_t elements = getElements(shape);
            memory::PtrHost<float> array(elements);
            memory::PtrHost<float> array_reordered_expected(elements);
            memory::PtrHost<float> array_reordered_results(elements);
            file_array.readAll(array.get());

            // fftshift
            file_array_reorder.open(path / "tmp_array_fftshift_2D.mrc", io::READ);
            file_array_reorder.readAll(array_reordered_expected.get());

            fourier::F2FC(array.get(), array_reordered_results.get(), shape);
            float diff = test::getDifference(array_reordered_expected.get(), array_reordered_results.get(), elements);
            REQUIRE_THAT(diff, Catch::WithinAbs(0., 1e-13));

            // ifftshift
            test::initDataZero(array_reordered_expected.get(), elements);
            file_array_reorder.open(path / "tmp_array_ifftshift_2D.mrc", io::READ);
            file_array_reorder.readAll(array_reordered_expected.get());

            fourier::FC2F(array.get(), array_reordered_results.get(), shape);
            diff = test::getDifference(array_reordered_expected.get(), array_reordered_results.get(), elements);
            REQUIRE_THAT(diff, Catch::WithinAbs(0., 1e-13));
        }

        AND_THEN("3D") {
            MRCFile file_array(path / "tmp_array_3D.mrc", io::READ);
            MRCFile file_array_reorder;
            size3_t shape = file_array.getShape();
            size_t elements = getElements(shape);
            memory::PtrHost<float> array(elements);
            memory::PtrHost<float> array_reordered_expected(elements);
            memory::PtrHost<float> array_reordered_results(elements);
            file_array.readAll(array.get());

            // fftshift
            file_array_reorder.open(path / "tmp_array_fftshift_3D.mrc", io::READ);
            file_array_reorder.readAll(array_reordered_expected.get());

            fourier::F2FC(array.get(), array_reordered_results.get(), shape);
            float diff = test::getDifference(array_reordered_expected.get(), array_reordered_results.get(), elements);
            REQUIRE_THAT(diff, Catch::WithinAbs(0., 1e-13));

            // ifftshift
            test::initDataZero(array_reordered_expected.get(), elements);
            file_array_reorder.open(path / "tmp_array_ifftshift_3D.mrc", io::READ);
            file_array_reorder.readAll(array_reordered_expected.get());

            fourier::FC2F(array.get(), array_reordered_results.get(), shape);
            diff = test::getDifference(array_reordered_expected.get(), array_reordered_results.get(), elements);
            REQUIRE_THAT(diff, Catch::WithinAbs(0., 1e-13));
        }
    }
}

TEMPLATE_TEST_CASE("Fourier: HC2H <-> H2HC", "[noa][cpu][fourier]", float, double, cfloat_t, cdouble_t) {
    test::IntRandomizer<size_t> randomizer(1, 128);
    test::RealRandomizer<TestType> randomizer_data(1., 128.);
    uint ndim = GENERATE(1U, 2U, 3U);

    AND_THEN("hc > h > hc") {
        size3_t shape = test::getRandomShape(ndim);
        size_t elements = getElementsFFT(shape);
        memory::PtrHost<TestType> half_centered_in(elements);
        memory::PtrHost<TestType> half_centered_out(elements);
        memory::PtrHost<TestType> half(elements);

        test::initDataRandom(half_centered_in.get(), half_centered_in.elements(), randomizer_data);
        test::initDataZero(half.get(), half.elements());
        test::initDataZero(half_centered_out.get(), half_centered_out.elements());

        fourier::HC2H(half_centered_in.get(), half.get(), shape);
        fourier::H2HC(half.get(), half_centered_out.get(), shape);
        TestType diff = test::getDifference(half_centered_in.get(), half_centered_out.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-13));
    }

    AND_THEN("h > hc > h") {
        size3_t shape = test::getRandomShape(ndim);
        size_t elements = getElementsFFT(shape);
        memory::PtrHost<TestType> half_in(elements);
        memory::PtrHost<TestType> half_out(elements);
        memory::PtrHost<TestType> half_centered(elements);

        test::initDataRandom(half_in.get(), half_in.elements(), randomizer_data);
        test::initDataZero(half_centered.get(), half_centered.elements());
        test::initDataZero(half_out.get(), half_out.elements());

        fourier::H2HC(half_in.get(), half_centered.get(), shape);
        fourier::HC2H(half_centered.get(), half_out.get(), shape);
        TestType diff = test::getDifference(half_in.get(), half_out.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-13));
    }
}

TEMPLATE_TEST_CASE("Fourier: H2F <-> F2H", "[noa][cpu][fourier]", float, double, cfloat_t, cdouble_t) {
    test::RealRandomizer<TestType> randomizer_data(1., 128.);

    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = test::getRandomShape(ndim);
    size_t elements = getElements(shape);
    size_t elements_fft = getElementsFFT(shape);

    AND_THEN("h > f > h") {
        memory::PtrHost<TestType> half_in(elements_fft);
        memory::PtrHost<TestType> half_out(elements_fft);
        memory::PtrHost<TestType> full(elements);

        test::initDataRandom(half_in.get(), half_in.elements(), randomizer_data);
        test::initDataZero(half_out.get(), half_out.elements());

        fourier::H2F(half_in.get(), full.get(), shape);
        fourier::F2H(full.get(), half_out.get(), shape);

        TestType diff = test::getAverageDifference(half_in.get(), half_out.get(), elements_fft);
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
    }
}

TEMPLATE_TEST_CASE("Fourier: FC2H", "[noa][cpu][fourier]", float, double, cfloat_t, cdouble_t) {
    test::RealRandomizer<TestType> randomizer_data(1., 128.);

    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = test::getRandomShape(ndim);
    size_t elements = getElements(shape);
    size_t elements_fft = getElementsFFT(shape);

    AND_THEN("h > f > fc > h") { // h2fc is not added so we have to fftshift separately.
        AND_THEN("complex") {
            memory::PtrHost<TestType> half_in(elements_fft);
            memory::PtrHost<TestType> half_out(elements_fft);
            memory::PtrHost<TestType> full(elements);
            memory::PtrHost<TestType> full_centered(elements);

            test::initDataRandom(half_in.get(), half_in.elements(), randomizer_data);
            test::initDataZero(half_out.get(), half_out.elements());

            fourier::H2F(half_in.get(), full.get(), shape);
            fourier::F2FC(full.get(), full_centered.get(), shape);
            fourier::FC2H(full_centered.get(), half_out.get(), shape);

            TestType diff = test::getAverageDifference(half_in.get(), half_out.get(), elements_fft);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }

        AND_THEN("real") {
            memory::PtrHost<TestType> half_in(elements_fft);
            memory::PtrHost<TestType> half_out(elements_fft);
            memory::PtrHost<TestType> full(elements);
            memory::PtrHost<TestType> full_centered(elements);

            test::initDataRandom(half_in.get(), half_in.elements(), randomizer_data);
            test::initDataZero(half_out.get(), half_out.elements());

            fourier::H2F(half_in.get(), full.get(), shape);
            fourier::F2FC(full.get(), full_centered.get(), shape);
            fourier::FC2H(full_centered.get(), half_out.get(), shape);

            TestType diff = test::getAverageDifference(half_in.get(), half_out.get(), elements_fft);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }
    }
}
