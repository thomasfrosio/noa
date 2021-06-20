#include <noa/cpu/fourier/Resize.h>
#include <noa/cpu/memory/PtrHost.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

// These tests uses that fact cropping after padding cancels the padding and returns the input array.
// These are not very good tests but it is better than nothing.
TEMPLATE_TEST_CASE("Fourier: pad / crop", "[noa][fourier]", float, cfloat_t, double, cdouble_t) {
    test::RealRandomizer<TestType> randomizer(1., 5.);
    test::IntRandomizer<size_t> randomizer_int(0, 32);
    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = test::getRandomShape(ndim);
    size3_t shape_padded(shape);
    if (ndim > 2) shape_padded.z += randomizer_int.get();
    if (ndim > 1) shape_padded.y += randomizer_int.get();
    shape_padded.x += randomizer_int.get();

    INFO(shape);
    INFO(shape_padded);

    AND_THEN("pad then crop") {
        size_t elements_fft = getElementsFFT(shape);
        size_t elements_fft_padded = getElementsFFT(shape_padded);
        memory::PtrHost<TestType> original(elements_fft);
        memory::PtrHost<TestType> padded(elements_fft_padded);
        memory::PtrHost<TestType> cropped(elements_fft);

        test::initDataRandom(original.get(), elements_fft, randomizer);
        fourier::pad(original.get(), shape, padded.get(), shape_padded);
        fourier::crop(padded.get(), shape_padded, cropped.get(), shape);

        TestType diff = test::getDifference(original.get(), cropped.get(), elements_fft);
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-13));
    }

    AND_THEN("padFull then cropFull") {
        size_t elements = getElements(shape);
        size_t elements_padded = getElements(shape_padded);
        memory::PtrHost<TestType> original(elements);
        memory::PtrHost<TestType> padded(elements_padded);
        memory::PtrHost<TestType> cropped(elements);

        test::initDataRandom(original.get(), elements, randomizer);
        fourier::padFull(original.get(), shape, padded.get(), shape_padded);
        fourier::cropFull(padded.get(), shape_padded, cropped.get(), shape);

        TestType diff = test::getDifference(original.get(), cropped.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-13));
    }
}
