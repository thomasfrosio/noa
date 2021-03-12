#include <noa/cpu/fourier/Plan.h>
#include <noa/cpu/fourier/Resize.h>

#include <noa/cpu/PtrHost.h>

#include <catch2/catch.hpp>
#include "../../../Helpers.h"

using namespace Noa;

// These tests uses that fact that padding and then cropping on the output returns the same input array.
TEMPLATE_TEST_CASE("Fourier: pad / crop", "[noa][fourier]", float) {
    Test::RealRandomizer<TestType> randomizer(1, 5);
    Test::IntRandomizer<size_t> rand_int(2, 32);
    using complex_t = Noa::Complex<TestType>;

    AND_THEN("pad then crop") {
        size3_t shape1D = {rand_int.get(), 1, 1};
        size3_t shape2D = {rand_int.get(), rand_int.get(), 1};
        size3_t shape3D = {rand_int.get(), rand_int.get(), rand_int.get()};
        size3_t shape1D_padded = {shape1D.x + rand_int.get(), 1, 1};
        size3_t shape2D_padded = {shape2D.x + rand_int.get(), shape2D.y + rand_int.get(), 1};
        size3_t shape3D_padded = {shape3D.x + rand_int.get(), shape3D.y + rand_int.get(), shape3D.z + rand_int.get()};
        std::vector<size3_t> shapes = {shape1D, shape1D_padded, shape2D, shape2D_padded, shape3D, shape3D_padded};

        for (size_t idx = 0; idx < 6; idx += 2) {
            size3_t shape = shapes[idx], shape_padded = shapes[idx + 1];
            size_t elements = getElementsFFT(shape), elements_padded = getElementsFFT(shape_padded);
            PtrHost<complex_t> original(elements);
            PtrHost<complex_t> padded(elements_padded);
            PtrHost<complex_t> cropped(elements);
            INFO(shape)
            INFO(shape_padded)
            Test::initDataRandom(original.get(), elements, randomizer);
            Fourier::pad(original.get(), shape, padded.get(), shape_padded);
            Fourier::crop(padded.get(), shape_padded, cropped.get(), shape);

            complex_t diff = Test::getDifference(original.get(), cropped.get(), elements);
            REQUIRE_THAT(diff.real(), Catch::WithinAbs(0., 1e-13));
            REQUIRE_THAT(diff.imag(), Catch::WithinAbs(0., 1e-13));
        }
    }

    AND_THEN("padFull then cropFull") {
        size3_t shape1D = {rand_int.get(), 1, 1};
        size3_t shape2D = {rand_int.get(), rand_int.get(), 1};
        size3_t shape3D = {rand_int.get(), rand_int.get(), rand_int.get()};
        size3_t shape1D_padded = {shape1D.x + rand_int.get(), 1, 1};
        size3_t shape2D_padded = {shape2D.x + rand_int.get(), shape2D.y + rand_int.get(), 1};
        size3_t shape3D_padded = {shape3D.x + rand_int.get(), shape3D.y + rand_int.get(), shape3D.z + rand_int.get()};
        std::vector<size3_t> shapes = {shape1D, shape1D_padded, shape2D, shape2D_padded, shape3D, shape3D_padded};

        for (size_t idx = 0; idx < 6; idx += 2) {
            size3_t shape = shapes[idx], shape_padded = shapes[idx + 1];
            size_t elements = getElements(shape), elements_padded = getElements(shape_padded);
            PtrHost<complex_t> original(elements);
            PtrHost<complex_t> padded(elements_padded);
            PtrHost<complex_t> cropped(elements);
            INFO(shape)
            INFO(shape_padded)
            Test::initDataRandom(original.get(), elements, randomizer);
            Fourier::padFull(original.get(), shape, padded.get(), shape_padded);
            Fourier::cropFull(padded.get(), shape_padded, cropped.get(), shape);

            complex_t diff = Test::getDifference(original.get(), cropped.get(), elements);
            REQUIRE_THAT(diff.real(), Catch::WithinAbs(0., 1e-13));
            REQUIRE_THAT(diff.imag(), Catch::WithinAbs(0., 1e-13));
        }
    }
}
