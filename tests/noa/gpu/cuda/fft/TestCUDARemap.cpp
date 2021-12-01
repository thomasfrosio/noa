#include <noa/cpu/fft/Remap.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/fft/Remap.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("cuda::fft::h2f(), f2h()", "[noa][cuda][fft]", float, cfloat_t) {
    test::Randomizer<TestType> randomizer_data(1., 128.);

    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = test::getRandomShape(ndim);
    size3_t shape_fft = shapeFFT(shape);
    size_t elements = noa::elements(shape);
    size_t elements_fft = noa::elements(shape_fft);

    cuda::Stream stream(cuda::Stream::SERIAL);
    INFO(shape);

    AND_THEN("h2f") {
        cpu::memory::PtrHost<TestType> h_half(elements_fft);
        cpu::memory::PtrHost<TestType> h_full(elements);
        test::randomize(h_half.get(), h_half.elements(), randomizer_data);
        test::memset(h_full.get(), h_full.elements(), 0);
        cpu::fft::remap(fft::H2F, h_half.get(), h_full.get(), shape, 1);

        AND_THEN("contiguous") {
            cuda::memory::PtrDevice<TestType> d_half(elements_fft);
            cuda::memory::PtrDevice<TestType> d_full(elements);
            cpu::memory::PtrHost<TestType> h_full_cuda(elements);

            cuda::memory::copy(h_half.get(), d_half.get(), h_half.size(), stream);
            cuda::fft::remap(fft::H2F, d_half.get(), shape.x / 2 + 1, d_full.get(), shape.x, shape, 1, stream);
            cuda::memory::copy(d_full.get(), h_full_cuda.get(), d_full.size(), stream);
            cuda::Stream::synchronize(stream);

            TestType diff = test::getAverageDifference(h_full.get(), h_full_cuda.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }

        AND_THEN("padded") {
            cuda::memory::PtrDevicePadded<TestType> d_half(shape_fft);
            cuda::memory::PtrDevicePadded<TestType> d_full(shape);
            cpu::memory::PtrHost<TestType> h_full_cuda(elements);

            cuda::memory::copy(h_half.get(), shape_fft.x, d_half.get(), d_half.pitch(), shape_fft, stream);
            cuda::fft::remap(fft::H2F, d_half.get(), d_half.pitch(), d_full.get(), d_full.pitch(), shape, 1, stream);
            cuda::memory::copy(d_full.get(), d_full.pitch(), h_full_cuda.get(), shape.x, shape, stream);
            cuda::Stream::synchronize(stream);

            TestType diff = test::getAverageDifference(h_full.get(), h_full_cuda.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }
    }

    AND_THEN("f2h") {
        cpu::memory::PtrHost<TestType> h_full(elements);
        cpu::memory::PtrHost<TestType> h_half(elements_fft);
        test::randomize(h_full.get(), h_full.elements(), randomizer_data);
        test::memset(h_half.get(), h_half.elements(), 0);
        cpu::fft::remap(fft::F2H, h_full.get(), h_half.get(), shape, 1);

        AND_THEN("contiguous") {
            cuda::memory::PtrDevice<TestType> d_full(elements);
            cuda::memory::PtrDevice<TestType> d_half(elements_fft);
            cpu::memory::PtrHost<TestType> h_half_cuda(elements_fft);
            cuda::memory::copy(h_full.get(), d_full.get(), h_full.size(), stream);
            cuda::fft::remap(fft::F2H, d_full.get(), shape.x, d_half.get(), shape.x / 2 + 1, shape, 1, stream);
            cuda::memory::copy(d_half.get(), h_half_cuda.get(), d_half.size(), stream);
            cuda::Stream::synchronize(stream);

            TestType diff = test::getAverageDifference(h_half.get(), h_half_cuda.get(), elements_fft);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }

        AND_THEN("padded") {
            cuda::memory::PtrDevicePadded<TestType> d_full(shape);
            cuda::memory::PtrDevicePadded<TestType> d_half(shape_fft);
            cpu::memory::PtrHost<TestType> h_half_cuda(elements_fft);

            cuda::memory::copy(h_full.get(), shape.x, d_full.get(), d_full.pitch(), shape, stream);
            cuda::fft::remap(fft::F2H, d_full.get(), d_full.pitch(), d_half.get(), d_half.pitch(), shape, 1, stream);
            cuda::memory::copy(d_half.get(), d_half.pitch(), h_half_cuda.get(), shape_fft.x, shape_fft, stream);
            cuda::Stream::synchronize(stream);

            TestType diff = test::getAverageDifference(h_half.get(), h_half_cuda.get(), elements_fft);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }
    }
}

TEMPLATE_TEST_CASE("cuda::fft::hc2f(), f2hc()", "[noa][cuda][fft]", float, cfloat_t) {
    test::Randomizer<TestType> randomizer_data(1., 128.);

    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = test::getRandomShape(ndim);
    size3_t shape_fft = shapeFFT(shape);
    size_t elements = noa::elements(shape);
    size_t elements_fft = noa::elements(shape_fft);

    cuda::Stream stream(cuda::Stream::SERIAL);
    INFO(shape);

    AND_THEN("hc2f") {
        cpu::memory::PtrHost<TestType> h_half(elements_fft);
        cpu::memory::PtrHost<TestType> h_full(elements);
        test::randomize(h_half.get(), h_half.elements(), randomizer_data);
        test::memset(h_full.get(), h_full.elements(), 0);
        cpu::fft::remap(fft::HC2F, h_half.get(), h_full.get(), shape, 1);

        AND_THEN("contiguous") {
            cuda::memory::PtrDevice<TestType> d_half(elements_fft);
            cuda::memory::PtrDevice<TestType> d_full(elements);
            cpu::memory::PtrHost<TestType> h_full_cuda(elements);

            cuda::memory::copy(h_half.get(), d_half.get(), h_half.size(), stream);
            cuda::fft::remap(fft::HC2F, d_half.get(), shape.x / 2 + 1, d_full.get(), shape.x, shape, 1, stream);
            cuda::memory::copy(d_full.get(), h_full_cuda.get(), d_full.size(), stream);
            cuda::Stream::synchronize(stream);

            TestType diff = test::getAverageDifference(h_full.get(), h_full_cuda.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }

        AND_THEN("padded") {
            cuda::memory::PtrDevicePadded<TestType> d_half(shape_fft);
            cuda::memory::PtrDevicePadded<TestType> d_full(shape);
            cpu::memory::PtrHost<TestType> h_full_cuda(elements);

            cuda::memory::copy(h_half.get(), shape_fft.x, d_half.get(), d_half.pitch(), shape_fft, stream);
            cuda::fft::remap(fft::HC2F, d_half.get(), d_half.pitch(), d_full.get(), d_full.pitch(), shape, 1, stream);
            cuda::memory::copy(d_full.get(), d_full.pitch(), h_full_cuda.get(), shape.x, shape, stream);
            cuda::Stream::synchronize(stream);

            TestType diff = test::getAverageDifference(h_full.get(), h_full_cuda.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }
    }

    AND_THEN("f2hc") {
        cpu::memory::PtrHost<TestType> h_full(elements);
        cpu::memory::PtrHost<TestType> h_half(elements_fft);
        test::randomize(h_full.get(), h_full.elements(), randomizer_data);
        test::memset(h_half.get(), h_half.elements(), 0);
        cpu::fft::remap(fft::F2HC, h_full.get(), h_half.get(), shape, 1);

        AND_THEN("contiguous") {
            cuda::memory::PtrDevice<TestType> d_full(elements);
            cuda::memory::PtrDevice<TestType> d_half(elements_fft);
            cpu::memory::PtrHost<TestType> h_half_cuda(elements_fft);
            cuda::memory::copy(h_full.get(), d_full.get(), h_full.size(), stream);
            cuda::fft::remap(fft::F2HC, d_full.get(), shape.x, d_half.get(), shape.x / 2 + 1, shape, 1, stream);
            cuda::memory::copy(d_half.get(), h_half_cuda.get(), d_half.size(), stream);
            cuda::Stream::synchronize(stream);

            TestType diff = test::getAverageDifference(h_half.get(), h_half_cuda.get(), elements_fft);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }

        AND_THEN("padded") {
            cuda::memory::PtrDevicePadded<TestType> d_full(shape);
            cuda::memory::PtrDevicePadded<TestType> d_half(shape_fft);
            cpu::memory::PtrHost<TestType> h_half_cuda(elements_fft);

            cuda::memory::copy(h_full.get(), shape.x, d_full.get(), d_full.pitch(), shape, stream);
            cuda::fft::remap(fft::F2HC, d_full.get(), d_full.pitch(), d_half.get(), d_half.pitch(), shape, 1, stream);
            cuda::memory::copy(d_half.get(), d_half.pitch(), h_half_cuda.get(), shape_fft.x, shape_fft, stream);
            cuda::Stream::synchronize(stream);

            TestType diff = test::getAverageDifference(h_half.get(), h_half_cuda.get(), elements_fft);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }
    }
}

TEMPLATE_TEST_CASE("cuda::fft::f2fc(), fc2f()", "[noa][cuda][fft]", float, cfloat_t) {
    test::Randomizer<TestType> randomizer_data(1., 128.);

    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = test::getRandomShape(ndim);
    size_t elements = noa::elements(shape);

    cuda::Stream stream(cuda::Stream::SERIAL);
    INFO(shape);

    AND_THEN("f2fc") {
        cpu::memory::PtrHost<TestType> h_full(elements);
        cpu::memory::PtrHost<TestType> h_full_centered(elements);
        test::randomize(h_full.get(), h_full.elements(), randomizer_data);
        test::memset(h_full_centered.get(), h_full_centered.elements(), 0);
        cpu::fft::remap(fft::F2FC, h_full.get(), h_full_centered.get(), shape, 1);

        AND_THEN("contiguous") {
            cuda::memory::PtrDevice<TestType> d_full(elements);
            cuda::memory::PtrDevice<TestType> d_full_centered(elements);
            cpu::memory::PtrHost<TestType> h_full_centered_cuda(elements);

            cuda::memory::copy(h_full.get(), d_full.get(), h_full.size(), stream);
            cuda::fft::remap(fft::F2FC, d_full.get(), shape.x, d_full_centered.get(), shape.x, shape, 1, stream);
            cuda::memory::copy(d_full_centered.get(), h_full_centered_cuda.get(), h_full.size(), stream);
            cuda::Stream::synchronize(stream);

            TestType diff = test::getAverageDifference(h_full_centered.get(), h_full_centered_cuda.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }

        AND_THEN("padded") {
            cuda::memory::PtrDevicePadded<TestType> d_full(shape);
            cuda::memory::PtrDevicePadded<TestType> d_full_centered(shape);
            cpu::memory::PtrHost<TestType> h_full_centered_cuda(elements);

            cuda::memory::copy(h_full.get(), shape.x,
                               d_full.get(), d_full.pitch(), shape, stream);
            cuda::fft::remap(fft::F2FC, d_full.get(), d_full.pitch(),
                             d_full_centered.get(), d_full_centered.pitch(),
                             shape, 1, stream);
            cuda::memory::copy(d_full_centered.get(), d_full_centered.pitch(),
                               h_full_centered_cuda.get(), shape.x, shape, stream);
            cuda::Stream::synchronize(stream);

            TestType diff = test::getAverageDifference(h_full_centered.get(), h_full_centered_cuda.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }
    }

    AND_THEN("f2fc") {
        cpu::memory::PtrHost<TestType> h_full_centered(elements);
        cpu::memory::PtrHost<TestType> h_full(elements);
        test::randomize(h_full_centered.get(), h_full_centered.elements(), randomizer_data);
        test::memset(h_full.get(), h_full.elements(), 0);
        cpu::fft::remap(fft::FC2F, h_full_centered.get(), h_full.get(), shape, 1);

        AND_THEN("contiguous") {
            cuda::memory::PtrDevice<TestType> d_full_centered(elements);
            cuda::memory::PtrDevice<TestType> d_full(elements);
            cpu::memory::PtrHost<TestType> h_full_cuda(elements);

            cuda::memory::copy(h_full_centered.get(), d_full_centered.get(), d_full_centered.elements(), stream);
            cuda::fft::remap(fft::FC2F, d_full_centered.get(), shape.x, d_full.get(), shape.x, shape, 1, stream);
            cuda::memory::copy(d_full.get(), h_full_cuda.get(), h_full.elements(), stream);
            cuda::Stream::synchronize(stream);

            TestType diff = test::getAverageDifference(h_full.get(), h_full_cuda.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }

        AND_THEN("padded") {
            cuda::memory::PtrDevicePadded<TestType> d_full_centered(shape);
            cuda::memory::PtrDevicePadded<TestType> d_full(shape);
            cpu::memory::PtrHost<TestType> h_full_cuda(elements);

            cuda::memory::copy(h_full_centered.get(), shape.x,
                               d_full_centered.get(), d_full_centered.pitch(), shape, stream);
            cuda::fft::remap(fft::FC2F, d_full_centered.get(), d_full_centered.pitch(),
                             d_full.get(), d_full.pitch(),
                             shape, 1, stream);
            cuda::memory::copy(d_full.get(), d_full.pitch(), h_full_cuda.get(), shape.x, shape, stream);
            cuda::Stream::synchronize(stream);

            TestType diff = test::getAverageDifference(h_full.get(), h_full_cuda.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }
    }
}

TEMPLATE_TEST_CASE("cuda::fft::h2hc(), hc2h()", "[noa][cuda][fft]", float, cfloat_t) {
    test::Randomizer<TestType> randomizer_data(1., 128.);

    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = test::getRandomShape(ndim);
    size3_t shape_fft = shapeFFT(shape);
    size_t elements_fft = noa::elements(shape_fft);

    cuda::Stream stream(cuda::Stream::SERIAL);
    INFO(shape);

    AND_THEN("h2hc") {
        cpu::memory::PtrHost<TestType> h_half(elements_fft);
        cpu::memory::PtrHost<TestType> h_half_centered(elements_fft);
        test::randomize(h_half.get(), h_half.elements(), randomizer_data);
        test::memset(h_half_centered.get(), h_half_centered.elements(), 0);
        cpu::fft::remap(fft::H2HC, h_half.get(), h_half_centered.get(), shape, 1);

        AND_THEN("contiguous") {
            cuda::memory::PtrDevice<TestType> d_half(elements_fft);
            cuda::memory::PtrDevice<TestType> d_half_centered(elements_fft);
            cpu::memory::PtrHost<TestType> h_half_centered_cuda(elements_fft);

            cuda::memory::copy(h_half.get(), d_half.get(), h_half.size(), stream);
            cuda::fft::remap(fft::H2HC, d_half.get(), shape.x / 2 + 1,
                             d_half_centered.get(), shape.x / 2 + 1,
                             shape, 1, stream);
            cuda::memory::copy(d_half_centered.get(), h_half_centered_cuda.get(), h_half.size(), stream);
            cuda::Stream::synchronize(stream);

            TestType diff = test::getAverageDifference(h_half_centered.get(), h_half_centered_cuda.get(), elements_fft);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }

        AND_THEN("padded") {
            cuda::memory::PtrDevicePadded<TestType> d_half(shape_fft);
            cuda::memory::PtrDevicePadded<TestType> d_half_centered(shape_fft);
            cpu::memory::PtrHost<TestType> h_half_centered_cuda(elements_fft);

            cuda::memory::copy(h_half.get(), shape_fft.x,
                               d_half.get(), d_half.pitch(),
                               shape_fft, stream);
            cuda::fft::remap(fft::H2HC, d_half.get(), d_half.pitch(),
                             d_half_centered.get(), d_half_centered.pitch(),
                             shape, 1, stream);
            cuda::memory::copy(d_half_centered.get(), d_half_centered.pitch(),
                               h_half_centered_cuda.get(), shape_fft.x,
                               shape_fft, stream);
            cuda::Stream::synchronize(stream);

            TestType diff = test::getAverageDifference(h_half_centered.get(), h_half_centered_cuda.get(), elements_fft);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }
    }

    AND_THEN("hc2h") {
        cpu::memory::PtrHost<TestType> h_half_centered(elements_fft);
        cpu::memory::PtrHost<TestType> h_half(elements_fft);
        test::randomize(h_half_centered.get(), h_half_centered.elements(), randomizer_data);
        test::memset(h_half.get(), h_half.elements(), 0);
        cpu::fft::remap(fft::HC2H, h_half_centered.get(), h_half.get(), shape, 1);

        AND_THEN("contiguous") {
            cuda::memory::PtrDevice<TestType> d_half_centered(elements_fft);
            cuda::memory::PtrDevice<TestType> d_half(elements_fft);
            cpu::memory::PtrHost<TestType> h_half_cuda(elements_fft);

            cuda::memory::copy(h_half_centered.get(), d_half_centered.get(), h_half.size(), stream);
            cuda::fft::remap(fft::HC2H, d_half_centered.get(), shape.x / 2 + 1,
                             d_half.get(), shape.x / 2 + 1,
                             shape, 1, stream);
            cuda::memory::copy(d_half.get(), h_half_cuda.get(), h_half.size(), stream);
            cuda::Stream::synchronize(stream);

            TestType diff = test::getAverageDifference(h_half.get(), h_half_cuda.get(), elements_fft);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }

        AND_THEN("padded") {
            cuda::memory::PtrDevicePadded<TestType> d_half_centered(shape_fft);
            cuda::memory::PtrDevicePadded<TestType> d_half(shape_fft);
            cpu::memory::PtrHost<TestType> h_half_cuda(elements_fft);

            cuda::memory::copy(h_half_centered.get(), shape_fft.x,
                               d_half_centered.get(), d_half_centered.pitch(),
                               shape_fft, stream);
            cuda::fft::remap(fft::HC2H, d_half_centered.get(), d_half_centered.pitch(),
                             d_half.get(), d_half.pitch(),
                             shape, 1, stream);
            cuda::memory::copy(d_half.get(), d_half.pitch(),
                               h_half_cuda.get(), shape_fft.x,
                               shape_fft, stream);
            cuda::Stream::synchronize(stream);

            TestType diff = test::getAverageDifference(h_half.get(), h_half_cuda.get(), elements_fft);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }
    }
}

TEMPLATE_TEST_CASE("cuda::fft::h2hc(), in-place", "[noa][cuda][fft]", float, cfloat_t) {
    test::Randomizer<TestType> randomizer_data(1., 128.);

    uint ndim = GENERATE(1U, 2U, 3U);
    size_t batches = 2;
    size3_t shape = test::getRandomShape(ndim, true);
    size3_t shape_fft = shapeFFT(shape);
    size_t elements_fft = noa::elements(shape_fft);

    cuda::Stream stream(cuda::Stream::SERIAL);
    INFO(shape);

    cpu::memory::PtrHost<TestType> h_half(elements_fft * batches);
    cpu::memory::PtrHost<TestType> h_half_centered(h_half.size());
    test::randomize(h_half.get(), h_half.size(), randomizer_data);
    cpu::fft::remap(fft::H2HC, h_half.get(), h_half_centered.get(), shape, batches);

    cuda::memory::PtrDevicePadded<TestType> d_half({shape_fft.x, shape_fft.y * shape_fft.z, batches});
    cpu::memory::PtrHost<TestType> h_half_centered_cuda(h_half.size());

    cuda::memory::copy(h_half.get(), shape_fft.x,
                       d_half.get(), d_half.pitch(),
                       shape_fft, batches, stream);
    cuda::fft::remap(fft::H2HC, d_half.get(), d_half.pitch(),
                     d_half.get(), d_half.pitch(),
                     shape, batches, stream);
    cuda::memory::copy(d_half.get(), d_half.pitch(),
                       h_half_centered_cuda.get(), shape_fft.x,
                       shape_fft, stream);
    cuda::Stream::synchronize(stream);

    TestType diff = test::getAverageDifference(h_half_centered.get(), h_half_centered_cuda.get(), elements_fft);
    REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
}

TEMPLATE_TEST_CASE("cuda::fft::fc2h()", "[noa][cuda][fft]", float, cfloat_t) {
    test::Randomizer<TestType> randomizer_data(1., 128.);

    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = test::getRandomShape(ndim);
    size3_t shape_fft = shapeFFT(shape);
    size_t elements = noa::elements(shape);
    size_t elements_fft = noa::elements(shape_fft);

    cuda::Stream stream(cuda::Stream::SERIAL);
    INFO(shape);

    AND_THEN("fc2h") {
        cpu::memory::PtrHost<TestType> h_full_centered(elements);
        cpu::memory::PtrHost<TestType> h_half(elements_fft);
        test::randomize(h_full_centered.get(), h_full_centered.elements(), randomizer_data);
        test::memset(h_half.get(), h_half.elements(), 0);
        cpu::fft::remap(fft::FC2H, h_full_centered.get(), h_half.get(), shape, 1);

        AND_THEN("contiguous") {
            cuda::memory::PtrDevice<TestType> d_full_centered(elements);
            cuda::memory::PtrDevice<TestType> d_half(elements_fft);
            cpu::memory::PtrHost<TestType> h_half_cuda(elements_fft);

            cuda::memory::copy(h_full_centered.get(), d_full_centered.get(), h_full_centered.size(), stream);
            cuda::fft::remap(fft::FC2H, d_full_centered.get(), shape.x,
                             d_half.get(), shape.x / 2 + 1,
                             shape, 1, stream);
            cuda::memory::copy(d_half.get(), h_half_cuda.get(), h_half.size(), stream);
            cuda::Stream::synchronize(stream);

            TestType diff = test::getAverageDifference(h_half.get(), h_half_cuda.get(), elements_fft);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }

        AND_THEN("padded") {
            cuda::memory::PtrDevicePadded<TestType> d_full_centered(shape);
            cuda::memory::PtrDevicePadded<TestType> d_half(shape_fft);
            cpu::memory::PtrHost<TestType> h_half_cuda(elements_fft);

            cuda::memory::copy(h_full_centered.get(), shape.x,
                               d_full_centered.get(), d_full_centered.pitch(),
                               shape, stream);
            cuda::fft::remap(fft::FC2H, d_full_centered.get(), d_full_centered.pitch(),
                             d_half.get(), d_half.pitch(),
                             shape, 1, stream);
            cuda::memory::copy(d_half.get(), d_half.pitch(),
                               h_half_cuda.get(), shape_fft.x,
                               shape_fft, stream);
            cuda::Stream::synchronize(stream);

            TestType diff = test::getAverageDifference(h_half.get(), h_half_cuda.get(), elements_fft);
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-14));
        }
    }
}
