#include <noa/common/io/ImageFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/fft/Remap.h>

#include "Helpers.h"
#include "Assets.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("cpu::fft::fc2f(), f2fc()", "[noa][cpu][fft]",
                   half_t, float, double, cfloat_t, cdouble_t, chalf_t) {
    test::Randomizer<size_t> randomizer(1, 128);
    test::Randomizer<TestType> randomizer_data(1., 128.);

    const uint ndim = GENERATE(1U, 2U, 3U);
    cpu::Stream stream;

    AND_THEN("fc > f > fc") {
        const size4_t shape = test::getRandomShapeBatched(ndim);
        const size4_t stride = shape.stride();
        const size_t elements = shape.elements();
        cpu::memory::PtrHost<TestType> full_centered_in(elements);
        cpu::memory::PtrHost<TestType> full_centered_out(elements);
        cpu::memory::PtrHost<TestType> full(elements);

        test::randomize(full_centered_in.get(), full_centered_in.size(), randomizer_data);
        test::memset(full_centered_out.get(), full_centered_out.size(), 0);
        test::memset(full.get(), full.size(), 0);

        cpu::fft::remap<TestType>(fft::FC2F, full_centered_in.share(), stride, full.share(), stride, shape, stream);
        cpu::fft::remap<TestType>(fft::F2FC, full.share(), stride, full_centered_out.share(), stride, shape, stream);
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, full_centered_in.get(), full_centered_out.get(), elements, 1e-10));
    }

    AND_THEN("f > fc > f") {
        const size4_t shape = test::getRandomShapeBatched(ndim);
        const size4_t stride = shape.stride();
        const size_t elements = shape.elements();
        cpu::memory::PtrHost<TestType> full_in(elements);
        cpu::memory::PtrHost<TestType> full_out(elements);
        cpu::memory::PtrHost<TestType> full_centered(elements);

        test::randomize(full_in.get(), full_in.size(), randomizer_data);
        test::memset(full_out.get(), full_out.size(), 0);
        test::memset(full_centered.get(), full_centered.size(), 0);

        cpu::fft::remap<TestType>(fft::F2FC, full_in.share(), stride, full_centered.share(), stride, shape, stream);
        cpu::fft::remap<TestType>(fft::FC2F, full_centered.share(), stride, full_out.share(), stride, shape, stream);
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, full_in.get(), full_out.get(), elements, 1e-10));
    }
}

TEST_CASE("cpu::fft::fc2f(), f2fc() -- vs numpy", "[assets][noa][cpu][fft]") {
    const fs::path path = test::NOA_DATA_PATH / "fft";
    YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["remap"];
    io::ImageFile file;
    cpu::Stream stream;

    AND_THEN("2D") {
        file.open(path / tests["2D"]["input"].as<path_t>(), io::READ);
        const size4_t shape{file.shape()};
        const size4_t stride{shape.stride()};
        const size_t size{shape.elements()};
        cpu::memory::PtrHost<float> array(size);
        file.readAll(array.get());

        cpu::memory::PtrHost<float> reordered_expected(size);
        cpu::memory::PtrHost<float> reordered_results(size);

        // fftshift
        file.open(path / tests["2D"]["fftshift"].as<path_t>(), io::READ);
        file.readAll(reordered_expected.get());

        cpu::fft::remap<float>(fft::F2FC, array.share(), stride, reordered_results.share(), stride, shape, stream);
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, reordered_expected.get(), reordered_results.get(), size, 1e-10));

        // ifftshift
        test::memset(reordered_expected.get(), size, 0);
        file.open(path / tests["2D"]["ifftshift"].as<path_t>(), io::READ);
        file.readAll(reordered_expected.get());

        cpu::fft::remap<float>(fft::FC2F, array.share(), stride, reordered_results.share(), stride, shape, stream);
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, reordered_expected.get(), reordered_results.get(), size, 1e-10));
    }

    AND_THEN("3D") {
        file.open(path / tests["3D"]["input"].as<path_t>(), io::READ);
        const size4_t shape{file.shape()};
        const size4_t stride{shape.stride()};
        const size_t size{shape.elements()};
        cpu::memory::PtrHost<float> array(size);
        file.readAll(array.get());

        cpu::memory::PtrHost<float> reordered_expected(size);
        cpu::memory::PtrHost<float> reordered_results(size);

        // fftshift
        file.open(path / tests["3D"]["fftshift"].as<path_t>(), io::READ);
        file.readAll(reordered_expected.get());

        cpu::fft::remap<float>(fft::F2FC, array.share(), stride, reordered_results.share(), stride, shape, stream);
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, reordered_expected.get(), reordered_results.get(), size, 1e-10));

        // ifftshift
        test::memset(reordered_expected.get(), size, 0);
        file.open(path / tests["3D"]["ifftshift"].as<path_t>(), io::READ);
        file.readAll(reordered_expected.get());

        cpu::fft::remap<float>(fft::FC2F, array.share(), stride, reordered_results.share(), stride, shape, stream);
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, reordered_expected.get(), reordered_results.get(), size, 1e-10));
    }
}

TEMPLATE_TEST_CASE("cpu::fft::hc2h(), h2hc()", "[noa][cpu][fft]",
                   half_t, float, double, cfloat_t, cdouble_t, chalf_t) {
    test::Randomizer<size_t> randomizer(1, 128);
    test::Randomizer<TestType> randomizer_data(-128., 128.);
    cpu::Stream stream;
    const uint ndim = GENERATE(1U, 2U, 3U);

    AND_THEN("hc > h > hc") {
        const size4_t shape = test::getRandomShapeBatched(ndim);
        const size4_t stride = shape.fft().stride();
        const size_t elements = shape.fft().elements();
        cpu::memory::PtrHost<TestType> half_centered_in(elements);
        cpu::memory::PtrHost<TestType> half_centered_out(elements);
        cpu::memory::PtrHost<TestType> half(elements);

        test::randomize(half_centered_in.get(), half_centered_in.size(), randomizer_data);
        test::memset(half.get(), half.size(), 0);
        test::memset(half_centered_out.get(), half_centered_out.size(), 0);

        cpu::fft::remap<TestType>(fft::HC2H, half_centered_in.share(), stride, half.share(), stride, shape, stream);
        cpu::fft::remap<TestType>(fft::H2HC, half.share(), stride, half_centered_out.share(), stride, shape, stream);
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, half_centered_in.get(), half_centered_out.get(), elements, 1e-10));
    }

    AND_THEN("h > hc > h") {
        const size4_t shape = test::getRandomShapeBatched(ndim);
        const size4_t stride = shape.fft().stride();
        const size_t elements = shape.fft().elements();
        cpu::memory::PtrHost<TestType> half_in(elements);
        cpu::memory::PtrHost<TestType> half_out(elements);
        cpu::memory::PtrHost<TestType> half_centered(elements);

        test::randomize(half_in.get(), half_in.size(), randomizer_data);
        test::memset(half_centered.get(), half_centered.size(), 0);
        test::memset(half_out.get(), half_out.size(), 0);

        cpu::fft::remap<TestType>(fft::H2HC, half_in.share(), stride, half_centered.share(), stride, shape, stream);
        cpu::fft::remap<TestType>(fft::HC2H, half_centered.share(), stride, half_out.share(), stride, shape, stream);
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, half_in.get(), half_out.get(), elements, 1e-10));
    }

    AND_THEN("in-place") {
        const size4_t shape = test::getRandomShapeBatched(ndim, true); // even
        const size4_t stride = shape.fft().stride();
        const size_t elements = shape.fft().elements();
        cpu::memory::PtrHost<TestType> half_in(elements);
        cpu::memory::PtrHost<TestType> half_out(elements);
        cpu::memory::PtrHost<TestType> half_centered(elements);

        test::randomize(half_in.get(), half_in.elements(), randomizer_data);

        cpu::fft::remap<TestType>(fft::H2HC, half_in.share(), stride, half_out.share(), stride, shape, stream);
        cpu::fft::remap<TestType>(fft::H2HC, half_in.share(), stride, half_in.share(), stride, shape, stream);
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, half_in.get(), half_out.get(), half_in.size(), 1e-10));
    }
}

TEMPLATE_TEST_CASE("cpu::fft::h2f(), f2h()", "[noa][cpu][fft]",
                   half_t, float, double, cfloat_t, cdouble_t, chalf_t) {
    test::Randomizer<TestType> randomizer_data(1., 128.);
    cpu::Stream stream;

    const uint ndim = GENERATE(1U, 2U, 3U);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t stride = shape.stride();
    const size4_t stride_fft = shape.fft().stride();
    const size_t elements = shape.elements();
    const size_t elements_fft = shape.fft().elements();

    AND_THEN("h > f > h") {
        cpu::memory::PtrHost<TestType> half_in(elements_fft);
        cpu::memory::PtrHost<TestType> half_out(elements_fft);
        cpu::memory::PtrHost<TestType> full(elements);

        test::randomize(half_in.get(), half_in.size(), randomizer_data);
        test::memset(half_out.get(), half_out.size(), 0);

        cpu::fft::remap<TestType>(fft::H2F, half_in.share(), stride_fft, full.share(), stride, shape, stream);
        cpu::fft::remap<TestType>(fft::F2H, full.share(), stride, half_out.share(), stride_fft, shape, stream);
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, half_in.get(), half_out.get(), half_in.elements(), 1e-10));
    }
}

TEMPLATE_TEST_CASE("cpu::fft::hc2f(), f2hc()", "[noa][cpu][fft]",
                   half_t, float, double, cfloat_t, cdouble_t, chalf_t) {
    test::Randomizer<TestType> randomizer_data(-128., 128.);
    cpu::Stream stream;
    const uint ndim = GENERATE(1U, 2U, 3U);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t stride = shape.stride();
    const size4_t stride_fft = shape.fft().stride();
    const size_t elements = shape.elements();
    const size_t elements_fft = shape.fft().elements();

    cpu::memory::PtrHost<TestType> half(elements_fft);
    cpu::memory::PtrHost<TestType> half_centered(elements_fft);
    cpu::memory::PtrHost<TestType> half_2(elements_fft);
    cpu::memory::PtrHost<TestType> full(elements);
    cpu::memory::PtrHost<TestType> full_2(elements);

    AND_THEN("hc > f") {
        test::randomize(half.get(), half.elements(), randomizer_data);
        cpu::fft::remap<TestType>(fft::H2HC, half.share(), stride_fft, half_centered.share(), stride_fft, shape, stream);
        cpu::fft::remap<TestType>(fft::HC2F, half_centered.share(), stride_fft, full.share(), stride, shape, stream);
        cpu::fft::remap<TestType>(fft::H2F, half.share(), stride_fft, full_2.share(), stride, shape, stream);
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, full.get(), full_2.get(), full_2.size(), 1e-10));
    }

    AND_THEN("f > hc") {
        test::randomize(full.get(), full.elements(), randomizer_data);
        cpu::fft::remap<TestType>(fft::F2H, full.share(), stride, half.share(), stride_fft, shape, stream);
        cpu::fft::remap<TestType>(fft::H2HC, half.share(), stride_fft, half_centered.share(), stride_fft, shape, stream);
        cpu::fft::remap<TestType>(fft::F2HC, full.share(), stride, half_2.share(), stride_fft, shape, stream);
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, half_centered.get(), half_2.get(), half.size(), 1e-10));
    }
}

TEMPLATE_TEST_CASE("cpu::fft::fc2h()", "[noa][cpu][fft]",
                   half_t, float, double, cfloat_t, cdouble_t, chalf_t) {
    test::Randomizer<TestType> randomizer_data(1., 128.);
    cpu::Stream stream;
    const uint ndim = GENERATE(1U, 2U, 3U);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t stride = shape.stride();
    const size4_t stride_fft = shape.fft().stride();
    const size_t elements = shape.elements();
    const size_t elements_fft = shape.fft().elements();

    AND_THEN("h > f > fc > h") { // h2fc is not supported so we have to fftshift separately.
        AND_THEN("complex") {
            cpu::memory::PtrHost<TestType> half_in(elements_fft);
            cpu::memory::PtrHost<TestType> half_out(elements_fft);
            cpu::memory::PtrHost<TestType> full(elements);
            cpu::memory::PtrHost<TestType> full_centered(elements);

            test::randomize(half_in.get(), half_in.elements(), randomizer_data);
            test::memset(half_out.get(), half_out.elements(), 0);

            cpu::fft::remap<TestType>(fft::H2F, half_in.share(), stride_fft, full.share(), stride, shape, stream);
            cpu::fft::remap<TestType>(fft::F2FC, full.share(), stride, full_centered.share(), stride, shape, stream);
            cpu::fft::remap<TestType>(fft::FC2H, full_centered.share(), stride, half_out.share(), stride_fft, shape, stream);
            stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS, half_in.get(), half_out.get(), elements_fft, 1e-10));
        }

        AND_THEN("real") {
            cpu::memory::PtrHost<TestType> half_in(elements_fft);
            cpu::memory::PtrHost<TestType> half_out(elements_fft);
            cpu::memory::PtrHost<TestType> full(elements);
            cpu::memory::PtrHost<TestType> full_centered(elements);

            test::randomize(half_in.get(), half_in.elements(), randomizer_data);
            test::memset(half_out.get(), half_out.elements(), 0);

            cpu::fft::remap<TestType>(fft::H2F, half_in.share(), stride_fft, full.share(), stride, shape, stream);
            cpu::fft::remap<TestType>(fft::F2FC, full.share(), stride, full_centered.share(), stride, shape, stream);
            cpu::fft::remap<TestType>(fft::FC2H, full_centered.share(), stride, half_out.share(), stride_fft, shape, stream);
            stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS, half_in.get(), half_out.get(), elements_fft, 1e-10));
        }
    }
}
