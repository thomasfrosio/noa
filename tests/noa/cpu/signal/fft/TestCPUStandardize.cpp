#include <noa/cpu/fft/Remap.h>
#include <noa/cpu/fft/Transforms.h>
#include <noa/cpu/math/Ewise.h>
#include <noa/cpu/math/Random.h>
#include <noa/cpu/math/Reduce.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/signal/fft/Standardize.h>

#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cpu::signal::fft::standardize(), half", "[noa][cpu]") {
    const size4_t shape{1, 1, 128, 128};
    const size4_t stride = shape.stride();
    const size4_t stride_fft = shape.fft().stride();
    const size_t elements = shape.elements();

    fft::Norm norm = GENERATE(fft::NORM_FORWARD, fft::NORM_BACKWARD, fft::NORM_ORTHO, fft::NORM_NONE);

    cpu::Stream stream(cpu::Stream::DEFAULT);
    cpu::memory::PtrHost<float> image(elements);
    cpu::memory::PtrHost<cfloat_t> image_fft(shape.fft().elements());

    cpu::math::randomize(math::normal_t{}, image.share(), image.elements(), 2.4f, 4.1f, stream);
    cpu::fft::r2c(image.share(), image_fft.share(), shape, cpu::fft::ESTIMATE, norm, stream);

    cpu::memory::PtrHost<cfloat_t> image_fft_centered(shape.fft().elements());
    cpu::fft::remap(fft::H2HC,
                    image_fft.share(), stride_fft,
                    image_fft_centered.share(), stride_fft, shape, stream);

    cpu::signal::fft::standardize<fft::HC2HC>(image_fft_centered.share(), stride_fft,
                                              image_fft_centered.share(), stride_fft,
                                              shape, norm, stream);

    cpu::fft::remap(fft::HC2H,
                    image_fft_centered.share(), stride_fft,
                    image_fft.share(), stride_fft, shape, stream);
    cpu::fft::c2r(image_fft.share(), image.share(), shape, cpu::fft::ESTIMATE, norm, stream);
    if (norm == fft::NORM_NONE)
        cpu::math::ewise(image.share(), stride, 1 / static_cast<float>(elements),
                         image.share(), stride, shape, math::multiply_t{}, stream);

    const float mean = cpu::math::mean(image.share(), stride, shape, stream);
    const float std = cpu::math::std(image.share(), stride, shape, stream);
    REQUIRE_THAT(mean, Catch::WithinAbs(0, 1e-6));
    REQUIRE_THAT(std, Catch::WithinAbs(1, 1e-5));
}

TEST_CASE("cpu::signal::fft::standardize(), full", "[noa][cpu]") {
    const size4_t shape{1, 1, 128, 128};
    const size4_t stride = shape.stride();
    const size4_t stride_fft = shape.fft().stride();
    const size_t elements = shape.elements();

    fft::Norm norm = GENERATE(fft::NORM_FORWARD, fft::NORM_BACKWARD, fft::NORM_ORTHO, fft::NORM_NONE);

    cpu::Stream stream(cpu::Stream::DEFAULT);
    cpu::memory::PtrHost<float> image(elements);
    cpu::memory::PtrHost<cfloat_t> image_fft(shape.fft().elements());

    cpu::math::randomize(math::normal_t{}, image.share(), image.elements(), 2.4f, 4.1f, stream);
    cpu::fft::r2c(image.share(), image_fft.share(), shape, cpu::fft::ESTIMATE, norm, stream);

    cpu::memory::PtrHost<cfloat_t> image_fft_centered(shape.elements());
    cpu::fft::remap(fft::H2FC,
                    image_fft.share(), stride_fft,
                    image_fft_centered.share(), stride, shape, stream);

    cpu::signal::fft::standardize<fft::FC2FC>(image_fft_centered.share(), stride,
                                              image_fft_centered.share(), stride,
                                              shape, norm, stream);

    cpu::fft::remap(fft::FC2H,
                    image_fft_centered.share(), stride,
                    image_fft.share(), stride_fft, shape, stream);
    cpu::fft::c2r(image_fft.share(), image.share(), shape, cpu::fft::ESTIMATE, norm, stream);
    if (norm == fft::NORM_NONE)
        cpu::math::ewise(image.share(), stride, 1 / static_cast<float>(elements),
                         image.share(), stride, shape, math::multiply_t{}, stream);

    const float mean = cpu::math::mean(image.share(), stride, shape, stream);
    const float std = cpu::math::std(image.share(), stride, shape, stream);
    REQUIRE_THAT(mean, Catch::WithinAbs(0, 1e-6));
    REQUIRE_THAT(std, Catch::WithinAbs(1, 1e-5));
}
