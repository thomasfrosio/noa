#include <noa/gpu/cuda/fft/Remap.h>
#include <noa/gpu/cuda/fft/Transforms.h>
#include <noa/gpu/cuda/math/Ewise.h>
#include <noa/gpu/cuda/math/Random.h>
#include <noa/gpu/cuda/math/Reduce.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/signal/fft/Standardize.h>

#include <catch2/catch.hpp>

using namespace ::noa;

TEMPLATE_TEST_CASE("cuda::signal::fft::standardize(), half", "[noa][cuda]", float, double) {
    const size4_t shape{1, 1, 128, 128};
    const size4_t stride = shape.stride();
    const size4_t stride_fft = shape.fft().stride();
    const size_t elements = shape.elements();

    fft::Norm norm = GENERATE(fft::NORM_FORWARD, fft::NORM_BACKWARD, fft::NORM_ORTHO, fft::NORM_NONE);

    using real_t = TestType;
    using complex_t = Complex<real_t>;

    cuda::Stream stream(cuda::Stream::DEFAULT);
    cuda::memory::PtrDevice<real_t> image(elements, stream);
    cuda::memory::PtrDevice<complex_t> image_fft(shape.fft().elements(), stream);

    cuda::math::randomize(math::normal_t{}, image.share(), image.elements(),
                          static_cast<real_t>(2.4), static_cast<real_t>(4.1), stream);
    cuda::fft::r2c(image.share(), image_fft.share(), shape, norm, stream);

    cuda::memory::PtrDevice<complex_t> image_fft_centered(shape.fft().elements(), stream);
    cuda::fft::remap(fft::H2HC,
                    image_fft.share(), stride_fft,
                    image_fft_centered.share(), stride_fft, shape, stream);

    cuda::signal::fft::standardize<fft::HC2HC>(image_fft_centered.share(), stride_fft,
                                               image_fft_centered.share(), stride_fft,
                                               shape, norm, stream);

    cuda::fft::remap(fft::HC2H,
                    image_fft_centered.share(), stride_fft,
                    image_fft.share(), stride_fft, shape, stream);
    cuda::fft::c2r(image_fft.share(), image.share(), shape, norm, stream);
    if (norm == fft::NORM_NONE)
        cuda::math::ewise(image.share(), stride, 1 / static_cast<real_t>(elements),
                          image.share(), stride, shape, math::multiply_t{}, stream);

    const real_t mean = cuda::math::mean(image.share(), stride, shape, stream);
    const real_t std = cuda::math::std(image.share(), stride, shape, stream);
    REQUIRE_THAT(mean, Catch::WithinAbs(0, 1e-6));
    REQUIRE_THAT(std, Catch::WithinAbs(1, 1e-5));
}

TEMPLATE_TEST_CASE("cuda::signal::fft::standardize(), full", "[noa][cuda]", float, double) {
    const size4_t shape{1, 1, 128, 128};
    const size4_t stride = shape.stride();
    const size4_t stride_fft = shape.fft().stride();
    const size_t elements = shape.elements();

    fft::Norm norm = GENERATE(fft::NORM_FORWARD, fft::NORM_BACKWARD, fft::NORM_ORTHO, fft::NORM_NONE);

    using real_t = TestType;
    using complex_t = Complex<real_t>;

    cuda::Stream stream(cuda::Stream::DEFAULT);
    cuda::memory::PtrDevice<real_t> image(elements);
    cuda::memory::PtrDevice<complex_t> image_fft(shape.fft().elements());

    cuda::math::randomize(math::normal_t{}, image.share(), image.elements(),
                          static_cast<real_t>(2.4), static_cast<real_t>(4.1), stream);
    cuda::fft::r2c(image.share(), image_fft.share(), shape, norm, stream);

    cuda::memory::PtrDevice<complex_t> image_fft_centered(shape.elements());
    cuda::fft::remap(fft::H2FC,
                    image_fft.share(), stride_fft,
                    image_fft_centered.share(), stride, shape, stream);

    cuda::signal::fft::standardize<fft::FC2FC>(image_fft_centered.share(), stride,
                                               image_fft_centered.share(), stride,
                                               shape, norm, stream);

    cuda::fft::remap(fft::FC2H,
                    image_fft_centered.share(), stride,
                    image_fft.share(), stride_fft, shape, stream);
    cuda::fft::c2r(image_fft.share(), image.share(), shape, norm, stream);
    if (norm == fft::NORM_NONE)
        cuda::math::ewise(image.share(), stride, 1 / static_cast<real_t>(elements),
                          image.share(), stride, shape, math::multiply_t{}, stream);

    const real_t mean = cuda::math::mean(image.share(), stride, shape, stream);
    const real_t std = cuda::math::std(image.share(), stride, shape, stream);
    REQUIRE_THAT(mean, Catch::WithinAbs(0, 1e-6));
    REQUIRE_THAT(std, Catch::WithinAbs(1, 1e-5));
}