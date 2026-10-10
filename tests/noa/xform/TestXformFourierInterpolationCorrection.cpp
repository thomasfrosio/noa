#include <noa/Xform.hpp>
#include <noa/Runtime.hpp>
#include <noa/FFT.hpp>
#include <noa/Signal.hpp>
#include <noa/IO.hpp>

#include "Catch.hpp"

TEST_CASE("xform:: correct Fourier interpolation", "[.]") {
    using namespace noa::types;
    namespace nx = noa::xform;
    namespace nf = noa::fft;
    namespace ns = noa::signal;

    auto input = noa::random<f64, 2>(noa::Uniform(1.4, 1.5), Shape2{512, 512});
    nx::draw_2d(input, input, nx::Rectangle{.center = Vec{256., 256.,}, .radius = Vec{100., 100.}, .smoothness = 20.}.get());
    noa::write_image(input, "~/tmp/noa/input.mrc");

    const auto interp = nx::Interp::LANCZOS4;
    // nx::fourier_interpolation_correction_2d(input, input, {.interp = interp});

    auto input_rfft = nf::rfft2(input);
    auto output_rfft = noa::empty_like(input_rfft);
    auto center = (input.shape().vec / 2).as<f64>();
    ns::phase_shift_2d<"h">(input_rfft, input_rfft, input.shape(), -center);
    nx::transform_spectrum_2d<"h">(
        input_rfft, output_rfft, input.shape(), nx::rotate(noa::deg2rad(45.)), {}, {.interp = interp});
    ns::phase_shift_2d<"h">(output_rfft, output_rfft, input.shape(), center);
    auto output = nf::irfft2(output_rfft, input.shape());
    noa::write_image(output, "~/tmp/noa/output_4.mrc");

    // nx::fourier_interpolation_correction_2d(output, output, {.interp = interp});
    // noa::write_image(output, "~/tmp/noa/output.mrc");

    auto ra = noa::empty<f32, 1>(257);
    nx::rotational_average<"fc2h">(output, output.shape(), ra, {});
    noa::write_image(ra, "~/tmp/noa/ra.mrc");
}
