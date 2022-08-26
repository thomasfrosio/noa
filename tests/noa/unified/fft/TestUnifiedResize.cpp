#include <noa/IO.h>
#include <noa/FFT.h>

#include <catch2/catch.hpp>
#include "Helpers.h"
using namespace ::noa;

TEST_CASE("unified::fft::resize, crop", "[.]") {
    using real_t = float;
    using complex_t = Complex<real_t>;

    const Device cpu{};
    StreamGuard stream(cpu);
    ArrayOption options{cpu, Allocator::DEFAULT};

    Array src = io::load<real_t>(test::NOA_DATA_PATH / "signal" / "fft" / "tilt1_slice21.mrc", false, options);
    Array src_fft = fft::r2c(src.release());

    const size_t reduce_factor = 6;
    const size4_t shape = src.shape();
    const size4_t shape_bin{shape[0], shape[1], shape[2] / reduce_factor, shape[3] / reduce_factor};

    Array<complex_t> dst_fft(shape_bin.fft(), options);
    fft::resize<fft::H2H>(src_fft.release(), shape, dst_fft, shape_bin);

    Array dst = fft::c2r(dst_fft.release(), shape_bin);
    io::save(dst, test::NOA_DATA_PATH / "signal" / "fft" / "tilt1_slice21_cropped2.mrc");
}

TEST_CASE("unified::fft::resize, crop entire stack", "[.]") {
    using real_t = float;
    using complex_t = Complex<real_t>;

    const Device cpu{};
    StreamGuard stream(cpu);
    ArrayOption options{cpu, Allocator::DEFAULT};

    const size_t reduce_factor = 6;

    io::ImageFile file(test::NOA_DATA_PATH / "tilt1.mrc", io::READ);
    const size4_t shape = file.shape();
    float3_t pixel_size = file.pixelSize();
    pixel_size[1] *= reduce_factor;
    pixel_size[2] *= reduce_factor;

    Array src = file.read<real_t>(options);
    Array src_fft = fft::r2c(src.release());

    const size4_t shape_binned{shape[0], shape[1], shape[2] / reduce_factor, shape[3] / reduce_factor};
    Array<complex_t> dst_fft{shape_binned.fft(), options};
    fft::resize<fft::H2H>(src_fft.release(), shape, dst_fft, shape_binned);

    Array dst = fft::c2r(dst_fft.release(), shape_binned);

    file.open(test::NOA_DATA_PATH / "tilt1_cropped.mrc", io::WRITE);
    file.pixelSize(pixel_size);
    file.write(dst);
}
