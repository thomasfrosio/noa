#include <noa/common/io/ImageFile.h>
#include <noa/unified/FFT.h>

#include <catch2/catch.hpp>
#include "Helpers.h"
using namespace ::noa;

TEST_CASE("unified::fft::resize, crop", "[.]") {
    using real_t = float;
    using complex_t = Complex<real_t>;

    const Device cpu{};
    StreamGuard stream{cpu};
    ArrayOption options{cpu, Allocator::DEFAULT};

    io::ImageFile file{test::NOA_DATA_PATH / "signal" / "fft" / "tilt1_slice21.mrc", io::READ};
    const size4_t shape = file.shape();

    Array<real_t> src{shape, options};
    Array<complex_t> src_fft{shape.fft(), options};
    file.readAll(src.get());

    fft::r2c(src, src_fft);
    src.release();

    const size4_t shape_bin2{shape[0], shape[1], shape[2] / 7, shape[3] / 7};
    Array<complex_t> dst_fft{shape_bin2.fft(), options};
    fft::resize<fft::H2H>(src_fft, shape, dst_fft, shape_bin2);
    src_fft.release();

    Array<real_t> dst{shape_bin2, options};
    fft::c2r(dst_fft, dst);
    dst_fft.release();

    file.open(test::NOA_DATA_PATH / "signal" / "fft" / "tilt1_slice21_cropped2.mrc", io::WRITE);
    file.shape(dst.shape());
    file.writeAll(dst.eval().get());
}

TEST_CASE("unified::fft::resize, crop entire stack", "[.]") {
    using real_t = float;
    using complex_t = Complex<real_t>;

    const Device cpu{};
    StreamGuard stream{cpu};
    ArrayOption options{cpu, Allocator::DEFAULT};

    const size_t reduce_factor = 6;

    io::ImageFile file{test::NOA_DATA_PATH / "tilt1.mrc", io::READ};
    const size4_t shape = file.shape();
    float3_t pixel_size = file.pixelSize();
    pixel_size[1] *= reduce_factor;
    pixel_size[2] *= reduce_factor;

    Array<real_t> src{shape, options};
    Array<complex_t> src_fft{shape.fft(), options};
    file.readAll(src.get());

    fft::r2c(src, src_fft);
    src.release();

    const size4_t shape_binned{shape[0], shape[1], shape[2] / reduce_factor, shape[3] / reduce_factor};
    Array<complex_t> dst_fft{shape_binned.fft(), options};
    fft::resize<fft::H2H>(src_fft, shape, dst_fft, shape_binned);
    src_fft.release();

    Array<real_t> dst{shape_binned, options};
    fft::c2r(dst_fft, dst);
    dst_fft.release();

    file.open(test::NOA_DATA_PATH / "tilt1_cropped.mrc", io::WRITE);
    file.pixelSize(pixel_size);
    file.shape(dst.shape());
    file.writeAll(dst.eval().get());
}
