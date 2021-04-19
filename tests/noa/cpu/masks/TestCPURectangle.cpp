#include <noa/cpu/masks/Rectangle.h>

#include <noa/cpu/PtrHost.h>
#include <noa/io/files/MRCFile.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

TEMPLATE_TEST_CASE("CPU::Mask - rectangle", "[noa][cpu][masks]", float) {
    size3_t shape(512, 512, 1);
    PtrHost<float> mask(getElements(shape));
    Mask::rectangle<true>(mask.get(), shape, {-250, 50, 0}, {50, 100, 50}, 20);

    MRCFile file("test_mask.mrc", IO::WRITE);
    file.setShape(shape);
    file.writeAll(mask.get());
}
