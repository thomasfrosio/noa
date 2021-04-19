#include <noa/cpu/masks/Cylinder.h>

#include <noa/cpu/PtrHost.h>
#include <noa/io/files/MRCFile.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

TEMPLATE_TEST_CASE("CPU::Mask - cylinder", "[noa][cpu][masks]", float) {
    size3_t shape(512, 512, 512);
    PtrHost<float> mask(getElements(shape));
    Mask::cylinder<true>(mask.get(), shape, {0, 0, 0}, 100, 80, 0);

    MRCFile file("test_mask.mrc", IO::WRITE);
    file.setShape(shape);
    file.writeAll(mask.get());
}
