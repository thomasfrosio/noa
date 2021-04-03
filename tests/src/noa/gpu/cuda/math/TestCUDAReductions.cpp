#include "noa/gpu/cuda/math/Reductions.h"

#include <noa/cpu/PtrHost.h>
#include <noa/gpu/cuda/PtrDevice.h>
#include <noa/gpu/cuda/Memory.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

TEMPLATE_TEST_CASE("CUDA::Math: Reduction - contiguous", "[noa][cuda][math]",
                   int, uint, float, double, cfloat_t, cdouble_t) {

}

