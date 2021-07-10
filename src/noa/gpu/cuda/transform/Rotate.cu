#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/transform/Interpolate.h"

namespace {
    using namespace ::noa;

    template<InterpMode MODE>
    __global__ void test(cudaTextureObject_t t, float* out) {
        cuda::transform::tex3D<MODE>(out, t, 1., 1., 0.);
    }
}
