#include <noa/gpu/cuda/Stream.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>

#include <catch2/catch.hpp>

using namespace noa;

TEST_CASE("cuda::Stream", "[noa][cuda]") {
    using namespace cuda;

    memory::PtrDevice<float> a{100};
    memory::PtrDevice<float> b{100};
    {
        Stream stream;
        memory::PtrDevice<float> c{100, stream};
        stream.attach(b.share(), c.share(), a.share());
        stream.attach(c.share());
        c = nullptr;
        stream.attach(b.share());
    }
}
