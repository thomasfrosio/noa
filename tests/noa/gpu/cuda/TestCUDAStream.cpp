#include <noa/gpu/cuda/Stream.hpp>
#include <noa/gpu/cuda/memory/PtrDevice.hpp>

#include <catch2/catch.hpp>

using namespace noa;

TEST_CASE("cuda::Stream", "[noa][cuda]") {

    const cuda::memory::PtrDevice<float> a{100};
    const cuda::memory::PtrDevice<float> b{100};
    {
        cuda::Stream stream;
        cuda::memory::PtrDevice<float> c{100, stream};
        stream.enqueue_attach(b.share(), c.share(), a.share());
        stream.enqueue_attach(c.share());
        c = nullptr;
        stream.enqueue_attach(b.share());
    }
}
