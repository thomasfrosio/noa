#include <noa/unified/Array.hpp>
#include <noa/unified/Iwise.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

using namespace ::noa::types;

namespace {
    struct Arange4d {
        Span<f32, 4> span;

        constexpr void operator()(const Vec<isize, 4>& indices) {
            span(indices) = static_cast<f32>(noa::indexing::offset_at(span.strides(), indices));
        }
    };

    struct ThreadPoolLike {
        SpanContiguous<i32> array;
        void operator()(i32 index) const {
            #ifdef NOA_ENABLE_OPENMP
            array[index] = omp_in_parallel() ? omp_get_thread_num() : 1;
            #elif
            array[index] = 1;
            #endif
        }
    };
}

TEST_CASE("unified::iwise") {
    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    const auto shape = test::random_shape_batched(3);
    for (auto& device: devices) {
        const auto stream = StreamGuard(device, Stream::ASYNC);
        const auto options = ArrayOption{device, "managed"};

        const auto a0 = Array<f32>(shape, options);
        noa::iwise(a0.shape(), a0.device(), Arange4d{a0.span()}, a0);

        const auto a1 = Array<f32>(shape, options);
        test::arange(a1.get(), a1.n_elements());

        REQUIRE(test::allclose_abs(a0, a1, 1e-6));
    }
}

TEST_CASE("unified::iwise - threadpool-like") {
    auto data = Vec<i32, 4>{};
    auto span = Span(data.data(), data.ssize());
    auto op = ThreadPoolLike{span};

    auto stream = StreamGuard(Device(), Stream::SYNC);
    stream.set_thread_limit(2);

    // By default, only one thread should be launched for this shape.
    constexpr auto OPTIONS1 = noa::IwiseOptions{.generate_gpu = false};
    noa::iwise<OPTIONS1>(Shape{4}, stream.device(), op);
    for (auto e: data)
        REQUIRE(e == 1);

    // But we can enforce the number of threads.
    constexpr auto OPTIONS2 = noa::IwiseOptions{.generate_gpu = false, .cpu_launch_n_threads = 4};
    noa::iwise<OPTIONS2>(Shape{4}, stream.device(), op);
    for (i32 i{}; auto e: data)
        REQUIRE(e == i++);
}
