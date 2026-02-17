#include <noa/runtime/Array.hpp>
#include <noa/runtime/Iwise.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

using namespace ::noa::types;
namespace nt = noa::traits;

namespace {
    struct Arange4d {
        Span<f32, 4> span;

        constexpr void operator()(const Vec<isize, 4>& indices) {
            span(indices) = static_cast<f32>(noa::offset_at(span.strides(), indices));
        }
    };

    std::atomic<i32> iwise_init{};
    std::atomic<i32> iwise_deinit{};

    struct ThreadPoolLike {
        SpanContiguous<i32> array;
        i32 value{-1};

        void init(nt::compute_handle auto& handle) {
            iwise_init += 1;
            value = handle.thread().gid();
        }

        void operator()(i32 index) const {
            array[index] = value;
        }

        static void deinit() {
            iwise_deinit += 1;
        }
    };
}

TEST_CASE("runtime::iwise") {
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

TEST_CASE("runtime::iwise - threadpool-like") {
    auto data = Vec<i32, 4>::from_value(-1);
    auto span = Span(data.data(), data.ssize());
    auto op = ThreadPoolLike{span};

    auto stream = StreamGuard(Device(), Stream::SYNC);
    stream.set_thread_limit(2);

    // By default, only one thread should be launched for this shape.
    constexpr auto OPTIONS1 = noa::IwiseOptions{.generate_gpu = false};
    REQUIRE((iwise_init == 0 and iwise_deinit == 0));
    noa::iwise<OPTIONS1>(Shape{4}, stream.device(), op);
    REQUIRE((iwise_init == 1 and iwise_deinit == 1));
    REQUIRE(data == 0);

    // But we can enforce the number of threads.
    constexpr auto OPTIONS2 = noa::IwiseOptions{.generate_gpu = false, .cpu_launch_n_threads = 4};
    iwise_init = 0;
    iwise_deinit = 0;
    noa::iwise<OPTIONS2>(Shape{4}, stream.device(), op);
    REQUIRE((iwise_init == 4 and iwise_deinit == 4));
    REQUIRE(data == Vec{0, 1, 2, 3});
}
