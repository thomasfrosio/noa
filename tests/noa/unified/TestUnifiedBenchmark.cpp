#include <noa/core/utils/Timer.hpp>
#include <noa/unified/Reduce.hpp>
#include <noa/unified/Factory.hpp>

#include <catch2/catch.hpp>
#include "Utils.hpp"

using namespace noa::types;

namespace test1 {
    auto get_min0(float* ptr, size_t size) -> float;
    auto get_min1(float* ptr, Shape<i64, 4> shape, int n_threads) -> float;
    auto get_min2(noa::View<float> ptr) -> float;
}

TEST_CASE("unified::benchmark", "[noa][unified]") {
    const auto shape = Shape4<i64>{3, 256, 256, 300};

    auto stream = Stream(Device{}, Stream::DEFAULT);
    stream.set_thread_limit(1);
    Stream::set_current(stream);

    Array input = noa::empty<f32>(shape);
    const i64 n_elements = input.n_elements();
    test::randomize(input.get(), n_elements, test::Randomizer<f32>(-1., 1.));

    noa::Timer t0;
    // t0.start();
    // f32 output0{};
    // for (auto _: noa::irange(100)) {
    //     output0 += test1::get_min0(input.get(), static_cast<size_t>(n_elements));
    // }
    // fmt::println("raw-ptr done in {}, out={}", t0.elapsed() / 100, output0);
    //
    // t0.start();
    // f32 output1{};
    // for (auto _: noa::irange(100)) {
    //     output1 += test1::get_min1(input.get(), Shape<i64, 4>{1, 1, 1, n_elements}, 1);
    // }
    // fmt::println("cpu-core done in {}, out={}", t0.elapsed() / 100, output1);

    t0.start();
    auto view = input.view().subregion(noa::indexing::Ellipsis{}, noa::indexing::Slice{0, -2});
    f32 output2{};
    for (auto _: noa::irange(100))
        output2 += test1::get_min2(view);
    fmt::println("unified done in {}, out={}", t0.elapsed() / 100, output2);
}

TEST_CASE("unified::benchmark rr", "[noa][unified]") {
    const auto shape = Shape4<i64>{3, 256, 256, 300};
    const auto size = static_cast<size_t>(shape.n_elements());
    std::unique_ptr input = std::make_unique<f32[]>(size);

    test::randomize(input.get(), size, test::Randomizer<f32>(-1., 1.));

    noa::Timer t0;
    t0.start();
    f32 out2{};
    for (auto _: noa::irange(100)) {
        f32 m = 300;
        for (size_t i{}; i < size; ++i)
            m = std::min(input[i], m);
        out2 += m;
    }
    fmt::println("0 done in {}, out={}", t0.elapsed() / 100, out2);
}

//TEST_CASE("unified::argmax/argmin ", "[noa][unified]") {
//    const auto shape = Shape4<i64>{3, 256, 256, 300};
//    const auto n_elements_per_batch = shape.pop_front().template as<u32>().n_elements();
//
//    auto stream = Stream(Device{}, Stream::DEFAULT);
//    stream.set_thread_limit(1);
//    Stream::set_current(stream);
//
//    AND_THEN("reduce entire array") {
//        Array<f32> input(shape);
//        test::randomize(input.get(), input.n_elements(), test::Randomizer<f32>(-100., 100.));
//
////        auto expected_min_offset = test::Randomizer<u32>(i64{}, input.n_elements() - 2).get();
////        auto expected_max_offset = test::Randomizer<u32>(i64{}, input.n_elements() - 2).get();
////        if (expected_min_offset == expected_max_offset)
////            expected_max_offset += 1;
////
////        input.span_1d()[expected_min_offset] = -101;
////        input.span_1d()[expected_max_offset] = 101;
//
////        const auto options = ArrayOption{"cpu", "managed"};
////        input = input.device().is_cpu() ? input : input.to(options);
//
//        noa::Timer t0;
//        t0.start();
//        for (auto _: noa::irange(10)) {
//            const auto [min, min_offset] = noa::argmin(input);
//            fmt::println("{}", min);
//        }
//        fmt::println("0 done in {}", t0.elapsed() / 10);
//    }
//
////    AND_THEN("per batch") {
////        Array<f32> input(shape);
////        test::randomize(input.get(), input.n_elements(), test::Randomizer<f32>(-100., 100.));
////
////        Array min_values = noa::empty<f32>({shape.batch(), 1, 1, 1});
////        Array min_offsets = noa::like<i32>(min_values);
////        Array expected_min_values = noa::like<f32>(min_values);
////        Array expected_min_offsets = noa::like<i32>(min_values);
////
////        test::randomize(expected_min_values.get(), expected_min_values.n_elements(),
////                        test::Randomizer<f32>(-200, -101));
////        test::randomize(expected_min_offsets.get(), expected_min_offsets.n_elements(),
////                        test::Randomizer<i32>(0u, n_elements_per_batch - 1));
////
////        const auto input_2d = input.reshape({shape.batch(), 1, 1, -1});
////        for (i64 batch = 0; batch < shape.batch(); ++batch) {
////            auto& offset = expected_min_offsets(batch, 0, 0, 0);
////            input_2d(batch, 0, 0, offset) = expected_min_values(batch, 0, 0, 0);
////        }
////
////        for (auto& device: devices) {
////            INFO(device);
////            const auto options = ArrayOption{device, "managed"};
////            if (input.device() != device) {
////                input = input.to(options);
////                min_values = min_values.to(options);
////                min_offsets = min_offsets.to(options);
////            }
////
////            noa::Timer t0;
////            t0.start();
////            for (auto _: noa::irange(10))
////                noa::argmin(input, min_values, min_offsets);
////            fmt::println("1 done in {}", t0.elapsed() / 10);
//////            REQUIRE(test::allclose_abs(min_values, expected_min_values, 1e-7));
//////
//////            // Offsets are not relative to each batch...
//////            for (i64 batch = 0; auto& value: expected_min_offsets.span_1d_contiguous()) {
//////                value += static_cast<i32>(batch * n_elements_per_batch);
//////                ++batch;
//////            }
//////            REQUIRE(test::allclose_abs(min_offsets, expected_min_offsets, 1e-7));
////        } // FIXME check performance of reduce_axes_iwise, with batch?
////    }
//}
