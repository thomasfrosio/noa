#include <noa/cpu/memory/Resize.h>

#include <noa/cpu/PtrHost.h>
#include <noa/io/files/MRCFile.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

// Test against manually checked data.
namespace {
    constexpr bool COMPUTE_TEST_DATA_INSTEAD = false;

    void initInput(int test_number, float* input, size3_t shape, uint batches) {
        size_t elements = getElements(shape);
        for (uint i = 0; i < elements * batches; ++i)
            input[i] = static_cast<float>(i);
        if (test_number > 10) {
            size3_t center = shape / size_t{2};
            for (uint batch = 0; batch < batches; ++batch)
                input[batch * elements + (center.z * shape.y + center.y) * shape.x + center.x] = 0;
        }
    }

    void getParam(int test_number, path_t* filename, uint* batches, size3_t* i_shape, size3_t* o_shape,
                  int3_t* border_left, int3_t* border_right, BorderMode* mode, float* value) {
        *filename /= Test::PATH_TEST_DATA / "memory";
        if (test_number == 1) {
            *batches = 3;
            *i_shape = {64, 64, 1};
            *o_shape = {81, 59, 1};
            *border_left = {11, -5, 0};
            *border_right = {6, 0, 0};
            *mode = BORDER_VALUE;
            *value = 5.f;
            *filename /= "resize_01.mrc";
        } else if (test_number == 2) {
            *batches = 1;
            *i_shape = {127, 128, 1};
            *o_shape = {108, 130, 1};
            *border_left = {-20, 1, 0};
            *border_right = {1, 1, 0};
            *mode = BORDER_ZERO;
            *value = 5.f;
            *filename /= "resize_02.mrc";
        } else if (test_number == 3) {
            *batches = 1;
            *i_shape = {63, 64, 1};
            *o_shape = {255, 256, 1};
            *border_left = {192, 100, 0};
            *border_right = {0, 92, 0};
            *mode = BORDER_PERIODIC;
            *value = 0.f;
            *filename /= "resize_03.mrc";
        } else if (test_number == 4) {
            *batches = 2;
            *i_shape = {127, 128, 1};
            *o_shape = {68, 128, 1};
            *border_left = {-50, 100, 0};
            *border_right = {-9, -100, 0};
            *mode = BORDER_CLAMP;
            *value = 0.f;
            *filename /= "resize_04.mrc";
        } else if (test_number == 5) {
            *batches = 2;
            *i_shape = {256, 256, 1};
            *o_shape = {256, 300, 1};
            *border_left = {0, 4, 0};
            *border_right = {0, 40, 0};
            *mode = BORDER_MIRROR;
            *value = 0.f;
            *filename /= "resize_05.mrc";
        } else if (test_number == 6) {
            *batches = 1;
            *i_shape = {64, 64, 64};
            *o_shape = {81, 59, 38};
            *border_left = {11, -5, -30};
            *border_right = {6, 0, 4};
            *mode = BORDER_VALUE;
            *value = 1.f;
            *filename /= "resize_06.mrc";
        } else if (test_number == 7) {
            *batches = 1;
            *i_shape = {127, 128, 66};
            *o_shape = {108, 130, 66};
            *border_left = {-20, 1, 0};
            *border_right = {1, 1, 0};
            *mode = BORDER_ZERO;
            *value = 5.f;
            *filename /= "resize_07.mrc";
        } else if (test_number == 8) {
            *batches = 1;
            *i_shape = {63, 64, 65};
            *o_shape = {255, 256, 100};
            *border_left = {192, 100, 25};
            *border_right = {0, 92, 10};
            *mode = BORDER_PERIODIC;
            *value = 0.f;
            *filename /= "resize_08.mrc";
        } else if (test_number == 9) {
            *batches = 1;
            *i_shape = {127, 128, 1};
            *o_shape = {68, 128, 5};
            *border_left = {-50, 128, 4};
            *border_right = {-9, -128, 0};
            *mode = BORDER_CLAMP;
            *value = 0.f;
            *filename /= "resize_09.mrc";
        } else if (test_number == 10) {
            *batches = 1;
            *i_shape = {256, 256, 30};
            *o_shape = {256, 300, 1};
            *border_left = {0, 4, -10};
            *border_right = {0, 40, -19};
            *mode = BORDER_MIRROR;
            *value = 0.f;
            *filename /= "resize_10.mrc";
        } else if (test_number == 11) {
            *batches = 3;
            *i_shape = {64, 64, 1};
            *o_shape = {81, 59, 1};
            *mode = BORDER_VALUE;
            *value = 5.f;
            *filename /= "resize_11.mrc";
        } else if (test_number == 12) {
            *batches = 1;
            *i_shape = {64, 64, 64};
            *o_shape = {81, 59, 40};
            *mode = BORDER_VALUE;
            *value = 1.f;
            *filename /= "resize_12.mrc";
        } else if (test_number == 13) {
            *batches = 1;
            *i_shape = {127, 128, 1};
            *o_shape = {108, 130, 1};
            *mode = BORDER_ZERO;
            *value = 5.f;
            *filename /= "resize_13.mrc";
        } else if (test_number == 14) {
            *batches = 1;
            *i_shape = {127, 128, 30};
            *o_shape = {130, 128, 1};
            *mode = BORDER_ZERO;
            *value = 5.f;
            *filename /= "resize_14.mrc";
        } else if (test_number == 15) {
            *batches = 1;
            *i_shape = {80, 1, 1};
            *o_shape = {80, 80, 40};
            *mode = BORDER_CLAMP;
            *value = 0.f;
            *filename /= "resize_15.mrc";
        } else if (test_number == 16) {
            *batches = 1;
            *i_shape = {1, 50, 50};
            *o_shape = {20, 31, 5};
            *mode = BORDER_CLAMP;
            *value = 0.f;
            *filename /= "resize_16.mrc";
        } else if (test_number == 17) {
            *batches = 1;
            *i_shape = {30, 30, 30};
            *o_shape = {90, 90, 90};
            *mode = BORDER_PERIODIC;
            *value = 0.f;
            *filename /= "resize_17.mrc";
        } else if (test_number == 18) {
            *batches = 1;
            *i_shape = {64, 128, 32};
            *o_shape = {128, 256, 32};
            *mode = BORDER_MIRROR;
            *value = 0.f;
            *filename /= "resize_18.mrc";
        }
    }
}

TEST_CASE("Memory::Resize", "[noa][cpu]") {
    uint batches;
    size3_t i_shape;
    size3_t o_shape;
    int3_t border_left;
    int3_t border_right;
    path_t filename;
    BorderMode mode;
    float value;

    int test_number = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18);
    getParam(test_number, &filename, &batches, &i_shape, &o_shape, &border_left, &border_right, &mode, &value);
    INFO(test_number);

    size_t i_elements = getElements(i_shape);
    size_t o_elements = getElements(o_shape);
    PtrHost<float> input(i_elements * batches);
    PtrHost<float> output(o_elements * batches);
    initInput(test_number, input.get(), i_shape, batches);

    if (test_number < 11)
        Memory::resize(input.get(), i_shape, output.get(), o_shape, border_left, border_right, mode, value, batches);
    else
        Memory::resize(input.get(), i_shape, output.get(), o_shape, mode, value, batches);

    if (COMPUTE_TEST_DATA_INSTEAD) {
        MRCFile file(filename, IO::WRITE);
        file.setShape(batches > 1 ? size3_t{o_shape.x, o_shape.y, batches} : o_shape);
        file.writeAll(output.get());
        file.close();
        return;
    }

    PtrHost<float> expected(o_elements * batches);
    MRCFile file(filename, IO::READ);
    file.readAll(expected.get());
    float diff = Test::getAverageNormalizedDifference(expected.get(), output.get(), o_elements * batches);
    REQUIRE_THAT(diff, Test::isWithinAbs(0.f, 1e-6));
}
