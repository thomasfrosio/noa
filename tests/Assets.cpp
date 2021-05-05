#include "Assets.h"
#include "Helpers.h"

namespace Test::Assets::Memory {
    void initResizeInput(int test_number, float* input, size3_t shape, uint batches) {
        size_t elements = getElements(shape);
        for (uint i = 0; i < elements * batches; ++i)
            input[i] = static_cast<float>(i);
        if (test_number > 10 && test_number < 19) {
            size3_t center = shape / size_t{2};
            for (uint batch = 0; batch < batches; ++batch)
                input[batch * elements + (center.z * shape.y + center.y) * shape.x + center.x] = 0;
        }
    }

    void initResizeOutput(float* input, size3_t shape, uint batches) {
        for (uint i = 0; i < getElements(shape) * batches; ++i)
            input[i] = static_cast<float>(2);
    }

    void getResizeParams(int test_number, path_t* filename, uint* batches, size3_t* i_shape, size3_t* o_shape,
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
        } else if (test_number == 19) {
            *batches = 3;
            *i_shape = {64, 64, 1};
            *o_shape = {81, 59, 1};
            *border_left = {11, -5, 0};
            *border_right = {6, 0, 0};
            *mode = BORDER_NOTHING;
            *value = 0.f;
            *filename /= "resize_19.mrc";
        } else if (test_number == 20) {
            *batches = 1;
            *i_shape = {127, 128, 1};
            *o_shape = {68, 128, 5};
            *border_left = {-50, 100, 4};
            *border_right = {-9, -100, 0};
            *mode = BORDER_NOTHING;
            *value = 0.f;
            *filename /= "resize_20.mrc";
        }
    }
}

namespace Test::Assets::Fourier {
    void getLowpassParams(int test_number, path_t* filename, size3_t* shape, float* cutoff, float* width) {
        *filename = Test::PATH_TEST_DATA / "fourier";
        if (test_number == 1) {
            *shape = {256, 256, 1};
            *cutoff = 0;
            *width = 0;
            *filename /= "lowpass_01.mrc";
        } else if (test_number == 2) {
            *shape = {256, 256, 1};
            *cutoff = 0.5f;
            *width = 0;
            *filename /= "lowpass_02.mrc";
        } else if (test_number == 3) {
            *shape = {256, 256, 1};
            *cutoff = 0.35f;
            *width = 0.1f;
            *filename /= "lowpass_03.mrc";
        } else if (test_number == 4) {
            *shape = {512, 256, 1};
            *cutoff = 0.2f;
            *width = 0.3f;
            *filename /= "lowpass_04.mrc";
        } else if (test_number == 5) {
            *shape = {128, 128, 128};
            *cutoff = 0;
            *width = 0;
            *filename /= "lowpass_11.mrc";
        } else if (test_number == 6) {
            *shape = {128, 128, 128};
            *cutoff = 0.5f;
            *width = 0;
            *filename /= "lowpass_12.mrc";
        } else if (test_number == 7) {
            *shape = {64, 128, 128};
            *cutoff = 0.2f;
            *width = 0.3f;
            *filename /= "lowpass_13.mrc";
        }
    }

    void getHighpassParams(int test_number, path_t* filename, size3_t* shape, float* cutoff, float* width) {
        *filename = Test::PATH_TEST_DATA / "fourier";
        if (test_number == 1) {
            *shape = {256, 256, 1};
            *cutoff = 0;
            *width = 0;
            *filename /= "highpass_01.mrc";
        } else if (test_number == 2) {
            *shape = {256, 256, 1};
            *cutoff = 0.5f;
            *width = 0;
            *filename /= "highpass_02.mrc";
        } else if (test_number == 3) {
            *shape = {256, 256, 1};
            *cutoff = 0.35f;
            *width = 0.1f;
            *filename /= "highpass_03.mrc";
        } else if (test_number == 4) {
            *shape = {512, 256, 1};
            *cutoff = 0.2f;
            *width = 0.3f;
            *filename /= "highpass_04.mrc";
        } else if (test_number == 5) {
            *shape = {128, 128, 128};
            *cutoff = 0;
            *width = 0;
            *filename /= "highpass_11.mrc";
        } else if (test_number == 6) {
            *shape = {128, 128, 128};
            *cutoff = 0.5f;
            *width = 0;
            *filename /= "highpass_12.mrc";
        } else if (test_number == 7) {
            *shape = {64, 128, 128};
            *cutoff = 0.2f;
            *width = 0.3f;
            *filename /= "highpass_13.mrc";
        }
    }

    void getBandpassParams(int test_number, path_t* filename, size3_t* shape,
                           float* cutoff1, float* cutoff2, float* width1, float* width2) {
        *filename = Test::PATH_TEST_DATA / "fourier";
        if (test_number == 1) {
            *shape = {256, 256, 1};
            *cutoff1 = 0.4f;
            *cutoff2 = 0.5f;
            *width1 = 0;
            *width2 = 0;
            *filename /= "bandpass_01.mrc";
        } else if (test_number == 2) {
            *shape = {256, 512, 1};
            *cutoff1 = 0.3f;
            *cutoff2 = 0.45f;
            *width1 = 0.3f;
            *width2 = 0.05f;
            *filename /= "bandpass_02.mrc";
        } else if (test_number == 3) {
            *shape = {128, 128, 1};
            *cutoff1 = 0.3f;
            *cutoff2 = 0.4f;
            *width1 = 0.05f;
            *width2 = 0.05f;
            *filename /= "bandpass_03.mrc";
        } else if (test_number == 4) {
            *shape = {128, 128, 128};
            *cutoff1 = 0.2f;
            *cutoff2 = 0.45f;
            *width1 = 0.1f;
            *width2 = 0.05f;
            *filename /= "bandpass_11.mrc";
        } else if (test_number == 5) {
            *shape = {64, 128, 128};
            *cutoff1 = 0.1f;
            *cutoff2 = 0.3f;
            *width1 = 0;
            *width2 = 0.1f;
            *filename /= "bandpass_12.mrc";
        }
    }
}

namespace Test::Assets::Mask {
    void getSphereParams(int test_number, path_t* filename, size3_t* shape,
                         float3_t* shifts, float* radius, float* taper) {
        *filename = Test::PATH_TEST_DATA / "masks";
        if (test_number == 1) {
            *shape = {128, 128, 1};
            *shifts = {0, 0, 0};
            *radius = 40;
            *taper = 0;
            *filename /= "sphere_01.mrc";
        } else if (test_number == 2) {
            *shape = {128, 128, 1};
            *shifts = {0, 0, 0};
            *radius = 41;
            *taper = 0;
            *filename /= "sphere_02.mrc";
        } else if (test_number == 3) {
            *shape = {256, 256, 1};
            *shifts = {-127, 0, 0};
            *radius = 108;
            *taper = 19;
            *filename /= "sphere_03.mrc";
        } else if (test_number == 4) {
            *shape = {100, 100, 100};
            *shifts = {20, 0, -20};
            *radius = 30;
            *taper = 0;
            *filename /= "sphere_04.mrc";
        } else if (test_number == 5) {
            *shape = {100, 100, 100};
            *shifts = {20, 0, -20};
            *radius = 20;
            *taper = 10;
            *filename /= "sphere_05.mrc";
        }
    }

    void getCylinderParams(int test_number, path_t* filename, size3_t* shape,
                           float3_t* shifts, float* radius_xy, float* radius_z, float* taper) {
        *filename = Test::PATH_TEST_DATA / "masks";
        if (test_number == 1) {
            *shape = {256, 256, 64};
            *shifts = {0, 0, 0};
            *radius_xy = 60;
            *radius_z = 20;
            *taper = 0;
            *filename /= "cylinder_01.mrc";
        } else if (test_number == 2) {
            *shape = {128, 128, 128};
            *shifts = {-11, 11, 0};
            *radius_xy = 31;
            *radius_z = 45;
            *taper = 11;
            *filename /= "cylinder_02.mrc";
        } else if (test_number == 3) {
            *shape = {80, 91, 180};
            *shifts = {-6, 0, 10};
            *radius_xy = 10;
            *radius_z = 50;
            *taper = 6;
            *filename /= "cylinder_03.mrc";
        }
    }

    void getRectangleParams(int test_number, path_t* filename, size3_t* shape,
                            float3_t* shifts, float3_t* radius, float* taper) {
        *filename = Test::PATH_TEST_DATA / "masks";
        if (test_number == 1) {
            *shape = {512, 512, 1};
            *shifts = {0, 0, 0};
            *radius = {50, 51, 1};
            *taper = 0;
            *filename /= "rectangle_01.mrc";
        } else if (test_number == 2) {
            *shape = {231, 230, 1};
            *shifts = {-11, 11, 0};
            *radius = {50, 51, 1};
            *taper = 0;
            *filename /= "rectangle_02.mrc";
        } else if (test_number == 3) {
            *shape = {128, 256, 1};
            *shifts = {12, 10, 0};
            *radius = {30, 80, 1};
            *taper = 10;
            *filename /= "rectangle_03.mrc";
        } else if (test_number == 4) {
            *shape = {128, 128, 64};
            *shifts = {20, 0, 0};
            *radius = {30, 80, 5};
            *taper = 10;
            *filename /= "rectangle_04.mrc";
        } else if (test_number == 5) {
            *shape = {64, 64, 64};
            *shifts = {0, -10, 0};
            *radius = {10, 10, 15};
            *taper = 15;
            *filename /= "rectangle_05.mrc";
        }
    }
}
