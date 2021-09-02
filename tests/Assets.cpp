#include "Assets.h"
#include "Helpers.h"

namespace test::assets::memory {
    void initResizeInput(int test_number, float* input, size3_t shape, uint batches) {
        size_t elements = getElements(shape);
        for (uint i = 0; i < elements * batches; ++i)
            input[i] = static_cast<float>(i);
        if (test_number >= 20 && test_number < 29) {
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
        *filename /= test::PATH_TEST_DATA / "memory";
        if (test_number == 0) {
            *batches = 3;
            *i_shape = {64, 64, 1};
            *o_shape = {81, 59, 1};
            *border_left = {11, -5, 0};
            *border_right = {6, 0, 0};
            *mode = BORDER_VALUE;
            *value = 5.f;
            *filename /= "resize_00.mrc";
        } else if (test_number == 1) {
            *batches = 1;
            *i_shape = {127, 128, 1};
            *o_shape = {108, 130, 1};
            *border_left = {-20, 1, 0};
            *border_right = {1, 1, 0};
            *mode = BORDER_ZERO;
            *value = 5.f;
            *filename /= "resize_01.mrc";
        } else if (test_number == 2) {
            *batches = 1;
            *i_shape = {63, 64, 1};
            *o_shape = {255, 256, 1};
            *border_left = {192, 100, 0};
            *border_right = {0, 92, 0};
            *mode = BORDER_PERIODIC;
            *value = 0.f;
            *filename /= "resize_02.mrc";
        } else if (test_number == 3) {
            *batches = 2;
            *i_shape = {127, 128, 1};
            *o_shape = {68, 128, 1};
            *border_left = {-50, 100, 0};
            *border_right = {-9, -100, 0};
            *mode = BORDER_CLAMP;
            *value = 0.f;
            *filename /= "resize_03.mrc";
        } else if (test_number == 4) {
            *batches = 2;
            *i_shape = {256, 256, 1};
            *o_shape = {256, 300, 1};
            *border_left = {0, 4, 0};
            *border_right = {0, 40, 0};
            *mode = BORDER_MIRROR;
            *value = 0.f;
            *filename /= "resize_04.mrc";
        } else if (test_number == 5) {
            *batches = 2;
            *i_shape = {256, 256, 1};
            *o_shape = {256, 300, 1};
            *border_left = {0, 4, 0};
            *border_right = {0, 40, 0};
            *mode = BORDER_REFLECT;
            *value = 0.f;
            *filename /= "resize_05.mrc";
        } else if (test_number == 10) {
            *batches = 1;
            *i_shape = {64, 64, 64};
            *o_shape = {81, 59, 38};
            *border_left = {11, -5, -30};
            *border_right = {6, 0, 4};
            *mode = BORDER_VALUE;
            *value = 1.f;
            *filename /= "resize_10.mrc";
        } else if (test_number == 11) {
            *batches = 1;
            *i_shape = {127, 128, 66};
            *o_shape = {108, 130, 66};
            *border_left = {-20, 1, 0};
            *border_right = {1, 1, 0};
            *mode = BORDER_ZERO;
            *value = 5.f;
            *filename /= "resize_11.mrc";
        } else if (test_number == 12) {
            *batches = 1;
            *i_shape = {63, 64, 65};
            *o_shape = {255, 256, 100};
            *border_left = {192, 100, 25};
            *border_right = {0, 92, 10};
            *mode = BORDER_PERIODIC;
            *value = 0.f;
            *filename /= "resize_12.mrc";
        } else if (test_number == 13) {
            *batches = 1;
            *i_shape = {127, 128, 1};
            *o_shape = {68, 128, 5};
            *border_left = {-50, 128, 4};
            *border_right = {-9, -128, 0};
            *mode = BORDER_CLAMP;
            *value = 0.f;
            *filename /= "resize_13.mrc";
        } else if (test_number == 14) {
            *batches = 1;
            *i_shape = {256, 256, 30};
            *o_shape = {256, 300, 1};
            *border_left = {0, 4, -10};
            *border_right = {0, 40, -19};
            *mode = BORDER_MIRROR;
            *value = 0.f;
            *filename /= "resize_14.mrc";
        } else if (test_number == 15) {
            *batches = 1;
            *i_shape = {256, 256, 30};
            *o_shape = {256, 300, 1};
            *border_left = {0, 4, -10};
            *border_right = {0, 40, -19};
            *mode = BORDER_REFLECT;
            *value = 0.f;
            *filename /= "resize_15.mrc";
        } else if (test_number == 20) {
            *batches = 3;
            *i_shape = {64, 64, 1};
            *o_shape = {81, 59, 1};
            *mode = BORDER_VALUE;
            *value = 5.f;
            *filename /= "resize_20.mrc";
        } else if (test_number == 21) {
            *batches = 1;
            *i_shape = {64, 64, 64};
            *o_shape = {81, 59, 40};
            *mode = BORDER_VALUE;
            *value = 1.f;
            *filename /= "resize_21.mrc";
        } else if (test_number == 22) {
            *batches = 1;
            *i_shape = {127, 128, 1};
            *o_shape = {108, 130, 1};
            *mode = BORDER_ZERO;
            *value = 5.f;
            *filename /= "resize_22.mrc";
        } else if (test_number == 23) {
            *batches = 1;
            *i_shape = {127, 128, 30};
            *o_shape = {130, 128, 1};
            *mode = BORDER_ZERO;
            *value = 5.f;
            *filename /= "resize_23.mrc";
        } else if (test_number == 24) {
            *batches = 1;
            *i_shape = {80, 1, 1};
            *o_shape = {80, 80, 40};
            *mode = BORDER_CLAMP;
            *value = 0.f;
            *filename /= "resize_24.mrc";
        } else if (test_number == 25) {
            *batches = 1;
            *i_shape = {1, 50, 50};
            *o_shape = {20, 31, 5};
            *mode = BORDER_CLAMP;
            *value = 0.f;
            *filename /= "resize_25.mrc";
        } else if (test_number == 26) {
            *batches = 1;
            *i_shape = {30, 30, 30};
            *o_shape = {90, 90, 90};
            *mode = BORDER_PERIODIC;
            *value = 0.f;
            *filename /= "resize_26.mrc";
        } else if (test_number == 27) {
            *batches = 1;
            *i_shape = {64, 128, 32};
            *o_shape = {128, 256, 32};
            *mode = BORDER_MIRROR;
            *value = 0.f;
            *filename /= "resize_27.mrc";
        } else if (test_number == 28) {
            *batches = 1;
            *i_shape = {64, 128, 32};
            *o_shape = {128, 256, 32};
            *mode = BORDER_REFLECT;
            *value = 0.f;
            *filename /= "resize_28.mrc";
        } else if (test_number == 30) {
            *batches = 3;
            *i_shape = {64, 64, 1};
            *o_shape = {81, 59, 1};
            *border_left = {11, -5, 0};
            *border_right = {6, 0, 0};
            *mode = BORDER_NOTHING;
            *value = 0.f;
            *filename /= "resize_30.mrc";
        } else if (test_number == 31) {
            *batches = 1;
            *i_shape = {127, 128, 1};
            *o_shape = {68, 128, 5};
            *border_left = {-50, 100, 4};
            *border_right = {-9, -100, 0};
            *mode = BORDER_NOTHING;
            *value = 0.f;
            *filename /= "resize_31.mrc";
        }
    }

    void initExtractInput(float* input, size_t elements) {
        for (uint i = 0; i < elements; ++i)
            input[i] = static_cast<float>(i);
    }

    void initInsertOutput(float* output, size_t elements) {
        for (uint i = 0; i < elements; ++i)
            output[i] = static_cast<float>(4);
    }

    path_t getExtractFilename(int test_number, uint subregion_idx) {
        path_t tmp = test::PATH_TEST_DATA / "memory" / "extract_";
        tmp += string::format("{}{}.mrc", test_number, subregion_idx);
        return tmp;
    }

    path_t getInsertFilename(int test_number) {
        path_t tmp = test::PATH_TEST_DATA / "memory" / "insert_";
        tmp += string::format("{}.mrc", test_number);
        return tmp;
    }

    void getExtractParams(int test_number,
                          size3_t* i_shape, size3_t* sub_shape, size3_t* sub_centers, uint* sub_count,
                          BorderMode* mode, float* value) {
        if (test_number == 1) {
            *i_shape = {512, 513, 1};
            *sub_shape = {62, 63, 1};
            sub_centers[0] = {30, 31, 0};
            sub_centers[1] = {500, 500, 0};
            sub_centers[2] = {128, 32, 0};
            sub_centers[3] = {350, 451, 0};
            sub_centers[4] = {512, 0, 0};
            *sub_count = 5;
            *mode = BORDER_VALUE;
            *value = 3.5;
        } else if (test_number == 2) {
            *i_shape = {256, 255, 50};
            *sub_shape = {55, 60, 1};
            sub_centers[0] = {0, 0, 0};
            sub_centers[1] = {128, 32, 24};
            sub_centers[2] = {0, 255, 0};
            *sub_count = 3;
            *mode = BORDER_NOTHING;
            *value = 3;
        } else if (test_number == 3) {
            *i_shape = {128, 127, 126};
            *sub_shape = {40, 42, 43};
            sub_centers[0] = {30, 31, 20};
            sub_centers[1] = {127, 126, 125};
            sub_centers[2] = {64, 117, 120};
            *sub_count = 3;
            *mode = BORDER_ZERO;
            *value = 5;
        }
    }

    void getTransposeParams(int test_number, path_t* filename_data, path_t* filename_expected,
                            size3_t* shape, uint3_t* permutation, bool* in_place) {
        *filename_data /= PATH_TEST_DATA / "memory";
        *filename_expected = PATH_TEST_DATA / "memory";

        if (test_number == 1) {
            *shape = {125, 120, 1};
            *permutation = {1, 0, 2};
            *filename_data /= "tmp_transpose_data_2d.mrc";
            *filename_expected /= "tmp_transpose_2d_102.mrc";
            *in_place = false;
        } else if (test_number == 2) {
            *shape = {125, 120, 121};
            *permutation = {0, 2, 1};
            *filename_data /= "tmp_transpose_data_3d.mrc";
            *filename_expected /= "tmp_transpose_3d_021.mrc";
            *in_place = false;
        } else if (test_number == 3) {
            *shape = {125, 120, 121};
            *permutation = {1, 0, 2};
            *filename_data /= "tmp_transpose_data_3d.mrc";
            *filename_expected /= "tmp_transpose_3d_102.mrc";
            *in_place = false;
        } else if (test_number == 4) {
            *shape = {125, 120, 121};
            *permutation = {1, 2, 0};
            *filename_data /= "tmp_transpose_data_3d.mrc";
            *filename_expected /= "tmp_transpose_3d_120.mrc";
            *in_place = false;
        } else if (test_number == 5) {
            *shape = {125, 120, 121};
            *permutation = {2, 0, 1};
            *filename_data /= "tmp_transpose_data_3d.mrc";
            *filename_expected /= "tmp_transpose_3d_201.mrc";
            *in_place = false;
        } else if (test_number == 6) {
            *shape = {125, 120, 121};
            *permutation = {2, 1, 0};
            *filename_data /= "tmp_transpose_data_3d.mrc";
            *filename_expected /= "tmp_transpose_3d_210.mrc";
            *in_place = false;
        } else if (test_number == 7) {
            *shape = {64, 64, 1};
            *permutation = {1, 0, 2};
            *filename_data /= "tmp_transpose_2d_in_place_102_data.mrc";
            *filename_expected /= "tmp_transpose_2d_in_place_102_expected.mrc";
            *in_place = true;
        } else if (test_number == 8) {
            *shape = {65, 64, 64};
            *permutation = {0, 2, 1};
            *filename_data /= "tmp_transpose_3d_in_place_021_data.mrc";
            *filename_expected /= "tmp_transpose_3d_in_place_021_expected.mrc";
            *in_place = true;
        } else if (test_number == 9) {
            *shape = {65, 65, 64};
            *permutation = {1, 0, 2};
            *filename_data /= "tmp_transpose_3d_in_place_102_data.mrc";
            *filename_expected /= "tmp_transpose_3d_in_place_102_expected.mrc";
            *in_place = true;
        } else if (test_number == 10) {
            *shape = {64, 66, 64};
            *permutation = {2, 1, 0};
            *filename_data /= "tmp_transpose_3d_in_place_210_data.mrc";
            *filename_expected /= "tmp_transpose_3d_in_place_210_expected.mrc";
            *in_place = true;
        }
    }
}

namespace test::assets::fourier {
    void getLowpassParams(int test_number, path_t* filename, size3_t* shape, float* cutoff, float* width) {
        *filename = test::PATH_TEST_DATA / "fourier";
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
        *filename = test::PATH_TEST_DATA / "fourier";
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
        *filename = test::PATH_TEST_DATA / "fourier";
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

namespace test::assets::filter {
    void getSphereParams(int test_number, path_t* filename, size3_t* shape,
                         float3_t* shifts, float* radius, float* taper) {
        *filename = test::PATH_TEST_DATA / "filter";
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
        *filename = test::PATH_TEST_DATA / "filter";
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
        *filename = test::PATH_TEST_DATA / "filter";
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

    void getMedianData(int test_number, path_t* filename) {
        switch (test_number) {
            default:
            case 1:
            case 2:
            case 5:
            case 6:
                *filename = test::PATH_TEST_DATA / "filter" / "tmp_medfilt_img_2D.mrc";
                break;
            case 3:
            case 4:
            case 7:
            case 8:
            case 9:
            case 10:
                *filename = test::PATH_TEST_DATA / "filter" / "tmp_medfilt_img_3D.mrc";
                break;
        }
    }

    void getMedianParams(int test_number, path_t* filename, size3_t* shape, BorderMode* mode, uint* window) {
        *filename = test::PATH_TEST_DATA / "filter";
        if (test_number == 1) {
            *shape = {250, 250, 1};
            *mode = BORDER_REFLECT;
            *window = 3;
            *filename /= "tmp_medfilt_1.mrc";
        } else if (test_number == 2) {
            *shape = {250, 250, 1};
            *mode = BORDER_ZERO;
            *window = 5;
            *filename /= "tmp_medfilt_2.mrc";
        } else if (test_number == 3) {
            *shape = {150, 150, 150};
            *mode = BORDER_REFLECT;
            *window = 7;
            *filename /= "tmp_medfilt_3.mrc";
        } else if (test_number == 4) {
            *shape = {150, 150, 150};
            *mode = BORDER_ZERO;
            *window = 9;
            *filename /= "tmp_medfilt_4.mrc";
        } else if (test_number == 5) {
            *shape = {250, 250, 1};
            *mode = BORDER_REFLECT;
            *window = 11;
            *filename /= "tmp_medfilt_5.mrc";
        } else if (test_number == 6) {
            *shape = {250, 250, 1};
            *mode = BORDER_ZERO;
            *window = 9;
            *filename /= "tmp_medfilt_6.mrc";
        } else if (test_number == 7) {
            *shape = {150, 150, 150};
            *mode = BORDER_REFLECT;
            *window = 7;
            *filename /= "tmp_medfilt_7.mrc";
        } else if (test_number == 8) {
            *shape = {150, 150, 150};
            *mode = BORDER_ZERO;
            *window = 3;
            *filename /= "tmp_medfilt_8.mrc";
        } else if (test_number == 9) {
            *shape = {150, 150, 150};
            *mode = BORDER_REFLECT;
            *window = 5;
            *filename /= "tmp_medfilt_9.mrc";
        } else if (test_number == 10) {
            *shape = {150, 150, 150};
            *mode = BORDER_ZERO;
            *window = 3;
            *filename /= "tmp_medfilt_10.mrc";
        }
    }

    void getConvData(int test_number, path_t* filename) {
        switch (test_number) {
            default:
            case 1:
            case 3:
            case 8:
            case 9:
            case 10:
                *filename = test::PATH_TEST_DATA / "filter" / "tmp_conv_img_2D.mrc";
                break;
            case 2:
            case 4:
            case 5:
            case 6:
            case 7:
            case 11:
            case 12:
            case 13:
            case 14:
            case 15:
            case 16:
            case 17:
                *filename = test::PATH_TEST_DATA / "filter" / "tmp_conv_img_3D.mrc";
                break;
        }
    }

    void getConvFilter(int test_number, path_t* filename) {
        switch (test_number) {
            default:
            case 1:
            case 2:
                *filename = test::PATH_TEST_DATA / "filter" / "tmp_conv_filter_1D.mrc";
                break;
            case 3:
            case 4:
                *filename = test::PATH_TEST_DATA / "filter" / "tmp_conv_filter_2D.mrc";
                break;
            case 5:
                *filename = test::PATH_TEST_DATA / "filter" / "tmp_conv_filter_3D_3x3x3.mrc";
                break;
            case 6:
                *filename = test::PATH_TEST_DATA / "filter" / "tmp_conv_filter_3D_5x5x5.mrc";
                break;
            case 7:
                *filename = test::PATH_TEST_DATA / "filter" / "tmp_conv_filter_3D_5x3x3.mrc";
                break;
            case 8:
            case 9:
            case 10:
            case 11:
            case 12:
            case 13:
            case 14:
            case 15:
            case 16:
            case 17:
                *filename = test::PATH_TEST_DATA / "filter" / "tmp_conv_filter_separable.mrc";
                break;
        }
    }

    void getConvParams(int test_number, path_t* filename, size3_t* shape, uint3_t* filter_size) {
        *filename = test::PATH_TEST_DATA / "filter";
        if (test_number == 1) {
            *shape = {251, 250, 1};
            *filter_size = {31, 1, 1};
            *filename /= "tmp_conv_1.mrc";
        } else if (test_number == 2) {
            *shape = {150, 151, 152};
            *filter_size = {31, 1, 1};
            *filename /= "tmp_conv_2.mrc";
        } else if (test_number == 3) {
            *shape = {251, 250, 1};
            *filter_size = {17, 9, 1};
            *filename /= "tmp_conv_3.mrc";
        } else if (test_number == 4) {
            *shape = {150, 151, 152};
            *filter_size = {17, 9, 1};
            *filename /= "tmp_conv_4.mrc";
        } else if (test_number == 5) {
            *shape = {150, 151, 152};
            *filter_size = {3, 3, 3};
            *filename /= "tmp_conv_5.mrc";
        } else if (test_number == 6) {
            *shape = {150, 151, 152};
            *filter_size = {5, 5, 5};
            *filename /= "tmp_conv_6.mrc";
        } else if (test_number == 7) {
            *shape = {150, 151, 152};
            *filter_size = {5, 3, 3};
            *filename /= "tmp_conv_7.mrc";
        } else if (test_number == 8) {
            *shape = {251, 250, 1};
            *filter_size = {21, 21, 1};
            *filename /= "tmp_conv_8.mrc";
        } else if (test_number == 9) {
            *shape = {251, 250, 1};
            *filter_size = {21, 1, 1};
            *filename /= "tmp_conv_9.mrc";
        } else if (test_number == 10) {
            *shape = {251, 250, 1};
            *filter_size = {1, 21, 1};
            *filename /= "tmp_conv_10.mrc";
        } else if (test_number == 11) {
            *shape = {150, 151, 152};
            *filter_size = {21, 21, 21};
            *filename /= "tmp_conv_11.mrc";
        } else if (test_number == 12) {
            *shape = {150, 151, 152};
            *filter_size = {21, 21, 1};
            *filename /= "tmp_conv_12.mrc";
        } else if (test_number == 13) {
            *shape = {150, 151, 152};
            *filter_size = {1, 21, 21};
            *filename /= "tmp_conv_13.mrc";
        } else if (test_number == 14) {
            *shape = {150, 151, 152};
            *filter_size = {21, 1, 21};
            *filename /= "tmp_conv_14.mrc";
        } else if (test_number == 15) {
            *shape = {150, 151, 152};
            *filter_size = {21, 1, 1};
            *filename /= "tmp_conv_15.mrc";
        } else if (test_number == 16) {
            *shape = {150, 151, 152};
            *filter_size = {1, 21, 1};
            *filename /= "tmp_conv_16.mrc";
        } else if (test_number == 17) {
            *shape = {150, 151, 152};
            *filter_size = {1, 1, 21};
            *filename /= "tmp_conv_17.mrc";
        }
    }
}

namespace test::assets::transform {
    void getRotate2DParams(int test_number, path_t* filename_data, path_t* filename_expected,
                           InterpMode* interp, BorderMode* border, float* value,
                           float* rotation, float2_t* rotation_center) {
        path_t assets = test::PATH_TEST_DATA / "transform";
        *filename_data = assets / "tmp_image2D.mrc";
        *value = 1.3f;
        *rotation = math::toRad(-45.f);
        *rotation_center = {0, 0};
        if (test_number == 0) {
            *filename_expected = assets / "tmp_rotate2D_test00.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_ZERO;
        } else if (test_number == 1) {
            *filename_expected = assets / "tmp_rotate2D_test01.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_VALUE;
        } else if (test_number == 2) {
            *filename_expected = assets / "tmp_rotate2D_test02.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_CLAMP;
        } else if (test_number == 3) {
            *filename_expected = assets / "tmp_rotate2D_test03.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_REFLECT;
        } else if (test_number == 4) {
            *filename_expected = assets / "tmp_rotate2D_test04.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_MIRROR;
        } else if (test_number == 5) {
            *filename_expected = assets / "tmp_rotate2D_test05.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_PERIODIC;
        } else if (test_number == 6) {
            *filename_expected = assets / "tmp_rotate2D_test06.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_ZERO;
        } else if (test_number == 7) {
            *filename_expected = assets / "tmp_rotate2D_test07.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_VALUE;
        } else if (test_number == 8) {
            *filename_expected = assets / "tmp_rotate2D_test08.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_CLAMP;
        } else if (test_number == 9) {
            *filename_expected = assets / "tmp_rotate2D_test09.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_REFLECT;
        } else if (test_number == 10) {
            *filename_expected = assets / "tmp_rotate2D_test10.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_MIRROR;
        } else if (test_number == 11) {
            *filename_expected = assets / "tmp_rotate2D_test11.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_PERIODIC;
        }
    }

    void getScale2DParams(int test_number, path_t* filename_data, path_t* filename_expected,
                          InterpMode* interp, BorderMode* border, float* value,
                          float2_t* scale_factor, float2_t* scale_center) {
        path_t assets = test::PATH_TEST_DATA / "transform";
        *filename_data = assets / "tmp_image2D.mrc";
        *value = 1.3f;
        *scale_factor = {0.6f, 0.7f};
        *scale_center = {128.f, 128.f};
        if (test_number == 0) {
            *filename_expected = assets / "tmp_scale2D_test00.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_ZERO;
        } else if (test_number == 1) {
            *filename_expected = assets / "tmp_scale2D_test01.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_VALUE;
        } else if (test_number == 2) {
            *filename_expected = assets / "tmp_scale2D_test02.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_CLAMP;
        } else if (test_number == 3) {
            *filename_expected = assets / "tmp_scale2D_test03.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_REFLECT;
        } else if (test_number == 4) {
            *filename_expected = assets / "tmp_scale2D_test04.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_MIRROR;
        } else if (test_number == 5) {
            *filename_expected = assets / "tmp_scale2D_test05.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_PERIODIC;
        } else if (test_number == 6) {
            *filename_expected = assets / "tmp_scale2D_test06.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_ZERO;
        } else if (test_number == 7) {
            *filename_expected = assets / "tmp_scale2D_test07.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_VALUE;
        } else if (test_number == 8) {
            *filename_expected = assets / "tmp_scale2D_test08.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_CLAMP;
        } else if (test_number == 9) {
            *filename_expected = assets / "tmp_scale2D_test09.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_REFLECT;
        } else if (test_number == 10) {
            *filename_expected = assets / "tmp_scale2D_test10.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_MIRROR;
        } else if (test_number == 11) {
            *filename_expected = assets / "tmp_scale2D_test11.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_PERIODIC;
        }
    }

    void getTranslate2DParams(int test_number, path_t* filename_data, path_t* filename_expected,
                              InterpMode* interp, BorderMode* border, float* value,
                              float2_t* shifts) {
        path_t assets = test::PATH_TEST_DATA / "transform";
        *filename_data = assets / "tmp_image2D.mrc";
        *value = 1.3f;
        *shifts = {10, -20.6};
        if (test_number == 0) {
            *filename_expected = assets / "tmp_translate2D_test00.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_ZERO;
        } else if (test_number == 1) {
            *filename_expected = assets / "tmp_translate2D_test01.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_VALUE;
        } else if (test_number == 2) {
            *filename_expected = assets / "tmp_translate2D_test02.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_CLAMP;
        } else if (test_number == 3) {
            *filename_expected = assets / "tmp_translate2D_test03.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_REFLECT;
        } else if (test_number == 4) {
            *filename_expected = assets / "tmp_translate2D_test04.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_MIRROR;
        } else if (test_number == 5) {
            *filename_expected = assets / "tmp_translate2D_test05.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_PERIODIC;
        } else if (test_number == 6) {
            *filename_expected = assets / "tmp_translate2D_test06.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_ZERO;
        } else if (test_number == 7) {
            *filename_expected = assets / "tmp_translate2D_test07.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_VALUE;
        } else if (test_number == 8) {
            *filename_expected = assets / "tmp_translate2D_test08.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_CLAMP;
        } else if (test_number == 9) {
            *filename_expected = assets / "tmp_translate2D_test09.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_REFLECT;
        } else if (test_number == 10) {
            *filename_expected = assets / "tmp_translate2D_test10.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_MIRROR;
        } else if (test_number == 11) {
            *filename_expected = assets / "tmp_translate2D_test11.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_PERIODIC;
        }
    }

    void getApply2DParams(int test_number, path_t* filename_data, path_t* filename_expected,
                          InterpMode* interp, BorderMode* border, float* value,
                          path_t* filename_matrix) {
        path_t assets = test::PATH_TEST_DATA / "transform";
        *filename_data = assets / "tmp_image2D.mrc";
        *value = 1.3f;
        if (test_number == 0) {
            *filename_expected = assets / "tmp_apply2D_test00.mrc";
            *filename_matrix = assets / "tmp_apply2D_test00_matrix33.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_ZERO;
        } else if (test_number == 1) {
            *filename_expected = assets / "tmp_apply2D_test01.mrc";
            *filename_matrix = assets / "tmp_apply2D_test01_matrix33.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_VALUE;
        } else if (test_number == 2) {
            *filename_expected = assets / "tmp_apply2D_test02.mrc";
            *filename_matrix = assets / "tmp_apply2D_test02_matrix33.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_CLAMP;
        } else if (test_number == 3) {
            *filename_expected = assets / "tmp_apply2D_test03.mrc";
            *filename_matrix = assets / "tmp_apply2D_test03_matrix33.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_REFLECT;
        } else if (test_number == 4) {
            *filename_expected = assets / "tmp_apply2D_test04.mrc";
            *filename_matrix = assets / "tmp_apply2D_test04_matrix33.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_MIRROR;
        } else if (test_number == 5) {
            *filename_expected = assets / "tmp_apply2D_test05.mrc";
            *filename_matrix = assets / "tmp_apply2D_test05_matrix33.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_PERIODIC;
        } else if (test_number == 6) {
            *filename_expected = assets / "tmp_apply2D_test06.mrc";
            *filename_matrix = assets / "tmp_apply2D_test06_matrix33.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_ZERO;
        } else if (test_number == 7) {
            *filename_expected = assets / "tmp_apply2D_test07.mrc";
            *filename_matrix = assets / "tmp_apply2D_test07_matrix33.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_VALUE;
        } else if (test_number == 8) {
            *filename_expected = assets / "tmp_apply2D_test08.mrc";
            *filename_matrix = assets / "tmp_apply2D_test08_matrix33.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_CLAMP;
        } else if (test_number == 9) {
            *filename_expected = assets / "tmp_apply2D_test09.mrc";
            *filename_matrix = assets / "tmp_apply2D_test09_matrix33.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_REFLECT;
        } else if (test_number == 10) {
            *filename_expected = assets / "tmp_apply2D_test10.mrc";
            *filename_matrix = assets / "tmp_apply2D_test10_matrix33.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_MIRROR;
        } else if (test_number == 11) {
            *filename_expected = assets / "tmp_apply2D_test11.mrc";
            *filename_matrix = assets / "tmp_apply2D_test11_matrix33.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_PERIODIC;
        }
    }

    void getCubic2D(int test_number, path_t* filename_data, path_t* filename_expected, path_t* filename_matrix,
                    InterpMode* interp, BorderMode* border) {
        path_t assets = test::PATH_TEST_DATA / "transform";
        *filename_data = assets / "apply2D_cubic_input.mrc";
        *filename_matrix = assets / "apply2D_cubic_matrix.mrc";
        if (test_number == 0) {
            *filename_expected = assets / "apply2D_cubic_test00.mrc";
            *interp = INTERP_CUBIC;
            *border = BORDER_ZERO;
        } else if (test_number == 1) {
            *filename_expected = assets / "apply2D_cubic_test01.mrc";
            *interp = INTERP_CUBIC_BSPLINE;
            *border = BORDER_ZERO;
        } else if (test_number == 2) {
            *filename_expected = assets / "apply2D_cubic_test02.mrc";
            *interp = INTERP_CUBIC;
            *border = BORDER_CLAMP;
        } else if (test_number == 3) {
            *filename_expected = assets / "apply2D_cubic_test03.mrc";
            *interp = INTERP_CUBIC_BSPLINE;
            *border = BORDER_CLAMP;
        } else if (test_number == 4) {
            *filename_expected = assets / "apply2D_cubic_test04.mrc";
            *interp = INTERP_CUBIC;
            *border = BORDER_MIRROR;
        } else if (test_number == 5) {
            *filename_expected = assets / "apply2D_cubic_test05.mrc";
            *interp = INTERP_CUBIC_BSPLINE;
            *border = BORDER_MIRROR;
        }
    }

    void getRotate3DParams(int test_number, path_t* filename_data, path_t* filename_expected,
                           InterpMode* interp, BorderMode* border, float* value,
                           float3_t* euler, float3_t* rotation_center) {
        path_t assets = test::PATH_TEST_DATA / "transform";
        *filename_data = assets / "tmp_image3D.mrc";
        *value = 1.3f;
        *euler = {math::toRad(-60.f), 0, math::toRad(20.f)};
        *rotation_center = {31.5, 31.5, 31.5};
        if (test_number == 0) {
            *filename_expected = assets / "tmp_rotate3D_test00.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_ZERO;
        } else if (test_number == 1) {
            *filename_expected = assets / "tmp_rotate3D_test01.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_VALUE;
        } else if (test_number == 2) {
            *filename_expected = assets / "tmp_rotate3D_test02.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_CLAMP;
        } else if (test_number == 3) {
            *filename_expected = assets / "tmp_rotate3D_test03.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_REFLECT;
        } else if (test_number == 4) {
            *filename_expected = assets / "tmp_rotate3D_test04.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_MIRROR;
        } else if (test_number == 5) {
            *filename_expected = assets / "tmp_rotate3D_test05.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_PERIODIC;
        } else if (test_number == 6) {
            *filename_expected = assets / "tmp_rotate3D_test06.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_ZERO;
        } else if (test_number == 7) {
            *filename_expected = assets / "tmp_rotate3D_test07.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_VALUE;
        } else if (test_number == 8) {
            *filename_expected = assets / "tmp_rotate3D_test08.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_CLAMP;
        } else if (test_number == 9) {
            *filename_expected = assets / "tmp_rotate3D_test09.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_REFLECT;
        } else if (test_number == 10) {
            *filename_expected = assets / "tmp_rotate3D_test10.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_MIRROR;
        } else if (test_number == 11) {
            *filename_expected = assets / "tmp_rotate3D_test11.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_PERIODIC;
        }
    }

    void getScale3DParams(int test_number, path_t* filename_data, path_t* filename_expected,
                          InterpMode* interp, BorderMode* border, float* value,
                          float3_t* scale_factor, float3_t* scale_center) {
        path_t assets = test::PATH_TEST_DATA / "transform";
        *filename_data = assets / "tmp_image3D.mrc";
        *value = 1.3f;
        *scale_factor = {0.8f, 0.6f, 1.1f};
        *scale_center = {31.5f, 31.5f, 31.5f};
        if (test_number == 0) {
            *filename_expected = assets / "tmp_scale3D_test00.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_ZERO;
        } else if (test_number == 1) {
            *filename_expected = assets / "tmp_scale3D_test01.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_VALUE;
        } else if (test_number == 2) {
            *filename_expected = assets / "tmp_scale3D_test02.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_CLAMP;
        } else if (test_number == 3) {
            *filename_expected = assets / "tmp_scale3D_test03.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_REFLECT;
        } else if (test_number == 4) {
            *filename_expected = assets / "tmp_scale3D_test04.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_MIRROR;
        } else if (test_number == 5) {
            *filename_expected = assets / "tmp_scale3D_test05.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_PERIODIC;
        } else if (test_number == 6) {
            *filename_expected = assets / "tmp_scale3D_test06.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_ZERO;
        } else if (test_number == 7) {
            *filename_expected = assets / "tmp_scale3D_test07.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_VALUE;
        } else if (test_number == 8) {
            *filename_expected = assets / "tmp_scale3D_test08.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_CLAMP;
        } else if (test_number == 9) {
            *filename_expected = assets / "tmp_scale3D_test09.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_REFLECT;
        } else if (test_number == 10) {
            *filename_expected = assets / "tmp_scale3D_test10.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_MIRROR;
        } else if (test_number == 11) {
            *filename_expected = assets / "tmp_scale3D_test11.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_PERIODIC;
        }
    }

    void getTranslate3DParams(int test_number, path_t* filename_data, path_t* filename_expected,
                              InterpMode* interp, BorderMode* border, float* value,
                              float3_t* shifts) {
        path_t assets = test::PATH_TEST_DATA / "transform";
        *filename_data = assets / "tmp_image3D.mrc";
        *value = 1.3f;
        *shifts = {-5.4, 10, -20.6};
        if (test_number == 0) {
            *filename_expected = assets / "tmp_translate3D_test00.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_ZERO;
        } else if (test_number == 1) {
            *filename_expected = assets / "tmp_translate3D_test01.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_VALUE;
        } else if (test_number == 2) {
            *filename_expected = assets / "tmp_translate3D_test02.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_CLAMP;
        } else if (test_number == 3) {
            *filename_expected = assets / "tmp_translate3D_test03.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_REFLECT;
        } else if (test_number == 4) {
            *filename_expected = assets / "tmp_translate3D_test04.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_MIRROR;
        } else if (test_number == 5) {
            *filename_expected = assets / "tmp_translate3D_test05.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_PERIODIC;
        } else if (test_number == 6) {
            *filename_expected = assets / "tmp_translate3D_test06.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_ZERO;
        } else if (test_number == 7) {
            *filename_expected = assets / "tmp_translate3D_test07.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_VALUE;
        } else if (test_number == 8) {
            *filename_expected = assets / "tmp_translate3D_test08.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_CLAMP;
        } else if (test_number == 9) {
            *filename_expected = assets / "tmp_translate3D_test09.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_REFLECT;
        } else if (test_number == 10) {
            *filename_expected = assets / "tmp_translate3D_test10.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_MIRROR;
        } else if (test_number == 11) {
            *filename_expected = assets / "tmp_translate3D_test11.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_PERIODIC;
        }
    }

    void getApply3DParams(int test_number, path_t* filename_data, path_t* filename_expected,
                          InterpMode* interp, BorderMode* border, float* value,
                          path_t* filename_matrix) {
        path_t assets = test::PATH_TEST_DATA / "transform";
        *filename_data = assets / "tmp_image3D.mrc";
        *value = 1.3f;
        if (test_number == 0) {
            *filename_expected = assets / "tmp_apply3D_test00.mrc";
            *filename_matrix = assets / "tmp_apply3D_test00_matrix44.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_ZERO;
        } else if (test_number == 1) {
            *filename_expected = assets / "tmp_apply3D_test01.mrc";
            *filename_matrix = assets / "tmp_apply3D_test01_matrix44.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_VALUE;
        } else if (test_number == 2) {
            *filename_expected = assets / "tmp_apply3D_test02.mrc";
            *filename_matrix = assets / "tmp_apply3D_test02_matrix44.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_CLAMP;
        } else if (test_number == 3) {
            *filename_expected = assets / "tmp_apply3D_test03.mrc";
            *filename_matrix = assets / "tmp_apply3D_test03_matrix44.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_REFLECT;
        } else if (test_number == 4) {
            *filename_expected = assets / "tmp_apply3D_test04.mrc";
            *filename_matrix = assets / "tmp_apply3D_test04_matrix44.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_MIRROR;
        } else if (test_number == 5) {
            *filename_expected = assets / "tmp_apply3D_test05.mrc";
            *filename_matrix = assets / "tmp_apply3D_test05_matrix44.mrc";
            *interp = INTERP_NEAREST;
            *border = BORDER_PERIODIC;
        } else if (test_number == 6) {
            *filename_expected = assets / "tmp_apply3D_test06.mrc";
            *filename_matrix = assets / "tmp_apply3D_test06_matrix44.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_ZERO;
        } else if (test_number == 7) {
            *filename_expected = assets / "tmp_apply3D_test07.mrc";
            *filename_matrix = assets / "tmp_apply3D_test07_matrix44.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_VALUE;
        } else if (test_number == 8) {
            *filename_expected = assets / "tmp_apply3D_test08.mrc";
            *filename_matrix = assets / "tmp_apply3D_test08_matrix44.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_CLAMP;
        } else if (test_number == 9) {
            *filename_expected = assets / "tmp_apply3D_test09.mrc";
            *filename_matrix = assets / "tmp_apply3D_test09_matrix44.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_REFLECT;
        } else if (test_number == 10) {
            *filename_expected = assets / "tmp_apply3D_test10.mrc";
            *filename_matrix = assets / "tmp_apply3D_test10_matrix44.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_MIRROR;
        } else if (test_number == 11) {
            *filename_expected = assets / "tmp_apply3D_test11.mrc";
            *filename_matrix = assets / "tmp_apply3D_test11_matrix44.mrc";
            *interp = INTERP_LINEAR;
            *border = BORDER_PERIODIC;
        }
    }

    void getCubic3D(int test_number, path_t* filename_data, path_t* filename_expected, path_t* filename_matrix,
                    InterpMode* interp, BorderMode* border) {
        path_t assets = test::PATH_TEST_DATA / "transform";
        *filename_data = assets / "apply3D_cubic_input.mrc";
        *filename_matrix = assets / "apply3D_cubic_matrix.mrc";
        if (test_number == 0) {
            *filename_expected = assets / "apply3D_cubic_test00.mrc";
            *interp = INTERP_CUBIC;
            *border = BORDER_ZERO;
        } else if (test_number == 1) {
            *filename_expected = assets / "apply3D_cubic_test01.mrc";
            *interp = INTERP_CUBIC_BSPLINE;
            *border = BORDER_ZERO;
        } else if (test_number == 2) {
            *filename_expected = assets / "apply3D_cubic_test02.mrc";
            *interp = INTERP_CUBIC;
            *border = BORDER_CLAMP;
        } else if (test_number == 3) {
            *filename_expected = assets / "apply3D_cubic_test03.mrc";
            *interp = INTERP_CUBIC_BSPLINE;
            *border = BORDER_CLAMP;
        } else if (test_number == 4) {
            *filename_expected = assets / "apply3D_cubic_test04.mrc";
            *interp = INTERP_CUBIC;
            *border = BORDER_MIRROR;
        } else if (test_number == 5) {
            *filename_expected = assets / "apply3D_cubic_test05.mrc";
            *interp = INTERP_CUBIC_BSPLINE;
            *border = BORDER_MIRROR;
        }
    }
}
