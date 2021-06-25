// Assets from noa-data.

#pragma once
#include <noa/Types.h>

// memory::resize(), extract(), insert()
namespace test::assets::memory {
    using namespace noa;

    void initResizeInput(int test_number, float* input, size3_t shape, uint batches);
    void initResizeOutput(float* input, size3_t shape, uint batches); // test 19 and 20
    void getResizeParams(int test_number, path_t* filename, uint* batches, size3_t* i_shape, size3_t* o_shape,
                         int3_t* border_left, int3_t* border_right, BorderMode* mode, float* value);

    void initExtractInput(float* input, size_t elements);
    void initInsertOutput(float* output, size_t elements);
    path_t getExtractFilename(int test_number, uint subregion_idx);
    path_t getInsertFilename(int test_number);
    void getExtractParams(int test_number, size3_t* i_shape,
                          size3_t* sub_shape, size3_t* sub_centers, uint* sub_count,
                          BorderMode* mode, float* value);
}

// fourier::lowpass(), highpass(), bandpass()
namespace test::assets::fourier {
    using namespace noa;

    void getLowpassParams(int test_number, path_t* filename, size3_t* shape, float* cutoff, float* width);
    void getHighpassParams(int test_number, path_t* filename, size3_t* shape, float* cutoff, float* width);
    void getBandpassParams(int test_number, path_t* filename, size3_t* shape,
                           float* cutoff1, float* cutoff2, float* width1, float* width2);
}

// mask::sphere(), cylinder(), rectangle()
namespace test::assets::mask {
    using namespace noa;

    void getSphereParams(int test_number, path_t* filename, size3_t* shape,
                         float3_t* shifts, float* radius, float* taper);
    void getCylinderParams(int test_number, path_t* filename, size3_t* shape,
                           float3_t* shifts, float* radius_xy, float* radius_z, float* taper);
    void getRectangleParams(int test_number, path_t* filename, size3_t* shape,
                            float3_t* shifts, float3_t* radius, float* taper);
}

namespace test::assets::filter {
    using namespace noa;

    void getMedianData(int test_number, path_t* filename);
    void getMedianParams(int test_number, path_t* filename, size3_t* shape, BorderMode* mode, uint* window);

    void getConvData(int test_number, path_t* filename);
    void getConvFilter(int test_number, path_t* filename);
    void getConvParams(int test_number, path_t* filename, size3_t* shape, uint3_t* filter_size);
}
