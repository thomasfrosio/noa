#include "noa/common/Assert.h"
#include "noa/cpu/memory/Transpose.h"

namespace noa::cpu::memory::details {
    template<typename T>
    void transpose021(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch,
                      size3_t shape, size_t batches) {
        NOA_ASSERT(inputs != outputs);
        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + elements(input_pitch) * batch;
            T* output = outputs + elements(output_pitch) * batch;

            size3_t offset;
            for (size_t z = 0; z < shape.z; ++z) {
                offset.z = z * output_pitch.x; // z becomes y
                for (size_t y = 0; y < shape.y; ++y) {
                    offset.y = y * output_pitch.x * output_pitch.z; // y becomes z
                    for (size_t x = 0; x < shape.x; ++x) {
                        offset.x = x; // x stays x
                        output[math::sum(offset)] = input[index(x, y, z, input_pitch)];
                    }
                }
            }
        }
    }

    template<typename T>
    void transpose102(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch,
                      size3_t shape, size_t batches) {
        NOA_ASSERT(inputs != outputs);
        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + elements(input_pitch) * batch;
            T* output = outputs + elements(output_pitch) * batch;

            size3_t offset;
            for (size_t z = 0; z < shape.z; ++z) {
                offset.z = z * output_pitch.x * output_pitch.y; // z stays z
                for (size_t y = 0; y < shape.y; ++y) {
                    offset.y = y; // y becomes x
                    for (size_t x = 0; x < shape.x; ++x) {
                        offset.x = x * output_pitch.y; // x becomes y
                        output[math::sum(offset)] = input[index(x, y, z, input_pitch)];
                    }
                }
            }
        }
    }

    template<typename T>
    void transpose120(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch,
                      size3_t shape, size_t batches) {
        NOA_ASSERT(inputs != outputs);
        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + elements(input_pitch) * batch;
            T* output = outputs + elements(output_pitch) * batch;

            size3_t offset;
            for (size_t z = 0; z < shape.z; ++z) {
                offset.z = z * output_pitch.y; // z becomes y
                for (size_t y = 0; y < shape.y; ++y) {
                    offset.y = y; // y becomes x
                    for (size_t x = 0; x < shape.x; ++x) {
                        offset.x = x * output_pitch.y * output_pitch.z; // x becomes z
                        output[math::sum(offset)] = input[index(x, y, z, input_pitch)];
                    }
                }
            }
        }
    }

    template<typename T>
    void transpose201(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch,
                      size3_t shape, size_t batches) {
        NOA_ASSERT(inputs != outputs);
        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + elements(input_pitch) * batch;
            T* output = outputs + elements(output_pitch) * batch;

            size3_t offset;
            for (size_t z = 0; z < shape.z; ++z) {
                offset.z = z; // z becomes x
                for (size_t y = 0; y < shape.y; ++y) {
                    offset.y = y * output_pitch.z * output_pitch.x; // y becomes z
                    for (size_t x = 0; x < shape.x; ++x) {
                        offset.x = x * output_pitch.z; // x becomes y
                        output[math::sum(offset)] = input[index(x, y, z, input_pitch)];
                    }
                }
            }
        }
    }

    template<typename T>
    void transpose210(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch,
                      size3_t shape, size_t batches) {
        NOA_ASSERT(inputs != outputs);
        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + elements(input_pitch) * batch;
            T* output = outputs + elements(output_pitch) * batch;

            size3_t offset;
            for (size_t z = 0; z < shape.z; ++z) {
                offset.z = z; // z becomes x
                for (size_t y = 0; y < shape.y; ++y) {
                    offset.y = y * output_pitch.z; // y stays y
                    for (size_t x = 0; x < shape.x; ++x) {
                        offset.x = x * output_pitch.z * output_pitch.y; // x becomes z
                        output[math::sum(offset)] = input[index(x, y, z, input_pitch)];
                    }
                }
            }
        }
    }
}

namespace noa::cpu::memory::details::inplace {
    template<typename T>
    void transpose021(T* outputs, size3_t pitch, size3_t shape, size_t batches) {
        if (shape.y != shape.z)
            NOA_THROW("For a \"021\" in-place permutation, shape[1] should be equal to shape[2]. Got {}", shape);

        for (size_t batch = 0; batch < batches; ++batch) {
            T* output = outputs + elements(pitch) * batch;
            for (size_t x = 0; x < shape.x; ++x) {

                // Transpose YZ: swap bottom triangle with upper triangle.
                for (size_t z = 0; z < shape.z; ++z)
                    for (size_t y = z + 1; y < shape.y; ++y)
                        std::swap(output[index(x, y, z, pitch)],
                                  output[index(x, z, y, pitch)]);
            }
        }
    }

    template<typename T>
    void transpose102(T* outputs, size3_t pitch, size3_t shape, size_t batches) {
        if (shape.x != shape.y)
            NOA_THROW("For a \"102\" in-place permutation, shape[0] should be equal to shape[1]. Got {}", shape);

        for (size_t batch = 0; batch < batches; ++batch) {
            T* output = outputs + elements(pitch) * batch;
            for (size_t z = 0; z < shape.z; ++z) {

                // Transpose XY: swap bottom triangle with upper triangle.
                for (size_t y = 0; y < shape.y; ++y)
                    for (size_t x = y + 1; x < shape.x; ++x)
                        std::swap(output[index(x, y, z, pitch)],
                                  output[index(y, x, z, pitch)]);
            }
        }
    }

    template<typename T>
    void transpose210(T* outputs, size3_t pitch, size3_t shape, size_t batches) {
        if (shape.x != shape.z)
            NOA_THROW("For a \"210\" in-place permutation, shape[0] should be equal to shape[2]. Got {}", shape);

        for (size_t batch = 0; batch < batches; ++batch) {
            T* output = outputs + elements(pitch) * batch;
            for (size_t y = 0; y < shape.y; ++y) {

                // Transpose XZ: swap bottom triangle with upper triangle.
                for (size_t z = 0; z < shape.z; ++z)
                    for (size_t x = z + 1; x < shape.x; ++x)
                        std::swap(output[index(x, y, z, pitch)],
                                  output[index(z, y, x, pitch)]);
            }
        }
    }
}

namespace noa::cpu::memory::details {
    #define NOA_INSTANTIATE_TRANSPOSE_(T)                                           \
    template void transpose021<T>(const T*, size3_t, T*, size3_t, size3_t, size_t); \
    template void transpose102<T>(const T*, size3_t, T*, size3_t, size3_t, size_t); \
    template void transpose120<T>(const T*, size3_t, T*, size3_t, size3_t, size_t); \
    template void transpose201<T>(const T*, size3_t, T*, size3_t, size3_t, size_t); \
    template void transpose210<T>(const T*, size3_t, T*, size3_t, size3_t, size_t); \
    template void inplace::transpose021<T>(T*, size3_t, size3_t, size_t);           \
    template void inplace::transpose102<T>(T*, size3_t, size3_t, size_t);           \
    template void inplace::transpose210<T>(T*, size3_t, size3_t, size_t)

    NOA_INSTANTIATE_TRANSPOSE_(unsigned char);
    NOA_INSTANTIATE_TRANSPOSE_(unsigned short);
    NOA_INSTANTIATE_TRANSPOSE_(unsigned int);
    NOA_INSTANTIATE_TRANSPOSE_(unsigned long);
    NOA_INSTANTIATE_TRANSPOSE_(unsigned long long);
    NOA_INSTANTIATE_TRANSPOSE_(char);
    NOA_INSTANTIATE_TRANSPOSE_(short);
    NOA_INSTANTIATE_TRANSPOSE_(int);
    NOA_INSTANTIATE_TRANSPOSE_(long);
    NOA_INSTANTIATE_TRANSPOSE_(long long);
    NOA_INSTANTIATE_TRANSPOSE_(half_t);
    NOA_INSTANTIATE_TRANSPOSE_(float);
    NOA_INSTANTIATE_TRANSPOSE_(double);
    NOA_INSTANTIATE_TRANSPOSE_(chalf_t);
    NOA_INSTANTIATE_TRANSPOSE_(cfloat_t);
    NOA_INSTANTIATE_TRANSPOSE_(cdouble_t);
}
