#include "noa/cpu/memory/Transpose.h"

namespace noa::cpu::memory::details {
    template<typename T>
    void transpose021(const T* inputs, T* outputs, size3_t shape, uint batches) {
        size_t elements = getElements(shape);
        for (uint batch = 0; batch < batches; ++batch) {
            const T* input = inputs + elements * batch;
            T* output = outputs + elements * batch;

            size3_t offset;
            for (size_t z = 0; z < shape.z; ++z) {
                offset.z = z * shape.x; // z becomes y
                for (size_t y = 0; y < shape.y; ++y) {
                    offset.y = y * shape.x * shape.z; // y becomes z
                    for (size_t x = 0; x < shape.x; ++x, ++input) {
                        offset.x = x; // x stays x
                        output[math::sum(offset)] = *input;
                    }
                }
            }
        }
    }

    template<typename T>
    void transpose102(const T* inputs, T* outputs, size3_t shape, uint batches) {
        size_t elements = getElements(shape);
        for (uint batch = 0; batch < batches; ++batch) {
            const T* input = inputs + elements * batch;
            T* output = outputs + elements * batch;

            size3_t offset;
            for (size_t z = 0; z < shape.z; ++z) {
                offset.z = z * shape.x * shape.y; // z stays z
                for (size_t y = 0; y < shape.y; ++y) {
                    offset.y = y; // y becomes x
                    for (size_t x = 0; x < shape.x; ++x, ++input) {
                        offset.x = x * shape.y; // x becomes y
                        output[math::sum(offset)] = *input;
                    }
                }
            }
        }
    }

    template<typename T>
    void transpose120(const T* inputs, T* outputs, size3_t shape, uint batches) {
        size_t elements = getElements(shape);
        for (uint batch = 0; batch < batches; ++batch) {
            const T* input = inputs + elements * batch;
            T* output = outputs + elements * batch;

            size3_t offset;
            for (size_t z = 0; z < shape.z; ++z) {
                offset.z = z * shape.y; // z becomes y
                for (size_t y = 0; y < shape.y; ++y) {
                    offset.y = y; // y becomes x
                    for (size_t x = 0; x < shape.x; ++x, ++input) {
                        offset.x = x * shape.y * shape.z; // x becomes z
                        output[math::sum(offset)] = *input;
                    }
                }
            }
        }
    }

    template<typename T>
    void transpose201(const T* inputs, T* outputs, size3_t shape, uint batches) {
        size_t elements = getElements(shape);
        for (uint batch = 0; batch < batches; ++batch) {
            const T* input = inputs + elements * batch;
            T* output = outputs + elements * batch;

            size3_t offset;
            for (size_t z = 0; z < shape.z; ++z) {
                offset.z = z; // z becomes x
                for (size_t y = 0; y < shape.y; ++y) {
                    offset.y = y * shape.z * shape.x; // y becomes z
                    for (size_t x = 0; x < shape.x; ++x, ++input) {
                        offset.x = x * shape.z; // x becomes y
                        output[math::sum(offset)] = *input;
                    }
                }
            }
        }
    }

    template<typename T>
    void transpose210(const T* inputs, T* outputs, size3_t shape, uint batches) {
        size_t elements = getElements(shape);
        for (uint batch = 0; batch < batches; ++batch) {
            const T* input = inputs + elements * batch;
            T* output = outputs + elements * batch;

            size3_t offset;
            for (size_t z = 0; z < shape.z; ++z) {
                offset.z = z; // z becomes x
                for (size_t y = 0; y < shape.y; ++y) {
                    offset.y = y * shape.z; // y stays y
                    for (size_t x = 0; x < shape.x; ++x, ++input) {
                        offset.x = x * shape.z * shape.y; // x becomes z
                        output[math::sum(offset)] = *input;
                    }
                }
            }
        }
    }
}

namespace noa::cpu::memory::details::inplace {
    template<typename T>
    void transpose021(T* outputs, size3_t shape, uint batches) {
        if (shape.y != shape.z)
            NOA_THROW("For a \"021\" in-place permutation, shape[1] should be equal to shape[2]. Got {}", shape);

        size_t elements = getElements(shape);
        for (uint batch = 0; batch < batches; ++batch) {
            T* output = outputs + elements * batch;
            for (size_t x = 0; x < shape.x; ++x) {

                // Transpose YZ: swap bottom triangle with upper triangle.
                for (size_t z = 0; z < shape.z; ++z) {
                    for (size_t y = z + 1; y < shape.y; ++y) {
                        size_t src_idx = getIdx(x, y, z, shape.x, shape.y);
                        size_t dst_idx = getIdx(x, z, y, shape.x, shape.y);
                        std::swap(output[src_idx], output[dst_idx]);
                    }
                }
            }
        }
    }

    template<typename T>
    void transpose102(T* outputs, size3_t shape, uint batches) {
        if (shape.x != shape.y)
            NOA_THROW("For a \"102\" in-place permutation, shape[0] should be equal to shape[1]. Got {}", shape);

        size_t elements = getElements(shape);
        for (uint batch = 0; batch < batches; ++batch) {
            T* output = outputs + elements * batch;
            for (size_t z = 0; z < shape.z; ++z) {

                // Transpose XY: swap bottom triangle with upper triangle.
                for (size_t y = 0; y < shape.y; ++y) {
                    for (size_t x = y + 1; x < shape.x; ++x) {
                        size_t src_idx = getIdx(x, y, z, shape.x, shape.y);
                        size_t dst_idx = getIdx(y, x, z, shape.x, shape.y);
                        std::swap(output[src_idx], output[dst_idx]);
                    }
                }
            }
        }
    }

    template<typename T>
    void transpose210(T* outputs, size3_t shape, uint batches) {
        if (shape.x != shape.z)
            NOA_THROW("For a \"210\" in-place permutation, shape[0] should be equal to shape[2]. Got {}", shape);

        size_t elements = getElements(shape);
        for (uint batch = 0; batch < batches; ++batch) {
            T* output = outputs + elements * batch;
            for (size_t y = 0; y < shape.y; ++y) {

                // Transpose XZ: swap bottom triangle with upper triangle.
                for (size_t z = 0; z < shape.z; ++z) {
                    for (size_t x = z + 1; x < shape.x; ++x) {
                        size_t src_idx = getIdx(x, y, z, shape.x, shape.y);
                        size_t dst_idx = getIdx(z, y, x, shape.x, shape.y);
                        std::swap(output[src_idx], output[dst_idx]);
                    }
                }
            }
        }
    }
}

#define NOA_INSTANTIATE_TRANSPOSE_(T)                                                    \
template void noa::cpu::memory::details::transpose021<T>(const T*, T*, size3_t, uint);   \
template void noa::cpu::memory::details::transpose102<T>(const T*, T*, size3_t, uint);   \
template void noa::cpu::memory::details::transpose120<T>(const T*, T*, size3_t, uint);   \
template void noa::cpu::memory::details::transpose201<T>(const T*, T*, size3_t, uint);   \
template void noa::cpu::memory::details::transpose210<T>(const T*, T*, size3_t, uint);   \
template void noa::cpu::memory::details::inplace::transpose021<T>(T*, size3_t, uint);    \
template void noa::cpu::memory::details::inplace::transpose102<T>(T*, size3_t, uint);    \
template void noa::cpu::memory::details::inplace::transpose210<T>(T*, size3_t, uint)

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
NOA_INSTANTIATE_TRANSPOSE_(float);
NOA_INSTANTIATE_TRANSPOSE_(double);
