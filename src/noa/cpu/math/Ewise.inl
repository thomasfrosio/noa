#pragma once

#ifndef NOA_EWISE_INL_
#error "This is an internal header; it should not be included."
#endif

namespace noa::cpu::math {
    template<typename T, typename U, typename UnaryOp>
    void ewise(const T* inputs, size3_t input_pitch, U* outputs, size3_t output_pitch,
               size3_t shape, size_t batches, UnaryOp unary_op, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        stream.enqueue([=]() {
            const size_t iffset = elements(input_pitch);
            const size_t offset = elements(output_pitch);
            for (size_t batch = 0; batch < batches; ++batch)
                for (size_t z = 0; z < shape.z; ++z)
                    for (size_t y = 0; y < shape.y; ++y)
                        for (size_t x = 0; x < shape.x; ++x)
                            outputs[batch * offset + index(x, y, z, output_pitch)] =
                                    unary_op(inputs[batch * iffset + index(x, y, z, input_pitch)]);
        });
    }

    template<typename T, typename U, typename V, typename BinaryOp>
    void ewise(const T* inputs, size3_t input_pitch, U values,
               V* outputs, size3_t output_pitch,
               size3_t shape, size_t batches, BinaryOp binary_op, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        stream.enqueue([=]() {
            const size_t iffset = elements(input_pitch);
            const size_t offset = elements(output_pitch);
            if constexpr (std::is_pointer_v<U>) {
                for (size_t batch = 0; batch < batches; ++batch)
                    for (size_t z = 0; z < shape.z; ++z)
                        for (size_t y = 0; y < shape.y; ++y)
                            for (size_t x = 0; x < shape.x; ++x)
                                outputs[offset * batch + index(x, y, z, output_pitch)] =
                                        binary_op(inputs[iffset * batch + index(x, y, z, input_pitch)], values[batch]);
            } else {
                for (size_t batch = 0; batch < batches; ++batch)
                    for (size_t z = 0; z < shape.z; ++z)
                        for (size_t y = 0; y < shape.y; ++y)
                            for (size_t x = 0; x < shape.x; ++x)
                                outputs[offset * batch + index(x, y, z, output_pitch)] =
                                        binary_op(inputs[iffset * batch + index(x, y, z, input_pitch)], values);
            }
        });
    }

    template<typename T, typename U, typename V, typename BinaryOp>
    void ewise(const T* inputs, size3_t input_pitch, const U* arrays, size3_t array_pitch,
               V* output, size3_t output_pitch,
               size3_t shape, size_t batches, BinaryOp binary_op, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        stream.enqueue([=]() {
            const size_t iffset = elements(input_pitch);
            const size_t affset = elements(array_pitch);
            const size_t offset = elements(output_pitch);
            for (size_t batch = 0; batch < batches; ++batch)
                for (size_t z = 0; z < shape.z; ++z)
                    for (size_t y = 0; y < shape.y; ++y)
                        for (size_t x = 0; x < shape.x; ++x)
                            output[offset * batch + index(x, y, z, output_pitch)] =
                                    binary_op(inputs[iffset * batch + index(x, y, z, input_pitch)],
                                              arrays[affset * batch + index(x, y, z, array_pitch)]);
        });
    }

    template<typename T, typename U, typename V, typename TrinaryOp>
    void ewise(const T* inputs, size3_t input_pitch, U v1, U v2,
               V* outputs, size3_t output_pitch,
               size3_t shape, size_t batches, TrinaryOp trinary_op, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        stream.enqueue([=]() {
            const size_t iffset = elements(input_pitch);
            const size_t offset = elements(output_pitch);
            if constexpr (std::is_pointer_v<U>) {
                for (size_t batch = 0; batch < batches; ++batch)
                    for (size_t z = 0; z < shape.z; ++z)
                        for (size_t y = 0; y < shape.y; ++y)
                            for (size_t x = 0; x < shape.x; ++x)
                                outputs[offset * batch + index(x, y, z, output_pitch)] =
                                        trinary_op(inputs[iffset * batch + index(x, y, z, input_pitch)],
                                                   v1[batch], v2[batch]);
            } else {
                for (size_t batch = 0; batch < batches; ++batch)
                    for (size_t z = 0; z < shape.z; ++z)
                        for (size_t y = 0; y < shape.y; ++y)
                            for (size_t x = 0; x < shape.x; ++x)
                                outputs[offset * batch + index(x, y, z, output_pitch)] =
                                        trinary_op(inputs[iffset * batch + index(x, y, z, input_pitch)],
                                                   v1, v2);
            }
        });
    }

    template<typename T, typename U, typename V, typename W, typename TrinaryOp>
    void ewise(const T* inputs, size3_t input_pitch, const U* a1, size3_t a1_pitch, const V* a2, size3_t a2_pitch,
               W* output, size3_t output_pitch,
               size3_t shape, size_t batches, TrinaryOp trinary_op, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        stream.enqueue([=]() {
            const size_t iffset = elements(input_pitch);
            const size_t a1_offset = elements(a1_pitch);
            const size_t a2_offset = elements(a2_pitch);
            const size_t offset = elements(output_pitch);
            for (size_t batch = 0; batch < batches; ++batch)
                for (size_t z = 0; z < shape.z; ++z)
                    for (size_t y = 0; y < shape.y; ++y)
                        for (size_t x = 0; x < shape.x; ++x)
                            output[offset * batch + index(x, y, z, output_pitch)] =
                                    trinary_op(inputs[iffset * batch + index(x, y, z, input_pitch)],
                                               a1[a1_offset * batch + index(x, y, z, a1_pitch)],
                                               a2[a2_offset * batch + index(x, y, z, a2_pitch)]);
        });
    }
}
