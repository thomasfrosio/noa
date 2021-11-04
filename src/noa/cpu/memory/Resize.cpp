#include "noa/common/Assert.h"
#include "noa/common/Exception.h"
#include "noa/common/Math.h"
#include "noa/common/Profiler.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/Resize.h"

namespace {
    using namespace ::noa;

    // Sets the elements within the padding to a given value.
    template<typename T>
    void applyBorderValue_(T* outputs, size3_t shape, size_t elements,
                           int3_t pad_left, int3_t pad_right, T value, size_t batches) {
        const int3_t int_shape(shape);
        const int3_t valid_end = int_shape - pad_right;

        for (size_t batch = 0; batch < batches; ++batch) {
            T* output = outputs + batch * elements;
            for (int z = 0; z < int_shape.z; ++z) {
                const bool skip_z = z >= pad_left.z && z < valid_end.z;
                for (int y = 0; y < int_shape.y; ++y) {
                    const bool skip_y = y >= pad_left.y && y < valid_end.y;
                    for (int x = 0; x < int_shape.x; ++x) {
                        const bool skip_x = x >= pad_left.x && x < valid_end.x;
                        if (skip_x && skip_y && skip_z)
                            continue;
                        output[index(x, y, z, shape)] = value;
                    }
                }
            }
        }
    }

    template<BorderMode MODE, typename T>
    void applyBorder_(const T* inputs, size3_t input_shape, size_t input_elements,
                      T* outputs, size3_t output_shape, size_t output_elements,
                      int3_t pad_left, int3_t pad_right, int3_t crop_left, size_t batches) {
        const int3_t int_input_shape(input_shape);
        const int3_t int_output_shape(output_shape);
        const int3_t valid_end = int_output_shape - pad_right;

        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + batch * input_elements;
            T* output = outputs + batch * output_elements;

            for (int o_z = 0; o_z < int_output_shape.z; ++o_z) {
                const bool skip_z = o_z >= pad_left.z && o_z < valid_end.z;
                const int i_z = getBorderIndex<MODE>(o_z - pad_left.z + crop_left.z, int_input_shape.z);

                for (int o_y = 0; o_y < int_output_shape.y; ++o_y) {
                    const bool skip_y = o_y >= pad_left.y && o_y < valid_end.y;
                    const int i_y = getBorderIndex<MODE>(o_y - pad_left.y + crop_left.y, int_input_shape.y);

                    for (int o_x = 0; o_x < int_output_shape.x; ++o_x) {
                        const bool skip_x = o_x >= pad_left.x && o_x < valid_end.x;
                        if (skip_x && skip_y && skip_z)
                            continue;
                        const int i_x = getBorderIndex<MODE>(o_x - pad_left.x + crop_left.x, int_input_shape.x);

                        output[index(o_x, o_y, o_z, output_shape)] =
                                input[index(i_x, i_y, i_z, input_shape)];
                    }
                }
            }
        }
    }
}

namespace noa::cpu::memory {
    template<typename T>
    void resize(const T* inputs, size3_t input_shape, int3_t border_left, int3_t border_right,
                T* outputs, BorderMode mode, T border_value, size_t batches) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);

        if (all(border_left == 0) && all(border_right == 0))
            return copy(inputs, outputs, elements(input_shape) * batches);

        const size3_t output_shape(int3_t(input_shape) + border_left + border_right); // assumed to be > 0
        const size_t input_elements = elements(input_shape);
        const size_t output_elements = elements(output_shape);
        const int3_t crop_left(math::min(border_left, 0) * -1);
        const int3_t crop_right(math::min(border_right, 0) * -1);
        const int3_t pad_left(math::max(border_left, 0));
        const int3_t pad_right(math::max(border_right, 0));

        // Copy the valid elements in the inputs into the outputs.
        const int3_t valid_end(int3_t(input_shape) - crop_right);
        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + batch * input_elements;
            T* output = outputs + batch * output_elements;

            for (int i_z = crop_left.z; i_z < valid_end.z; ++i_z) {
                int o_z = i_z + border_left.z; // positive offset for padding or remove the cropped elements
                for (int i_y = crop_left.y; i_y < valid_end.y; ++i_y) {
                    int o_y = i_y + border_left.y;

                    // Offset to the current row.
                    copy(input + index(crop_left.x, i_y, i_z, input_shape),
                         output + index(pad_left.x, o_y, o_z, output_shape),
                         static_cast<uint>(valid_end.x - crop_left.x));
                }
            }
        }

        // Shortcut: if there's nothing to pad, we are done here.
        if (mode == BORDER_NOTHING || (all(pad_left == 0) && all(pad_right == 0)))
            return;

        // Set the padded elements to the correct values.
        switch (mode) {
            case BORDER_ZERO:
                applyBorderValue_(outputs, output_shape, output_elements, pad_left, pad_right, T{0}, batches);
                break;
            case BORDER_VALUE:
                applyBorderValue_(outputs, output_shape, output_elements, pad_left, pad_right, border_value, batches);
                break;
            case BORDER_CLAMP:
                applyBorder_<BORDER_CLAMP>(inputs, input_shape, input_elements,
                                           outputs, output_shape, output_elements,
                                           pad_left, pad_right, crop_left, batches);
                break;
            case BORDER_PERIODIC:
                applyBorder_<BORDER_PERIODIC>(inputs, input_shape, input_elements,
                                              outputs, output_shape, output_elements,
                                              pad_left, pad_right, crop_left, batches);
                break;
            case BORDER_REFLECT:
                applyBorder_<BORDER_REFLECT>(inputs, input_shape, input_elements,
                                             outputs, output_shape, output_elements,
                                             pad_left, pad_right, crop_left, batches);
                break;
            case BORDER_MIRROR:
                applyBorder_<BORDER_MIRROR>(inputs, input_shape, input_elements,
                                            outputs, output_shape, output_elements,
                                            pad_left, pad_right, crop_left, batches);
                break;
            default:
                NOA_THROW("BorderMode not recognized. Got: {}", mode);
        }
    }

    #define NOA_INSTANTIATE_RESIZE_(T) \
    template void resize<T>(const T*, size3_t, int3_t, int3_t, T*, BorderMode, T, size_t)

    NOA_INSTANTIATE_RESIZE_(float);
    NOA_INSTANTIATE_RESIZE_(double);
    NOA_INSTANTIATE_RESIZE_(bool);
    NOA_INSTANTIATE_RESIZE_(char);
    NOA_INSTANTIATE_RESIZE_(short);
    NOA_INSTANTIATE_RESIZE_(int);
    NOA_INSTANTIATE_RESIZE_(long);
    NOA_INSTANTIATE_RESIZE_(long long);
    NOA_INSTANTIATE_RESIZE_(unsigned char);
    NOA_INSTANTIATE_RESIZE_(unsigned short);
    NOA_INSTANTIATE_RESIZE_(unsigned int);
    NOA_INSTANTIATE_RESIZE_(unsigned long);
    NOA_INSTANTIATE_RESIZE_(unsigned long long);
}
