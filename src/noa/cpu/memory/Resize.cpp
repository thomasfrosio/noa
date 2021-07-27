#include "noa/common/Exception.h"
#include "noa/common/Math.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/Resize.h"

namespace {
    using namespace ::noa;

    inline size_t getOffset_(size3_t shape, int idx_y, int idx_z) {
        return (static_cast<size_t>(idx_z) * shape.y + static_cast<size_t>(idx_y)) * shape.x;
    }

    inline size_t getOffset_(size3_t shape, int idx_x, int idx_y, int idx_z) {
        return getOffset_(shape, idx_y, idx_z) + static_cast<size_t>(idx_x);
    }

    // Sets the elements within the padding to a given value.
    template<typename T>
    void applyBorderValue_(T* outputs, size3_t shape, size_t elements,
                           int3_t pad_left, int3_t pad_right, T value, uint batches) {
        int3_t int_shape(shape);
        int3_t valid_end = int_shape - pad_right;
        for (int z = 0; z < int_shape.z; ++z) {
            bool skip_z = z >= pad_left.z && z < valid_end.z;

            for (int y = 0; y < int_shape.y; ++y) {
                bool skip_y = y >= pad_left.y && y < valid_end.y;

                for (int x = 0; x < int_shape.x; ++x) {
                    bool skip_x = x >= pad_left.x && x < valid_end.x;

                    if (skip_x && skip_y && skip_z) continue;
                    size_t offset = getOffset_(shape, x, y, z);
                    for (uint batch = 0; batch < batches; ++batch)
                        outputs[batch * elements + offset] = value;
                }
            }
        }
    }

    template<BorderMode MODE, typename T>
    void applyBorder_(const T* inputs, size3_t input_shape, size_t input_elements,
                      T* outputs, size3_t output_shape, size_t output_elements,
                      int3_t pad_left, int3_t pad_right, int3_t crop_left, uint batches) {
        int3_t int_input_shape(input_shape);
        int3_t int_output_shape(output_shape);
        int3_t valid_end = int_output_shape - pad_right;
        for (int o_z = 0; o_z < int_output_shape.z; ++o_z) {
            bool skip_z = o_z >= pad_left.z && o_z < valid_end.z;
            int i_z = getBorderIndex<MODE>(o_z - pad_left.z + crop_left.z, int_input_shape.z);

            for (int o_y = 0; o_y < int_output_shape.y; ++o_y) {
                bool skip_y = o_y >= pad_left.y && o_y < valid_end.y;
                int i_y = getBorderIndex<MODE>(o_y - pad_left.y + crop_left.y, int_input_shape.y);

                for (int o_x = 0; o_x < int_output_shape.x; ++o_x) {
                    bool skip_x = o_x >= pad_left.x && o_x < valid_end.x;
                    if (skip_x && skip_y && skip_z) continue;
                    int i_x = getBorderIndex<MODE>(o_x - pad_left.x + crop_left.x, int_input_shape.x);

                    size_t o_offset = getOffset_(output_shape, o_x, o_y, o_z);
                    size_t i_offset = getOffset_(input_shape, i_x, i_y, i_z);
                    for (uint batch = 0; batch < batches; ++batch)
                        outputs[batch * output_elements + o_offset] = inputs[batch * input_elements + i_offset];
                }
            }
        }
    }
}

namespace noa::memory {
    template<typename T>
    void resize(const T* inputs, size3_t input_shape, int3_t border_left, int3_t border_right,
                T* outputs, BorderMode mode, T border_value, uint batches) {
        if (all(border_left == 0) && all(border_right == 0)) {
            memory::copy(inputs, outputs, getElements(input_shape) * batches);
            return;
        }

        size3_t output_shape(int3_t(input_shape) + border_left + border_right); // assumed to be > 0
        size_t input_elements = getElements(input_shape);
        size_t output_elements = getElements(output_shape);
        int3_t crop_left(math::min(border_left, 0) * -1);
        int3_t crop_right(math::min(border_right, 0) * -1);
        int3_t pad_left(math::max(border_left, 0));
        int3_t pad_right(math::max(border_right, 0));

        // Copy the valid elements in the inputs into the outputs.
        int3_t valid_end(int3_t(input_shape) - crop_right);
        for (int i_z = crop_left.z; i_z < valid_end.z; ++i_z) {
            int o_z = i_z + border_left.z; // positive offset for padding or remove the cropped elements

            for (int i_y = crop_left.y; i_y < valid_end.y; ++i_y) {
                int o_y = i_y + border_left.y;

                // Offset to the current row.
                const T* tmp_inputs = inputs + getOffset_(input_shape, crop_left.x, i_y, i_z);
                T* tmp_outputs = outputs + getOffset_(output_shape, pad_left.x, o_y, o_z);
                auto elements_to_copy = static_cast<uint>(valid_end.x - crop_left.x);

                for (uint batch = 0; batch < batches; ++batch)
                    memory::copy(tmp_inputs + batch * input_elements,
                                 tmp_outputs + batch * output_elements,
                                 elements_to_copy);
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
    template void resize<T>(const T*, size3_t, int3_t, int3_t, T*, BorderMode, T, uint)

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
