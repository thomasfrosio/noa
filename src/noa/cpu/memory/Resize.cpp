#include "noa/common/Assert.h"
#include "noa/common/Exception.h"
#include "noa/common/Math.h"
#include "noa/common/Profiler.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/Resize.h"

namespace {
    using namespace ::noa;

    // Copy the valid elements in the inputs into the outputs.
    // So it performs the cropping, and padding with BORDER_NOTHING.
    template<typename T>
    void copyValidRegion_(const T* inputs, size3_t input_pitch, size3_t input_shape,
                          int3_t border_left, int3_t crop_left, int3_t crop_right, int3_t pad_left,
                          T* outputs, size3_t output_pitch, size3_t output_shape, size_t batches) {
        const size_t iffset = elements(input_pitch);
        const size_t offset = elements(output_pitch);

        const int3_t valid_end(int3_t(input_shape) - crop_right);
        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + batch * iffset;
            T* output = outputs + batch * offset;

            for (int i_z = crop_left.z; i_z < valid_end.z; ++i_z) {
                int o_z = i_z + border_left.z; // positive offset for padding or remove the cropped elements
                for (int i_y = crop_left.y; i_y < valid_end.y; ++i_y) {
                    int o_y = i_y + border_left.y;

                    // Offset to the current row.
                    cpu::memory::copy(input + index(crop_left.x, i_y, i_z, input_shape),
                                      output + index(pad_left.x, o_y, o_z, output_shape),
                                      static_cast<uint>(valid_end.x - crop_left.x));
                }
            }
        }
    }

    // Sets the elements within the padding to a given value.
    template<typename T>
    void applyBorderValue_(T* outputs, size3_t pitch, size3_t shape, size_t batches,
                           int3_t pad_left, int3_t pad_right, T value) {
        const int3_t int_shape(shape);
        const int3_t valid_end = int_shape - pad_right;
        const size_t offset = elements(pitch);
        for (size_t batch = 0; batch < batches; ++batch) {
            T* output = outputs + batch * offset;
            for (int z = 0; z < int_shape.z; ++z) {
                const bool skip_z = z >= pad_left.z && z < valid_end.z;
                for (int y = 0; y < int_shape.y; ++y) {
                    const bool skip_y = y >= pad_left.y && y < valid_end.y;
                    for (int x = 0; x < int_shape.x; ++x) {
                        const bool skip_x = x >= pad_left.x && x < valid_end.x;
                        if (!skip_x || !skip_y || !skip_z)
                            output[index(x, y, z, pitch)] = value;
                    }
                }
            }
        }
    }

    template<BorderMode MODE, typename T>
    void applyBorder_(const T* inputs, size3_t input_pitch, size3_t input_shape,
                      T* outputs, size3_t output_pitch, size3_t output_shape, size_t batches,
                      int3_t pad_left, int3_t pad_right, int3_t crop_left) {
        const int3_t int_input_shape(input_shape);
        const int3_t int_output_shape(output_shape);
        const int3_t valid_end = int_output_shape - pad_right;
        const size_t iffset = elements(input_pitch);
        const size_t offset = elements(output_pitch);

        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + batch * iffset;
            T* output = outputs + batch * offset;

            for (int o_z = 0; o_z < int_output_shape.z; ++o_z) {
                const bool skip_z = o_z >= pad_left.z && o_z < valid_end.z;
                const int i_z = getBorderIndex<MODE>(o_z - pad_left.z + crop_left.z, int_input_shape.z);

                for (int o_y = 0; o_y < int_output_shape.y; ++o_y) {
                    const bool skip_y = o_y >= pad_left.y && o_y < valid_end.y;
                    const int i_y = getBorderIndex<MODE>(o_y - pad_left.y + crop_left.y, int_input_shape.y);

                    for (int o_x = 0; o_x < int_output_shape.x; ++o_x) {
                        const bool skip_x = o_x >= pad_left.x && o_x < valid_end.x;
                        if (!skip_x || !skip_y || !skip_z) {
                            const int i_x = getBorderIndex<MODE>(o_x - pad_left.x + crop_left.x, int_input_shape.x);
                            output[index(o_x, o_y, o_z, output_pitch)] = input[index(i_x, i_y, i_z, input_pitch)];
                        }
                    }
                }
            }
        }
    }
}

namespace noa::cpu::memory {
    template<typename T>
    void resize(const T* inputs, size3_t input_pitch, size3_t input_shape, int3_t border_left, int3_t border_right,
                T* outputs, size3_t output_pitch, size_t batches, BorderMode border_mode, T border_value,
                Stream& stream) {
        NOA_PROFILE_FUNCTION();

        if (all(border_left == 0) && all(border_right == 0))
            return copy(inputs, input_pitch, outputs, output_pitch, input_shape, batches, stream);

        stream.enqueue([=]() {
            NOA_PROFILE_FUNCTION();
            NOA_ASSERT(inputs != outputs);

            const size3_t output_shape(int3_t(input_shape) + border_left + border_right); // assumed to be > 0
            const int3_t crop_left(math::min(border_left, 0) * -1);
            const int3_t crop_right(math::min(border_right, 0) * -1);
            const int3_t pad_left(math::max(border_left, 0));
            const int3_t pad_right(math::max(border_right, 0));

            // Copy the valid elements in the inputs into the outputs.
            copyValidRegion_(inputs, input_pitch, input_shape,
                             border_left, crop_left, crop_right, pad_left,
                             outputs, output_pitch, output_shape, batches);

            // Shortcut: if there's nothing to pad, we are done here.
            if (border_mode == BORDER_NOTHING || (all(pad_left == 0) && all(pad_right == 0)))
                return;

            // Set the padded elements to the correct values.
            switch (border_mode) {
                case BORDER_ZERO:
                    return applyBorderValue_(
                            outputs, output_pitch, output_shape, batches, pad_left, pad_right, T{0});
                case BORDER_VALUE:
                    return applyBorderValue_(
                            outputs, output_pitch, output_shape, batches, pad_left, pad_right, border_value);
                case BORDER_CLAMP:
                    return applyBorder_<BORDER_CLAMP>(
                            inputs, input_pitch, input_shape,
                            outputs, output_pitch, output_shape, batches,
                            pad_left, pad_right, crop_left);
                case BORDER_PERIODIC:
                    return applyBorder_<BORDER_PERIODIC>(
                            inputs, input_pitch, input_shape,
                            outputs, output_pitch, output_shape, batches,
                            pad_left, pad_right, crop_left);
                case BORDER_REFLECT:
                    return applyBorder_<BORDER_REFLECT>(
                            inputs, input_pitch, input_shape,
                            outputs, output_pitch, output_shape, batches,
                            pad_left, pad_right, crop_left);
                case BORDER_MIRROR:
                    return applyBorder_<BORDER_MIRROR>(
                            inputs, input_pitch, input_shape,
                            outputs, output_pitch, output_shape, batches,
                            pad_left, pad_right, crop_left);
                default:
                    NOA_THROW("BorderMode not supported. Got: {}", border_mode);
            }
        });
    }

    #define NOA_INSTANTIATE_RESIZE_(T) \
    template void resize<T>(const T*, size3_t, size3_t, int3_t, int3_t, T*, size3_t, size_t, BorderMode, T, Stream&)

    NOA_INSTANTIATE_RESIZE_(half_t);
    NOA_INSTANTIATE_RESIZE_(float);
    NOA_INSTANTIATE_RESIZE_(double);
    NOA_INSTANTIATE_RESIZE_(chalf_t);
    NOA_INSTANTIATE_RESIZE_(cfloat_t);
    NOA_INSTANTIATE_RESIZE_(cdouble_t);
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
