#include "noa/common/Assert.h"
#include "noa/common/Exception.h"
#include "noa/common/Math.h"
#include "noa/common/Profiler.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/Resize.h"

namespace {
    using namespace ::noa;

    // Copy the valid elements in the input into the output.
    // So it performs the cropping, and padding with BORDER_NOTHING.
    template<typename T>
    void copyValidRegion_(const T* input, size4_t input_stride, size4_t input_shape,
                          int4_t border_left, int4_t crop_left, int4_t crop_right,
                          T* output, size4_t output_stride) {
        const int4_t valid_end(int4_t(input_shape) - crop_right);
        for (int ii = crop_left[0]; ii < valid_end[0]; ++ii) {
            for (int ij = crop_left[1]; ij < valid_end[1]; ++ij) {
                for (int ik = crop_left[2]; ik < valid_end[2]; ++ik) {
                    for (int il = crop_left[3]; il < valid_end[3]; ++il) {
                        int oi = ii + border_left[0]; // positive offset for padding or remove the cropped elements
                        int oj = ij + border_left[1];
                        int ok = ik + border_left[2];
                        int ol = il + border_left[3];
                        output[indexing::at(oi, oj, ok, ol, output_stride)] =
                                input[indexing::at(ii, ij, ik, il, input_stride)];
                    }
                }
            }
        }
    }

    // TODO benchmark to see if copying rows contiguously really helps
    template<typename T>
    void copyValidRegionContiguous_(const T* input, size4_t input_stride, size4_t input_shape,
                                    int4_t border_left, int4_t crop_left, int4_t crop_right,
                                    T* output, size4_t output_stride) {
        const int4_t valid_end(int4_t(input_shape) - crop_right);
        const int ol = crop_left[3] + border_left[3];
        for (int ii = crop_left[0]; ii < valid_end[0]; ++ii) {
            for (int ij = crop_left[1]; ij < valid_end[1]; ++ij) {
                for (int ik = crop_left[2]; ik < valid_end[2]; ++ik) {

                    int oi = ii + border_left[0];
                    int oj = ij + border_left[1];
                    int ok = ik + border_left[2];
                    cpu::memory::copy(input + indexing::at(ii, ij, ik, crop_left[3], input_stride),
                                      output + indexing::at(oi, oj, ok, ol, output_stride),
                                      static_cast<uint>(valid_end[3] - crop_left[3]));
                }
            }
        }
    }

    // Sets the elements within the padding to a given value.
    template<typename T>
    void applyBorderValue_(T* output, size4_t stride, size4_t shape,
                           int4_t pad_left, int4_t pad_right, T value) {
        const int4_t int_shape(shape);
        const int4_t valid_end = int_shape - pad_right;
        for (int i = 0; i < int_shape[0]; ++i) {
            for (int j = 0; j < int_shape[1]; ++j) {
                for (int k = 0; k < int_shape[2]; ++k) {
                    for (int l = 0; l < int_shape[3]; ++l) {

                        const bool skip_i = i >= pad_left[0] && i < valid_end[0];
                        const bool skip_j = j >= pad_left[1] && j < valid_end[1];
                        const bool skip_k = k >= pad_left[2] && k < valid_end[2];
                        const bool skip_l = l >= pad_left[3] && l < valid_end[3];
                        if (!skip_i || !skip_j || !skip_k || !skip_l)
                            output[indexing::at(i, j, k, l, stride)] = value;
                    }
                }
            }
        }
    }

    template<BorderMode MODE, typename T>
    void applyBorder_(const T* input, size4_t input_stride, size4_t input_shape,
                      T* output, size4_t output_stride, size4_t output_shape,
                      int4_t pad_left, int4_t pad_right, int4_t crop_left) {
        const int4_t int_input_shape(input_shape);
        const int4_t int_output_shape(output_shape);
        const int4_t valid_end = int_output_shape - pad_right;

        for (int oi = 0; oi < int_output_shape[0]; ++oi) {
            for (int oj = 0; oj < int_output_shape[1]; ++oj) {
                for (int ok = 0; ok < int_output_shape[2]; ++ok) {
                    for (int ol = 0; ol < int_output_shape[3]; ++ol) {

                        const int ii = getBorderIndex<MODE>(oi - pad_left[0] + crop_left[0], int_input_shape[0]);
                        const int ij = getBorderIndex<MODE>(oj - pad_left[1] + crop_left[1], int_input_shape[1]);
                        const int ik = getBorderIndex<MODE>(ok - pad_left[2] + crop_left[2], int_input_shape[2]);
                        const int il = getBorderIndex<MODE>(ol - pad_left[3] + crop_left[3], int_input_shape[3]);

                        const bool skip_i = oi >= pad_left[0] && oi < valid_end[0];
                        const bool skip_j = oj >= pad_left[1] && oj < valid_end[1];
                        const bool skip_k = ok >= pad_left[2] && ok < valid_end[2];
                        const bool skip_l = ol >= pad_left[3] && ol < valid_end[3];

                        if (!skip_i || !skip_j || !skip_k || !skip_l)
                            output[indexing::at(oi, oj, ok, ol, output_stride)] =
                                    input[indexing::at(ii, ij, ik, il, input_stride)];
                    }
                }
            }
        }
    }
}

namespace noa::cpu::memory {
    template<typename T>
    void resize(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
                int4_t border_left, int4_t border_right, const shared_t<T[]>& output, size4_t output_stride,
                BorderMode border_mode, T border_value, Stream& stream) {
        if (all(border_left == 0) && all(border_right == 0))
            return copy(input, input_stride, output, output_stride, input_shape, stream);

        stream.enqueue([=]() {
            NOA_PROFILE_FUNCTION();
            NOA_ASSERT(input != output);

            const size4_t output_shape(int4_t(input_shape) + border_left + border_right); // assumed to be > 0
            const int4_t crop_left(math::min(border_left, 0) * -1);
            const int4_t crop_right(math::min(border_right, 0) * -1);
            const int4_t pad_left(math::max(border_left, 0));
            const int4_t pad_right(math::max(border_right, 0));

            // Copy the valid elements in the input into the output.
            if (indexing::isContiguous(input_stride, input_shape)[3] &&
                indexing::isContiguous(output_stride, output_shape)[3]) {
                copyValidRegionContiguous_(input.get(), input_stride, input_shape,
                                           border_left, crop_left, crop_right,
                                           output.get(), output_stride);
            } else {
                copyValidRegion_(input.get(), input_stride, input_shape,
                                 border_left, crop_left, crop_right,
                                 output.get(), output_stride);
            }

            // Shortcut: if there's nothing to pad, we are done here.
            if (border_mode == BORDER_NOTHING || (all(pad_left == 0) && all(pad_right == 0)))
                return;

            // Set the padded elements to the correct values.
            switch (border_mode) {
                case BORDER_ZERO:
                    return applyBorderValue_(
                            output.get(), output_stride, output_shape, pad_left, pad_right, T{0});
                case BORDER_VALUE:
                    return applyBorderValue_(
                            output.get(), output_stride, output_shape, pad_left, pad_right, border_value);
                case BORDER_CLAMP:
                    return applyBorder_<BORDER_CLAMP>(
                            input.get(), input_stride, input_shape,
                            output.get(), output_stride, output_shape,
                            pad_left, pad_right, crop_left);
                case BORDER_PERIODIC:
                    return applyBorder_<BORDER_PERIODIC>(
                            input.get(), input_stride, input_shape,
                            output.get(), output_stride, output_shape,
                            pad_left, pad_right, crop_left);
                case BORDER_REFLECT:
                    return applyBorder_<BORDER_REFLECT>(
                            input.get(), input_stride, input_shape,
                            output.get(), output_stride, output_shape,
                            pad_left, pad_right, crop_left);
                case BORDER_MIRROR:
                    return applyBorder_<BORDER_MIRROR>(
                            input.get(), input_stride, input_shape,
                            output.get(), output_stride, output_shape,
                            pad_left, pad_right, crop_left);
                default:
                    NOA_THROW("BorderMode not supported. Got: {}", border_mode);
            }
        });
    }

    #define NOA_INSTANTIATE_RESIZE_(T) \
    template void resize<T>(const shared_t<T[]>&, size4_t, size4_t, int4_t, int4_t, const shared_t<T[]>&, size4_t, BorderMode, T, Stream&)

    NOA_INSTANTIATE_RESIZE_(bool);
    NOA_INSTANTIATE_RESIZE_(int8_t);
    NOA_INSTANTIATE_RESIZE_(int16_t);
    NOA_INSTANTIATE_RESIZE_(int32_t);
    NOA_INSTANTIATE_RESIZE_(int64_t);
    NOA_INSTANTIATE_RESIZE_(uint8_t);
    NOA_INSTANTIATE_RESIZE_(uint16_t);
    NOA_INSTANTIATE_RESIZE_(uint32_t);
    NOA_INSTANTIATE_RESIZE_(uint64_t);
    NOA_INSTANTIATE_RESIZE_(half_t);
    NOA_INSTANTIATE_RESIZE_(float);
    NOA_INSTANTIATE_RESIZE_(double);
    NOA_INSTANTIATE_RESIZE_(chalf_t);
    NOA_INSTANTIATE_RESIZE_(cfloat_t);
    NOA_INSTANTIATE_RESIZE_(cdouble_t);
}
