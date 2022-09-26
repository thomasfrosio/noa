#include "noa/common/Assert.h"
#include "noa/common/Exception.h"
#include "noa/common/Math.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/Resize.h"

namespace {
    using namespace ::noa;

    // Copy the valid elements in the input into the output.
    // So it performs the cropping, and padding with BORDER_NOTHING.
    template<typename T>
    void copyValidRegion_(AccessorRestrict<const T, 4, dim_t> input, dim4_t input_shape,
                          long4_t border_left, long4_t crop_left, long4_t crop_right,
                          AccessorRestrict<T, 4, dim_t> output) {
        const long4_t valid_end(long4_t(input_shape) - crop_right);
        for (int64_t ii = crop_left[0]; ii < valid_end[0]; ++ii) {
            for (int64_t ij = crop_left[1]; ij < valid_end[1]; ++ij) {
                for (int64_t ik = crop_left[2]; ik < valid_end[2]; ++ik) {
                    for (int64_t il = crop_left[3]; il < valid_end[3]; ++il) {
                        const int64_t oi = ii + border_left[0]; // positive offset for padding or remove the cropped elements
                        const int64_t oj = ij + border_left[1];
                        const int64_t ok = ik + border_left[2];
                        const int64_t ol = il + border_left[3];
                        output(oi, oj, ok, ol) = input(ii, ij, ik, il);
                    }
                }
            }
        }
    }

    // Sets the elements within the padding to a given value.
    template<typename T>
    void applyBorderValue_(T* output, dim4_t strides, dim4_t shape,
                           long4_t pad_left, long4_t pad_right, T value) {
        const long4_t int_shape(shape);
        const long4_t valid_end = int_shape - pad_right;
        for (int64_t i = 0; i < int_shape[0]; ++i) {
            for (int64_t j = 0; j < int_shape[1]; ++j) {
                for (int64_t k = 0; k < int_shape[2]; ++k) {
                    for (int64_t l = 0; l < int_shape[3]; ++l) {

                        const bool skip_i = i >= pad_left[0] && i < valid_end[0];
                        const bool skip_j = j >= pad_left[1] && j < valid_end[1];
                        const bool skip_k = k >= pad_left[2] && k < valid_end[2];
                        const bool skip_l = l >= pad_left[3] && l < valid_end[3];
                        if (!skip_i || !skip_j || !skip_k || !skip_l)
                            output[indexing::at(i, j, k, l, strides)] = value;
                    }
                }
            }
        }
    }

    template<BorderMode MODE, typename T>
    void applyBorder_(AccessorRestrict<const T, 4, dim_t> input, dim4_t input_shape,
                      AccessorRestrict<T, 4, dim_t> output, dim4_t output_shape,
                      long4_t pad_left, long4_t pad_right, long4_t crop_left) {
        const long4_t int_input_shape(input_shape);
        const long4_t int_output_shape(output_shape);
        const long4_t valid_end = int_output_shape - pad_right;

        for (int64_t oi = 0; oi < int_output_shape[0]; ++oi) {
            for (int64_t oj = 0; oj < int_output_shape[1]; ++oj) {
                for (int64_t ok = 0; ok < int_output_shape[2]; ++ok) {
                    for (int64_t ol = 0; ol < int_output_shape[3]; ++ol) {

                        const int64_t ii = indexing::at<MODE>(oi - pad_left[0] + crop_left[0], int_input_shape[0]);
                        const int64_t ij = indexing::at<MODE>(oj - pad_left[1] + crop_left[1], int_input_shape[1]);
                        const int64_t ik = indexing::at<MODE>(ok - pad_left[2] + crop_left[2], int_input_shape[2]);
                        const int64_t il = indexing::at<MODE>(ol - pad_left[3] + crop_left[3], int_input_shape[3]);

                        const bool skip_i = oi >= pad_left[0] && oi < valid_end[0];
                        const bool skip_j = oj >= pad_left[1] && oj < valid_end[1];
                        const bool skip_k = ok >= pad_left[2] && ok < valid_end[2];
                        const bool skip_l = ol >= pad_left[3] && ol < valid_end[3];

                        if (!skip_i || !skip_j || !skip_k || !skip_l)
                            output(oi, oj, ok, ol) = input(ii, ij, ik, il);
                    }
                }
            }
        }
    }
}

namespace noa::cpu::memory {
    template<typename T, typename>
    void resize(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
                int4_t border_left, int4_t border_right, const shared_t<T[]>& output, dim4_t output_strides,
                BorderMode border_mode, T border_value, Stream& stream) {
        if (all(border_left == 0) && all(border_right == 0))
            return copy(input, input_strides, output, output_strides, input_shape, stream);

        stream.enqueue([=]() mutable {
            NOA_ASSERT(input != output);
            const int4_t tmp = int4_t(input_shape) + border_left + border_right;
            NOA_ASSERT(all(tmp >= 1));
            dim4_t output_shape(tmp);
            NOA_ASSERT(!indexing::isOverlap(input.get(), input_strides, input_shape,
                                            output.get(), output_strides, output_shape));

            // Optimize reads/writes for output:
            const dim4_t order = indexing::order(output_strides, output_shape);
            if (!all(order == dim4_t{0, 1, 2, 3})) {
                input_strides = indexing::reorder(input_strides, order);
                input_shape = indexing::reorder(input_shape, order);
                border_left = indexing::reorder(border_left, order);
                border_right = indexing::reorder(border_right, order);
                output_strides = indexing::reorder(output_strides, order);
                output_shape = indexing::reorder(output_shape, order);
            }

            const long4_t crop_left(math::min(border_left, 0) * -1);
            const long4_t crop_right(math::min(border_right, 0) * -1);
            const long4_t pad_left(math::max(border_left, 0));
            const long4_t pad_right(math::max(border_right, 0));

            // Copy the valid elements in the input into the output.
            copyValidRegion_<T>({input.get(), input_strides}, input_shape,
                                long4_t(border_left), crop_left, crop_right,
                                {output.get(), output_strides});

            // Shortcut: if there's nothing to pad, we are done here.
            if (border_mode == BORDER_NOTHING || (all(pad_left == 0) && all(pad_right == 0)))
                return;

            // Set the padded elements to the correct values.
            switch (border_mode) {
                case BORDER_ZERO:
                    return applyBorderValue_(
                            output.get(), output_strides, output_shape, pad_left, pad_right, T{0});
                case BORDER_VALUE:
                    return applyBorderValue_(
                            output.get(), output_strides, output_shape, pad_left, pad_right, border_value);
                case BORDER_CLAMP:
                    return applyBorder_<BORDER_CLAMP, T>(
                            {input.get(), input_strides}, input_shape,
                            {output.get(), output_strides}, output_shape,
                            pad_left, pad_right, crop_left);
                case BORDER_PERIODIC:
                    return applyBorder_<BORDER_PERIODIC, T>(
                            {input.get(), input_strides}, input_shape,
                            {output.get(), output_strides}, output_shape,
                            pad_left, pad_right, crop_left);
                case BORDER_REFLECT:
                    return applyBorder_<BORDER_REFLECT, T>(
                            {input.get(), input_strides}, input_shape,
                            {output.get(), output_strides}, output_shape,
                            pad_left, pad_right, crop_left);
                case BORDER_MIRROR:
                    return applyBorder_<BORDER_MIRROR, T>(
                            {input.get(), input_strides}, input_shape,
                            {output.get(), output_strides}, output_shape,
                            pad_left, pad_right, crop_left);
                default:
                    NOA_THROW("BorderMode not supported. Got: {}", border_mode);
            }
        });
    }

    #define NOA_INSTANTIATE_RESIZE_(T) \
    template void resize<T, void>(const shared_t<T[]>&, dim4_t, dim4_t, int4_t, int4_t, const shared_t<T[]>&, dim4_t, BorderMode, T, Stream&)

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
