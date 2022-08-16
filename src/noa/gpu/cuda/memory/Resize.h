#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::memory {
    // Sets the number of element(s) to pad/crop for each border of each dimension to get from input_shape to
    // output_shape, while keeping the centers of the input and output array (defined as ``shape / 2``) aligned.
    inline std::pair<int4_t, int4_t> borders(size4_t input_shape, size4_t output_shape) {
        const int4_t o_shape(output_shape);
        const int4_t i_shape(input_shape);
        const int4_t diff(o_shape - i_shape);

        const int4_t border_left = o_shape / 2 - i_shape / 2;
        const int4_t border_right = diff - border_left;
        return {border_left, border_right};
    }

    // Resizes the input array(s) by padding and/or cropping the edges of the array.
    template<typename T, typename = std::enable_if_t<traits::is_restricted_data_v<T>>>
    void resize(const shared_t<T[]>& input, size4_t input_strides, size4_t input_shape,
                int4_t border_left, int4_t border_right,
                const shared_t<T[]>& output, size4_t output_strides,
                BorderMode border_mode, T border_value, Stream& stream);

    // Resizes the input array(s) to the desired shape while keeping the center (defined as shape / 2) aligned.
    template<typename T, typename = std::enable_if_t<traits::is_restricted_data_v<T>>>
    inline void resize(const shared_t<T[]>& input, size4_t input_strides, size4_t input_shape,
                       const shared_t<T[]>& output, size4_t output_strides, size4_t output_shape,
                       BorderMode border_mode, T border_value, Stream& stream) {
        auto[border_left, border_right] = borders(input_shape, output_shape);
        resize(input, input_strides, input_shape,
               border_left, border_right,
               output, output_strides,
               border_mode, border_value, stream);
    }
}
