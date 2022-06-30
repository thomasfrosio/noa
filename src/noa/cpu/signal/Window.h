#pragma once

#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::signal {
    ///
    /// \tparam T
    /// \param input
    /// \param input_stride
    /// \param output
    /// \param output_stride
    /// \param output_shape
    /// \param stddev
    /// \param stream
    template<typename T>
    void gaussian(const shared_t<T[]>& input, size4_t input_stride,
                  const shared_t<T[]>& output, size4_t output_stride,
                  size4_t shape, float stddev, Stream& stream) {

    }

    template<typename T>
    void radius2D(const shared_t<T[]>& input, size4_t input_stride,
                  const shared_t<T[]>& output, size4_t output_stride,
                  size4_t shape, float2_t center, Stream& stream) {

    }

    template<typename T>
    void radius3D(const shared_t<T[]>& input, size4_t input_stride,
                  const shared_t<T[]>& output, size4_t output_stride,
                  size4_t shape, float3_t center, Stream& stream) {

    }


}
