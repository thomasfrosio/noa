#pragma once

#include "noa/unified/Ewise.hpp"
#include "noa/core/Operators.hpp"

namespace noa {
    /// Casts an array.
    /// \param[in] input    Array to convert.
    /// \param[out] output  Array with the casted values.
    /// \param clamp        Whether the input values should be clamped to the output range before casting.
    template<typename Input, typename Output>
    requires nt::are_varray_v<Input, Output>
    void cast(const Input& input, const Output& output, bool clamp = false) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        check(not ni::are_overlapped(input, output), "The input and output arrays should not overlap");

        auto input_strides = input.strides();
        if (not ni::broadcast(input.shape(), input_strides, output.shape())) {
            panic("Cannot broadcast an array of shape {} into an array of shape {}",
                  input.shape(), output.shape());
        }

        const Device device = output.device();
        check(device == input.device(),
              "The input and output arrays must be on the same device, but got input:{} and output:{}",
              device, input.device());

        ewise(forward_as_tuple(input), forward_as_tuple(output), Cast{clamp});
    }
}
