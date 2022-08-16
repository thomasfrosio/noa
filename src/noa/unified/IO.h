#pragma once

#include "noa/common/io/ImageFile.h"
#include "noa/unified/Array.h"

namespace noa::io {
    /// Loads a file into a new array.
    /// \tparam T           Any data type (integer, floating-point, complex).
    /// \param[in] filename Path of the file to read.
    /// \param clamp        Whether the values in the file should be clamped to fit the output \p T type.
    ///                     If false, out of range values are undefined.
    /// \param option       Options for the output array.
    /// \return             BDHW C-contiguous output array containing the whole data array of the file.
    /// \note This is an utility function. For better flexibility, use ImageFile directly.
    template<typename T>
    Array<T> load(const path_t& filename, bool clamp = false, ArrayOption option = {}) {
        ImageFile file(filename, io::READ);
        if (option.dereferenceable()) {
            Array<T> out(file.shape(), option);
            file.readAll(out.eval().view(), clamp);
            return out;
        } else {
            Array<T> tmp(file.shape());
            file.readAll(tmp.view(), clamp);
            return tmp.to(option);
        }
    }

    /// Saves the input array into a file.
    /// \tparam T           Any data type (integer, floating-point, complex).
    /// \param[in] input    Array to serialize.
    /// \param[in] filename Path of the new file.
    /// \param dtype        Data type of the file. If DATA_UNKNOWN, let the file format decide the best data type
    ///                     for \p T values, so that no truncation or loss of precision happens.
    /// \param clamp        Whether the values of the array should be clamped to fit the file data type.
    ///                     If false, out of range values are undefined.
    /// \note This is an utility function. For better flexibility, use ImageFile directly.
    template<typename T>
    void save(const Array<T> input, const path_t& filename, DataType dtype = DATA_UNKNOWN, bool clamp = false) {
        ImageFile file(filename, io::WRITE);
        if (dtype != DATA_UNKNOWN)
            file.dtype(dtype);
        if (input.dereferenceable()) {
            file.writeAll(input.eval().view(), clamp);
        } else {
            const ArrayOption default_options{};
            file.writeAll(input.to(default_options).eval().view(), clamp);
        }
    }
}
