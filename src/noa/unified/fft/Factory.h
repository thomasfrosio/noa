#pragma once

#include "noa/unified/Array.h"
#include "noa/unified/memory/Factory.h"

namespace noa::fft {
    /// Returns the optimum even size, greater or equal than \p size.
    /// \note A optimum size is an even integer satisfying (2^a)*(3^b)*(5^c)*(7^d).
    /// \note If \p size is >16800, this function will simply return the next even number and will not necessarily
    ///       satisfy the aforementioned requirements.
    inline dim_t nextFastSize(dim_t size);

    /// Returns the next optimum BDHW shape.
    /// \note Dimensions of size 0 or 1 are ignored as well as the batch dimension, e.g. {3,1,53,53}
    ///       is rounded up to {3,1,54,54}.
    template<typename T>
    Int4<T> nextFastShape(Int4<T> shape);

    /// Returns an alias of \p input that fits the corresponding real-space array.
    /// \tparam T           float, double.
    /// \param[in] input    Non-redundant FFT(s) to alias.
    /// \param shape        BDHW logical shape of \p input.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, double>>>
    Array<T> alias(const Array<Complex<T>>& input, dim4_t shape);

    /// Returns a "padded" array (as real and complex) filled with zeros.
    /// \tparam T       Any data type.
    /// \param shape    Shape of the array.
    /// \param option   Options of the created array.
    /// \return         The allocated array. Both the real and complex view are pointing to the same memory,
    ///                 i.e. the real array has enough padding and alignment to supports inplace r2c transforms.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, double>>>
    std::pair<Array<T>, Array<Complex<T>>> zeros(dim4_t shape, ArrayOption option = {});

    /// Returns a "padded" array (as real and complex) filled with ones.
    /// \tparam T       Any data type.
    /// \param shape    Shape of the array.
    /// \param option   Options of the created array.
    /// \return         The allocated array. Both the real and complex view are pointing to the same memory,
    ///                 i.e. the real array has enough padding and alignment to supports inplace r2c transforms.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, double>>>
    std::pair<Array<T>, Array<Complex<T>>> ones(dim4_t shape, ArrayOption option = {});

    /// Returns a "padded" (as real and complex) uninitialized array.
    /// \tparam T       Any data type.
    /// \param shape    Shape of the array.
    /// \param option   Options of the created array.
    /// \return         The allocated array. Both the real and complex view are pointing to the same memory,
    ///                 i.e. the real array has enough padding and alignment to supports inplace r2c transforms.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, double>>>
    std::pair<Array<T>, Array<Complex<T>>> empty(dim4_t shape, ArrayOption option = {});
}

#define NOA_FFT_FACTORY_
#include "noa/unified/fft/Factory.inl"
#undef NOA_FFT_FACTORY_
