#pragma once

#include "noa/cpu/fft/Transforms.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/fft/Transforms.hpp"
#endif

#include "noa/unified/Array.hpp"
#include "noa/unified/memory/Factory.hpp"

namespace noa::fft {
    /// Returns the optimum even size, greater or equal than \p size.
    /// \note A optimum size is an even integer satisfying (2^a)*(3^b)*(5^c)*(7^d).
    /// \note If \p size is >16800, this function will simply return the next even number and will not necessarily
    ///       satisfy the aforementioned requirements.
    inline i64 fast_size(i64 size) {
        #ifdef NOA_ENABLE_CUDA
        return noa::cuda::fft::fast_size(size);
        #else
        return noa::cpu::fft::fast_size(size);
        #endif
    }

    /// Returns the next optimum BDHW shape.
    /// \note Dimensions of size 0 or 1 are ignored as well as the batch dimension, e.g. {3,1,53,53}
    ///       is rounded up to {3,1,54,54}.
    [[nodiscard]] inline Shape4<i64> fast_shape(const Shape4<i64>& shape) {
        #ifdef NOA_ENABLE_CUDA
        return noa::cuda::fft::fast_shape(shape);
        #else
        return noa::cpu::fft::fast_shape(shape);
        #endif
    }

    /// Returns the real-valued alias of \p rfft.
    /// \param[in] rfft Array or View of the rfft(s) to alias.
    /// \param shape    BDHW logical shape of \p rfft.
    template<typename Complex, typename = std::enable_if_t<noa::traits::is_array_or_view_of_complex_v<Complex>>>
    [[nodiscard]] auto alias_to_real(const Complex& rfft, const Shape4<i64>& shape) {
        NOA_CHECK(noa::all(rfft.shape() == shape.fft()),
                  "Given the {} logical shape, the rfft should have a shape of {}, but got {}",
                  shape, rfft.shape(), shape.fft());
        using real_t = noa::traits::value_type_t<typename Complex::value_type>;
        auto tmp = rfft.template as<real_t>();
        return decltype(tmp)(tmp.share(), shape, tmp.strides(), tmp.options());
    }

    /// Returns a rfft (and its real-valued alias) filled with zeros.
    /// \param shape    BDHW shape of the real-space array.
    /// \param option   Options of the created array.
    /// \return         The allocated array. Both the real and complex view are pointing to the same memory,
    ///                 i.e. the real array has enough padding and alignment to supports inplace r2c transforms.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, f32, f64>>>
    [[nodiscard]] std::pair<Array<T>, Array<Complex<T>>> zeros(const Shape4<i64>& shape, ArrayOption option = {}) {
        Array out1 = noa::memory::zeros<Complex<T>>(shape.fft(), option);
        Array out0 = fft::alias_to_real(out1, shape);
        return {out0, out1};
    }

    /// Returns a rfft (and its real-valued alias) filled with ones.
    /// \param shape    BDHW shape of the real-space array.
    /// \param option   Options of the created array.
    /// \return         The allocated array. Both the real and complex view are pointing to the same memory,
    ///                 i.e. the real array has enough padding and alignment to supports inplace r2c transforms.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, f32, f64>>>
    [[nodiscard]] std::pair<Array<T>, Array<Complex<T>>> ones(const Shape4<i64>& shape, ArrayOption option = {}) {
        Array out1 = noa::memory::ones<Complex<T>>(shape.fft(), option);
        Array out0 = fft::alias_to_real(out1, shape);
        return {out0, out1};
    }

    /// Returns a "padded" (as real and complex) uninitialized array.
    /// \param shape    BDHW shape of the real-space array.
    /// \param option   Options of the created array.
    /// \return         The allocated array. Both the real and complex view are pointing to the same memory,
    ///                 i.e. the real array has enough padding and alignment to supports inplace r2c transforms.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, f32, f64>>>
    [[nodiscard]] std::pair<Array<T>, Array<Complex<T>>> empty(const Shape4<i64>& shape, ArrayOption option = {}) {
        Array out1 = noa::memory::empty<Complex<T>>(shape.fft(), option);
        Array out0 = fft::alias_to_real(out1, shape);
        return {out0, out1};
    }
}
