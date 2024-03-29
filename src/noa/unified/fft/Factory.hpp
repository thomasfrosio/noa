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
    inline i64 next_fast_size(i64 size) {
        #ifdef NOA_ENABLE_CUDA
        return noa::cuda::fft::fast_size(size);
        #else
        return noa::cpu::fft::fast_size(size);
        #endif
    }

    /// Returns the next optimum BDHW shape.
    /// \note Dimensions of size 0 or 1 are ignored as well as the batch dimension, e.g. {3,1,53,53}
    ///       is rounded up to {3,1,54,54}.
    template<size_t N>
    [[nodiscard]] inline Shape<i64, N> next_fast_shape(Shape<i64, N> shape) {
        constexpr size_t START_INDEX = N == 4 ? 1 : 0; // BDHW -> ignore batch
        for (size_t i = START_INDEX; i < N; ++i)
            if (shape[i] > 1)
                shape[i] = next_fast_size(shape[i]);
        return shape;
    }

    /// Returns the real-valued alias of \p rfft.
    /// \param[in] rfft VArray of the rfft(s) to alias.
    /// \param shape    BDHW logical shape of \p rfft.
    template<typename Complex, typename = std::enable_if_t<nt::is_varray_of_complex_v<Complex>>>
    [[nodiscard]] auto alias_to_real(const Complex& rfft, const Shape4<i64>& shape) {
        NOA_CHECK(noa::all(rfft.shape() == shape.rfft()),
                  "Given the {} logical shape, the rfft should have a shape of {}, but got {}",
                  shape, rfft.shape(), shape.rfft());
        using real_t = nt::value_type_t<typename Complex::value_type>;
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
        Array out1 = noa::memory::zeros<Complex<T>>(shape.rfft(), option);
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
        Array out1 = noa::memory::ones<Complex<T>>(shape.rfft(), option);
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
        Array out1 = noa::memory::empty<Complex<T>>(shape.rfft(), option);
        Array out0 = fft::alias_to_real(out1, shape);
        return {out0, out1};
    }
}
