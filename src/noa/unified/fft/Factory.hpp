#pragma once

#include "noa/unified/Array.hpp"
#include "noa/unified/Factory.hpp"

#include "noa/cpu/fft/Plan.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/fft/Plan.hpp"
#endif

namespace noa::fft {
    /// Returns the optimum even size, greater or equal than \p size.
    /// \note A optimum size is an even integer satisfying (2^a)*(3^b)*(5^c)*(7^d).
    /// \note If \p size is >16800, this function will simply return the next even number and will not necessarily
    ///       satisfy the aforementioned requirements.
    inline auto next_fast_size(isize size) -> isize {
        #ifdef NOA_ENABLE_CUDA
        return noa::cuda::fft::fast_size(size);
        #else
        return noa::cpu::fft::fast_size(size);
        #endif
    }

    /// Returns the next optimum (((B)D)H)W shape.
    /// \note Dimensions of size 0 or 1 are ignored as well as the batch dimension,
    ///       e.g. {3,1,53,53} is rounded up to {3,1,54,54}.
    template<usize N>
    [[nodiscard]] auto next_fast_shape(Shape<isize, N> shape) -> Shape<isize, N> {
        constexpr usize START_INDEX = N == 4 ? 1 : 0; // BDHW -> ignore batch
        for (usize i = START_INDEX; i < N; ++i)
            if (shape[i] > 1)
                shape[i] = next_fast_size(shape[i]);
        return shape;
    }

    /// Returns the real-valued alias of \p rfft.
    /// \param[in] rfft VArray of the rfft(s) to alias.
    /// \param shape    BDHW logical shape of \p rfft.
    template<nt::varray_decay_of_complex Complex>
    [[nodiscard]] auto alias_to_real(Complex&& rfft, const Shape4& shape) {
        check(rfft.shape() == shape.rfft(),
              "Given the {} logical shape, the rfft should have a shape of {}, but got {}",
              shape, shape.rfft(), rfft.shape());
        using real_t = nt::mutable_value_type_twice_t<Complex>;
        auto tmp = std::forward<Complex>(rfft).template reinterpret_as<real_t>();
        auto strides = tmp.strides();
        auto options = tmp.options();
        return decltype(tmp)(std::move(tmp).share(), shape, strides, options);
    }

    /// Returns a rfft (and its real-valued alias) filled with zeros.
    /// \param shape    BDHW shape of the real-space array.
    /// \param option   Options of the created array.
    /// \return         The allocated array. Both the real and complex views are pointing to the same memory,
    ///                 i.e. the real array has enough padding and alignment to support inplace r2c transforms.
    template<nt::any_of<f32, f64> T>
    [[nodiscard]] auto zeros(const Shape4& shape, ArrayOption option = {}) -> Pair<Array<T>, Array<Complex<T>>> {
        Array complex = noa::zeros<Complex<T>>(shape.rfft(), option);
        Array real = alias_to_real(complex, shape);
        return {std::move(real), std::move(complex)};
    }

    /// Returns a rfft (and its real-valued alias) filled with ones.
    /// \param shape    BDHW shape of the real-space array.
    /// \param option   Options of the created array.
    /// \return         The allocated array. Both the real and complex views are pointing to the same memory,
    ///                 i.e. the real array has enough padding and alignment to support inplace r2c transforms.
    template<nt::any_of<f32, f64> T>
    [[nodiscard]] auto ones(const Shape4& shape, ArrayOption option = {}) -> Pair<Array<T>, Array<Complex<T>>> {
        Array complex = noa::ones<Complex<T>>(shape.rfft(), option);
        Array real = alias_to_real(complex, shape);
        return {std::move(real), std::move(complex)};
    }

    /// Returns a "padded" (as real and complex) uninitialized array.
    /// \param shape    BDHW shape of the real-space array.
    /// \param option   Options of the created array.
    /// \return         The allocated array. Both the real and complex views are pointing to the same memory,
    ///                 i.e. the real array has enough padding and alignment to support inplace r2c transforms.
    template<nt::any_of<f32, f64> T>
    [[nodiscard]] auto empty(const Shape4& shape, ArrayOption option = {}) -> Pair<Array<T>, Array<Complex<T>>> {
        Array complex = noa::empty<Complex<T>>(shape.rfft(), option);
        Array real = alias_to_real(complex, shape);
        return {std::move(real), std::move(complex)};
    }
}
