#pragma once

#include "noa/runtime/Array.hpp"
#include "noa/runtime/Factory.hpp"

#include "noa/fft/cpu/Plan.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/fft/cuda/Plan.hpp"
#endif

namespace noa::fft {
    /// Returns the optimum even size, greater or equal than \p size.
    /// \note A optimum size is an even integer satisfying (2^a)*(3^b)*(5^c)*(7^d).
    /// \note If \p size is >16800, this function will simply return the next even number and will not necessarily
    ///       satisfy the aforementioned requirements.
    inline auto next_fast_size(isize size) -> isize {
        #ifdef NOA_ENABLE_CUDA
        return noa::fft::cuda::fast_size(size);
        #else
        return noa::fft::cpu::fast_size(size);
        #endif
    }

    /// Returns the next optimal transform shape.
    template<typename T, usize N>
    [[nodiscard]] auto next_fast_shape(Shape<T, N> shape, usize rank = 0) -> Shape<T, N> {
        rank = shape.rank_checked(rank);
        const auto start_index = std::max(usize{0}, N - rank); // ignore batch axes
        for (usize i = start_index; i < N; ++i)
            if (shape[i] > 1)
                shape[i] = static_cast<T>(next_fast_size(static_cast<isize>(shape[i])));
        return shape;
    }

    /// Returns the real-valued alias of rfft.
    /// \param[in] rfft Array of the rfft(s) to alias.
    /// \param shape    Logical shape of rfft.
    template<nt::array_decay_of_complex Complex, usize N>
        requires nt::array_decay_nd<Complex, N>
    [[nodiscard]] auto alias_to_real(Complex&& rfft, const Shape<isize, N>& shape) {
        check(rfft.shape() == shape.rfft(),
              "Given the {} logical shape, the rfft should have a shape of {}, but got {}",
              shape, shape.rfft(), rfft.shape());
        using real_t = nt::value_type_twice_t<Complex>;
        auto tmp = std::forward<Complex>(rfft).template as<real_t>();
        auto strides = tmp.strides();
        auto options = tmp.options();
        return decltype(tmp)(std::move(tmp).share(), shape, strides, options);
    }

    /// Returns a rfft (and its real-valued alias) filled with zeros.
    /// \param shape    Shape of the real-space array.
    /// \param option   Options of the created array.
    /// \return         The allocated array. Both the real and complex views are pointing to the same memory,
    ///                 i.e. the real array has enough padding and alignment to support inplace r2c transforms.
    template<nt::any_of<f32, f64> T, usize N>
    [[nodiscard]] auto zeros(const Shape<isize, N>& shape, ArrayOption option = {}) -> Pair<Array<T, N>, Array<Complex<T>, N>> {
        Array complex = noa::zeros<Complex<T>, N>(shape.rfft(), option);
        Array real = alias_to_real(complex, shape);
        return {std::move(real), std::move(complex)};
    }

    /// Returns a rfft (and its real-valued alias) filled with ones.
    /// \param shape    Shape of the real-space array.
    /// \param option   Options of the created array.
    /// \return         The allocated array. Both the real and complex views are pointing to the same memory,
    ///                 i.e. the real array has enough padding and alignment to support inplace r2c transforms.
    template<nt::any_of<f32, f64> T, usize N>
    [[nodiscard]] auto ones(const Shape<isize, N>& shape, ArrayOption option = {}) -> Pair<Array<T, N>, Array<Complex<T>, N>> {
        Array complex = noa::ones<Complex<T>, N>(shape.rfft(), option);
        Array real = alias_to_real(complex, shape);
        return {std::move(real), std::move(complex)};
    }

    /// Returns a "padded" (as real and complex) uninitialized array.
    /// \param shape    Shape of the real-space array.
    /// \param option   Options of the created array.
    /// \return         The allocated array. Both the real and complex views are pointing to the same memory,
    ///                 i.e. the real array has enough padding and alignment to support inplace r2c transforms.
    template<nt::any_of<f32, f64> T, usize N>
    [[nodiscard]] auto empty(const Shape<isize, N>& shape, ArrayOption option = {}) -> Pair<Array<T, N>, Array<Complex<T>, N>> {
        Array complex = noa::empty<Complex<T>, N>(shape.rfft(), option);
        Array real = alias_to_real(complex, shape);
        return {std::move(real), std::move(complex)};
    }
}
