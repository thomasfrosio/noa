#pragma once

#include "noa/core/Config.hpp"

#ifdef NOA_IS_OFFLINE
#include "noa/unified/Array.hpp"
#include "noa/unified/Ewise.hpp"

namespace noa {
    /// Extracts the real and imaginary part of complex numbers.
    /// \param[in] input    Complex array to decompose.
    /// \param[out] real    Real elements.
    /// \param[out] imag    Imaginary elements.
    template<typename C, typename R, typename I>
    requires (nt::is_varray_of_complex_v<C> and nt::are_varray_of_real_v<R, I>)
    void decompose(const C& input, const R& real, const I& imag) {
        check(not ni::are_overlapped(real, imag), "The output arrays should not overlap");
        ewise(input, wrap(real, imag), ComplexDecompose{});
    }

    /// Extracts the real part of complex numbers.
    /// \param[in] input    Complex array to decompose.
    /// \param[out] real    Real elements.
    template<typename C, typename R>
    requires (nt::is_varray_of_complex_v<C> and nt::is_varray_of_real_v<R>)
    void real(const C& input, const R& real) {
        check(not ni::are_overlapped(real, input), "The arrays should not overlap");
        ewise(input, real, Real{});
    }

    /// Extracts the imaginary part of complex numbers.
    /// \param[in] input    Complex array to decompose.
    /// \param[out] imag    Imaginary elements.
    template<typename C, typename I>
    requires (nt::is_varray_of_complex_v<C> and nt::is_varray_of_real_v<I>)
    void imag(const C& input, const I& imag) {
        check(not ni::are_overlapped(imag, input), "The arrays should not overlap");
        ewise(input, imag, Imag{});
    }

    /// Fuses the real and imaginary components.
    /// \param[in] real Real elements to interleave.
    /// \param[in] imag Imaginary elements to interleave.
    /// \param output   Complex array.
    template<typename C, typename R, typename I>
    requires (nt::is_varray_of_complex_v<C> and nt::are_varray_of_real_v<R, I>)
    void complex(const R& real, const I& imag, const C& output) {
        check(not ni::are_overlapped(real, imag), "The output arrays should not overlap");
        ewise(wrap(real, imag), output, ComplexFuse{});
    }
}

namespace noa {
    /// Extracts the real and imaginary part of complex numbers.
    template<typename C> requires nt::is_varray_of_complex_v<C>
    [[nodiscard]] auto decompose(const C& input) {
        using complex_t = nt::value_type_t<C>;
        using real_t = nt::value_type_t<complex_t>;
        Pair output{Array<real_t>(input.shape(), input.options()),
                    Array<real_t>(input.shape(), input.options())};
        decompose(input, output.first, output.second);
        return output;
    }

    /// Extracts the real part of complex numbers.
    template<typename C> requires nt::is_varray_of_complex_v<C>
    [[nodiscard]] auto real(const C& input) {
        using complex_t = nt::value_type_t<C>;
        using real_t = nt::value_type_t<complex_t>;
        Array<real_t> output(input.shape(), input.options());
        real(input, output);
        return output;
    }

    /// Extracts the imaginary part of complex numbers.
    template<typename C> requires nt::is_varray_of_complex_v<C>
    [[nodiscard]] auto imag(const C& input) {
        using complex_t = nt::value_type_t<C>;
        using real_t = nt::value_type_t<complex_t>;
        Array<real_t> output(input.shape(), input.options());
        imag(input, output);
        return output;
    }

    /// Fuses the real and imaginary components.
    template<typename R, typename I> requires nt::are_varray_of_real_v<R, I>
    [[nodiscard]] auto complex(const R& real, const I& imag) {
        using real_t = nt::mutable_value_type_t<R>;
        using complex_t = Complex<real_t>;
        Array<complex_t> output(real.shape(), real.options());
        complex(real, imag, output);
        return output;
    }
}
#endif
