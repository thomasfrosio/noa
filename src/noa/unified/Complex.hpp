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
    template<nt::readable_varray_decay_of_complex C,
             nt::writable_varray_decay_of_real R,
             nt::writable_varray_decay_of_real I>
    void decompose(C&& input, R&& real, I&& imag) {
        check(not ni::are_overlapped(real, imag), "The output arrays should not overlap");
        ewise(std::forward<C>(input), wrap(std::forward<R>(real), std::forward<I>(imag)), ComplexDecompose{});
    }

    /// Extracts the real part of complex numbers.
    /// \param[in] input    Complex array to decompose.
    /// \param[out] real    Real elements.
    template<nt::readable_varray_decay_of_complex C,
             nt::writable_varray_decay_of_real R>
    void real(C&& input, R&& real) {
        check(not ni::are_overlapped(real, input), "The arrays should not overlap");
        ewise(std::forward<C>(input), std::forward<R>(real), Real{});
    }

    /// Extracts the imaginary part of complex numbers.
    /// \param[in] input    Complex array to decompose.
    /// \param[out] imag    Imaginary elements.
    template<nt::readable_varray_decay_of_complex C,
             nt::writable_varray_decay_of_real I>
    void imag(C&& input, I&& imag) {
        check(not ni::are_overlapped(imag, input), "The arrays should not overlap");
        ewise(std::forward<C>(input), std::forward<I>(imag), Imag{});
    }

    /// Fuses the real and imaginary components.
    /// \param[in] real Real elements to interleave.
    /// \param[in] imag Imaginary elements to interleave.
    /// \param output   Complex array.
    template<nt::readable_varray_decay_of_real R,
             nt::readable_varray_decay_of_real I,
             nt::writable_varray_decay_of_complex C>
    void complex(R&& real, I&& imag, C&& output) {
        check(not ni::are_overlapped(real, imag), "The output arrays should not overlap");
        ewise(wrap(std::forward<R>(real), std::forward<I>(imag)), std::forward<C>(output), ComplexFuse{});
    }
}

namespace noa {
    /// Extracts the real and imaginary part of complex numbers.
    template<nt::readable_varray_decay_of_complex C>
    [[nodiscard]] auto decompose(C&& input) {
        using real_t = nt::value_type_twice_t<C>;
        Pair output{Array<real_t>(input.shape(), input.options()),
                    Array<real_t>(input.shape(), input.options())};
        decompose(std::forward<C>(input), output.first, output.second);
        return output;
    }

    /// Extracts the real part of complex numbers.
    template<nt::readable_varray_decay_of_complex C>
    [[nodiscard]] auto real(C&& input) {
        using real_t = nt::value_type_twice_t<C>;
        Array<real_t> output(input.shape(), input.options());
        real(std::forward<C>(input), output);
        return output;
    }

    /// Extracts the imaginary part of complex numbers.
    template<nt::readable_varray_decay_of_complex C>
    [[nodiscard]] auto imag(C&& input) {
        using real_t = nt::value_type_twice_t<C>;
        Array<real_t> output(input.shape(), input.options());
        imag(std::forward<C>(input), output);
        return output;
    }

    /// Fuses the real and imaginary components.
    template<nt::readable_varray_decay_of_real R,
             nt::readable_varray_decay_of_real I>
    [[nodiscard]] auto complex(R&& real, I&& imag) {
        using real_t = nt::mutable_value_type_t<R>;
        using complex_t = Complex<real_t>;
        Array<complex_t> output(real.shape(), real.options());
        complex(std::forward<R>(real), std::forward<I>(imag), output);
        return output;
    }
}
#endif
