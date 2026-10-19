#pragma once

#include "noa/runtime/Array.hpp"
#include "noa/runtime/Ewise.hpp"

namespace noa {
    /// Extracts the real and imaginary part of complex numbers.
    /// \param[in] input    Complex array to decompose.
    /// \param[out] real    Real elements.
    /// \param[out] imag    Imaginary elements.
    template<nt::readable_array_decay_of_complex C,
             nt::writable_array_decay_of_real R,
             nt::writable_array_decay_of_real I>
    void decompose(C&& input, R&& real, I&& imag) {
        check(not are_overlapped(real, imag), "The output arrays should not overlap");
        ewise(NOA_FWD(input), wrap(NOA_FWD(real), NOA_FWD(imag)), ComplexDecompose{});
    }

    /// Extracts the real part of complex numbers.
    /// \param[in] input    Complex array to decompose.
    /// \param[out] real    Real elements.
    template<nt::readable_array_decay_of_complex C,
             nt::writable_array_decay_of_real R>
    void real(C&& input, R&& real) {
        ewise(NOA_FWD(input), NOA_FWD(real), Real{});
    }

    /// Extracts the imaginary part of complex numbers.
    /// \param[in] input    Complex array to decompose.
    /// \param[out] imag    Imaginary elements.
    template<nt::readable_array_decay_of_complex C,
             nt::writable_array_decay_of_real I>
    void imag(C&& input, I&& imag) {
        ewise(NOA_FWD(input), NOA_FWD(imag), Imag{});
    }

    /// Fuses the real and imaginary components.
    /// \param[in] real Real elements to interleave.
    /// \param[in] imag Imaginary elements to interleave.
    /// \param output   Complex array.
    template<nt::readable_array_decay_of_real R,
             nt::readable_array_decay_of_real I,
             nt::writable_array_decay_of_complex C>
    void complex(R&& real, I&& imag, C&& output) {
        ewise(wrap(NOA_FWD(real), NOA_FWD(imag)), NOA_FWD(output), ComplexFuse{});
    }
}

namespace noa {
    /// Extracts the real and imaginary part of complex numbers.
    template<nt::readable_array_decay_of_complex C>
    [[nodiscard]] auto decompose(C&& input) {
        using real_t = nt::value_type_twice_t<C>;
        constexpr auto N = std::remove_reference_t<C>::SIZE;
        Pair output{Array<real_t, N>(input.shape(), input.options()),
                    Array<real_t, N>(input.shape(), input.options())};
        decompose(NOA_FWD(input), output.first, output.second);
        return output;
    }

    /// Extracts the real part of complex numbers.
    template<nt::readable_array_decay_of_complex C>
    [[nodiscard]] auto real(C&& input) {
        using real_t = nt::value_type_twice_t<C>;
        constexpr auto N = std::remove_reference_t<C>::SIZE;
        Array<real_t, N> output(input.shape(), input.options());
        real(NOA_FWD(input), output);
        return output;
    }

    /// Extracts the imaginary part of complex numbers.
    template<nt::readable_array_decay_of_complex C>
    [[nodiscard]] auto imag(C&& input) {
        using real_t = nt::value_type_twice_t<C>;
        constexpr auto N = std::remove_reference_t<C>::SIZE;
        Array<real_t, N> output(input.shape(), input.options());
        imag(NOA_FWD(input), output);
        return output;
    }

    /// Fuses the real and imaginary components.
    template<nt::readable_array_decay_of_real R,
             nt::readable_array_decay_of_real I>
    [[nodiscard]] auto complex(R&& real, I&& imag) {
        using real_t = nt::mutable_value_type_t<R>;
        using complex_t = Complex<real_t>;
        constexpr auto N = nd::maximum_nd_axes_of_arrays<R, I>();
        Array<complex_t, N> output(real.shape(), real.options());
        complex(NOA_FWD(real), NOA_FWD(imag), output);
        return output;
    }
}
