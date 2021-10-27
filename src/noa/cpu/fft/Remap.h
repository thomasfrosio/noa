/// \file noa/cpu/fft/Remap.h
/// \brief Remap FFTs.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"

namespace noa::cpu::fft::details {
    template<typename T>
    NOA_HOST void hc2h(const T* inputs, T* outputs, size3_t shape, size_t batches);

    template<typename T>
    NOA_HOST void h2hc(const T* inputs, T* outputs, size3_t shape, size_t batches);

    template<typename T>
    NOA_HOST void fc2f(const T* inputs, T* outputs, size3_t shape, size_t batches);

    template<typename T>
    NOA_HOST void f2fc(const T* inputs, T* outputs, size3_t shape, size_t batches);

    template<typename T>
    NOA_HOST void h2f(const T* inputs, T* outputs, size3_t shape, size_t batches);

    template<typename T>
    NOA_HOST void f2h(const T* inputs, T* outputs, size3_t shape, size_t batches);

    template<typename T>
    NOA_HOST void fc2h(const T* inputs, T* outputs, size3_t shape, size_t batches);

    template<typename T>
    NOA_HOST void hc2f(const T* inputs, T* outputs, size3_t shape, size_t batches);

    template<typename T>
    NOA_HOST void f2hc(const T* inputs, T* outputs, size3_t shape, size_t batches);
}

namespace noa::cpu::fft {
    using Remap = ::noa::fft::Remap;

    /// Remaps FFT(s).
    /// \tparam T               float, double, cfloat_t or cdouble_t.
    /// \param[in] inputs       On the \b host. Input FFT to remap.
    /// \param[out] outputs     On the \b host. Remapped FFT.
    /// \param shape            Logical {fast, medium, slow} shape, in \p T elements.
    /// \param batches          Number of contiguous batches to compute.
    /// \param remap            Remapping operation. \p H2FC is not supported. See noa::fft::Remap for more details.
    /// \note If \p remap is \c H2FC, \p inputs can be equal to \p outputs, only if \p shape.y is even,
    ///       and if \p shape.z is even or 1, otherwise, they should not overlap.
    template<typename T>
    NOA_IH void remap(const T* inputs, T* outputs, size3_t shape, uint batches, fft::Remap remap) {
        switch (remap) {
            case Remap::H2HC:
                return details::h2hc(inputs, outputs, shape, batches);
            case Remap::HC2H:
                return details::hc2h(inputs, outputs, shape, batches);
            case Remap::H2F:
                return details::h2f(inputs, outputs, shape, batches);
            case Remap::F2H:
                return details::f2h(inputs, outputs, shape, batches);
            case Remap::F2FC:
                return details::f2fc(inputs, outputs, shape, batches);
            case Remap::FC2F:
                return details::fc2f(inputs, outputs, shape, batches);
            case Remap::HC2F:
                return details::hc2f(inputs, outputs, shape, batches);
            case Remap::F2HC:
                return details::f2hc(inputs, outputs, shape, batches);
            case Remap::FC2H:
                return details::fc2h(inputs, outputs, shape, batches);
            case Remap::H2FC:
                NOA_THROW("{} is currently not supported");
                // TODO H2FC is missing, since it seems a bit more complicated and it would be surprising
                //      if we ever use it. Moreover, the same can be achieved with h2f and then f2fc.
        }
    }
}
