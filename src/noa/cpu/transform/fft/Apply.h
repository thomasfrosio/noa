#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/common/transform/Symmetry.h"

namespace noa::cpu::transform::fft::details {
    using Remap = noa::fft::Remap;

    template<Remap REMAP, typename T, typename SHAPE, typename SHIFT, typename ROT>
    NOA_HOST void applyND(const T* input, T* outputs, SHAPE shape,
                          const ROT* transforms, const SHIFT* shifts, size_t nb_transforms,
                          float max_frequency, InterpMode interp_mode);

    template<Remap REMAP, typename T, typename SHAPE, typename SHIFT, typename ROT>
    NOA_HOST void applyND(const T* input, T* outputs, SHAPE shape,
                          const ROT* transforms, SHIFT shift, size_t nb_transforms,
                          float max_frequency, InterpMode interp_mode);
}

namespace noa::cpu::transform::fft {
    using Remap = noa::fft::Remap;

    /// Rotates/scales a non-redundant FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b host. Non-redundant FFT to transform.
    /// \param[out] outputs     On the \b host. Non-redundant transformed FFT. One per transformation.
    /// \param shape            Logical {fast, medium} shape, in \p T elements, of \p input and \p outputs.
    /// \param[in] transforms   On the \b host. 2x2 inverse rotation/scaling matrix.
    ///                         For a final transformation `A` in the output array, we need to apply `inverse(A)`
    ///                         on the output array coordinates. This function assumes \p transforms are already
    ///                         inverted and pre-multiplies the coordinates with these matrices directly.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S in real
    ///                         space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] shifts       On the \b host. One per transformation. If nullptr or if \p T is real, it is ignored.
    ///                         2D real-space shift to apply (as phase shift) after the transformation.
    /// \param nb_transforms    Number of transformations.
    /// \param max_frequency    Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are left unchanged.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \note \p input and \p outputs should not overlap.
    template<Remap REMAP, typename T>
    NOA_IH void apply2D(const T* input, T* outputs, size2_t shape,
                        const float22_t* transforms, const float2_t* shifts, size_t nb_transforms,
                        float max_frequency, InterpMode interp_mode) {
        details::applyND<REMAP>(input, outputs, shape, transforms, shifts, nb_transforms, max_frequency, interp_mode);
    }

    /// Rotates/scales a non-redundant FFT.
    /// Overload applying the same shift for all transforms.
    template<Remap REMAP, typename T>
    NOA_IH void apply2D(const T* input, T* outputs, size2_t shape,
                        const float22_t* transforms, float2_t shift, size_t nb_transforms,
                        float max_frequency, InterpMode interp_mode) {
        details::applyND<REMAP>(input, outputs, shape, transforms, shift, nb_transforms, max_frequency, interp_mode);
    }

    /// Rotates/scales a non-redundant FFT.
    /// Overload for one single transformation.
    template<Remap REMAP, typename T>
    NOA_IH void apply2D(const T* input, T* output, size2_t shape,
                        float22_t transform, float2_t shift,
                        float max_frequency, InterpMode interp_mode) {
        apply2D<REMAP>(input, output, shape, &transform, &shift, 1, max_frequency, interp_mode);
    }

    /// Rotates/scales a non-redundant FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b host. Non-redundant FFT to transform.
    /// \param[out] outputs     On the \b host. Non-redundant transformed FFT. One per transformation.
    /// \param shape            Logical {fast, medium, slow} shape, in \p T elements, of \p input and \p outputs.
    /// \param[in] transforms   On the \b host. 3x3 inverse rotation/scaling matrix.
    ///                         For a final transformation `A` in the output array, we need to apply `inverse(A)`
    ///                         on the output array coordinates. This function assumes \p transforms are already
    ///                         inverted and pre-multiplies the coordinates with these matrices directly.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S in real
    ///                         space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] shifts       On the \b host. One per transformation. If nullptr or if \p T is real, it is ignored.
    ///                         3D real-space shift to apply (as phase shift) after the transformation.
    /// \param nb_transforms    Number of transformations.
    /// \param max_frequency    Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are left unchanged.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \note \p input and \p outputs should not overlap.
    template<Remap REMAP, typename T>
    NOA_IH void apply3D(const T* input, T* outputs, size3_t shape,
                        const float33_t* transforms, const float3_t* shifts, size_t nb_transforms,
                        float max_frequency, InterpMode interp_mode) {
        details::applyND<REMAP>(input, outputs, shape, transforms, shifts, nb_transforms, max_frequency, interp_mode);
    }

    /// Rotates/scales a non-redundant FFT.
    /// Overload applying the same shift for all transforms.
    template<Remap REMAP, typename T>
    NOA_IH void apply3D(const T* input, T* outputs, size3_t shape,
                        const float33_t* transforms, float3_t shift, size_t nb_transforms,
                        float max_frequency, InterpMode interp_mode) {
        details::applyND<REMAP>(input, outputs, shape, transforms, shift, nb_transforms, max_frequency, interp_mode);
    }

    /// Rotates/scales a non-redundant FFT.
    /// Overload for one single transformation.
    template<Remap REMAP, typename T>
    NOA_IH void apply3D(const T* input, T* output, size3_t shape,
                        float33_t transform, float3_t shift,
                        float max_frequency, InterpMode interp_mode) {
        apply3D<REMAP>(input, output, shape, &transform, &shift, 1, max_frequency, interp_mode);
    }
}

namespace noa::cpu::transform::fft {
    using Symmetry = ::noa::transform::Symmetry;

    /// Rotates/scales and then symmetrizes a non-redundant FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b host. Non-redundant FFT to transform and symmetrize.
    /// \param[out] output      On the \b host. Non-redundant transformed and symmetrized FFT.
    /// \param shape            Logical {fast, medium} shape, in \p T elements, of \p input and \p output.
    /// \param transform        On the \b host. 2x2 inverse rotation/scaling matrix.
    ///                         For a final transformation `A` in the output array, we need to apply `inverse(A)`
    ///                         on the output array coordinates. This function assumes \p transform is already
    ///                         inverted and pre-multiplies the coordinates with this matrix directly.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S in real
    ///                         space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] symmetry     Symmetry operator to apply after the rotation/scaling.
    /// \param max_frequency    Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are left unchanged.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \param normalize        Whether \p output should be normalized to have the same range as \p input.
    ///                         If false, output values end up being scaled by the symmetry count.
    /// \note \p input and \p output should not overlap.
    template<Remap REMAP, typename T>
    NOA_HOST void apply2D(const T* input, T* output, size2_t shape,
                          float22_t transform, const Symmetry& symmetry,
                          float max_frequency, InterpMode interp_mode, bool normalize);

    /// Rotates/scales and then symmetrizes a non-redundant FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b host. Non-redundant FFT to transform and symmetrize.
    /// \param[out] output      On the \b host. Non-redundant transformed and symmetrized FFT.
    /// \param shape            Logical {fast, medium, slow} shape, in \p T elements, of \p input and \p output.
    /// \param transform        On the \b host. 3x3 inverse rotation/scaling matrix.
    ///                         For a final transformation `A` in the output array, we need to apply `inverse(A)`
    ///                         on the output array coordinates. This function assumes \p transform is already
    ///                         inverted and pre-multiplies the coordinates with this matrix directly.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S in real
    ///                         space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] symmetry     Symmetry operator to apply after the rotation/scaling.
    /// \param max_frequency    Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are left unchanged.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \param normalize        Whether \p output should be normalized to have the same range as \p input.
    ///                         If false, output values end up being scaled by the symmetry count.
    /// \note \p input and \p output should not overlap.
    template<Remap REMAP, typename T>
    NOA_HOST void apply3D(const T* input, T* output, size3_t shape,
                          float33_t transform, const Symmetry& symmetry,
                          float max_frequency, InterpMode interp_mode, bool normalize);
}
