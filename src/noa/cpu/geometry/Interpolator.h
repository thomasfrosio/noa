/// \file noa/cpu/geometry/Interpolator.h
/// \brief 1D, 2D and 3D interpolators.
/// \author Thomas - ffyr2w
/// \date 20 Jul 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/common/Math.h"
#include "noa/cpu/geometry/Interpolate.h"

// On of the main difference between these Interpolators and what we can find on other cryoEM packages,
// is that the interpolation window can be partially OOB, that is, elements that are OOB are replaced
// according to a BorderMode. cryoEM packages usually check that all elements are in bound and if there's
// even one element OOB, they don't interpolate.
// Note: These Interpolators are for real space interpolation or redundant and centered Fourier transforms.

// The coordinate system matches the indexing. The coordinate is the floating-point passed to `get<>()`.
// For instance the first data sample at index 0 is located at the coordinate 0 and the coordinate 0.5
// is just in between the first and second element. As such, the fractional part of the coordinate
// corresponds to the ratio/weight used by the interpolation function. In other words,
// the coordinate system locates the data between -0.5 and N-1 + 0.5.

namespace noa::cpu::geometry {
    // Interpolates 1D data.
    template<typename T, AccessorTraits TRAITS = AccessorTraits::DEFAULT>
    class Interpolator1D {
    public:
        using index_t = int64_t; // must be signed
        using accessor_t = Accessor<T, 1, index_t, TRAITS>;
        using accessor_reference_t = AccessorReference<T, 1, index_t, TRAITS>;
        using mutable_value_t = traits::remove_ref_cv_t<T>;

    public:
        // Sets the data points.
        template<typename U>
        Interpolator1D(T* input, dim_t stride, dim_t size, U value) noexcept;

        template<typename I, typename U>
        Interpolator1D(const Accessor<T, 1, I, TRAITS>& input, dim_t size, U value) noexcept;

        template<typename I, typename U>
        Interpolator1D(const AccessorReference<T, 1, I, TRAITS>& input, dim_t size, U value) noexcept;

        // Returns the interpolated value at the coordinate x.
        template<InterpMode INTERP, BorderMode BORDER>
        auto get(float x) const;

        // Returns the interpolated value at the coordinate x.
        // Offset is the temporary memory offset to apply to the underlying array.
        // This is used for instance to change batches.
        template<InterpMode INTERP, BorderMode BORDER>
        auto get(float x, dim_t offset) const;

    private:
        template<BorderMode BORDER>
        auto nearest_(accessor_reference_t data, float x) const;

        template<BorderMode BORDER, bool COSINE>
        auto linear_(accessor_reference_t data, float x) const;

        template<BorderMode BORDER, bool BSPLINE>
        auto cubic_(accessor_reference_t data, float x) const;

    private:
        accessor_t m_data{};
        mutable_value_t m_value{};
        index_t m_size{};
    };

    // Interpolates 2D data.
    template<typename T, AccessorTraits TRAITS = AccessorTraits::DEFAULT>
    class Interpolator2D {
    public:
        using index_t = int64_t;
        using index2_t = Int2<index_t>;
        using mutable_value_t = traits::remove_ref_cv_t<T>;
        using const_value_t = const mutable_value_t;
        using accessor_t = Accessor<const_value_t, 2, index_t, TRAITS>;
        using accessor_reference_t = AccessorReference<const_value_t, 2, index_t, TRAITS>;

    public:
        // Sets the data points.
        template<typename U>
        Interpolator2D(T* input, dim2_t strides, dim2_t shape, U value) noexcept;

        template<typename I0, typename T1>
        Interpolator2D(const Accessor<T, 2, I0, TRAITS>& input, dim2_t shape, T1 value) noexcept;

        template<typename I0, typename T1>
        Interpolator2D(const AccessorReference<T, 2, I0, TRAITS>& input, dim2_t shape, T1 value) noexcept;

        // Returns the interpolated value at the coordinate x and y.
        template<InterpMode INTERP, BorderMode BORDER>
        auto get(float2_t coords) const;

        // Returns the interpolated value at the coordinate x and y.
        // Offset is the temporary memory offset to apply to the underlying array.
        // This is used for instance to change batches.
        template<InterpMode INTERP, BorderMode BORDER>
        auto get(float2_t coords, dim_t offset) const;

    private:
        template<BorderMode BORDER>
        auto nearest_(accessor_reference_t data, float y, float x) const;
        template<BorderMode BORDER, bool COSINE>
        auto linear_(accessor_reference_t data, float y, float x) const;
        template<BorderMode BORDER, bool BSPLINE>
        auto cubic_(accessor_reference_t data, float y, float x) const;

    private:
        accessor_t m_data{};
        index2_t m_shape{};
        mutable_value_t m_value{};
    };

    // Interpolates 3D data.
    template<typename T, AccessorTraits TRAITS = AccessorTraits::DEFAULT>
    class Interpolator3D {
    public:
        using index_t = int64_t;
        using index3_t = Int3<index_t>;
        using accessor_t = Accessor<T, 3, index_t, TRAITS>;
        using accessor_reference_t = AccessorReference<T, 3, index_t, TRAITS>;
        using mutable_value_t = traits::remove_ref_cv_t<T>;

    public:
        // Sets the data points.
        template<typename U>
        Interpolator3D(T* input, dim3_t strides, dim3_t shape, U value) noexcept;

        template<typename I, typename U>
        Interpolator3D(const Accessor<T, 3, I, TRAITS>& input, dim3_t shape, U value) noexcept;

        template<typename I, typename U>
        Interpolator3D(const AccessorReference<T, 3, I, TRAITS>& input, dim3_t shape, U value) noexcept;

        // Returns the interpolated value at the coordinate \p x, \p y, \p z.
        template<InterpMode INTERP, BorderMode BORDER>
        auto get(float3_t coords) const;

        // Returns the interpolated value at the coordinate \p x, \p y, \p z.
        // Offset is the temporary memory offset to apply to the underlying array.
        // This is used for instance to change batches.
        template<InterpMode INTERP, BorderMode BORDER>
        auto get(float3_t coords, dim_t offset) const;

    private:
        template<BorderMode BORDER>
        auto nearest_(accessor_reference_t data, float z, float y, float x) const;
        template<BorderMode BORDER, bool COSINE>
        auto linear_(accessor_reference_t data, float z, float y, float x) const;
        template<BorderMode BORDER, bool BSPLINE>
        auto cubic_(accessor_reference_t data, float z, float y, float x) const;

    private:
        accessor_t m_data{};
        index3_t m_shape{};
        mutable_value_t m_value{};
    };
}

#define NOA_CPU_INTERPOLATOR_
#include "noa/cpu/geometry/Interpolator.inl"
#undef NOA_CPU_INTERPOLATOR_
