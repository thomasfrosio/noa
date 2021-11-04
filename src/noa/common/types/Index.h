/// \file noa/common/types/Index.h
/// \brief Indexing utilities.
/// \author Thomas - ffyr2w
/// \date 10 Dec 2020

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/types/IntX.h"

namespace noa {
    /// Returns the number of rows in a array with a given \a shape.
    template<typename T>
    NOA_FHD constexpr T rows(Int3<T> shape) {
        return shape.y * shape.z;
    }

    /// Returns the number of dimensions encoded in \p shape. Can be either 1, 2 or 3.
    template<typename T>
    NOA_FHD constexpr T ndim(Int3<T> shape) {
        NOA_ASSERT(all(shape >= T{1}));
        return shape.z > 1 ? 3 : shape.y > 1 ? 2 : 1;
    }

    /// Returns the number of dimensions encoded in \p shape. Can be either 1 or 2.
    template<typename T>
    NOA_FHD constexpr T ndim(Int2<T> shape) {
        NOA_ASSERT(all(shape >= T{1}));
        return shape.y > 1 ? 2 : 1;
    }

    /// Returns the linear index corresponding to the (\p i0, \p i1) element in an array
    /// with \p s0 elements in its first dimension.
    /// \tparam O       Compute and output type. Any integral type.
    /// \tparam T, U, V Any integral type.
    /// \tparam U       Any integral type.
    /// \param i0, i1   Index in first (ie innermost) and the second dimension.
    /// \param s0       Size, in elements, of the first dimension.
    /// \return         The linear index.
    template<typename O = size_t, typename T, typename U, typename V>
    NOA_FHD constexpr O index(T i0, U i1, V s0) noexcept {
        static_assert(sizeof(O) >= 4); // indexing with 16-bit integers is too dangerous...
        static_assert(std::is_integral_v<O> || std::is_integral_v<T> ||
                      std::is_integral_v<U> || std::is_integral_v<V>);
        return static_cast<O>(i1) * static_cast<O>(s0) + static_cast<O>(i0);
    }

    template<typename O = size_t, typename T, typename U>
    NOA_FHD constexpr auto index(Int2<T> i, U s) noexcept {
        return index<O>(i.x, i.y, s);
    }

    /// Returns the linear index corresponding to the (\p i0, \p i1, \p i2) element in an array
    /// with \p s0 elements in its first (ie innermost) dimension and \p s1 elements in its second dimension.
    /// \tparam O               Compute and output type. Any integral type.
    /// \tparam T, U, V, W, Y   Any integral type.
    /// \param i0, i1, i2       Index in the first (ie innermost), second and third dimension.
    /// \param s0, s1           Size, in elements, of the first and second dimension.
    /// \return                 The linear index.
    template<typename O = size_t, typename T, typename U, typename V, typename W, typename Y>
    NOA_FHD constexpr auto index(T i0, U i1, V i2, W s0, Y s1) noexcept {
        static_assert(sizeof(O) >= 4); // indexing with 16-bit integers is too dangerous...
        static_assert(std::is_integral_v<O> || std::is_integral_v<T> || std::is_integral_v<U> ||
                      std::is_integral_v<V> || std::is_integral_v<W> || std::is_integral_v<Y>);
        return (static_cast<O>(i2) * static_cast<O>(s1) + static_cast<O>(i1)) *
               static_cast<O>(s0) + static_cast<O>(i0);
    }

    template<typename O = size_t, typename T, typename U>
    NOA_FHD constexpr auto index(Int3<T> i, Int2<U> s) noexcept {
        return index<O>(i.x, i.y, i.z, s.x, s.y);
    }

    template<typename O = size_t, typename T, typename U>
    NOA_FHD constexpr auto index(Int3<T> i, Int3<U> s) noexcept {
        return index<O>(i.x, i.y, i.z, s.x, s.y);
    }

    template<typename O = size_t, typename T, typename U, typename V, typename W>
    NOA_FHD constexpr auto index(T i0, U i1, V i2, Int3<W> s) noexcept {
        return index<O>(i0, i1, i2, s.x, s.y);
    }

    /// Returns the linear index corresponding to the (0, \p i1, \p i2) element in an array
    /// with \p s0 elements in its first (ie innermost) dimension and \p s1 elements in its second dimension.
    /// \tparam O           Compute and output type. Any integral type.
    /// \tparam U, V, W, Y  Any integral type.
    /// \param i1, i2       Index in second and third dimension.
    /// \param s0, s1       Size, in elements, of the first and second dimension.
    /// \return             The linear index.
    template<typename O = size_t, typename U, typename V, typename W, typename Y>
    NOA_FHD constexpr auto index(U i1, V i2, W s0, Y s1) noexcept {
        static_assert(sizeof(O) >= 4); // indexing with 16-bit integers is too dangerous...
        static_assert(std::is_integral_v<O> || std::is_integral_v<U> ||
                      std::is_integral_v<V> || std::is_integral_v<W> || std::is_integral_v<Y>);
        return (static_cast<O>(i2) * static_cast<O>(s1) + static_cast<O>(i1)) *
               static_cast<O>(s0);
    }

    template<typename O = size_t, typename T, typename U, typename V>
    NOA_FHD constexpr auto index(T i1, U i2, Int2<V> s) noexcept {
        return index<O>(i1, i2, s.x, s.y);
    }

    template<typename O = size_t, typename T, typename U, typename V>
    NOA_FHD constexpr auto index(T i1, U i2, Int3<V> s) noexcept {
        return index<O>(i1, i2, s.x, s.y);
    }

    /// Returns the {i0, i1} coordinates corresponding to the linear index \p i.
    /// \tparam T   Any integral type.
    /// \param i    Linear index to decompose.
    /// \param s0   Size of the first (ie innermost) dimension.
    /// \return     2D coordinates {i0, i1}.
    template<typename T>
    NOA_FHD constexpr Int2<T> coordinates(T i, T s0) noexcept {
        T cy = i / s0;
        T cx = i - cy * s0;
        return {cx, cy};
    }

    /// Returns the {i0, i1, i2} coordinates corresponding to the linear index \p i.
    /// \tparam T       Any integral type.
    /// \param i        Linear index to decompose.
    /// \param s0, s1   Size of the first (ie innermost) and second dimension.
    /// \return         3D coordinates {i0, i1, i2}.
    template<typename T>
    NOA_FHD constexpr Int3<T> coordinates(T i, T s0, T s1) noexcept {
        T i2 = i / (s1 * s0);
        T tmp = i - i2 * s1 * s0;
        T i1 = tmp / s0;
        T i0 = tmp - i1 * s0;
        return {i0, i1, i2};
    }
}
