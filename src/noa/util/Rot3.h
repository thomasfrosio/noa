#pragma once

#include "noa/Definitions.h"
#include "noa/util/Float3.h"
#include "noa/util/traits/BaseTypes.h"

namespace Noa {
    template<typename T>
    class Rot3 {
    private:
        static constexpr uint ROWS = 3U;
        static constexpr uint COLS = 3U;
        Float3<T> m_data[ROWS];

    public: // Type definitions
        typedef T value_type;

    public: // Component accesses
        [[nodiscard]] NOA_HD static constexpr size_t elements() noexcept { return 3; }
        [[nodiscard]] NOA_HD static constexpr size_t size() noexcept { return elements(); }
        NOA_HD constexpr Float3<T>& operator[](size_t i) { return m_data[i]; }
        NOA_HD constexpr const Float3<T>& operator[](size_t i) const { return m_data[i]; }

    public: // (Conversion) Constructors
        NOA_HD constexpr Rot3() noexcept;
        template<typename U> NOA_HD constexpr explicit Rot3(U s) noexcept;

        template<typename X0, typename Y0, typename Z0,
                 typename X1, typename Y1, typename Z1,
                 typename X2, typename Y2, typename Z2>
        NOA_HD constexpr Rot3(X0 x0, Y0 y0, Z0 z0,
                              X1 x1, Y1 y1, Z1 z1,
                              X2 x2, Y2 y2, Z2 z2) noexcept;

        template<typename V0, typename V1, typename V2>
        NOA_HD constexpr Rot3(const Float3<V0>& v0,
                              const Float3<V1>& v1,
                              const Float3<V2>& v2) noexcept;

    public: // Assignment operators
        template<typename U> NOA_HD constexpr Rot3<T>& operator=(const Rot3<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Rot3<T>& operator+=(const Rot3<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Rot3<T>& operator-=(const Rot3<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Rot3<T>& operator*=(const Rot3<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Rot3<T>& operator/=(const Rot3<U>& rhs) noexcept;

        template<typename U> NOA_HD constexpr Rot3<T>& operator+=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Rot3<T>& operator-=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Rot3<T>& operator*=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Rot3<T>& operator/=(U rhs) noexcept;

    public: // Factory functions
        NOA_HD static constexpr Rot3<T> eye() noexcept;
        NOA_HD static constexpr Rot3<T> identity() noexcept;

        template<typename U> NOA_HD static constexpr Rot3<T> rotationX(U angle);
        template<typename U> NOA_HD static constexpr Rot3<T> rotationY(U angle);
        template<typename U> NOA_HD static constexpr Rot3<T> rotationZ(U angle);
        template<typename U> NOA_HD static constexpr Rot3<T> scale(U mag) noexcept;

    public: // Public functions
        template<typename U> NOA_HD constexpr void rotationX(U angle);
        template<typename U> NOA_HD constexpr void rotationY(U angle);
        template<typename U> NOA_HD constexpr void rotationZ(U angle);
        template<typename U> NOA_HD constexpr void scale(U mag) noexcept;

        NOA_HD constexpr void transpose();
        NOA_HD constexpr void inverse();
    };

    template<typename T> NOA_HD constexpr Rot3<T> operator+(const Rot3<U>& rhs) noexcept;

    using rot3_t = Rot3<float>;

    // -- Conversion constructors --

    template<typename T>
    NOA_HD constexpr Rot3<T>::Rot3() noexcept
            : m_data{{1, 0, 0},
                     {0, 1, 0},
                     {0, 0, 1}} {}

    template<typename T>
    template<typename U>
    NOA_HD constexpr explicit Rot3<T>::Rot3(U s) noexcept
            : m_data{{s, 0, 0},
                     {0, s, 0},
                     {0, 0, s}} {}

    template<typename T>
    template<typename X0, typename Y0, typename Z0,
             typename X1, typename Y1, typename Z1,
             typename X2, typename Y2, typename Z2>
    NOA_HD constexpr Rot3<T>::Rot3(X0 x0, Y0 y0, Z0 z0,
                                   X1 x1, Y1 y1, Z1 z1,
                                   X2 x2, Y2 y2, Z2 z2) noexcept
            : m_data{Float3<T>(x0, y0, z0), Float3<T>(x1, y1, z1), Float3<T>(x2, y2, z2)} {}

    template<typename T>
    template<typename V0, typename V1, typename V2>
    NOA_HD constexpr Rot3<T>::Rot3(const Float3<V0>& v0, const Float3<V1>& v1, const Float3<V2>& v2) noexcept
            : m_data{v0, v1, v2} {}



    template<typename T>
    template<typename U>
    constexpr Rot3<T>& Rot3<T>::operator=(const Rot3<U>& rhs) noexcept {
        m_data[0] = rhs[0];
        m_data[1] = rhs[1];
        m_data[2] = rhs[2];
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Rot3<T>& Rot3<T>::operator+=(const Rot3<U>& rhs) noexcept {
        m_data[0] += rhs[0];
        m_data[1] += rhs[1];
        m_data[2] += rhs[2];
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Rot3<T>& Rot3<T>::operator-=(const Rot3<U>& rhs) noexcept {
        m_data[0] -= rhs[0];
        m_data[1] -= rhs[1];
        m_data[2] -= rhs[2];
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Rot3<T>& Rot3<T>::operator*=(const Rot3<U>& rhs) noexcept {
        const T A00 = m_data[0][0];
        const T A01 = m_data[0][1];
        const T A02 = m_data[0][2];
        const T A10 = m_data[1][0];
        const T A11 = m_data[1][1];
        const T A12 = m_data[1][2];
        const T A20 = m_data[2][0];
        const T A21 = m_data[2][1];
        const T A22 = m_data[2][2];

        const T B00 = static_cast<T>(rhs[0][0]);
        const T B01 = static_cast<T>(rhs[0][1]);
        const T B02 = static_cast<T>(rhs[0][2]);
        const T B10 = static_cast<T>(rhs[1][0]);
        const T B11 = static_cast<T>(rhs[1][1]);
        const T B12 = static_cast<T>(rhs[1][2]);
        const T B20 = static_cast<T>(rhs[2][0]);
        const T B21 = static_cast<T>(rhs[2][1]);
        const T B22 = static_cast<T>(rhs[2][2]);

        m_data[0][0] = A00 * B00 + A10 * B01 + A20 * B02;
        m_data[0][1] = A01 * B00 + A11 * B01 + A21 * B02;
        m_data[0][2] = A02 * B00 + A12 * B01 + A22 * B02;
        m_data[1][0] = A00 * B10 + A10 * B11 + A20 * B12;
        m_data[1][1] = A01 * B10 + A11 * B11 + A21 * B12;
        m_data[1][2] = A02 * B10 + A12 * B11 + A22 * B12;
        m_data[2][0] = A00 * B20 + A10 * B21 + A20 * B22;
        m_data[2][1] = A01 * B20 + A11 * B21 + A21 * B22;
        m_data[2][2] = A02 * B20 + A12 * B21 + A22 * B22;
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Rot3<T>& Rot3<T>::operator+=(U rhs) noexcept {
        m_data[0] += rhs;
        m_data[1] += rhs;
        m_data[2] += rhs;
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Rot3<T>& Rot3<T>::operator-=(U rhs) noexcept {
        m_data[0] -= rhs;
        m_data[1] -= rhs;
        m_data[2] -= rhs;
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Rot3<T>& Rot3<T>::operator*=(U rhs) noexcept {
        m_data[0] *= rhs;
        m_data[1] *= rhs;
        m_data[2] *= rhs;
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Rot3<T>& Rot3<T>::operator/=(U rhs) noexcept {
        m_data[0] /= rhs;
        m_data[1] /= rhs;
        m_data[2] /= rhs;
        return *this;
    }

    // In Noa::Math:: or in Noa:: ? Probably Noa::Math::
    // dot
    // cross
    // transpose

    // Rot3<T> transpose(const Rot3<T>&)
}
