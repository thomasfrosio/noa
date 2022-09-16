/// \file noa/common/types/Mat44.h
/// \author Thomas - ffyr2w
/// \date 2 Jun 2021
/// A 4x4 floating-point matrix.

#pragma once

#include "noa/common/Assert.h"
#include "noa/common/Definitions.h"
#include "noa/common/Math.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/traits/ArrayTypes.h"
#include "noa/common/types/Float4.h"

// A few necessary forward declarations:
namespace noa {
    template<typename T>
    class Mat34;

    template<typename T>
    class Mat33;

    template<typename T>
    class Float3;

    template<typename T>
    class Int4;

    template<typename T>
    class Mat44;

    namespace math {
        template<typename T>
        NOA_IHD constexpr Mat44<T> transpose(Mat44<T> m) noexcept;

        template<typename T>
        NOA_IHD constexpr Mat44<T> inverse(Mat44<T> m) noexcept;
    }
}

namespace noa {
    /// A 4x4 floating-point matrix.
    /// \note The indexing is "row-first" (as opposed to "column-first", like in OpenGL Math),
    ///       i.e. M[r][c] with r = row index and c = column index. All indexes starts from 0.
    template<typename T>
    class Mat44 {
    public: // Type definitions
        using value_type = T;

    public: // Component accesses
        static constexpr size_t ROWS = 4;
        static constexpr size_t COLS = 4;
        static constexpr size_t COUNT = ROWS * COLS;

        template<typename I, typename = std::enable_if_t<traits::is_int_v<I>>>
        NOA_HD constexpr Float4<T>& operator[](I i) noexcept {
            NOA_ASSERT(static_cast<size_t>(i) < ROWS);
            return m_row[i];
        }

        template<typename I, typename = std::enable_if_t<traits::is_int_v<I>>>
        NOA_HD constexpr const Float4<T>& operator[](I i) const noexcept {
            NOA_ASSERT(static_cast<size_t>(i) < ROWS);
            return m_row[i];
        }

    public: // Default constructors
        NOA_HD constexpr Mat44() noexcept
                : m_row{Float4<T>(1, 0, 0, 0),
                        Float4<T>(0, 1, 0, 0),
                        Float4<T>(0, 0, 1, 0),
                        Float4<T>(0, 0, 0, 1)} {}

        constexpr Mat44(const Mat44&) noexcept = default;
        constexpr Mat44(Mat44&&) noexcept = default;

    public: // (Conversion) Constructors
        template<typename U, typename = std::enable_if_t<traits::is_scalar_v<U>>>
        NOA_HD constexpr explicit Mat44(U s) noexcept
                : m_row{Float4<T>(s, 0, 0, 0),
                        Float4<T>(0, s, 0, 0),
                        Float4<T>(0, 0, s, 0),
                        Float4<T>(0, 0, 0, s)} {}

        template<typename U>
        NOA_HD constexpr explicit Mat44(Float4<U> v) noexcept
                : m_row{Float4<T>(v[0], 0, 0, 0),
                        Float4<T>(0, v[1], 0, 0),
                        Float4<T>(0, 0, v[2], 0),
                        Float4<T>(0, 0, 0, v[3])} {}

        template<typename U>
        NOA_HD constexpr explicit Mat44(Float3<U> v) noexcept
                : m_row{Float4<T>(v[0], 0, 0, 0),
                        Float4<T>(0, v[1], 0, 0),
                        Float4<T>(0, 0, v[2], 0),
                        Float4<T>(0, 0, 0, 1)} {}

        template<typename U>
        NOA_HD constexpr explicit Mat44(Mat44<U> m) noexcept
                : m_row{Float4<T>(m[0]),
                        Float4<T>(m[1]),
                        Float4<T>(m[2]),
                        Float4<T>(m[3])} {}

        template<typename U>
        NOA_HD constexpr explicit Mat44(Mat34<U> m) noexcept
                : m_row{Float4<T>(m[0]),
                        Float4<T>(m[1]),
                        Float4<T>(m[2]),
                        Float4<T>(0, 0, 0, 1)} {}

        template<typename U>
        NOA_HD constexpr explicit Mat44(Mat33<U> m) noexcept
                : m_row{Float4<T>(m[0][0], m[0][1], m[0][2], 0),
                        Float4<T>(m[1][0], m[1][1], m[1][2], 0),
                        Float4<T>(m[2][0], m[2][1], m[2][2], 0),
                        Float4<T>(0, 0, 0, 1)} {}

        template<typename U, typename V>
        NOA_HD constexpr explicit Mat44(Mat33<U> m, Float3<V> v) noexcept
                : m_row{Float4<T>(m[0][0], m[0][1], m[0][2], v[0]),
                        Float4<T>(m[1][0], m[1][1], m[1][2], v[1]),
                        Float4<T>(m[2][0], m[2][1], m[2][2], v[2]),
                        Float4<T>(0, 0, 0, 1)} {}

        template<typename X00, typename X01, typename X02, typename X03,
                 typename Y10, typename Y11, typename Y12, typename Y13,
                 typename Z20, typename Z21, typename Z22, typename Z23,
                 typename W30, typename W31, typename W32, typename W33>
        NOA_HD constexpr Mat44(X00 x00, X01 x01, X02 x02, X03 x03,
                               Y10 y10, Y11 y11, Y12 y12, Y13 y13,
                               Z20 z20, Z21 z21, Z22 z22, Z23 z23,
                               W30 w30, W31 w31, W32 w32, W33 w33) noexcept
                : m_row{Float4<T>(x00, x01, x02, x03),
                        Float4<T>(y10, y11, y12, y13),
                        Float4<T>(z20, z21, z22, z23),
                        Float4<T>(w30, w31, w32, w33)} {}

        template<typename U, typename = std::enable_if_t<traits::is_scalar_v<U>>>
        NOA_HD constexpr explicit Mat44(U* ptr) noexcept
                : m_row{Float4<T>(ptr[0], ptr[1], ptr[2], ptr[3]),
                        Float4<T>(ptr[4], ptr[5], ptr[6], ptr[7]),
                        Float4<T>(ptr[8], ptr[9], ptr[10], ptr[11]),
                        Float4<T>(ptr[12], ptr[13], ptr[14], ptr[15])} {}

        template<typename V0, typename V1, typename V2, typename V3>
        NOA_HD constexpr Mat44(Float4<V0> r0,
                               Float4<V1> r1,
                               Float4<V2> r2,
                               Float4<V3> r3) noexcept
                : m_row{Float4<T>(r0),
                        Float4<T>(r1),
                        Float4<T>(r2),
                        Float4<T>(r3)} {}

        template<typename V0, typename V1, typename V2, typename V3>
        NOA_HD constexpr Mat44(Int4<V0> r0,
                               Int4<V1> r1,
                               Int4<V2> r2,
                               Int4<V3> r3) noexcept
                : m_row{Float4<T>(r0),
                        Float4<T>(r1),
                        Float4<T>(r2),
                        Float4<T>(r3)} {}

    public: // Assignment operators
        constexpr Mat44& operator=(const Mat44& v) noexcept = default;
        constexpr Mat44& operator=(Mat44&& v) noexcept = default;

        NOA_HD constexpr Mat44& operator+=(Mat44 m) noexcept {
            m_row[0] += m[0];
            m_row[1] += m[1];
            m_row[2] += m[2];
            m_row[3] += m[3];
            return *this;
        }

        NOA_HD constexpr Mat44& operator-=(Mat44 m) noexcept {
            m_row[0] -= m[0];
            m_row[1] -= m[1];
            m_row[2] -= m[2];
            m_row[3] -= m[3];
            return *this;
        }

        NOA_HD constexpr Mat44& operator*=(Mat44 m) noexcept {
            const Float4<T> A0 = m_row[0];
            const Float4<T> A1 = m_row[1];
            const Float4<T> A2 = m_row[2];
            const Float4<T> A3 = m_row[3];

            const Float4<T> B0 = m[0];
            const Float4<T> B1 = m[1];
            const Float4<T> B2 = m[2];
            const Float4<T> B3 = m[3];

            m_row[0] = A0[0] * B0 + A0[1] * B1 + A0[2] * B2 + A0[3] * B3;
            m_row[1] = A1[0] * B0 + A1[1] * B1 + A1[2] * B2 + A1[3] * B3;
            m_row[2] = A2[0] * B0 + A2[1] * B1 + A2[2] * B2 + A2[3] * B3;
            m_row[3] = A3[0] * B0 + A3[1] * B1 + A3[2] * B2 + A3[3] * B3;
            return *this;
        }

        NOA_HD constexpr Mat44& operator/=(Mat44 m) noexcept {
            *this *= math::inverse(m);
            return *this;
        }

        NOA_HD constexpr Mat44& operator+=(T s) noexcept {
            m_row[0] += s;
            m_row[1] += s;
            m_row[2] += s;
            m_row[3] += s;
            return *this;
        }

        NOA_HD constexpr Mat44& operator-=(T s) noexcept {
            m_row[0] -= s;
            m_row[1] -= s;
            m_row[2] -= s;
            m_row[3] -= s;
            return *this;
        }

        NOA_HD constexpr Mat44& operator*=(T s) noexcept {
            m_row[0] *= s;
            m_row[1] *= s;
            m_row[2] *= s;
            m_row[3] *= s;
            return *this;
        }

        NOA_HD constexpr Mat44& operator/=(T s) noexcept {
            m_row[0] /= s;
            m_row[1] /= s;
            m_row[2] /= s;
            m_row[3] /= s;
            return *this;
        }

    public: // Non-member functions
        // -- Unary operators --
        [[nodiscard]] friend NOA_HD constexpr Mat44 operator+(Mat44 m) noexcept {
            return m;
        }

        [[nodiscard]] friend NOA_HD constexpr Mat44 operator-(Mat44 m) noexcept {
            return Mat44(-m[0], -m[1], -m[2], -m[3]);
        }

        // -- Binary arithmetic operators --
        [[nodiscard]] friend NOA_HD constexpr Mat44 operator+(Mat44 m1, Mat44 m2) noexcept {
            return Mat44(m1[0] + m2[0], m1[1] + m2[1], m1[2] + m2[2], m1[3] + m2[3]);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat44 operator+(T s, Mat44 m) noexcept {
            return Mat44(s + m[0], s + m[1], s + m[2], s + m[3]);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat44 operator+(Mat44 m, T s) noexcept {
            return Mat44(m[0] + s, m[1] + s, m[2] + s, m[3] + s);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat44 operator-(Mat44 m1, Mat44 m2) noexcept {
            return Mat44(m1[0] - m2[0], m1[1] - m2[1], m1[2] - m2[2], m1[3] - m2[3]);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat44 operator-(T s, Mat44 m) noexcept {
            return Mat44(s - m[0], s - m[1], s - m[2], s - m[3]);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat44 operator-(Mat44 m, T s) noexcept {
            return Mat44(m[0] - s, m[1] - s, m[2] - s, m[3] - s);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat44 operator*(Mat44 m1, Mat44 m2) noexcept {
            Mat44 out(m1);
            out *= m2;
            return out;
        }

        [[nodiscard]] friend NOA_HD constexpr Mat44 operator*(T s, Mat44 m) noexcept {
            return Mat44(m[0] * s, m[1] * s, m[2] * s, m[3] * s);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat44 operator*(Mat44 m, T s) noexcept {
            return Mat44(m[0] * s, m[1] * s, m[2] * s, m[3] * s);
        }

        [[nodiscard]] friend NOA_HD constexpr Float4<T> operator*(Mat44 m, const Float4<T>& column) noexcept {
            return Float4<T>(math::dot(m[0], column),
                             math::dot(m[1], column),
                             math::dot(m[2], column),
                             math::dot(m[3], column));
        }

        [[nodiscard]] friend NOA_HD constexpr Float4<T> operator*(const Float4<T>& row, Mat44 m) noexcept {
            return Float4<T>(math::dot(Float4<T>(m[0][0], m[1][0], m[2][0], m[3][0]), row),
                             math::dot(Float4<T>(m[0][1], m[1][1], m[2][1], m[3][1]), row),
                             math::dot(Float4<T>(m[0][2], m[1][2], m[2][2], m[3][2]), row),
                             math::dot(Float4<T>(m[0][3], m[1][3], m[2][3], m[3][3]), row));
        }

        [[nodiscard]] friend NOA_HD constexpr Mat44 operator/(Mat44 m1, Mat44 m2) noexcept {
            Mat44 out(m1);
            out /= m2;
            return out;
        }

        [[nodiscard]] friend NOA_HD constexpr Mat44 operator/(T s, Mat44 m) noexcept {
            return Mat44(s / m[0], s / m[1], s / m[2], s / m[3]);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat44 operator/(Mat44 m, T s) noexcept {
            return Mat44(m[0] / s, m[1] / s, m[2] / s, m[3] / s);
        }

        [[nodiscard]] friend NOA_HD constexpr Float4<T> operator/(Mat44 m, const Float4<T>& column) noexcept {
            return math::inverse(m) * column;
        }

        [[nodiscard]] friend NOA_HD constexpr Float4<T> operator/(const Float4<T>& row, Mat44 m) noexcept {
            return row * math::inverse(m);
        }

        [[nodiscard]] friend NOA_HD constexpr bool operator==(Mat44 m1, Mat44 m2) noexcept {
            return all(m1[0] == m2[0]) && all(m1[1] == m2[1]) && all(m1[2] == m2[2]) && all(m1[3] == m2[3]);
        }

        [[nodiscard]] friend NOA_HD constexpr bool operator!=(Mat44 m1, Mat44 m2) noexcept {
            return all(m1[0] != m2[0]) && all(m1[1] != m2[1]) && all(m1[2] != m2[2]) && all(m1[3] != m2[3]);
        }

    public:
        [[nodiscard]] NOA_HD constexpr const T* get() const noexcept { return m_row[0].get(); }
        [[nodiscard]] NOA_HD constexpr T* get() noexcept { return m_row[0].get(); }

        [[nodiscard]] NOA_IHD constexpr Mat44 transpose() const noexcept {
            return math::transpose(*this);
        }

    private:
        Float4<T> m_row[ROWS];
    };

    template<typename T> struct traits::proclaim_is_float44<Mat44<T>> : std::true_type {};
    template<typename T> struct traits::proclaim_is_mat44<Mat44<T>> : std::true_type {};

    using float44_t = Mat44<float>;
    using double44_t = Mat44<double>;

    template<typename T>
    [[nodiscard]] NOA_IH constexpr std::array<T, 16> toArray(Mat44<T> v) noexcept {
        return {v[0][0], v[0][1], v[0][2], v[0][3],
                v[1][0], v[1][1], v[1][2], v[1][3],
                v[2][0], v[2][1], v[2][2], v[2][3],
                v[3][0], v[3][1], v[3][2], v[3][3]};
    }

    template<> [[nodiscard]] NOA_IH std::string string::human<float44_t>() { return "float44"; }
    template<> [[nodiscard]] NOA_IH std::string string::human<double44_t>() { return "double44"; }
}

namespace noa::math {
    /// Multiplies matrix \a lhs by matrix \a rhs element-wise, i.e. `out[i][j] = lhs[i][j] * rhs[i][j]`.
    template<typename T>
    [[nodiscard]] NOA_IHD constexpr Mat44<T> elementMultiply(Mat44<T> m1, Mat44<T> m2) noexcept {
        Mat44<T> out;
        for (size_t i = 0; i < Mat44<T>::ROWS; ++i)
            out[i] = m1[i] * m2[i];
        return out;
    }

    /// Given the column vector \a column and row vector \a row,
    /// computes the linear algebraic matrix multiply `c * r`.
    template<typename T>
    [[nodiscard]] NOA_IHD constexpr Mat44<T> outerProduct(const Float4<T>& column, const Float4<T>& row) noexcept {
        return Mat44<T>(column[0] * row[0], column[0] * row[1], column[0] * row[2], column[0] * row[3],
                        column[1] * row[0], column[1] * row[1], column[1] * row[2], column[1] * row[3],
                        column[2] * row[0], column[2] * row[1], column[2] * row[2], column[2] * row[3],
                        column[3] * row[0], column[3] * row[1], column[3] * row[2], column[3] * row[3]);
    }

    template<typename T>
    [[nodiscard]] NOA_IHD constexpr Mat44<T> transpose(Mat44<T> m) noexcept {
        return Mat44<T>(m[0][0], m[1][0], m[2][0], m[3][0],
                        m[0][1], m[1][1], m[2][1], m[3][1],
                        m[0][2], m[1][2], m[2][2], m[3][2],
                        m[0][3], m[1][3], m[2][3], m[3][3]);
    }

    template<typename T>
    [[nodiscard]] NOA_HD constexpr T determinant(Mat44<T> m) noexcept {
        T s00 = m[2][2] * m[3][3] - m[2][3] * m[3][2];
        T s01 = m[1][2] * m[3][3] - m[1][3] * m[3][2];
        T s02 = m[1][2] * m[2][3] - m[1][3] * m[2][2];
        T s03 = m[0][2] * m[3][3] - m[0][3] * m[3][2];
        T s04 = m[0][2] * m[2][3] - m[0][3] * m[2][2];
        T s05 = m[0][2] * m[1][3] - m[0][3] * m[1][2];

        Float4<T> c(+(m[1][1] * s00 - m[2][1] * s01 + m[3][1] * s02),
                    -(m[0][1] * s00 - m[2][1] * s03 + m[3][1] * s04),
                    +(m[0][1] * s01 - m[1][1] * s03 + m[3][1] * s05),
                    -(m[0][1] * s02 - m[1][1] * s04 + m[2][1] * s05));

        return m[0][0] * c[0] + m[1][0] * c[1] +
               m[2][0] * c[2] + m[3][0] * c[3];
    }

    template<typename T>
    [[nodiscard]] NOA_HD constexpr Mat44<T> inverse(Mat44<T> m) noexcept {
        // From https://stackoverflow.com/a/44446912 and https://github.com/willnode/N-Matrix-Programmer
        T A2323 = m[2][2] * m[3][3] - m[2][3] * m[3][2];
        T A1323 = m[2][1] * m[3][3] - m[2][3] * m[3][1];
        T A1223 = m[2][1] * m[3][2] - m[2][2] * m[3][1];
        T A0323 = m[2][0] * m[3][3] - m[2][3] * m[3][0];
        T A0223 = m[2][0] * m[3][2] - m[2][2] * m[3][0];
        T A0123 = m[2][0] * m[3][1] - m[2][1] * m[3][0];
        T A2313 = m[1][2] * m[3][3] - m[1][3] * m[3][2];
        T A1313 = m[1][1] * m[3][3] - m[1][3] * m[3][1];
        T A1213 = m[1][1] * m[3][2] - m[1][2] * m[3][1];
        T A2312 = m[1][2] * m[2][3] - m[1][3] * m[2][2];
        T A1312 = m[1][1] * m[2][3] - m[1][3] * m[2][1];
        T A1212 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
        T A0313 = m[1][0] * m[3][3] - m[1][3] * m[3][0];
        T A0213 = m[1][0] * m[3][2] - m[1][2] * m[3][0];
        T A0312 = m[1][0] * m[2][3] - m[1][3] * m[2][0];
        T A0212 = m[1][0] * m[2][2] - m[1][2] * m[2][0];
        T A0113 = m[1][0] * m[3][1] - m[1][1] * m[3][0];
        T A0112 = m[1][0] * m[2][1] - m[1][1] * m[2][0];

        T det = m[0][0] * (m[1][1] * A2323 - m[1][2] * A1323 + m[1][3] * A1223) -
                m[0][1] * (m[1][0] * A2323 - m[1][2] * A0323 + m[1][3] * A0223) +
                m[0][2] * (m[1][0] * A1323 - m[1][1] * A0323 + m[1][3] * A0123) -
                m[0][3] * (m[1][0] * A1223 - m[1][1] * A0223 + m[1][2] * A0123);
        NOA_ASSERT(!isEqual(det, static_cast<T>(0))); // non singular
        det = 1 / det;

        return Mat44<T>(det * +(m[1][1] * A2323 - m[1][2] * A1323 + m[1][3] * A1223),
                        det * -(m[0][1] * A2323 - m[0][2] * A1323 + m[0][3] * A1223),
                        det * +(m[0][1] * A2313 - m[0][2] * A1313 + m[0][3] * A1213),
                        det * -(m[0][1] * A2312 - m[0][2] * A1312 + m[0][3] * A1212),
                        det * -(m[1][0] * A2323 - m[1][2] * A0323 + m[1][3] * A0223),
                        det * +(m[0][0] * A2323 - m[0][2] * A0323 + m[0][3] * A0223),
                        det * -(m[0][0] * A2313 - m[0][2] * A0313 + m[0][3] * A0213),
                        det * +(m[0][0] * A2312 - m[0][2] * A0312 + m[0][3] * A0212),
                        det * +(m[1][0] * A1323 - m[1][1] * A0323 + m[1][3] * A0123),
                        det * -(m[0][0] * A1323 - m[0][1] * A0323 + m[0][3] * A0123),
                        det * +(m[0][0] * A1313 - m[0][1] * A0313 + m[0][3] * A0113),
                        det * -(m[0][0] * A1312 - m[0][1] * A0312 + m[0][3] * A0112),
                        det * -(m[1][0] * A1223 - m[1][1] * A0223 + m[1][2] * A0123),
                        det * +(m[0][0] * A1223 - m[0][1] * A0223 + m[0][2] * A0123),
                        det * -(m[0][0] * A1213 - m[0][1] * A0213 + m[0][2] * A0113),
                        det * +(m[0][0] * A1212 - m[0][1] * A0212 + m[0][2] * A0112)
        );
    }

    template<uint ULP = 2, typename T>
    [[nodiscard]] NOA_IHD constexpr bool isEqual(Mat44<T> m1, Mat44<T> m2, T e = 1e-6f) noexcept {
        return all(isEqual<ULP>(m1[0], m2[0], e)) &&
               all(isEqual<ULP>(m1[1], m2[1], e)) &&
               all(isEqual<ULP>(m1[2], m2[2], e));
    }
}
