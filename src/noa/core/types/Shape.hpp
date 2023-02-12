#pragma once

#include "noa/core/traits/Numerics.hpp"
#include "noa/core/traits/Shape.hpp"
#include "noa/core/types/Vec.hpp"

namespace noa {
    template<typename Int, size_t N>
    class Strides;
}

namespace noa {
    template<typename Int, size_t N>
    class Shape {
    public:
        static_assert(noa::traits::is_restricted_int_v<Int>);
        static_assert(N <= 4);
        using vector_type = Vec<Int, N>;
        using value_type = typename vector_type::value_type;
        static constexpr int64_t SSIZE = N;
        static constexpr size_t SIZE = N;

    public: // Default constructors
        constexpr Shape() noexcept = default;

        // Explicit element-wise conversion constructor.
        template<typename... Ts,
                 typename = std::enable_if_t<
                         sizeof...(Ts) == SSIZE && (sizeof...(Ts) > 1) &&
                         noa::traits::are_int_v<Ts...>>>
        NOA_HD constexpr Shape(Ts... ts) noexcept : m_vec(std::forward<Ts>(ts)...) {}

        // Explicit fill conversion constructor.
        template<typename T, typename = std::enable_if_t<noa::traits::are_int_v<T>>>
        NOA_HD constexpr explicit Shape(T value) noexcept : m_vec(value) {}

        // Explicit conversion constructor.
        template<typename T, typename = std::enable_if_t<noa::traits::are_int_v<T>>>
        NOA_HD constexpr explicit Shape(const Vec<T, N>& vector) noexcept : m_vec(vector) {}

        // Explicit conversion constructor.
        template<typename T, typename = std::enable_if_t<noa::traits::are_int_v<T>>>
        NOA_HD constexpr explicit Shape(const Shape<T, N>& shape) noexcept : m_vec(shape.vec()) {}

        // Explicit construction from a pointer.
        // This is not ideal (because it can segfault), but is truly useful in some cases.
        NOA_HD constexpr explicit Shape(const value_type* values) noexcept : m_vec(values) {}

    public: // Accessor operators and functions
        template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
        [[nodiscard]] NOA_HD constexpr value_type& operator[](I i) noexcept {
            return m_vec[i];
        }

        template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
        [[nodiscard]] NOA_HD constexpr const value_type& operator[](I i) const noexcept {
            return m_vec[i];
        }

        template<typename Void = void, typename = std::enable_if_t<N == 4 && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr value_type& batch() noexcept { return m_vec[0]; }

        template<typename Void = void, typename = std::enable_if_t<N == 4 && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr const value_type& batch() const noexcept { return m_vec[0]; }

        template<typename Void = void, typename = std::enable_if_t<N >= 3 && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr value_type& depth() noexcept { return m_vec[N - 1]; }

        template<typename Void = void, typename = std::enable_if_t<N >= 3 && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr const value_type& depth() const noexcept { return m_vec[N - 1]; }

        template<typename Void = void, typename = std::enable_if_t<N >= 2 && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr value_type& height() noexcept { return m_vec[N - 2]; }

        template<typename Void = void, typename = std::enable_if_t<N >= 2 && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr const value_type& height() const noexcept { return m_vec[N - 2]; }

        [[nodiscard]] NOA_HD constexpr value_type& row() noexcept { return m_vec[N - 1]; }
        [[nodiscard]] NOA_HD constexpr const value_type& row() const noexcept { return m_vec[N - 1]; }

        template<int I>
        [[nodiscard]] NOA_HD constexpr auto get() const noexcept { return m_vec[I]; } // structure binding support

        [[nodiscard]] NOA_HD constexpr const value_type* data() const noexcept { return m_vec.data(); }
        [[nodiscard]] NOA_HD constexpr value_type* data() noexcept { return m_vec.data(); }
        [[nodiscard]] NOA_HD constexpr size_t size() const noexcept { return SIZE; };

        [[nodiscard]] NOA_HD constexpr const vector_type& vec() const noexcept { return m_vec; }
        [[nodiscard]] NOA_HD constexpr vector_type& vec() noexcept { return m_vec; }

    public: // Iterators -- support for range loops
        [[nodiscard]] NOA_HD constexpr value_type* begin() noexcept { return m_vec.begin(); }
        [[nodiscard]] NOA_HD constexpr const value_type* begin() const noexcept { return m_vec.begin(); }
        [[nodiscard]] NOA_HD constexpr const value_type* cbegin() const noexcept { return m_vec.cbegin(); }
        [[nodiscard]] NOA_HD constexpr value_type* end() noexcept { return m_vec.end(); }
        [[nodiscard]] NOA_HD constexpr const value_type* end() const noexcept { return m_vec.end(); }
        [[nodiscard]] NOA_HD constexpr const value_type* cend() const noexcept { return m_vec.cend(); }

    public: // Assignment operators
        NOA_HD constexpr Shape& operator=(value_type size) noexcept {
            *this = Shape(size);
            return *this;
        }

        NOA_HD constexpr Shape& operator+=(const Shape& shape) noexcept {
            *this = *this + shape;
            return *this;
        }

        NOA_HD constexpr Shape& operator-=(const Shape& shape) noexcept {
            *this = *this - shape;
            return *this;
        }

        NOA_HD constexpr Shape& operator*=(const Shape& shape) noexcept {
            *this = *this * shape;
            return *this;
        }

        NOA_HD constexpr Shape& operator/=(const Shape& shape) noexcept {
            *this = *this / shape;
            return *this;
        }

        NOA_HD constexpr Shape& operator+=(value_type value) noexcept {
            *this = *this + value;
            return *this;
        }

        NOA_HD constexpr Shape& operator-=(value_type value) noexcept {
            *this = *this - value;
            return *this;
        }

        NOA_HD constexpr Shape& operator*=(value_type value) noexcept {
            *this = *this * value;
            return *this;
        }

        NOA_HD constexpr Shape& operator/=(value_type value) noexcept {
            *this = *this / value;
            return *this;
        }

    public: // Non-member functions
        // -- Unary operators --
        [[nodiscard]] friend NOA_HD constexpr Shape operator+(const Shape& shape) noexcept {
            return shape;
        }

        [[nodiscard]] friend NOA_HD constexpr Shape operator-(Shape shape) noexcept {
            return Shape(-shape.vec());
        }

        // -- Binary Arithmetic Operators --
        [[nodiscard]] friend NOA_HD constexpr Shape operator+(Shape lhs, Shape rhs) noexcept {
            return Shape(lhs.vec() + rhs.vec());
        }

        [[nodiscard]] friend NOA_HD constexpr Shape operator+(const Shape& lhs, value_type rhs) noexcept {
            return lhs + Shape(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr Shape operator+(value_type lhs, const Shape& rhs) noexcept {
            return Shape(lhs) + rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr Shape operator-(Shape lhs, Shape rhs) noexcept {
            return Shape(lhs.vec() - rhs.vec());
        }

        [[nodiscard]] friend NOA_HD constexpr Shape operator-(const Shape& lhs, value_type rhs) noexcept {
            return lhs - Shape(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr Shape operator-(value_type lhs, const Shape& rhs) noexcept {
            return Shape(lhs) - rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr Shape operator*(Shape lhs, Shape rhs) noexcept {
            return Shape(lhs.vec() * rhs.vec());
        }

        [[nodiscard]] friend NOA_HD constexpr Shape operator*(const Shape& lhs, value_type rhs) noexcept {
            return lhs * Shape(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr Shape operator*(value_type lhs, const Shape& rhs) noexcept {
            return Shape(lhs) * rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr Shape operator/(Shape lhs, Shape rhs) noexcept {
            return Shape(lhs.vec() / rhs.vec());
        }

        [[nodiscard]] friend NOA_HD constexpr Shape operator/(const Shape& lhs, value_type rhs) noexcept {
            return lhs / Shape(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr Shape operator/(value_type lhs, const Shape& rhs) noexcept {
            return Shape(lhs) / rhs;
        }

        // -- Comparison Operators --
        [[nodiscard]] friend NOA_HD constexpr auto operator>(Shape lhs, Shape rhs) noexcept {
            return lhs.vec() > rhs.vec();
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator>(const Shape& lhs, value_type rhs) noexcept {
            return lhs > Shape(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator>(value_type lhs, const Shape& rhs) noexcept {
            return Shape(lhs) > rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator<(Shape lhs, Shape rhs) noexcept {
            return lhs.vec() < rhs.vec();
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator<(const Shape& lhs, value_type rhs) noexcept {
            return lhs < Shape(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator<(value_type lhs, const Shape& rhs) noexcept {
            return Shape(lhs) < rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator>=(Shape lhs, Shape rhs) noexcept {
            return lhs.vec() >= rhs.vec();
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator>=(const Shape& lhs, value_type rhs) noexcept {
            return lhs >= Shape(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator>=(value_type lhs, const Shape& rhs) noexcept {
            return Shape(lhs) >= rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator<=(Shape lhs, Shape rhs) noexcept {
            return lhs.vec() <= rhs.vec();
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator<=(const Shape& lhs, value_type rhs) noexcept {
            return lhs <= Shape(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator<=(value_type lhs, const Shape& rhs) noexcept {
            return Shape(lhs) <= rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator==(Shape lhs, Shape rhs) noexcept {
            return lhs.vec() == rhs.vec();
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator==(const Shape& lhs, value_type rhs) noexcept {
            return lhs == Shape(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator==(value_type lhs, const Shape& rhs) noexcept {
            return Shape(lhs) == rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator!=(Shape lhs, Shape rhs) noexcept {
            return lhs.vec() != rhs.vec();
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator!=(const Shape& lhs, value_type rhs) noexcept {
            return lhs != Shape(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator!=(value_type lhs, const Shape& rhs) noexcept {
            return Shape(lhs) != rhs;
        }

    public: // Type casts
        template<typename TTo, typename = std::enable_if_t<noa::traits::is_restricted_int_v<TTo>>>
        [[nodiscard]] NOA_HD constexpr auto as() const noexcept {
            return static_cast<Shape<TTo, SIZE>>(*this);
        }

        template<typename TTo, typename = std::enable_if_t<noa::traits::is_restricted_int_v<TTo>>>
        [[nodiscard]] NOA_HD constexpr auto as_clamp() const noexcept {
            return clamp_cast<Shape<TTo, SIZE>>(*this);
        }

        template<typename TTo, typename = std::enable_if_t<noa::traits::is_restricted_int_v<TTo>>>
        constexpr auto as_safe() const {
            return safe_cast<Shape<TTo, SIZE>>(*this);
        }

    public:
        template<size_t S = 1, typename Void = void, typename = std::enable_if_t<(N > S) && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr auto pop_front() const noexcept {
            return Shape<value_type, N - S>(data() + S);
        }

        template<size_t S = 1, typename Void = void, typename = std::enable_if_t<(N > S) && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr auto pop_back() const noexcept {
            return Shape<value_type, N - S>(data());
        }

        [[nodiscard]] NOA_HD constexpr auto push_front(value_type value) const noexcept {
            return Shape<value_type, N + 1>(m_vec.push_front(value));
        }

        [[nodiscard]] NOA_HD constexpr auto push_back(value_type value) const noexcept {
            return Shape<value_type, N + 1>(m_vec.push_back(value));
        }

        template<size_t S>
        [[nodiscard]] NOA_HD constexpr auto push_front(const Vec<value_type, S>& vector) const noexcept {
            constexpr bool NEW_SIZE = N + S;
            return Shape<value_type, NEW_SIZE>(m_vec.push_front(vector));
        }

        template<size_t S>
        [[nodiscard]] NOA_HD constexpr auto push_back(const Vec<value_type, S>& vector) const noexcept {
            constexpr bool NEW_SIZE = N + S;
            return Shape<value_type, NEW_SIZE>(m_vec.push_back(vector));
        }

        template<typename... Ts,
                 typename = std::enable_if_t<sizeof...(Ts) <= N && noa::traits::are_restricted_int_v<Ts...>>>
        [[nodiscard]] NOA_HD constexpr auto filter(Ts... ts) const noexcept {
            return Shape<value_type, sizeof...(Ts)>((*this)[ts]...);
        }

        [[nodiscard]] NOA_HD constexpr Shape flip() const noexcept {
            Shape output;
            for (size_t i = 0; i < SIZE; ++i)
                output[i] = m_vec[(N - 1) - i];
            return output;
        }

        template<typename I, typename = std::enable_if_t<noa::traits::is_restricted_int_v<I>>>
        [[nodiscard]] NOA_HD constexpr Shape reorder(const Vec<I, SIZE>& order) const noexcept {
            return Shape(m_vec.reorder(order));
        }

        template<typename I, typename = std::enable_if_t<noa::traits::is_restricted_int_v<I>>>
        [[nodiscard]] NOA_HD constexpr Shape circular_shift(I count) {
            return Shape(m_vec.circular_shift(count));
        }

        [[nodiscard]] NOA_HD constexpr value_type elements() const noexcept {
            auto output = m_vec[0];
            for (size_t i = 1; i < N; ++i)
                output *= m_vec[i];
            return output;
        }

        // Whether the shape has at least one dimension equal to 0.
        [[nodiscard]] NOA_HD constexpr bool is_empty() const noexcept {
            return any(m_vec == 0);
        }

        // Returns the logical number of dimensions in the BDHW convention.
        // This returns a value from 1 to 3. The batch dimension is ignored,
        // and is_batched() should be used to know whether the array is batched.
        // Note that both row and column vectors are considered to be 1D, but
        // if the depth dimension is greater than 1, ndim() == 3 even if both the
        // height and width are 1.
        [[nodiscard]] NOA_HD constexpr value_type ndim() const noexcept {
            NOA_ASSERT(!is_empty());
            if constexpr (N == 1) {
                return 1;
            } else if constexpr (N == 2) {
                return m_vec[0] > 1 && m_vec[1] > 1 ? 2 : 1;
            } else if constexpr (N == 3) {
                return m_vec[0] > 1 ? 3 :
                       m_vec[1] > 1 && m_vec[2] > 1 ? 2 : 1;
            } else {
                return m_vec[1] > 1 ? 3 :
                       m_vec[2] > 1 && m_vec[3] > 1 ? 2 : 1;
            }
        }

        // Computes the strides, in elements, in C- or F-order.
        // Note that if the height and width dimensions are empty, 'C' and 'F' returns the same strides.
        template<char ORDER = 'C', typename void_ = void>
        [[nodiscard]] NOA_HD constexpr auto strides() const noexcept {
            using output_strides = Strides<value_type, SIZE>;

            if constexpr (ORDER == 'C' || ORDER == 'c') {
                if constexpr (SIZE == 4) {
                    return output_strides(m_vec[3] * m_vec[2] * m_vec[1],
                                          m_vec[3] * m_vec[2],
                                          m_vec[3],
                                          1);
                } else if constexpr (SIZE == 3) {
                    return output_strides(m_vec[2] * m_vec[1],
                                          m_vec[2],
                                          1);
                } else if constexpr (SIZE == 2) {
                    return output_strides(m_vec[1], 1);
                } else {
                    return output_strides(1);
                }
            } else if constexpr (ORDER == 'F' || ORDER == 'f') {
                if constexpr (SIZE == 4) {
                    return output_strides(m_vec[3] * m_vec[2] * m_vec[1],
                                          m_vec[3] * m_vec[2],
                                          1,
                                          m_vec[2]);
                } else if constexpr (SIZE == 3) {
                    return output_strides(m_vec[2] * m_vec[1],
                                          1,
                                          m_vec[1]);
                } else if constexpr (SIZE == 2) {
                    return output_strides(1, m_vec[0]);
                } else {
                    return output_strides(1);
                }
            } else {
                static_assert(traits::always_false_v<void_>);
            }
        }

        // Returns the shape of the non-redundant FFT, in elements,
        template<typename Void = void, typename = std::enable_if_t<SIZE == 4 && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr Shape fft() const noexcept {
            return {m_vec[0], m_vec[1], m_vec[2], m_vec[3] / 2 + 1};
        }

        // Whether the shape describes vector.
        // A vector has one dimension with a size >= 1 and all the other dimensions empty (i.e. size == 1).
        // By this definition, the shapes {1,1,1,1}, {5,1,1,1} and {1,1,1,5} are all vectors.
        // If "can_be_batched" is true, the shape can describe a batch of vectors,
        // e.g. {4,1,1,5} is describing 4 row vectors with a length of 5.
        template<typename Void = void, std::enable_if_t<SIZE == 4 && std::is_void_v<Void>, bool> = true>
        [[nodiscard]]  NOA_FHD constexpr bool is_vector(bool can_be_batched = false) const noexcept {
            int non_empty_dimension = 0;
            for (int i = 0; i < SIZE; ++i) {
                if (m_vec[i] == 0)
                    return false; // empty/invalid shape
                if ((!can_be_batched || i != 0) && m_vec[i] > 1)
                    ++non_empty_dimension;
            }
            return non_empty_dimension <= 1;
        }

        // Whether the shape describes vector.
        // A vector has one dimension with a size >= 1 and all the other dimensions empty (i.e. size == 1).
        // By this definition, the shapes {1,1,1}, {5,1,1} and {1,1,5} are all vectors.
        template<typename Void = void, std::enable_if_t<SIZE <= 3 && std::is_void_v<Void>, bool> = true>
        [[nodiscard]] NOA_FHD constexpr bool is_vector() const noexcept {
            int non_empty_dimension = 0;
            for (int i = 0; i < SIZE; ++i) {
                if (m_vec[i] == 0)
                    return false; // empty/invalid shape
                if (m_vec[i] > 1)
                    ++non_empty_dimension;
            }
            return non_empty_dimension <= 1;
        }

        // Whether this is a (batched) column vector.
        template<typename Void = void, typename = std::enable_if_t<SIZE >= 2 && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr bool is_column() const noexcept {
            return m_vec[N - 2] >= 1 && m_vec[N - 1] == 1;
        }

        // Whether this is a (batched) row vector.
        template<typename Void = void, typename = std::enable_if_t<SIZE >= 2 && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr bool is_row() const noexcept {
            return m_vec[N - 2] == 1 && m_vec[N - 1] >= 1;
        }

        // Whether this is a (batched) column vector.
        template<typename Void = void, typename = std::enable_if_t<SIZE ==4 && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr bool is_batched() const noexcept {
            return m_vec[0] > 1;
        }

        // Move the left-most non-empty dimension to the batch dimension.
        template<typename Void = void, typename = std::enable_if_t<SIZE ==4 && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr Shape to_batched() const noexcept {
            if (m_vec[0] > 1)
                return *this; // already batched
            if (m_vec[1] > 1)
                return Shape(m_vec[1], 1, m_vec[2], m_vec[3]);
            if (m_vec[2] > 1)
                return Shape(m_vec[2], 1, 1, m_vec[3]);
            if (m_vec[3] > 1)
                return Shape(m_vec[3], 1, 1, 1);
            return *this; // {1,1,1,1}
        }

    public: // Support for noa::string::human<Vec>();
        [[nodiscard]] static std::string name() {
            return noa::string::format("Shape<{},{}>", noa::string::human<value_type>(), SIZE);
        }

    private:
        vector_type m_vec;
    };
}

namespace noa {
    template<typename Int, size_t N>
    class Strides {
    public:
        static_assert(noa::traits::is_restricted_int_v<Int>);
        static_assert(N <= 4);
        using vector_type = Vec<Int, N>;
        using value_type = typename vector_type::value_type;
        static constexpr int64_t SSIZE = N;
        static constexpr size_t SIZE = N;

    public: // Default constructors
        constexpr Strides() noexcept = default;

        // Explicit element-wise conversion constructor.
        template<typename... Ts,
                 typename = std::enable_if_t<
                         sizeof...(Ts) == SSIZE &&
                         (sizeof...(Ts) > 1) &&
                         noa::traits::are_int_v<Ts...>>>
        NOA_HD constexpr Strides(Ts... ts) noexcept : m_vec(std::forward<Ts>(ts)...) {}

        // Explicit fill conversion constructor.
        template<typename T, typename = std::enable_if_t<noa::traits::are_int_v<T>>>
        NOA_HD constexpr explicit Strides(T value) noexcept : m_vec(value) {}

        // Explicit conversion constructor.
        template<typename T, typename = std::enable_if_t<noa::traits::are_int_v<T>>>
        NOA_HD constexpr explicit Strides(const Vec<T, N>& vector) noexcept : m_vec(vector) {}

        // Explicit conversion constructor.
        template<typename T, typename = std::enable_if_t<noa::traits::are_int_v<T>>>
        NOA_HD constexpr explicit Strides(const Strides<T, N>& shape) noexcept : m_vec(shape.vec()) {}

        // Explicit construction from a pointer.
        // This is not ideal (because it can segfault), but is truly useful in some cases.
        NOA_HD constexpr explicit Strides(const value_type* values) noexcept : m_vec(values) {}

    public: // Accessor operators and functions
        template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
        [[nodiscard]] NOA_HD constexpr value_type& operator[](I i) noexcept {
            return m_vec[i];
        }

        template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
        [[nodiscard]] NOA_HD constexpr const value_type& operator[](I i) const noexcept {
            return m_vec[i];
        }

        template<typename Void = void, typename = std::enable_if_t<N == 4 && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr value_type& batch() noexcept { return m_vec[0]; }

        template<typename Void = void, typename = std::enable_if_t<N == 4 && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr const value_type& batch() const noexcept { return m_vec[0]; }

        template<typename Void = void, typename = std::enable_if_t<N >= 3 && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr value_type& depth() noexcept { return m_vec[N - 1]; }

        template<typename Void = void, typename = std::enable_if_t<N >= 3 && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr const value_type& depth() const noexcept { return m_vec[N - 1]; }

        template<typename Void = void, typename = std::enable_if_t<N >= 2 && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr value_type& height() noexcept { return m_vec[N - 2]; }

        template<typename Void = void, typename = std::enable_if_t<N >= 2 && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr const value_type& height() const noexcept { return m_vec[N - 2]; }

        [[nodiscard]] NOA_HD constexpr value_type& row() noexcept { return m_vec[N - 1]; }
        [[nodiscard]] NOA_HD constexpr const value_type& row() const noexcept { return m_vec[N - 1]; }

        template<int I>
        [[nodiscard]] NOA_HD constexpr auto get() const noexcept { return m_vec[I]; } // structure binding support

        [[nodiscard]] NOA_HD constexpr const value_type* data() const noexcept { return m_vec.data(); }
        [[nodiscard]] NOA_HD constexpr value_type* data() noexcept { return m_vec.data(); }
        [[nodiscard]] NOA_HD constexpr size_t size() const noexcept { return SIZE; };

        [[nodiscard]] NOA_HD constexpr const vector_type& vec() const noexcept { return m_vec; }
        [[nodiscard]] NOA_HD constexpr vector_type& vec() noexcept { return m_vec; }

    public: // Iterators -- support for range loops
        [[nodiscard]] NOA_HD constexpr value_type* begin() noexcept { return m_vec.begin(); }
        [[nodiscard]] NOA_HD constexpr const value_type* begin() const noexcept { return m_vec.begin(); }
        [[nodiscard]] NOA_HD constexpr const value_type* cbegin() const noexcept { return m_vec.cbegin(); }
        [[nodiscard]] NOA_HD constexpr value_type* end() noexcept { return m_vec.end(); }
        [[nodiscard]] NOA_HD constexpr const value_type* end() const noexcept { return m_vec.end(); }
        [[nodiscard]] NOA_HD constexpr const value_type* cend() const noexcept { return m_vec.cend(); }

    public: // Assignment operators
        NOA_HD constexpr Strides& operator=(value_type size) noexcept {
            *this = Strides(size);
            return *this;
        }

        NOA_HD constexpr Strides& operator+=(const Strides& strides) noexcept {
            *this = *this + strides;
            return *this;
        }

        NOA_HD constexpr Strides& operator-=(const Strides& strides) noexcept {
            *this = *this - strides;
            return *this;
        }

        NOA_HD constexpr Strides& operator*=(const Strides& strides) noexcept {
            *this = *this * strides;
            return *this;
        }

        NOA_HD constexpr Strides& operator/=(const Strides& strides) noexcept {
            *this = *this / strides;
            return *this;
        }

        NOA_HD constexpr Strides& operator+=(value_type value) noexcept {
            *this = *this + value;
            return *this;
        }

        NOA_HD constexpr Strides& operator-=(value_type value) noexcept {
            *this = *this - value;
            return *this;
        }

        NOA_HD constexpr Strides& operator*=(value_type value) noexcept {
            *this = *this * value;
            return *this;
        }

        NOA_HD constexpr Strides& operator/=(value_type value) noexcept {
            *this = *this / value;
            return *this;
        }

    public: // Non-member functions
        // -- Unary operators --
        [[nodiscard]] friend NOA_HD constexpr Strides operator+(const Strides& strides) noexcept {
            return strides;
        }

        [[nodiscard]] friend NOA_HD constexpr Strides operator-(Strides strides) noexcept {
            return Strides(-strides.vec());
        }

        // -- Binary Arithmetic Operators --
        [[nodiscard]] friend NOA_HD constexpr Strides operator+(Strides lhs, Strides rhs) noexcept {
            return Strides(lhs.vec() + rhs.vec());
        }

        [[nodiscard]] friend NOA_HD constexpr Strides operator+(const Strides& lhs, value_type rhs) noexcept {
            return lhs + Strides(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr Strides operator+(value_type lhs, const Strides& rhs) noexcept {
            return Strides(lhs) + rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr Strides operator-(Strides lhs, Strides rhs) noexcept {
            return Strides(lhs.vec() - rhs.vec());
        }

        [[nodiscard]] friend NOA_HD constexpr Strides operator-(const Strides& lhs, value_type rhs) noexcept {
            return lhs - Strides(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr Strides operator-(value_type lhs, const Strides& rhs) noexcept {
            return Strides(lhs) - rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr Strides operator*(Strides lhs, Strides rhs) noexcept {
            return Strides(lhs.vec() * rhs.vec());
        }

        [[nodiscard]] friend NOA_HD constexpr Strides operator*(const Strides& lhs, value_type rhs) noexcept {
            return lhs * Strides(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr Strides operator*(value_type lhs, const Strides& rhs) noexcept {
            return Strides(lhs) * rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr Strides operator/(Strides lhs, Strides rhs) noexcept {
            return Strides(lhs.vec() / rhs.vec());
        }

        [[nodiscard]] friend NOA_HD constexpr Strides operator/(const Strides& lhs, value_type rhs) noexcept {
            return lhs / Strides(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr Strides operator/(value_type lhs, const Strides& rhs) noexcept {
            return Strides(lhs) / rhs;
        }

        // -- Comparison Operators --
        [[nodiscard]] friend NOA_HD constexpr auto operator>(Strides lhs, Strides rhs) noexcept {
            return lhs.vec() > rhs.vec();
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator>(const Strides& lhs, value_type rhs) noexcept {
            return lhs > Strides(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator>(value_type lhs, const Strides& rhs) noexcept {
            return Strides(lhs) > rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator<(Strides lhs, Strides rhs) noexcept {
            return lhs.vec() < rhs.vec();
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator<(const Strides& lhs, value_type rhs) noexcept {
            return lhs < Strides(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator<(value_type lhs, const Strides& rhs) noexcept {
            return Strides(lhs) < rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator>=(Strides lhs, Strides rhs) noexcept {
            return lhs.vec() >= rhs.vec();
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator>=(const Strides& lhs, value_type rhs) noexcept {
            return lhs >= Strides(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator>=(value_type lhs, const Strides& rhs) noexcept {
            return Strides(lhs) >= rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator<=(Strides lhs, Strides rhs) noexcept {
            return lhs.vec() <= rhs.vec();
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator<=(const Strides& lhs, value_type rhs) noexcept {
            return lhs <= Strides(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator<=(value_type lhs, const Strides& rhs) noexcept {
            return Strides(lhs) <= rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator==(Strides lhs, Strides rhs) noexcept {
            return lhs.vec() == rhs.vec();
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator==(const Strides& lhs, value_type rhs) noexcept {
            return lhs == Strides(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator==(value_type lhs, const Strides& rhs) noexcept {
            return Strides(lhs) == rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator!=(Strides lhs, Strides rhs) noexcept {
            return lhs.vec() != rhs.vec();
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator!=(const Strides& lhs, value_type rhs) noexcept {
            return lhs != Strides(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator!=(value_type lhs, const Strides& rhs) noexcept {
            return Strides(lhs) != rhs;
        }

    public: // Type casts
        template<typename TTo, typename = std::enable_if_t<noa::traits::is_restricted_int_v<TTo>>>
        [[nodiscard]] NOA_HD constexpr auto as() const noexcept {
            return static_cast<Strides<TTo, SIZE>>(*this);
        }

        template<typename TTo, typename = std::enable_if_t<noa::traits::is_restricted_int_v<TTo>>>
        [[nodiscard]] NOA_HD constexpr auto as_clamp() const noexcept {
            return clamp_cast<Strides<TTo, SIZE>>(*this);
        }

        template<typename TTo, typename = std::enable_if_t<noa::traits::is_restricted_int_v<TTo>>>
        constexpr auto as_safe() const {
            return safe_cast<Strides<TTo, SIZE>>(*this);
        }

    public:
        template<size_t S = 1, typename Void = void, typename = std::enable_if_t<(N > S) && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr auto pop_front() const noexcept {
            return Strides<value_type, N - S>(data() + S);
        }

        template<size_t S = 1, typename Void = void, typename = std::enable_if_t<(N > S) && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr auto pop_back() const noexcept {
            return Strides<value_type, N - S>(data());
        }

        [[nodiscard]] NOA_HD constexpr auto push_front(value_type value) const noexcept {
            return Strides<value_type, N + 1>(m_vec.push_front(value));
        }

        [[nodiscard]] NOA_HD constexpr auto push_back(value_type value) const noexcept {
            return Strides<value_type, N + 1>(m_vec.push_back(value));
        }

        template<size_t S>
        [[nodiscard]] NOA_HD constexpr auto push_front(const Vec<value_type, S>& vector) const noexcept {
            constexpr bool NEW_SIZE = N + S;
            return Strides<value_type, NEW_SIZE>(m_vec.push_front(vector));
        }

        template<size_t S>
        [[nodiscard]] NOA_HD constexpr auto push_back(const Vec<value_type, S>& vector) const noexcept {
            constexpr bool NEW_SIZE = N + S;
            return Strides<value_type, NEW_SIZE>(m_vec.push_back(vector));
        }

        template<typename... Ts,
                 typename = std::enable_if_t<sizeof...(Ts) <= N && noa::traits::are_restricted_int_v<Ts...>>>
        [[nodiscard]] NOA_HD constexpr auto filter(Ts... ts) const noexcept {
            return Strides<value_type, sizeof...(Ts)>((*this)[ts]...);
        }

        [[nodiscard]] NOA_HD constexpr Strides flip() const noexcept {
            Strides output;
            for (size_t i = 0; i < SIZE; ++i)
                output[i] = m_vec[(N - 1) - i];
            return output;
        }

        template<typename I, typename = std::enable_if_t<noa::traits::is_restricted_int_v<I>>>
        [[nodiscard]] NOA_HD constexpr Strides reorder(const Vec<I, SIZE>& order) const noexcept {
            return Strides(m_vec.reorder(order));
        }

        template<typename I, typename = std::enable_if_t<noa::traits::is_restricted_int_v<I>>>
        [[nodiscard]] NOA_HD constexpr Strides circular_shift(I count) {
            return Strides(m_vec.circular_shift(count));
        }

        // Whether there's at least one dimension equal to 0.
        [[nodiscard]] NOA_HD constexpr bool is_broadcast() const noexcept {
            return any(m_vec == 0);
        }

        // Whether the strides are in the rightmost order.
        // Rightmost order is when the innermost stride (i.e. the dimension with the smallest stride)
        // is on the right, and strides increase right-to-left.
        [[nodiscard]] NOA_HD constexpr bool is_rightmost() const noexcept {
            for (size_t i = 0; i < SIZE - 1; ++i)
                if (m_vec[i] < m_vec[i + 1])
                    return false;
            return true;
        }

        // Computes the physical layout (the actual memory footprint) encoded in these strides.
        // Note that the left-most size is not-encoded in the strides, and therefore cannot be recovered.
        template<char ORDER = 'C', typename Void = void,
                 typename = std::enable_if_t<SIZE >= 2 && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr auto physical_shape() const noexcept {
            NOA_ASSERT(!is_broadcast() && "Cannot recover pitches from broadcast strides");
            using output_shape = Shape<value_type, N - 1>;

            if constexpr (ORDER == 'C' || ORDER == 'c') {
                if constexpr (SIZE == 4) {
                    return output_shape(m_vec[0] / m_vec[1],
                                        m_vec[1] / m_vec[2],
                                        m_vec[2]);
                } else if constexpr (SIZE == 3) {
                    return output_shape(m_vec[0] / m_vec[1],
                                        m_vec[1]);
                } else {
                    return output_shape(m_vec[0]);
                }
            } else if constexpr (ORDER == 'F' || ORDER == 'f') {
                if constexpr (SIZE == 4) {
                    return output_shape(m_vec[0] / m_vec[1],
                                        m_vec[3],
                                        m_vec[1] / m_vec[3]);
                } else if constexpr (SIZE == 3) {
                    return output_shape(m_vec[2],
                                        m_vec[0] / m_vec[2]);
                } else {
                    return output_shape(m_vec[1]);
                }
            } else {
                static_assert(traits::always_false_v<Void>);
            }
        }

    public: // Support for noa::string::human<Vec>();
        [[nodiscard]] static std::string name() {
            return noa::string::format("Strides<{},{}>", noa::string::human<value_type>(), SIZE);
        }

    private:
        vector_type m_vec;
    };
}

// Support for structure bindings:
namespace std {
    template<typename T, size_t N>
    struct tuple_size<noa::Shape<T, N>> : std::integral_constant<size_t, N> {};

    template<size_t I, size_t N, typename T>
    struct tuple_element<I, noa::Shape<T, N>> { using type = T; };

    template<typename T, size_t N>
    struct tuple_size<noa::Strides<T, N>> : std::integral_constant<size_t, N> {};

    template<size_t I, size_t N, typename T>
    struct tuple_element<I, noa::Strides<T, N>> { using type = T; };
}

// Support for output stream:
namespace noa {
    template<typename T, size_t N>
    NOA_IH std::ostream& operator<<(std::ostream& os, const Shape<T, N>& v) {
        os << string::format("{}", v.vec());
        return os;
    }

    template<typename T, size_t N>
    NOA_IH std::ostream& operator<<(std::ostream& os, const Strides<T, N>& v) {
        os << string::format("{}", v.vec());
        return os;
    }
}

// Type aliases:
namespace noa {
    template<typename T> using Shape1 = Shape<T, 1>;
    template<typename T> using Shape2 = Shape<T, 2>;
    template<typename T> using Shape3 = Shape<T, 3>;
    template<typename T> using Shape4 = Shape<T, 4>;

    template<typename T> using Strides1 = Strides<T, 1>;
    template<typename T> using Strides2 = Strides<T, 2>;
    template<typename T> using Strides3 = Strides<T, 3>;
    template<typename T> using Strides4 = Strides<T, 4>;
}

// Type traits:
namespace noa::traits {
    static_assert(noa::traits::is_detected_convertible_v<std::string, has_name, Shape<int, 1>>);
    static_assert(noa::traits::is_detected_convertible_v<std::string, has_name, Strides<int, 1>>);

    template<typename T, size_t N> struct proclaim_is_shape<Shape<T, N>> : std::true_type {};
    template<typename V1, size_t N, typename V2> struct proclaim_is_shape_of_type<Shape<V1, N>, V2> : std::bool_constant<std::is_same_v<V1, V2>> {};
    template<typename V, size_t N1, size_t N2> struct proclaim_is_shape_of_size<Shape<V, N1>, N2> : std::bool_constant<N1 == N2> {};

    template<typename T, size_t N> struct proclaim_is_strides<Strides<T, N>> : std::true_type {};
    template<typename V1, size_t N, typename V2> struct proclaim_is_strides_of_type<Strides<V1, N>, V2> : std::bool_constant<std::is_same_v<V1, V2>> {};
    template<typename V, size_t N1, size_t N2> struct proclaim_is_strides_of_size<Strides<V, N1>, N2> : std::bool_constant<N1 == N2> {};
}

namespace noa {
    // -- Modulo Operator --
    template<typename ShapeOrStrides,
             typename std::enable_if_t<noa::traits::is_shape_or_strides_v<ShapeOrStrides>, bool> =  true>
    [[nodiscard]] NOA_HD constexpr ShapeOrStrides operator%(ShapeOrStrides lhs, const ShapeOrStrides& rhs) noexcept {
        for (int64_t i = 0; i < ShapeOrStrides::SSIZE; ++i)
            lhs[i] %= rhs[i];
        return lhs;
    }

    template<typename ShapeOrStrides, typename Int,
             typename std::enable_if_t<noa::traits::is_restricted_int_v<Int> &&
                                       noa::traits::is_shape_or_strides_v<ShapeOrStrides>, bool> = true>
    [[nodiscard]] NOA_HD constexpr ShapeOrStrides operator%(const ShapeOrStrides& lhs, Int rhs) noexcept {
        return lhs % ShapeOrStrides(rhs);
    }

    template<typename ShapeOrStrides, typename Int,
             typename std::enable_if_t<noa::traits::is_restricted_int_v<Int> &&
                                       noa::traits::is_shape_or_strides_v<ShapeOrStrides>, bool> = true>
    [[nodiscard]] NOA_HD constexpr ShapeOrStrides operator%(Int lhs, const ShapeOrStrides& rhs) noexcept {
        return ShapeOrStrides(lhs) % rhs;
    }

    // Cast Shape->Shape
    template<typename TTo, typename TFrom, size_t N,
             std::enable_if_t<noa::traits::is_shapeN_v<TTo, N>, bool> = true>
    [[nodiscard]] NOA_FHD constexpr bool is_safe_cast(const Shape<TFrom, N>& src) noexcept {
        return is_safe_cast<typename TTo::vector_type>(src.vec());
    }

    template<typename TTo, typename TFrom, size_t N,
             std::enable_if_t<noa::traits::is_shapeN_v<TTo, N>, bool> = true>
    [[nodiscard]] NOA_FHD constexpr TTo clamp_cast(const Shape<TFrom, N>& src) noexcept {
        return TTo(clamp_cast<typename TTo::vector_type>(src.vec()));
    }

    // Cast Strides->Strides
    template<typename TTo, typename TFrom, size_t N,
             std::enable_if_t<noa::traits::is_stridesN_v<TTo, N>, bool> = true>
    [[nodiscard]] NOA_FHD constexpr bool is_safe_cast(const Strides<TFrom, N>& src) noexcept {
        return is_safe_cast<typename TTo::vector_type>(src.vec());
    }

    template<typename TTo, typename TFrom, size_t N,
             std::enable_if_t<noa::traits::is_stridesN_v<TTo, N>, bool> = true>
    [[nodiscard]] NOA_FHD constexpr TTo clamp_cast(const Strides<TFrom, N>& src) noexcept {
        return TTo(clamp_cast<typename TTo::vector_type>(src.vec()));
    }

    // Cast Vec->Shape/Strides
    template<typename TTo, typename TFrom, size_t N,
             std::enable_if_t<noa::traits::is_shapeN_or_stridesN_v<TTo, N>, bool> = true>
    [[nodiscard]] NOA_FHD constexpr bool is_safe_cast(const Vec<TFrom, N>& src) noexcept {
        return is_safe_cast<typename TTo::vector_type>(src);
    }

    template<typename TTo, typename TFrom, size_t N,
             std::enable_if_t<noa::traits::is_shapeN_or_stridesN_v<TTo, N>, bool> = true>
    [[nodiscard]] NOA_FHD constexpr TTo clamp_cast(const Vec<TFrom, N>& src) noexcept {
        return TTo(clamp_cast<typename TTo::vector_type>(src));
    }
}

namespace noa::math {
    template<typename ShapeOrStrides,
             std::enable_if_t<noa::traits::is_shape_or_strides_v<ShapeOrStrides>, bool> = true>
    [[nodiscard]] NOA_FHD constexpr auto abs(ShapeOrStrides shape) noexcept {
        return ShapeOrStrides(abs(shape.vec()));
    }

    template<typename ShapeOrStrides,
             std::enable_if_t<noa::traits::is_shape_or_strides_v<ShapeOrStrides>, bool> = true>
    [[nodiscard]] NOA_FHD constexpr auto sum(const ShapeOrStrides& shape) noexcept {
        return sum(shape.vec());
    }

    template<typename ShapeOrStrides,
             std::enable_if_t<noa::traits::is_shape_or_strides_v<ShapeOrStrides>, bool> = true>
    [[nodiscard]] NOA_FHD constexpr auto product(const ShapeOrStrides& shape) noexcept {
        return product(shape.vec());
    }

    template<typename ShapeOrStrides,
             std::enable_if_t<noa::traits::is_shape_or_strides_v<ShapeOrStrides>, bool> = true>
    [[nodiscard]] NOA_FHD constexpr auto min(const ShapeOrStrides& shape) noexcept {
        return ShapeOrStrides(min(shape.vec()));
    }

    template<typename ShapeOrStrides,
             std::enable_if_t<noa::traits::is_shape_or_strides_v<ShapeOrStrides>, bool> = true>
    [[nodiscard]] NOA_FHD constexpr auto min(const ShapeOrStrides& lhs, const ShapeOrStrides& rhs) noexcept {
        return ShapeOrStrides(min(lhs.vec(), rhs.vec()));
    }

    template<typename Int, typename ShapeOrStrides,
             std::enable_if_t<
                     noa::traits::is_shape_or_strides_v<ShapeOrStrides> &&
                     noa::traits::is_almost_same_v<noa::traits::value_type_t<ShapeOrStrides>, Int>, bool> = true>
    [[nodiscard]] NOA_FHD constexpr auto min(const ShapeOrStrides& lhs, Int rhs) noexcept {
        return min(lhs, ShapeOrStrides(rhs));
    }

    template<typename Int, typename ShapeOrStrides,
             std::enable_if_t<
                     noa::traits::is_shape_or_strides_v<ShapeOrStrides> &&
                     noa::traits::is_almost_same_v<noa::traits::value_type_t<ShapeOrStrides>, Int>, bool> = true>
    [[nodiscard]] NOA_FHD constexpr auto min(Int lhs, const ShapeOrStrides& rhs) noexcept {
        return min(ShapeOrStrides(lhs), rhs);
    }

    template<typename ShapeOrStrides,
             std::enable_if_t<noa::traits::is_shape_or_strides_v<ShapeOrStrides>, bool> = true>
    [[nodiscard]] NOA_FHD constexpr auto max(const ShapeOrStrides& shape) noexcept {
        return ShapeOrStrides(max(shape.vec()));
    }

    template<typename ShapeOrStrides,
             std::enable_if_t<noa::traits::is_shape_or_strides_v<ShapeOrStrides>, bool> = true>
    [[nodiscard]] NOA_FHD constexpr auto max(const ShapeOrStrides& lhs, const ShapeOrStrides& rhs) noexcept {
        return ShapeOrStrides(max(lhs.vec(), rhs.vec()));
    }

    template<typename Int, typename ShapeOrStrides,
             std::enable_if_t<
                     noa::traits::is_shape_or_strides_v<ShapeOrStrides> &&
                     noa::traits::is_almost_same_v<noa::traits::value_type_t<ShapeOrStrides>, Int>, bool> = true>
    [[nodiscard]] NOA_FHD constexpr auto max(const ShapeOrStrides& lhs, Int rhs) noexcept {
        return max(lhs, ShapeOrStrides(rhs));
    }

    template<typename Int, typename ShapeOrStrides,
             std::enable_if_t<
                     noa::traits::is_shape_or_strides_v<ShapeOrStrides> &&
                     noa::traits::is_almost_same_v<noa::traits::value_type_t<ShapeOrStrides>, Int>, bool> = true>
    [[nodiscard]] NOA_FHD constexpr auto max(Int lhs, const ShapeOrStrides& rhs) noexcept {
        return max(ShapeOrStrides(lhs), rhs);
    }

    template<typename ShapeOrStrides,
             std::enable_if_t<noa::traits::is_shape_or_strides_v<ShapeOrStrides>, bool> = true>
    [[nodiscard]] NOA_FHD constexpr auto clamp(const ShapeOrStrides& lhs,
                                               const ShapeOrStrides& low,
                                               const ShapeOrStrides& high) noexcept {
        return min(max(lhs, low), high);
    }

    template<typename Int, typename ShapeOrStrides,
             std::enable_if_t<
                     noa::traits::is_shape_or_strides_v<ShapeOrStrides> &&
                     noa::traits::is_almost_same_v<noa::traits::value_type_t<ShapeOrStrides>, Int>, bool> = true>
    [[nodiscard]] NOA_FHD constexpr auto clamp(const ShapeOrStrides& lhs, Int low, Int high) noexcept {
        return min(max(lhs, low), high);
    }
}

// Sort:
namespace noa {
    template<typename T, size_t N, typename Comparison>
    [[nodiscard]] NOA_IHD constexpr auto stable_sort(Shape<T, N> shape, Comparison&& comp) noexcept {
        small_stable_sort<N>(shape.data(), std::forward<Comparison>(comp));
        return shape;
    }

    template<typename T, size_t N, typename Comparison>
    [[nodiscard]] NOA_IHD constexpr auto sort(Shape<T, N> shape, Comparison&& comp) noexcept {
        small_stable_sort<N>(shape.data(), std::forward<Comparison>(comp));
        return shape;
    }

    template<typename T, size_t N>
    [[nodiscard]] NOA_IHD constexpr auto stable_sort(Shape<T, N> shape) noexcept {
        small_stable_sort<N>(shape.data(), [](const T& a, const T& b) { return a < b; });
        return shape;
    }

    template<typename T, size_t N>
    [[nodiscard]] NOA_IHD constexpr auto sort(Shape<T, N> shape) noexcept {
        small_stable_sort<N>(shape.data(), [](const T& a, const T& b) { return a < b; });
        return shape;
    }

    template<typename T, size_t N, typename Comparison>
    [[nodiscard]] NOA_IHD constexpr auto stable_sort(Strides<T, N> shape, Comparison&& comp) noexcept {
        small_stable_sort<N>(shape.data(), std::forward<Comparison>(comp));
        return shape;
    }

    template<typename T, size_t N, typename Comparison>
    [[nodiscard]] NOA_IHD constexpr auto sort(Strides<T, N> shape, Comparison&& comp) noexcept {
        small_stable_sort<N>(shape.data(), std::forward<Comparison>(comp));
        return shape;
    }

    template<typename T, size_t N>
    [[nodiscard]] NOA_IHD constexpr auto stable_sort(Strides<T, N> shape) noexcept {
        small_stable_sort<N>(shape.data(), [](const T& a, const T& b) { return a < b; });
        return shape;
    }

    template<typename T, size_t N>
    [[nodiscard]] NOA_IHD constexpr auto sort(Strides<T, N> shape) noexcept {
        small_stable_sort<N>(shape.data(), [](const T& a, const T& b) { return a < b; });
        return shape;
    }
}
