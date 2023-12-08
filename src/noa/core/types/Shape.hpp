#pragma once

#include "noa/core/Traits.hpp"
#include "noa/core/types/Vec.hpp"
#include "noa/core/math/Ops.hpp"

namespace noa::inline types {
    template<typename Int, size_t N>
    class Strides;
}

namespace noa::inline types {
    template<typename Int, size_t N>
    class Shape {
    public:
        static_assert(nt::is_int_v<Int>);
        static_assert(N <= 4);
        using vector_type = Vec<Int, N>;
        using value_type = typename vector_type::value_type;
        using mutable_value_type = value_type;
        static constexpr int64_t SSIZE = N;
        static constexpr size_t SIZE = N;

    public:
        vector_type vec; // uninitialized

    public: // Static factory functions
        template<typename T, typename = std::enable_if_t<nt::is_int_v<T>>>
        [[nodiscard]] NOA_HD static constexpr Shape from_value(T value) noexcept {
            return {vector_type::from_value(value)};
        }

        template<typename T, typename = std::enable_if_t<nt::is_int_v<T>>>
        [[nodiscard]] NOA_HD static constexpr Shape filled_with(T value) noexcept {
            return {vector_type::filled_with(value)}; // same as from_value
        }

        template<typename... Args, typename = std::enable_if_t<sizeof...(Args) == SIZE && nt::are_int_v<Args...>>>
        [[nodiscard]] NOA_HD static constexpr Shape from_values(Args... values) noexcept {
            return {vector_type::from_values(values...)};
        }

        template<typename T, size_t A, typename = std::enable_if_t<nt::is_int_v<T>>>
        [[nodiscard]] NOA_HD static constexpr Shape from_vec(const Vec<T, SIZE, A>& vector) noexcept {
            return {vector_type::from_vector(vector)};
        }

        template<typename T>
        [[nodiscard]] NOA_HD static constexpr Shape from_shape(const Shape<T, SIZE>& shape) noexcept {
            return from_vec(shape.vec);
        }

        template<typename T, typename = std::enable_if_t<nt::is_int_v<T>>>
        [[nodiscard]] NOA_HD static constexpr Shape from_pointer(const T* values) noexcept {
            return {vector_type::from_pointer(values)};
        }

    public:
        // Allow explicit conversion constructor (while still being an aggregate)
        // and add support for static_cast<Shape<U>>(Shape<T>{}).
        template<typename U>
        [[nodiscard]] NOA_HD constexpr explicit operator Shape<U, SIZE>() const noexcept {
            return Shape<U, SIZE>::from_shape(*this);
        }

    public: // Accessor operators and functions
        template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
        [[nodiscard]] NOA_HD constexpr value_type& operator[](I i) noexcept { return vec[i]; }

        template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
        [[nodiscard]] NOA_HD constexpr const value_type& operator[](I i) const noexcept { return vec[i]; }

        template<typename Void = void, typename = std::enable_if_t<(SIZE == 4) && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr value_type& batch() noexcept { return vec[0]; }

        template<typename Void = void, typename = std::enable_if_t<(SIZE == 4) && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr const value_type& batch() const noexcept { return vec[0]; }

        template<typename Void = void, typename = std::enable_if_t<(SIZE >= 3) && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr value_type& depth() noexcept { return vec[SIZE - 1]; }

        template<typename Void = void, typename = std::enable_if_t<(SIZE >= 3) && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr const value_type& depth() const noexcept { return vec[SIZE - 1]; }

        template<typename Void = void, typename = std::enable_if_t<(SIZE >= 2) && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr value_type& height() noexcept { return vec[SIZE - 2]; }

        template<typename Void = void, typename = std::enable_if_t<(SIZE >= 2) && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr const value_type& height() const noexcept { return vec[SIZE - 2]; }

        template<typename Void = void, typename = std::enable_if_t<(SIZE >= 1) && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr value_type& width() noexcept { return vec[SIZE - 1]; }

        template<typename Void = void, typename = std::enable_if_t<(SIZE >= 1) && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr const value_type& width() const noexcept { return vec[SIZE - 1]; }

        // Structure binding support.
        template<int I> [[nodiscard]] NOA_HD constexpr const value_type& get() const noexcept { return vec[I]; }
        template<int I> [[nodiscard]] NOA_HD constexpr value_type& get() noexcept { return vec[I]; }

        [[nodiscard]] NOA_HD constexpr const value_type* data() const noexcept { return vec.data(); }
        [[nodiscard]] NOA_HD constexpr value_type* data() noexcept { return vec.data(); }
        [[nodiscard]] NOA_HD constexpr size_t size() const noexcept { return SIZE; };
        [[nodiscard]] NOA_HD constexpr int64_t ssize() const noexcept { return SSIZE; };

    public: // Iterators -- support for range loops
        [[nodiscard]] NOA_HD constexpr value_type* begin() noexcept { return vec.begin(); }
        [[nodiscard]] NOA_HD constexpr const value_type* begin() const noexcept { return vec.begin(); }
        [[nodiscard]] NOA_HD constexpr const value_type* cbegin() const noexcept { return vec.cbegin(); }
        [[nodiscard]] NOA_HD constexpr value_type* end() noexcept { return vec.end(); }
        [[nodiscard]] NOA_HD constexpr const value_type* end() const noexcept { return vec.end(); }
        [[nodiscard]] NOA_HD constexpr const value_type* cend() const noexcept { return vec.cend(); }

    public: // Assignment operators
        NOA_HD constexpr Shape& operator=(value_type size) noexcept {
            *this = Shape::filled_with(size);
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
            return {-shape.vec};
        }

        // -- Binary Arithmetic Operators --
        [[nodiscard]] friend NOA_HD constexpr Shape operator+(Shape lhs, Shape rhs) noexcept {
            return {lhs.vec + rhs.vec};
        }

        [[nodiscard]] friend NOA_HD constexpr Shape operator+(const Shape& lhs, value_type rhs) noexcept {
            return lhs + Shape::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr Shape operator+(value_type lhs, const Shape& rhs) noexcept {
            return Shape::filled_with(lhs) + rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr Shape operator-(Shape lhs, Shape rhs) noexcept {
            return Shape{lhs.vec - rhs.vec};
        }

        [[nodiscard]] friend NOA_HD constexpr Shape operator-(const Shape& lhs, value_type rhs) noexcept {
            return lhs - Shape::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr Shape operator-(value_type lhs, const Shape& rhs) noexcept {
            return Shape::filled_with(lhs) - rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr Shape operator*(Shape lhs, Shape rhs) noexcept {
            return Shape{lhs.vec * rhs.vec};
        }

        [[nodiscard]] friend NOA_HD constexpr Shape operator*(const Shape& lhs, value_type rhs) noexcept {
            return lhs * Shape::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr Shape operator*(value_type lhs, const Shape& rhs) noexcept {
            return Shape::filled_with(lhs) * rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr Shape operator/(Shape lhs, Shape rhs) noexcept {
            return Shape{lhs.vec / rhs.vec};
        }

        [[nodiscard]] friend NOA_HD constexpr Shape operator/(const Shape& lhs, value_type rhs) noexcept {
            return lhs / Shape::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr Shape operator/(value_type lhs, const Shape& rhs) noexcept {
            return Shape::filled_with(lhs) / rhs;
        }

        // -- Comparison Operators --
        [[nodiscard]] friend NOA_HD constexpr auto operator>(Shape lhs, Shape rhs) noexcept {
            return lhs.vec > rhs.vec;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator>(const Shape& lhs, value_type rhs) noexcept {
            return lhs > Shape::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator>(value_type lhs, const Shape& rhs) noexcept {
            return Shape::filled_with(lhs) > rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator<(Shape lhs, Shape rhs) noexcept {
            return lhs.vec < rhs.vec;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator<(const Shape& lhs, value_type rhs) noexcept {
            return lhs < Shape::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator<(value_type lhs, const Shape& rhs) noexcept {
            return Shape::filled_with(lhs) < rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator>=(Shape lhs, Shape rhs) noexcept {
            return lhs.vec >= rhs.vec;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator>=(const Shape& lhs, value_type rhs) noexcept {
            return lhs >= Shape::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator>=(value_type lhs, const Shape& rhs) noexcept {
            return Shape::filled_with(lhs) >= rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator<=(Shape lhs, Shape rhs) noexcept {
            return lhs.vec <= rhs.vec;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator<=(const Shape& lhs, value_type rhs) noexcept {
            return lhs <= Shape::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator<=(value_type lhs, const Shape& rhs) noexcept {
            return Shape::filled_with(lhs) <= rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator==(Shape lhs, Shape rhs) noexcept {
            return lhs.vec == rhs.vec;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator==(const Shape& lhs, value_type rhs) noexcept {
            return lhs == Shape::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator==(value_type lhs, const Shape& rhs) noexcept {
            return Shape::filled_with(lhs) == rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator!=(Shape lhs, Shape rhs) noexcept {
            return lhs.vec != rhs.vec;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator!=(const Shape& lhs, value_type rhs) noexcept {
            return lhs != Shape::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator!=(value_type lhs, const Shape& rhs) noexcept {
            return Shape::filled_with(lhs) != rhs;
        }

    public: // Type casts
        template<typename TTo, typename = std::enable_if_t<nt::is_int_v<TTo>>>
        [[nodiscard]] NOA_HD constexpr auto as() const noexcept {
            return static_cast<Shape<TTo, SIZE>>(*this);
        }

        template<typename TTo, typename = std::enable_if_t<nt::is_int_v<TTo>>>
        [[nodiscard]] NOA_HD constexpr auto as_clamp() const noexcept {
            return clamp_cast<Shape<TTo, SIZE>>(*this);
        }

#if defined(NOA_IS_OFFLINE)
        template<typename TTo, typename = std::enable_if_t<nt::is_int_v<TTo>>>
        [[nodiscard]] constexpr auto as_safe() const {
            return safe_cast<Shape<TTo, SIZE>>(*this);
        }
#endif

    public:
        template<size_t S = 1, typename = std::enable_if_t<(SIZE >= S)>>
        [[nodiscard]] NOA_HD constexpr auto pop_front() const noexcept {
            return Shape<value_type, SIZE - S>::from_pointer(data() + S);
        }

        template<size_t S = 1, typename = std::enable_if_t<(SIZE >= S)>>
        [[nodiscard]] NOA_HD constexpr auto pop_back() const noexcept {
            return Shape<value_type, SIZE - S>::from_pointer(data());
        }

        [[nodiscard]] NOA_HD constexpr auto push_front(value_type value) const noexcept {
            return Shape<value_type, SIZE + 1>{vec.push_front(value)};
        }

        [[nodiscard]] NOA_HD constexpr auto push_back(value_type value) const noexcept {
            return Shape<value_type, SIZE + 1>{vec.push_back(value)};
        }

        template<size_t S, size_t A>
        [[nodiscard]] NOA_HD constexpr auto push_front(const Vec<value_type, S, A>& vector) const noexcept {
            constexpr size_t NEW_SIZE = SIZE + S;
            return Shape<value_type, NEW_SIZE>{vec.push_front(vector)};
        }

        template<size_t S, size_t A>
        [[nodiscard]] NOA_HD constexpr auto push_back(const Vec<value_type, S, A>& vector) const noexcept {
            constexpr size_t NEW_SIZE = SIZE + S;
            return Shape<value_type, NEW_SIZE>{vec.push_back(vector)};
        }

        template<typename... Ts, typename = std::enable_if_t<nt::are_int_v<Ts...>>>
        [[nodiscard]] NOA_HD constexpr auto filter(Ts... ts) const noexcept {
            return Shape<value_type, sizeof...(Ts)>{(*this)[ts]...};
        }

        [[nodiscard]] NOA_HD constexpr Shape flip() const noexcept {
            return {vec.flip()};
        }

        template<typename I = value_type, size_t A, typename = std::enable_if_t<nt::is_int_v<I>>>
        [[nodiscard]] NOA_HD constexpr Shape reorder(const Vec<I, SIZE, A>& order) const noexcept {
            return {vec.reorder(order)};
        }

        [[nodiscard]] NOA_HD constexpr Shape circular_shift(int64_t count) {
            return {vec.circular_shift(count)};
        }

        [[nodiscard]] NOA_HD constexpr Shape copy() const noexcept {
            return *this;
        }

        template<size_t INDEX>
        [[nodiscard]] NOA_HD constexpr Shape set(value_type value) const noexcept {
            static_assert(INDEX < SIZE);
            auto output = *this;
            output[INDEX] = value;
            return output;
        }

    public:
        [[nodiscard]] NOA_HD constexpr value_type elements() const noexcept {
            if constexpr (SIZE == 0) {
                return 0;
            } else {
                auto output = vec[0];
                for (size_t i = 1; i < SIZE; ++i)
                    output *= vec[i];
                return output;
            }
        }

        // Whether the shape has at least one dimension equal to 0.
        [[nodiscard]] NOA_HD constexpr bool is_empty() const noexcept {
            return any(vec == 0);
        }

        /// Returns the logical number of dimensions in the BDHW convention.
        /// This returns a value from 0 to 3. The batch dimension is ignored,
        /// and is_batched() should be used to know whether the array is batched.
        /// Note that both row and column vectors are considered to be 1d, but
        /// if the depth dimension is greater than 1, ndim() == 3 even if both the
        /// height and width are 1.
        [[nodiscard]] NOA_HD constexpr value_type ndim() const noexcept {
            NOA_ASSERT(!is_empty());
            if constexpr (SIZE <= 1) {
                return static_cast<value_type>(SIZE);
            } else if constexpr (SIZE == 2) {
                return vec[0] > 1 && vec[1] > 1 ? 2 : 1;
            } else if constexpr (SIZE == 3) {
                return vec[0] > 1 ? 3 :
                       vec[1] > 1 && vec[2] > 1 ? 2 : 1;
            } else {
                return vec[1] > 1 ? 3 :
                       vec[2] > 1 && vec[3] > 1 ? 2 : 1;
            }
        }

        /// Computes the strides, in elements, in C- or F-order.
        /// Note that if the height and width dimensions are empty, 'C' and 'F' returns the same strides.
        template<char ORDER = 'C', typename Void = void, nt::enable_if_bool_t<std::is_void_v<Void> && (SIZE > 0)> = true>
        [[nodiscard]] NOA_HD constexpr auto strides() const noexcept {
            using output_strides = Strides<value_type, SIZE>;

            if constexpr (ORDER == 'C' || ORDER == 'c') {
                if constexpr (SIZE == 4) {
                    return output_strides{vec[3] * vec[2] * vec[1],
                                          vec[3] * vec[2],
                                          vec[3],
                                          1};
                } else if constexpr (SIZE == 3) {
                    return output_strides{vec[2] * vec[1],
                                          vec[2],
                                          1};
                } else if constexpr (SIZE == 2) {
                    return output_strides{vec[1], 1};
                } else {
                    return output_strides{1};
                }
            } else if constexpr (ORDER == 'F' || ORDER == 'f') {
                if constexpr (SIZE == 4) {
                    return output_strides{vec[3] * vec[2] * vec[1],
                                          vec[3] * vec[2],
                                          1,
                                          vec[2]};
                } else if constexpr (SIZE == 3) {
                    return output_strides{vec[2] * vec[1],
                                          1,
                                          vec[1]};
                } else if constexpr (SIZE == 2) {
                    return output_strides{1, vec[0]};
                } else {
                    return output_strides{1};
                }
            } else {
                static_assert(nt::always_false_v<Void>);
            }
        }

        // Returns the shape of the non-redundant FFT, in elements,
        [[nodiscard]] NOA_HD constexpr Shape rfft() const noexcept {
            Shape output = *this;
            if constexpr (SIZE > 0)
                output[SIZE - 1] = output[SIZE - 1] / 2 + 1;
            return output;
        }

        // Whether the shape describes vector.
        // A vector has one dimension with a size >= 1 and all the other dimensions empty (i.e. size == 1).
        // By this definition, the shapes {1,1,1,1}, {5,1,1,1} and {1,1,1,5} are all vectors.
        // If "can_be_batched" is true, the shape can describe a batch of vectors,
        // e.g. {4,1,1,5} is describing 4 row vectors with a length of 5.
        template<typename Void = void, nt::enable_if_bool_t<(SIZE == 4) && std::is_void_v<Void>> = true>
        [[nodiscard]]  NOA_FHD constexpr bool is_vector(bool can_be_batched = false) const noexcept {
            int non_empty_dimension = 0;
            for (size_t i = 0; i < SIZE; ++i) {
                if (vec[i] == 0)
                    return false; // empty/invalid shape
                if ((!can_be_batched || i != 0) && vec[i] > 1)
                    ++non_empty_dimension;
            }
            return non_empty_dimension <= 1;
        }

        // Whether the shape describes vector.
        // A vector has one dimension with a size >= 1 and all the other dimensions empty (i.e. size == 1).
        // By this definition, the shapes {1,1,1}, {5,1,1} and {1,1,5} are all vectors.
        template<typename Void = void, nt::enable_if_bool_t<(SIZE > 0 && SIZE <= 3) && std::is_void_v<Void>> = true>
        [[nodiscard]] NOA_FHD constexpr bool is_vector() const noexcept {
            int non_empty_dimension = 0;
            for (size_t i = 0; i < SIZE; ++i) {
                if (vec[i] == 0)
                    return false; // empty/invalid shape
                if (vec[i] > 1)
                    ++non_empty_dimension;
            }
            return non_empty_dimension <= 1;
        }

        // Whether this is a (batched) column vector.
        template<typename Void = void, typename = std::enable_if_t<(SIZE >= 2) && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr bool is_column() const noexcept {
            return vec[SIZE - 2] >= 1 && vec[SIZE - 1] == 1;
        }

        // Whether this is a (batched) row vector.
        template<typename Void = void, typename = std::enable_if_t<(SIZE >= 2) && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr bool is_row() const noexcept {
            return vec[SIZE - 2] == 1 && vec[SIZE - 1] >= 1;
        }

        // Whether this is a (batched) column vector.
        template<typename Void = void, typename = std::enable_if_t<(SIZE == 4) && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr bool is_batched() const noexcept {
            return vec[0] > 1;
        }

        // Move the first non-empty dimension (starting from the front) to the batch dimension.
        template<typename Void = void, typename = std::enable_if_t<(SIZE == 4) && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr Shape to_batched() const noexcept {
            if (vec[0] > 1)
                return *this; // already batched
            if (vec[1] > 1)
                return Shape{vec[1], 1, vec[2], vec[3]};
            if (vec[2] > 1)
                return Shape{vec[2], 1, 1, vec[3]};
            if (vec[3] > 1)
                return Shape{vec[3], 1, 1, 1};
            return *this; // {1,1,1,1}
        }

        template<typename Void = void, typename = std::enable_if_t<(SIZE == 4) && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr auto split_batch() const noexcept -> Pair<value_type, Shape<value_type, 3>> {
            return {batch(), pop_front()};
        }

#if defined(NOA_IS_OFFLINE)
    public:
        [[nodiscard]] static std::string name() {
            return fmt::format("Shape<{},{}>", ns::to_human_readable<value_type>(), SIZE);
        }
#endif
    };

    /// Deduction guide.
    template<typename T, typename... U>
    Shape(T, U...) -> Shape<std::enable_if_t<(std::is_same_v<T, U> && ...), T>, 1 + sizeof...(U)>;

    template<typename Int, size_t N>
    class Strides {
    public:
        static_assert(nt::is_int_v<Int>);
        static_assert(N <= 4);
        using vector_type = Vec<Int, N>;
        using value_type = typename vector_type::value_type;
        using mutable_value_type = value_type;
        static constexpr int64_t SSIZE = N;
        static constexpr size_t SIZE = N;

    public:
        vector_type vec; // uninitialized

    public:
        template<typename T, typename = std::enable_if_t<nt::is_int_v<T>>>
        [[nodiscard]] NOA_HD static constexpr Strides from_value(T value) noexcept {
            return {vector_type::from_value(value)};
        }

        template<typename T, typename = std::enable_if_t<nt::is_int_v<T>>>
        [[nodiscard]] NOA_HD static constexpr Strides filled_with(T value) noexcept {
            return {vector_type::filled_with(value)}; // same as from_value
        }

        template<typename... Args, typename = std::enable_if_t<sizeof...(Args) == SIZE && nt::are_int_v<Args...>>>
        [[nodiscard]] NOA_HD static constexpr Strides from_values(Args... values) noexcept {
            return {vector_type::from_values(values...)};
        }

        template<typename T, size_t A, typename = std::enable_if_t<nt::is_int_v<T>>>
        [[nodiscard]] NOA_HD static constexpr Strides from_vec(const Vec<T, SIZE, A>& vector) noexcept {
            return {vector_type::from_vector(vector)};
        }

        template<typename T>
        [[nodiscard]] NOA_HD static constexpr Strides from_strides(const Strides<T, SIZE>& strides) noexcept {
            return from_vec(strides.vec);
        }

        template<typename T, typename = std::enable_if_t<nt::is_int_v<T>>>
        [[nodiscard]] NOA_HD static constexpr Strides from_pointer(const T* values) noexcept {
            return {vector_type::from_pointer(values)};
        }

    public:
        // Allow explicit conversion constructor (while still being an aggregate)
        // and add support for static_cast<Strides<U>>(Strides<T>{}).
        template<typename U>
        [[nodiscard]] NOA_HD constexpr explicit operator Strides<U, SIZE>() const noexcept {
            return Strides<U, SIZE>::from_strides(*this);
        }

    public: // Accessor operators and functions
        template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
        [[nodiscard]] NOA_HD constexpr value_type& operator[](I i) noexcept { return vec[i]; }

        template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
        [[nodiscard]] NOA_HD constexpr const value_type& operator[](I i) const noexcept { return vec[i]; }

        template<typename Void = void, typename = std::enable_if_t<(SIZE == 4) && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr value_type& batch() noexcept { return vec[0]; }

        template<typename Void = void, typename = std::enable_if_t<(SIZE == 4) && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr const value_type& batch() const noexcept { return vec[0]; }

        template<typename Void = void, typename = std::enable_if_t<(SIZE >= 3) && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr value_type& depth() noexcept { return vec[SIZE - 1]; }

        template<typename Void = void, typename = std::enable_if_t<(SIZE >= 3) && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr const value_type& depth() const noexcept { return vec[SIZE - 1]; }

        template<typename Void = void, typename = std::enable_if_t<(SIZE >= 2) && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr value_type& height() noexcept { return vec[SIZE - 2]; }

        template<typename Void = void, typename = std::enable_if_t<(SIZE >= 2) && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr const value_type& height() const noexcept { return vec[SIZE - 2]; }

        template<typename Void = void, typename = std::enable_if_t<(SIZE >= 1) && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr value_type& width() noexcept { return vec[SIZE - 1]; }

        template<typename Void = void, typename = std::enable_if_t<(SIZE >= 1) && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr const value_type& width() const noexcept { return vec[SIZE - 1]; }

        // Structure binding support.
        template<int I> [[nodiscard]] NOA_HD constexpr const value_type& get() const noexcept { return vec[I]; }
        template<int I> [[nodiscard]] NOA_HD constexpr value_type& get() noexcept { return vec[I]; }

        [[nodiscard]] NOA_HD constexpr const value_type* data() const noexcept { return vec.data(); }
        [[nodiscard]] NOA_HD constexpr value_type* data() noexcept { return vec.data(); }
        [[nodiscard]] NOA_HD constexpr size_t size() const noexcept { return SIZE; };
        [[nodiscard]] NOA_HD constexpr int64_t ssize() const noexcept { return SSIZE; };

    public: // Iterators -- support for range loops
        [[nodiscard]] NOA_HD constexpr value_type* begin() noexcept { return vec.begin(); }
        [[nodiscard]] NOA_HD constexpr const value_type* begin() const noexcept { return vec.begin(); }
        [[nodiscard]] NOA_HD constexpr const value_type* cbegin() const noexcept { return vec.cbegin(); }
        [[nodiscard]] NOA_HD constexpr value_type* end() noexcept { return vec.end(); }
        [[nodiscard]] NOA_HD constexpr const value_type* end() const noexcept { return vec.end(); }
        [[nodiscard]] NOA_HD constexpr const value_type* cend() const noexcept { return vec.cend(); }

    public: // Assignment operators
        NOA_HD constexpr Strides& operator=(value_type size) noexcept {
            *this = Strides::filled_with(size);
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
            return {-strides.vec};
        }

        // -- Binary Arithmetic Operators --
        [[nodiscard]] friend NOA_HD constexpr Strides operator+(Strides lhs, Strides rhs) noexcept {
            return {lhs.vec + rhs.vec};
        }

        [[nodiscard]] friend NOA_HD constexpr Strides operator+(const Strides& lhs, value_type rhs) noexcept {
            return lhs + Strides::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr Strides operator+(value_type lhs, const Strides& rhs) noexcept {
            return Strides::filled_with(lhs) + rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr Strides operator-(Strides lhs, Strides rhs) noexcept {
            return {lhs.vec - rhs.vec};
        }

        [[nodiscard]] friend NOA_HD constexpr Strides operator-(const Strides& lhs, value_type rhs) noexcept {
            return lhs - Strides::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr Strides operator-(value_type lhs, const Strides& rhs) noexcept {
            return Strides::filled_with(lhs) - rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr Strides operator*(Strides lhs, Strides rhs) noexcept {
            return {lhs.vec * rhs.vec};
        }

        [[nodiscard]] friend NOA_HD constexpr Strides operator*(const Strides& lhs, value_type rhs) noexcept {
            return lhs * Strides::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr Strides operator*(value_type lhs, const Strides& rhs) noexcept {
            return Strides::filled_with(lhs) * rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr Strides operator/(Strides lhs, Strides rhs) noexcept {
            return {lhs.vec / rhs.vec};
        }

        [[nodiscard]] friend NOA_HD constexpr Strides operator/(const Strides& lhs, value_type rhs) noexcept {
            return lhs / Strides::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr Strides operator/(value_type lhs, const Strides& rhs) noexcept {
            return Strides::filled_with(lhs) / rhs;
        }

        // -- Comparison Operators --
        [[nodiscard]] friend NOA_HD constexpr auto operator>(Strides lhs, Strides rhs) noexcept {
            return lhs.vec > rhs.vec;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator>(const Strides& lhs, value_type rhs) noexcept {
            return lhs > Strides::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator>(value_type lhs, const Strides& rhs) noexcept {
            return Strides::filled_with(lhs) > rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator<(Strides lhs, Strides rhs) noexcept {
            return lhs.vec < rhs.vec;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator<(const Strides& lhs, value_type rhs) noexcept {
            return lhs < Strides::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator<(value_type lhs, const Strides& rhs) noexcept {
            return Strides::filled_with(lhs) < rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator>=(Strides lhs, Strides rhs) noexcept {
            return lhs.vec >= rhs.vec;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator>=(const Strides& lhs, value_type rhs) noexcept {
            return lhs >= Strides::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator>=(value_type lhs, const Strides& rhs) noexcept {
            return Strides::filled_with(lhs) >= rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator<=(Strides lhs, Strides rhs) noexcept {
            return lhs.vec <= rhs.vec;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator<=(const Strides& lhs, value_type rhs) noexcept {
            return lhs <= Strides::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator<=(value_type lhs, const Strides& rhs) noexcept {
            return Strides::filled_with(lhs) <= rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator==(Strides lhs, Strides rhs) noexcept {
            return lhs.vec == rhs.vec;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator==(const Strides& lhs, value_type rhs) noexcept {
            return lhs == Strides::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator==(value_type lhs, const Strides& rhs) noexcept {
            return Strides::filled_with(lhs) == rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator!=(Strides lhs, Strides rhs) noexcept {
            return lhs.vec != rhs.vec;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator!=(const Strides& lhs, value_type rhs) noexcept {
            return lhs != Strides::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator!=(value_type lhs, const Strides& rhs) noexcept {
            return Strides::filled_with(lhs) != rhs;
        }

    public: // Type casts
        template<typename TTo, typename = std::enable_if_t<nt::is_int_v<TTo>>>
        [[nodiscard]] NOA_HD constexpr auto as() const noexcept {
            return static_cast<Strides<TTo, SIZE>>(*this);
        }

        template<typename TTo, typename = std::enable_if_t<nt::is_int_v<TTo>>>
        [[nodiscard]] NOA_HD constexpr auto as_clamp() const noexcept {
            return clamp_cast<Strides<TTo, SIZE>>(*this);
        }

#if defined(NOA_IS_OFFLINE)
        template<typename TTo, typename = std::enable_if_t<nt::is_int_v<TTo>>>
        [[nodiscard]] constexpr auto as_safe() const {
            return safe_cast<Strides<TTo, SIZE>>(*this);
        }
#endif

    public:
        template<size_t S = 1, typename = std::enable_if_t<(SIZE >= S)>>
        [[nodiscard]] NOA_HD constexpr auto pop_front() const noexcept {
            return Strides<value_type, SIZE - S>::from_pointer(data() + S);
        }

        template<size_t S = 1, typename = std::enable_if_t<(SIZE >= S)>>
        [[nodiscard]] NOA_HD constexpr auto pop_back() const noexcept {
            return Strides<value_type, SIZE - S>::from_pointer(data());
        }

        [[nodiscard]] NOA_HD constexpr auto push_front(value_type value) const noexcept {
            return Strides<value_type, SIZE + 1>{vec.push_front(value)};
        }

        [[nodiscard]] NOA_HD constexpr auto push_back(value_type value) const noexcept {
            return Strides<value_type, SIZE + 1>{vec.push_back(value)};
        }

        template<size_t S, size_t A>
        [[nodiscard]] NOA_HD constexpr auto push_front(const Vec<value_type, S, A>& vector) const noexcept {
            constexpr size_t NEW_SIZE = SIZE + S;
            return Strides<value_type, NEW_SIZE>{vec.push_front(vector)};
        }

        template<size_t S, size_t A>
        [[nodiscard]] NOA_HD constexpr auto push_back(const Vec<value_type, S, A>& vector) const noexcept {
            constexpr size_t NEW_SIZE = SIZE + S;
            return Strides<value_type, NEW_SIZE>{vec.push_back(vector)};
        }

        template<typename... Ts, typename = std::enable_if_t<nt::are_int_v<Ts...>>>
        [[nodiscard]] NOA_HD constexpr auto filter(Ts... ts) const noexcept {
            return Strides<value_type, sizeof...(Ts)>{(*this)[ts]...};
        }

        [[nodiscard]] NOA_HD constexpr Strides flip() const noexcept {
            return {vec.flip()};
        }

        template<typename I = value_type, size_t A, typename = std::enable_if_t<nt::is_int_v<I>>>
        [[nodiscard]] NOA_HD constexpr Strides reorder(const Vec<I, SIZE, A>& order) const noexcept {
            return {vec.reorder(order)};
        }

        [[nodiscard]] NOA_HD constexpr Strides circular_shift(int64_t count) {
            return {vec.circular_shift(count)};
        }

        [[nodiscard]] NOA_HD constexpr Strides copy() const noexcept {
            return *this;
        }

        template<size_t INDEX>
        [[nodiscard]] NOA_HD constexpr Strides set(value_type value) const noexcept {
            static_assert(INDEX < SIZE);
            auto output = *this;
            output[INDEX] = value;
            return output;
        }

    public:
        // Whether there's at least one dimension equal to 0.
        [[nodiscard]] NOA_HD constexpr bool is_broadcast() const noexcept {
            return any(vec == 0);
        }

        // Whether the strides are in the rightmost order.
        // Rightmost order is when the innermost stride (i.e. the dimension with the smallest stride)
        // is on the right, and strides increase right-to-left.
        template<typename Void = void, nt::enable_if_bool_t<std::is_void_v<Void> && (SIZE > 0)> = true>
        [[nodiscard]] NOA_HD constexpr bool is_rightmost() const noexcept {
            for (size_t i = 0; i < SIZE - 1; ++i)
                if (vec[i] < vec[i + 1])
                    return false;
            return true;
        }

        // Computes the physical layout (the actual memory footprint) encoded in these strides.
        // Note that the left-most size is not-encoded in the strides, and therefore cannot be recovered.
        template<char ORDER = 'C', typename Void = void,
                 typename = std::enable_if_t<(SIZE >= 2) && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr auto physical_shape() const noexcept {
            NOA_ASSERT(!is_broadcast() && "Cannot recover the physical shape from broadcast strides");
            using output_shape = Shape<value_type, SIZE - 1>;

            if constexpr (ORDER == 'C' || ORDER == 'c') {
                if constexpr (SIZE == 4) {
                    return output_shape{vec[0] / vec[1],
                                        vec[1] / vec[2],
                                        vec[2]};
                } else if constexpr (SIZE == 3) {
                    return output_shape{vec[0] / vec[1],
                                        vec[1]};
                } else {
                    return output_shape{vec[0]};
                }
            } else if constexpr (ORDER == 'F' || ORDER == 'f') {
                if constexpr (SIZE == 4) {
                    return output_shape{vec[0] / vec[1],
                                        vec[3],
                                        vec[1] / vec[3]};
                } else if constexpr (SIZE == 3) {
                    return output_shape{vec[2],
                                        vec[0] / vec[2]};
                } else {
                    return output_shape{vec[1]};
                }
            } else {
                static_assert(nt::always_false_v<Void>);
            }
        }

        template<typename Void = void, typename = std::enable_if_t<(SIZE == 4) && std::is_void_v<Void>>>
        [[nodiscard]] NOA_HD constexpr auto split_batch() const noexcept -> Pair<value_type, Strides<value_type, 3>> {
            return {batch(), pop_front()};
        }

#if defined(NOA_IS_OFFLINE)
    public:
        [[nodiscard]] static std::string name() {
            return fmt::format("Strides<{},{}>", ns::to_human_readable<value_type>(), SIZE);
        }
#endif
    };

    /// Deduction guide.
    template<typename T, typename... U>
    Strides(T, U...) -> Strides<std::enable_if_t<(std::is_same_v<T, U> && ...), T>, 1 + sizeof...(U)>;
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

    template<typename T, size_t N>
    struct tuple_size<const noa::Shape<T, N>> : std::integral_constant<size_t, N> {};

    template<size_t I, size_t N, typename T>
    struct tuple_element<I, const noa::Shape<T, N>> { using type = const T; };

    template<typename T, size_t N>
    struct tuple_size<const noa::Strides<T, N>> : std::integral_constant<size_t, N> {};

    template<size_t I, size_t N, typename T>
    struct tuple_element<I, const noa::Strides<T, N>> { using type = const T; };
}

#if defined(NOA_IS_OFFLINE)
// Support for output stream:
namespace noa::inline types {
    template<typename T, size_t N>
    NOA_IH std::ostream& operator<<(std::ostream& os, const Shape<T, N>& v) {
        os << fmt::format("{}", v.vec);
        return os;
    }

    template<typename T, size_t N>
    NOA_IH std::ostream& operator<<(std::ostream& os, const Strides<T, N>& v) {
        os << fmt::format("{}", v.vec);
        return os;
    }
}
#endif

// Type aliases:
namespace noa::inline types {
    template<typename T> using Shape1 = Shape<T, 1>;
    template<typename T> using Shape2 = Shape<T, 2>;
    template<typename T> using Shape3 = Shape<T, 3>;
    template<typename T> using Shape4 = Shape<T, 4>;

    template<typename T> using Strides1 = Strides<T, 1>;
    template<typename T> using Strides2 = Strides<T, 2>;
    template<typename T> using Strides3 = Strides<T, 3>;
    template<typename T> using Strides4 = Strides<T, 4>;
}

namespace noa::traits {
    template<typename T, size_t N> struct proclaim_is_shape<Shape<T, N>> : std::true_type {};
    template<typename V1, size_t N, typename V2> struct proclaim_is_shape_of_type<Shape<V1, N>, V2> : std::bool_constant<std::is_same_v<V1, V2>> {};
    template<typename V, size_t N1, size_t N2> struct proclaim_is_shape_of_size<Shape<V, N1>, N2> : std::bool_constant<N1 == N2> {};

    template<typename T, size_t N> struct proclaim_is_strides<Strides<T, N>> : std::true_type {};
    template<typename V1, size_t N, typename V2> struct proclaim_is_strides_of_type<Strides<V1, N>, V2> : std::bool_constant<std::is_same_v<V1, V2>> {};
    template<typename V, size_t N1, size_t N2> struct proclaim_is_strides_of_size<Strides<V, N1>, N2> : std::bool_constant<N1 == N2> {};
}

namespace noa::inline types {
    // -- Modulo Operator --
    template<typename T, nt::enable_if_bool_t<nt::is_shape_or_strides_v<T> && (T::SIZE > 0)> = true>
    [[nodiscard]] NOA_HD constexpr T operator%(T lhs, const T& rhs) noexcept {
        for (int64_t i = 0; i < T::SSIZE; ++i)
            lhs[i] %= rhs[i];
        return lhs;
    }

    template<typename T, typename Int,
             nt::enable_if_bool_t<nt::is_int_v<Int> && nt::is_shape_or_strides_v<T> && (T::SIZE > 0)> = true>
    [[nodiscard]] NOA_HD constexpr T operator%(const T& lhs, Int rhs) noexcept {
        return lhs % T::filled_with(rhs);
    }

    template<typename T, typename Int,
             nt::enable_if_bool_t<nt::is_int_v<Int> && nt::is_shape_or_strides_v<T> && (T::SIZE > 0)> = true>
    [[nodiscard]] NOA_HD constexpr T operator%(Int lhs, const T& rhs) noexcept {
        return T::filled_with(lhs) % rhs;
    }
}

namespace noa {
    // Cast Shape->Shape
    template<typename TTo, typename TFrom, size_t N, nt::enable_if_bool_t<nt::is_shapeN_v<TTo, N>> = true>
    [[nodiscard]] NOA_FHD constexpr bool is_safe_cast(const Shape<TFrom, N>& src) noexcept {
        return is_safe_cast<typename TTo::vector_type>(src.vec);
    }

    template<typename TTo, typename TFrom, size_t N, nt::enable_if_bool_t<nt::is_shapeN_v<TTo, N>> = true>
    [[nodiscard]] NOA_FHD constexpr TTo clamp_cast(const Shape<TFrom, N>& src) noexcept {
        return TTo{clamp_cast<typename TTo::vector_type>(src.vec)};
    }

    // Cast Strides->Strides
    template<typename TTo, typename TFrom, size_t N, nt::enable_if_bool_t<nt::is_stridesN_v<TTo, N>> = true>
    [[nodiscard]] NOA_FHD constexpr bool is_safe_cast(const Strides<TFrom, N>& src) noexcept {
        return is_safe_cast<typename TTo::vector_type>(src.vec);
    }

    template<typename TTo, typename TFrom, size_t N, nt::enable_if_bool_t<nt::is_stridesN_v<TTo, N>> = true>
    [[nodiscard]] NOA_FHD constexpr TTo clamp_cast(const Strides<TFrom, N>& src) noexcept {
        return TTo{clamp_cast<typename TTo::vector_type>(src.vec)};
    }

    template<typename T, nt::enable_if_bool_t<nt::is_shape_or_strides_v<T>> = true>
    [[nodiscard]] NOA_FHD constexpr T abs(T shape) noexcept {
        return {abs(shape.vec)};
    }

    template<typename T, nt::enable_if_bool_t<nt::is_shape_or_strides_v<T> && (T::SIZE > 0)> = true>
    [[nodiscard]] NOA_FHD constexpr auto sum(const T& shape) noexcept {
        return sum(shape.vec);
    }

    template<typename T, nt::enable_if_bool_t<nt::is_shape_or_strides_v<T> && (T::SIZE > 0)> = true>
    [[nodiscard]] NOA_FHD constexpr auto product(const T& shape) noexcept {
        return product(shape.vec);
    }

    template<typename T, nt::enable_if_bool_t<nt::is_shape_or_strides_v<T> && (T::SIZE > 0)> = true>
    [[nodiscard]] NOA_FHD constexpr auto min(const T& shape) noexcept {
        return min(shape.vec);
    }

    template<typename T, nt::enable_if_bool_t<nt::is_shape_or_strides_v<T>> = true>
    [[nodiscard]] NOA_FHD constexpr T min(const T& lhs, const T& rhs) noexcept {
        return {min(lhs.vec, rhs.vec)};
    }

    template<typename Int, typename T,
             nt::enable_if_bool_t<
                     nt::is_shape_or_strides_v<T> &&
                     nt::is_almost_same_v<nt::value_type_t<T>, Int>> = true>
    [[nodiscard]] NOA_FHD constexpr auto min(const T& lhs, Int rhs) noexcept {
        return min(lhs, T::filled_with(rhs));
    }

    template<typename Int, typename T,
             nt::enable_if_bool_t<
                     nt::is_shape_or_strides_v<T> &&
                     nt::is_almost_same_v<nt::value_type_t<T>, Int>> = true>
    [[nodiscard]] NOA_FHD constexpr auto min(Int lhs, const T& rhs) noexcept {
        return min(T::filled_with(lhs), rhs);
    }

    template<typename T, nt::enable_if_bool_t<nt::is_shape_or_strides_v<T>> = true>
    [[nodiscard]] NOA_FHD constexpr auto max(const T& shape) noexcept {
        return max(shape.vec);
    }

    template<typename T, nt::enable_if_bool_t<nt::is_shape_or_strides_v<T> && (T::SIZE > 0)> = true>
    [[nodiscard]] NOA_FHD constexpr T max(const T& lhs, const T& rhs) noexcept {
        return {max(lhs.vec, rhs.vec)};
    }

    template<typename Int, typename T,
             nt::enable_if_bool_t<
                     nt::is_shape_or_strides_v<T> &&
                     nt::is_almost_same_v<nt::value_type_t<T>, Int>> = true>
    [[nodiscard]] NOA_FHD constexpr auto max(const T& lhs, Int rhs) noexcept {
        return max(lhs, T::filled_with(rhs));
    }

    template<typename Int, typename T,
             nt::enable_if_bool_t<
                     nt::is_shape_or_strides_v<T> &&
                     nt::is_almost_same_v<nt::value_type_t<T>, Int>> = true>
    [[nodiscard]] NOA_FHD constexpr auto max(Int lhs, const T& rhs) noexcept {
        return max(T::filled_with(lhs), rhs);
    }

    template<typename T, nt::enable_if_bool_t<nt::is_shape_or_strides_v<T>> = true>
    [[nodiscard]] NOA_FHD constexpr auto clamp(const T& lhs, const T& low, const T& high) noexcept {
        return min(max(lhs, low), high);
    }

    template<typename Int, typename T,
             nt::enable_if_bool_t<
                     nt::is_shape_or_strides_v<T> &&
                     nt::is_almost_same_v<nt::value_type_t<T>, Int>> = true>
    [[nodiscard]] NOA_FHD constexpr auto clamp(const T& lhs, Int low, Int high) noexcept {
        return min(max(lhs, low), high);
    }

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
        small_stable_sort<N>(shape.data(), Less{});
        return shape;
    }

    template<typename T, size_t N>
    [[nodiscard]] NOA_IHD constexpr auto sort(Shape<T, N> shape) noexcept {
        small_stable_sort<N>(shape.data(), Less{});
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
        small_stable_sort<N>(shape.data(), Less{});
        return shape;
    }

    template<typename T, size_t N>
    [[nodiscard]] NOA_IHD constexpr auto sort(Strides<T, N> shape) noexcept {
        small_stable_sort<N>(shape.data(), Less{});
        return shape;
    }
}
