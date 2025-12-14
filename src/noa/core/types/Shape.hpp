#pragma once

#include "noa/core/Traits.hpp"
#include "noa/core/types/Vec.hpp"
#include "noa/core/types/Tuple.hpp"

namespace noa::inline types {
    template<typename Int, usize N, usize A>
    class Strides;
}

namespace noa::inline types {
    template<typename T, usize N, usize A = 0>
    class Shape {
    public:
        static_assert(nt::integer<T> and N <= 4);
        using vector_type = Vec<T, N, A>;
        using value_type = vector_type::value_type;
        using mutable_value_type = value_type;
        static constexpr isize SSIZE = N;
        static constexpr usize SIZE = N;

    public:
        NOA_NO_UNIQUE_ADDRESS vector_type vec; // uninitialized

    public: // Static factory functions
        template<nt::integer U>
        [[nodiscard]] NOA_HD static constexpr Shape from_value(U value) noexcept {
            return {vector_type::from_value(value)};
        }

        template<nt::integer U>
        [[nodiscard]] NOA_HD static constexpr Shape filled_with(U value) noexcept {
            return {vector_type::filled_with(value)}; // same as from_value
        }

        template<nt::integer... Args> requires (sizeof...(Args) == SIZE)
        [[nodiscard]] NOA_HD static constexpr Shape from_values(Args... values) noexcept {
            return {vector_type::from_values(values...)};
        }

        template<nt::integer U, usize AR>
        [[nodiscard]] NOA_HD static constexpr Shape from_vec(const Vec<U, SIZE, AR>& vector) noexcept {
            return {vector_type::from_vec(vector)};
        }

        template<typename U, usize AR>
        [[nodiscard]] NOA_HD static constexpr Shape from_shape(const Shape<U, SIZE, AR>& shape) noexcept {
            return from_vec(shape.vec);
        }

        template<nt::integer U>
        [[nodiscard]] NOA_HD static constexpr Shape from_pointer(const U* values) noexcept {
            return {vector_type::from_pointer(values)};
        }

    public:
        // Allow explicit conversion constructor (while still being an aggregate)
        // and add support for static_cast<Shape<U>>(Shape<T>{}).
        template<typename U, usize AR>
        [[nodiscard]] NOA_HD constexpr explicit operator Shape<U, SIZE, AR>() const noexcept {
            return Shape<U, SIZE, AR>::from_shape(*this);
        }

        // Allow implicit conversion from a shape with a different alignment.
        template<usize AR> requires (A != AR)
        [[nodiscard]] NOA_HD constexpr /*implicit*/ operator Shape<value_type, SIZE, AR>() const noexcept {
            return Shape<value_type, SIZE, AR>::from_shape(*this);
        }

    public: // Accessor operators and functions
        [[nodiscard]] NOA_HD constexpr auto operator[](nt::integer auto i)       noexcept ->       value_type& requires (SIZE > 0) { return vec[i]; }
        [[nodiscard]] NOA_HD constexpr auto operator[](nt::integer auto i) const noexcept -> const value_type& requires (SIZE > 0) { return vec[i]; }

        [[nodiscard]] NOA_HD constexpr auto batch()        noexcept ->       value_type& requires (SIZE == 4) { return vec[0]; }
        [[nodiscard]] NOA_HD constexpr auto batch()  const noexcept -> const value_type& requires (SIZE == 4) { return vec[0]; }
        [[nodiscard]] NOA_HD constexpr auto depth()        noexcept ->       value_type& requires (SIZE >= 3) { return vec[SIZE - 3]; }
        [[nodiscard]] NOA_HD constexpr auto depth()  const noexcept -> const value_type& requires (SIZE >= 3) { return vec[SIZE - 3]; }
        [[nodiscard]] NOA_HD constexpr auto height()       noexcept ->       value_type& requires (SIZE >= 2) { return vec[SIZE - 2]; }
        [[nodiscard]] NOA_HD constexpr auto height() const noexcept -> const value_type& requires (SIZE >= 2) { return vec[SIZE - 2]; }
        [[nodiscard]] NOA_HD constexpr auto width()        noexcept ->       value_type& requires (SIZE >= 1) { return vec[SIZE - 1]; }
        [[nodiscard]] NOA_HD constexpr auto width()  const noexcept -> const value_type& requires (SIZE >= 1) { return vec[SIZE - 1]; }

        // Structure binding support.
        template<int I> [[nodiscard]] NOA_HD constexpr       value_type& get() noexcept { return vec[I]; }
        template<int I> [[nodiscard]] NOA_HD constexpr const value_type& get() const noexcept { return vec[I]; }

        [[nodiscard]] NOA_HD constexpr auto data()   const noexcept -> const value_type* { return vec.data(); }
        [[nodiscard]] NOA_HD constexpr auto data()         noexcept -> value_type* { return vec.data(); }
        [[nodiscard]] NOA_HD static constexpr auto size()  noexcept -> usize { return SIZE; };
        [[nodiscard]] NOA_HD static constexpr auto ssize() noexcept -> isize { return SSIZE; };

    public: // Iterators -- support for range loops
        [[nodiscard]] NOA_HD constexpr       value_type* begin()        noexcept { return vec.begin(); }
        [[nodiscard]] NOA_HD constexpr const value_type* begin()  const noexcept { return vec.begin(); }
        [[nodiscard]] NOA_HD constexpr const value_type* cbegin() const noexcept { return vec.cbegin(); }
        [[nodiscard]] NOA_HD constexpr       value_type* end()          noexcept { return vec.end(); }
        [[nodiscard]] NOA_HD constexpr const value_type* end()    const noexcept { return vec.end(); }
        [[nodiscard]] NOA_HD constexpr const value_type* cend()   const noexcept { return vec.cend(); }

    public: // Assignment operators
        NOA_FHD constexpr auto operator=(value_type size) noexcept -> Shape& {
            *this = Shape::filled_with(size);
            return *this;
        }

        NOA_FHD constexpr auto operator+=(const Shape& shape) noexcept -> Shape& { *this = *this + shape; return *this; }
        NOA_FHD constexpr auto operator-=(const Shape& shape) noexcept -> Shape& { *this = *this - shape; return *this; }
        NOA_FHD constexpr auto operator*=(const Shape& shape) noexcept -> Shape& { *this = *this * shape; return *this; }
        NOA_FHD constexpr auto operator/=(const Shape& shape) noexcept -> Shape& { *this = *this / shape; return *this; }

        NOA_FHD constexpr auto operator+=(value_type value) noexcept -> Shape& { *this = *this + value; return *this; }
        NOA_FHD constexpr auto operator-=(value_type value) noexcept -> Shape& { *this = *this - value; return *this; }
        NOA_FHD constexpr auto operator*=(value_type value) noexcept -> Shape& { *this = *this * value; return *this; }
        NOA_FHD constexpr auto operator/=(value_type value) noexcept -> Shape& { *this = *this / value; return *this; }

    public: // Non-member functions
        [[nodiscard]] NOA_FHD friend constexpr Shape operator+(const Shape& shape) noexcept {
            return shape;
        }

        [[nodiscard]] NOA_FHD friend constexpr Shape operator-(Shape shape) noexcept {
            return {-shape.vec};
        }

        [[nodiscard]] NOA_FHD friend constexpr Shape operator+(Shape lhs, Shape rhs) noexcept { return {lhs.vec + rhs.vec}; }
        [[nodiscard]] NOA_FHD friend constexpr Shape operator-(Shape lhs, Shape rhs) noexcept { return {lhs.vec - rhs.vec}; }
        [[nodiscard]] NOA_FHD friend constexpr Shape operator*(Shape lhs, Shape rhs) noexcept { return {lhs.vec * rhs.vec}; }
        [[nodiscard]] NOA_FHD friend constexpr Shape operator/(Shape lhs, Shape rhs) noexcept { return {lhs.vec / rhs.vec}; }

        [[nodiscard]] NOA_FHD friend constexpr Shape operator+(const Shape& lhs, value_type rhs) noexcept { return lhs + Shape::filled_with(rhs); }
        [[nodiscard]] NOA_FHD friend constexpr Shape operator-(const Shape& lhs, value_type rhs) noexcept { return lhs - Shape::filled_with(rhs); }
        [[nodiscard]] NOA_FHD friend constexpr Shape operator*(const Shape& lhs, value_type rhs) noexcept { return lhs * Shape::filled_with(rhs); }
        [[nodiscard]] NOA_FHD friend constexpr Shape operator/(const Shape& lhs, value_type rhs) noexcept { return lhs / Shape::filled_with(rhs); }

        [[nodiscard]] NOA_FHD friend constexpr Shape operator+(value_type lhs, const Shape& rhs) noexcept { return Shape::filled_with(lhs) + rhs; }
        [[nodiscard]] NOA_FHD friend constexpr Shape operator-(value_type lhs, const Shape& rhs) noexcept { return Shape::filled_with(lhs) - rhs; }
        [[nodiscard]] NOA_FHD friend constexpr Shape operator*(value_type lhs, const Shape& rhs) noexcept { return Shape::filled_with(lhs) * rhs; }
        [[nodiscard]] NOA_FHD friend constexpr Shape operator/(value_type lhs, const Shape& rhs) noexcept { return Shape::filled_with(lhs) / rhs; }

    public: // comparison operators
        [[nodiscard]] NOA_FHD friend constexpr bool operator==(const Shape& lhs, const Shape& rhs) noexcept { return nd::vec_op_bool<Equal>(lhs.vec, rhs.vec); }
        [[nodiscard]] NOA_FHD friend constexpr bool operator!=(const Shape& lhs, const Shape& rhs) noexcept { return nd::vec_op_bool<NotEqual>(lhs.vec, rhs.vec); }
        [[nodiscard]] NOA_FHD friend constexpr bool operator<=(const Shape& lhs, const Shape& rhs) noexcept { return nd::vec_op_bool<LessEqual>(lhs.vec, rhs.vec); }
        [[nodiscard]] NOA_FHD friend constexpr bool operator>=(const Shape& lhs, const Shape& rhs) noexcept { return nd::vec_op_bool<GreaterEqual>(lhs.vec, rhs.vec); }
        [[nodiscard]] NOA_FHD friend constexpr bool operator<(const Shape& lhs, const Shape& rhs) noexcept { return nd::vec_op_bool<Less>(lhs.vec, rhs.vec); }
        [[nodiscard]] NOA_FHD friend constexpr bool operator>(const Shape& lhs, const Shape& rhs) noexcept { return nd::vec_op_bool<Greater>(lhs.vec, rhs.vec); }

        [[nodiscard]] NOA_FHD friend constexpr bool operator==(const Shape& lhs, const value_type& rhs) noexcept { return lhs == Shape::filled_with(rhs); }
        [[nodiscard]] NOA_FHD friend constexpr bool operator!=(const Shape& lhs, const value_type& rhs) noexcept { return lhs != Shape::filled_with(rhs); }
        [[nodiscard]] NOA_FHD friend constexpr bool operator<=(const Shape& lhs, const value_type& rhs) noexcept { return lhs <= Shape::filled_with(rhs); }
        [[nodiscard]] NOA_FHD friend constexpr bool operator>=(const Shape& lhs, const value_type& rhs) noexcept { return lhs >= Shape::filled_with(rhs); }
        [[nodiscard]] NOA_FHD friend constexpr bool operator<(const Shape& lhs, const value_type& rhs) noexcept { return lhs < Shape::filled_with(rhs); }
        [[nodiscard]] NOA_FHD friend constexpr bool operator>(const Shape& lhs, const value_type& rhs) noexcept { return lhs > Shape::filled_with(rhs); }

        [[nodiscard]] NOA_FHD friend constexpr bool operator==(const value_type& lhs, const Shape& rhs) noexcept { return Shape::filled_with(lhs) == rhs; }
        [[nodiscard]] NOA_FHD friend constexpr bool operator!=(const value_type& lhs, const Shape& rhs) noexcept { return Shape::filled_with(lhs) != rhs; }
        [[nodiscard]] NOA_FHD friend constexpr bool operator<=(const value_type& lhs, const Shape& rhs) noexcept { return Shape::filled_with(lhs) <= rhs; }
        [[nodiscard]] NOA_FHD friend constexpr bool operator>=(const value_type& lhs, const Shape& rhs) noexcept { return Shape::filled_with(lhs) >= rhs; }
        [[nodiscard]] NOA_FHD friend constexpr bool operator<(const value_type& lhs, const Shape& rhs) noexcept { return Shape::filled_with(lhs) < rhs; }
        [[nodiscard]] NOA_FHD friend constexpr bool operator>(const value_type& lhs, const Shape& rhs) noexcept { return Shape::filled_with(lhs) > rhs; }

    public: // element-wise comparison
        [[nodiscard]] NOA_FHD constexpr auto cmp_eq(const Shape& rhs) const noexcept { return nd::vec_cmp<Equal>(vec, rhs.vec); }
        [[nodiscard]] NOA_FHD constexpr auto cmp_ne(const Shape& rhs) const noexcept { return nd::vec_cmp<NotEqual>(vec, rhs.vec); }
        [[nodiscard]] NOA_FHD constexpr auto cmp_le(const Shape& rhs) const noexcept { return nd::vec_cmp<LessEqual>(vec, rhs.vec); }
        [[nodiscard]] NOA_FHD constexpr auto cmp_ge(const Shape& rhs) const noexcept { return nd::vec_cmp<GreaterEqual>(vec, rhs.vec); }
        [[nodiscard]] NOA_FHD constexpr auto cmp_lt(const Shape& rhs) const noexcept { return nd::vec_cmp<Less>(vec, rhs.vec); }
        [[nodiscard]] NOA_FHD constexpr auto cmp_gt(const Shape& rhs) const noexcept { return nd::vec_cmp<Greater>(vec, rhs.vec); }

        [[nodiscard]] NOA_FHD constexpr auto cmp_eq(const value_type& rhs) const noexcept { return nd::vec_cmp<Equal>(vec, vector_type::filled_with(rhs)); }
        [[nodiscard]] NOA_FHD constexpr auto cmp_ne(const value_type& rhs) const noexcept { return nd::vec_cmp<NotEqual>(vec, vector_type::filled_with(rhs)); }
        [[nodiscard]] NOA_FHD constexpr auto cmp_le(const value_type& rhs) const noexcept { return nd::vec_cmp<LessEqual>(vec, vector_type::filled_with(rhs)); }
        [[nodiscard]] NOA_FHD constexpr auto cmp_ge(const value_type& rhs) const noexcept { return nd::vec_cmp<GreaterEqual>(vec, vector_type::filled_with(rhs)); }
        [[nodiscard]] NOA_FHD constexpr auto cmp_lt(const value_type& rhs) const noexcept { return nd::vec_cmp<Less>(vec, vector_type::filled_with(rhs)); }
        [[nodiscard]] NOA_FHD constexpr auto cmp_gt(const value_type& rhs) const noexcept { return nd::vec_cmp<Greater>(vec, vector_type::filled_with(rhs)); }

        [[nodiscard]] NOA_FHD constexpr bool any_eq(const Shape& rhs) const noexcept { return vec.any_eq(rhs.vec); }
        [[nodiscard]] NOA_FHD constexpr bool any_ne(const Shape& rhs) const noexcept { return vec.any_ne(rhs.vec); }
        [[nodiscard]] NOA_FHD constexpr bool any_le(const Shape& rhs) const noexcept { return vec.any_le(rhs.vec); }
        [[nodiscard]] NOA_FHD constexpr bool any_ge(const Shape& rhs) const noexcept { return vec.any_ge(rhs.vec); }
        [[nodiscard]] NOA_FHD constexpr bool any_lt(const Shape& rhs) const noexcept { return vec.any_lt(rhs.vec); }
        [[nodiscard]] NOA_FHD constexpr bool any_gt(const Shape& rhs) const noexcept { return vec.any_gt(rhs.vec); }

        [[nodiscard]] NOA_FHD constexpr bool any_eq(const value_type& rhs) const noexcept { return vec.any_eq(rhs); }
        [[nodiscard]] NOA_FHD constexpr bool any_ne(const value_type& rhs) const noexcept { return vec.any_ne(rhs); }
        [[nodiscard]] NOA_FHD constexpr bool any_le(const value_type& rhs) const noexcept { return vec.any_le(rhs); }
        [[nodiscard]] NOA_FHD constexpr bool any_ge(const value_type& rhs) const noexcept { return vec.any_ge(rhs); }
        [[nodiscard]] NOA_FHD constexpr bool any_lt(const value_type& rhs) const noexcept { return vec.any_lt(rhs); }
        [[nodiscard]] NOA_FHD constexpr bool any_gt(const value_type& rhs) const noexcept { return vec.any_gt(rhs); }

    public: // Type casts
        template<nt::integer U, usize AR = 0>
        [[nodiscard]] NOA_HD constexpr auto as() const noexcept {
            return static_cast<Shape<U, SIZE, AR>>(*this);
        }

        template<nt::integer U, usize AR = 0>
        [[nodiscard]] NOA_HD constexpr auto as_clamp() const noexcept {
            return clamp_cast<Shape<U, SIZE, AR>>(*this);
        }

        template<nt::integer U, usize AR = 0>
        [[nodiscard]] constexpr auto as_safe() const {
            return safe_cast<Shape<U, SIZE, AR>>(*this);
        }

    public:
        template<usize S = 1, usize AR = 0> requires (SIZE >= S)
        [[nodiscard]] NOA_HD constexpr auto pop_front() const noexcept {
            return Shape<value_type, SIZE - S, AR>::from_pointer(data() + S);
        }

        template<usize S = 1, usize AR = 0> requires (SIZE >= S)
        [[nodiscard]] NOA_HD constexpr auto pop_back() const noexcept {
            return Shape<value_type, SIZE - S, AR>::from_pointer(data());
        }

        template<usize S = 1, usize AR = 0>
        [[nodiscard]] NOA_HD constexpr auto push_front(value_type value) const noexcept {
            return Shape<value_type, SIZE + S, AR>{vec.template push_front<S, AR>(value)};
        }

        template<usize S = 1, usize AR = 0>
        [[nodiscard]] NOA_HD constexpr auto push_back(value_type value) const noexcept {
            return Shape<value_type, SIZE + S, AR>{vec.template push_back<S, AR>(value)};
        }

        template<usize AR = 0, usize S, usize AR0>
        [[nodiscard]] NOA_HD constexpr auto push_front(const Vec<value_type, S, AR0>& vector) const noexcept {
            constexpr usize NEW_SIZE = SIZE + S;
            return Shape<value_type, NEW_SIZE, AR>{vec.template push_front<AR>(vector)};
        }

        template<usize AR = 0, usize S, usize AR0>
        [[nodiscard]] NOA_HD constexpr auto push_back(const Vec<value_type, S, AR0>& vector) const noexcept {
            constexpr usize NEW_SIZE = SIZE + S;
            return Shape<value_type, NEW_SIZE, AR>{vec.template push_back<AR>(vector)};
        }

        template<nt::integer... U>
        [[nodiscard]] NOA_HD constexpr auto filter(U... ts) const noexcept {
            return Shape<value_type, sizeof...(U)>{(*this)[ts]...};
        }

        template<usize S> requires (S < N and N == 4)
        [[nodiscard]] constexpr auto filter_nd() const noexcept {
            if constexpr (S == 1)
                return filter(0, 3);
            else if constexpr (S == 2)
                return filter(0, 2, 3);
            else if constexpr (S == 3)
                return *this;
            else
                static_assert(nt::always_false<T>);
        }

        [[nodiscard]] NOA_HD constexpr Shape flip() const noexcept {
            return {vec.flip()};
        }

        template<nt::integer I = value_type, usize AR>
        [[nodiscard]] NOA_HD constexpr Shape reorder(const Vec<I, SIZE, AR>& order) const noexcept {
            return {vec.reorder(order)};
        }

        [[nodiscard]] NOA_HD constexpr Shape circular_shift(isize count) const noexcept {
            return {vec.circular_shift(count)};
        }

        [[nodiscard]] NOA_HD constexpr Shape copy() const noexcept {
            return *this;
        }

        template<usize INDEX> requires (INDEX < SIZE)
        [[nodiscard]] NOA_HD constexpr Shape set(value_type value) const noexcept {
            auto output = *this;
            output[INDEX] = value;
            return output;
        }

    public:
        [[nodiscard]] NOA_HD constexpr value_type n_elements() const noexcept {
            if constexpr (SIZE == 0) {
                return 0;
            } else {
                auto output = vec[0];
                for (usize i{1}; i < SIZE; ++i)
                    output *= vec[i];
                return output;
            }
        }

        /// Whether the shape has at least one dimension equal to 0.
        [[nodiscard]] NOA_HD constexpr bool is_empty() const noexcept {
            for (usize i{}; i < SIZE; ++i)
                if (vec[i] == 0)
                    return true;
            return false;
        }

        /// Returns the logical number of dimensions in the BDHW convention.
        /// This returns a value from 0 to 3. The batch dimension is ignored,
        /// and is_batched() should be used to know whether the array is batched.
        /// Note that both row and column vectors are considered to be 1d, but
        /// if the depth dimension is greater than 1, ndim() == 3 even if both the
        /// height and width are 1.
        [[nodiscard]] NOA_HD constexpr value_type ndim() const noexcept {
            NOA_ASSERT(not is_empty());
            if constexpr (SIZE <= 1) {
                return static_cast<value_type>(SIZE);
            } else if constexpr (SIZE == 2) {
                if (vec[0] > 1 and vec[1] > 1)
                    return 2;
                return 1;
            } else if constexpr (SIZE == 3) {
                if (vec[0] > 1)
                    return 3;
                if (vec[1] > 1 and vec[2] > 1)
                    return 2;
                return 1;
            } else {
                if (vec[1] > 1)
                    return 3;
                if (vec[2] > 1 and vec[3] > 1)
                    return 2;
                return 1;
            }
        }

        /// Computes the strides, in elements, in C- or F-order.
        /// Note that if the height and width dimensions are empty, 'C' and 'F' returns the same strides.
        template<char ORDER = 'C'> requires (SIZE > 0)
        [[nodiscard]] NOA_HD constexpr auto strides() const noexcept {
            using output_strides = Strides<value_type, SIZE, A>;

            if constexpr (ORDER == 'C' or ORDER == 'c') {
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
            } else if constexpr (ORDER == 'F' or ORDER == 'f') {
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
                static_assert(nt::always_false<value_type>);
            }
        }

        /// Returns the shape of the non-redundant FFT, in elements,
        [[nodiscard]] NOA_HD constexpr Shape rfft() const noexcept {
            Shape output = *this;
            if constexpr (SIZE > 0)
                output[SIZE - 1] = output[SIZE - 1] / 2 + 1;
            return output;
        }

        /// Whether the shape describes vector.
        /// A vector has one dimension with a size >= 1 and all the other dimensions empty (i.e. size == 1).
        /// By this definition, the shapes {1,1,1,1}, {5,1,1,1} and {1,1,1,5} are all vectors.
        /// If "can_be_batched" is true, the shape can describe a batch of vectors,
        /// e.g. {4,1,1,5} is describing 4 row vectors with a length of 5.
        [[nodiscard]] NOA_FHD constexpr bool is_vector(bool can_be_batched = false) const noexcept requires (SIZE == 4) {
            int non_empty_dimension = 0;
            for (usize i{}; i < SIZE; ++i) {
                if (vec[i] == 0)
                    return false; // empty/invalid shape
                if ((not can_be_batched or i != 0) and vec[i] > 1)
                    ++non_empty_dimension;
            }
            return non_empty_dimension <= 1;
        }

        /// Whether the shape describes vector.
        /// A vector has one dimension with a size >= 1 and all the other dimensions empty (i.e. size == 1).
        /// By this definition, the shapes {1,1,1}, {5,1,1} and {1,1,5} are all vectors.
        [[nodiscard]] NOA_FHD constexpr bool is_vector() const noexcept requires (SIZE > 0 and SIZE <= 3) {
            int non_empty_dimension = 0;
            for (usize i{}; i < SIZE; ++i) {
                if (vec[i] == 0)
                    return false; // empty/invalid shape
                if (vec[i] > 1)
                    ++non_empty_dimension;
            }
            return non_empty_dimension <= 1;
        }

        /// Whether this is a (batched) column vector.
        [[nodiscard]] NOA_HD constexpr bool is_column() const noexcept requires (SIZE >= 2) {
            return vec[SIZE - 2] >= 1 and vec[SIZE - 1] == 1;
        }

        /// Whether this is a (batched) row vector.
        [[nodiscard]] NOA_HD constexpr bool is_row() const noexcept requires (SIZE >= 2) {
            return vec[SIZE - 2] == 1 and vec[SIZE - 1] >= 1;
        }

        /// Whether this is a (batched) column vector.
        [[nodiscard]] NOA_HD constexpr bool is_batched() const noexcept requires (SIZE == 4) {
            return vec[0] > 1;
        }

        /// Move the first non-empty dimension (starting from the front) to the batch dimension.
        [[nodiscard]] NOA_HD constexpr Shape to_batched() const noexcept requires (SIZE == 4) {
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

        [[nodiscard]] NOA_HD constexpr auto split_batch() const noexcept
            -> Pair<value_type, Shape<value_type, 3>> requires (SIZE == 4) {
            return {batch(), pop_front()};
        }
    };

    /// Deduction guide.
    template<nt::integer T, usize N, usize A>
    Shape(Vec<T, N, A>) -> Shape<T, N, A>;

    template<nt::integer T, nt::same_as<T>... U>
    Shape(T, U...) -> Shape<T, 1 + sizeof...(U)>;

    template<typename T, usize N, usize A = 0>
    class Strides {
    public:
        static_assert(nt::integer<T> and N <= 4);
        using vector_type = Vec<T, N, A>;
        using value_type = vector_type::value_type;
        using mutable_value_type = value_type;
        static constexpr isize SSIZE = N;
        static constexpr usize SIZE = N;

    public:
        NOA_NO_UNIQUE_ADDRESS vector_type vec; // uninitialized

    public:
        template<nt::integer U>
        [[nodiscard]] NOA_HD static constexpr Strides from_value(U value) noexcept {
            return {vector_type::from_value(value)};
        }

        template<nt::integer U>
        [[nodiscard]] NOA_HD static constexpr Strides filled_with(U value) noexcept {
            return {vector_type::filled_with(value)}; // same as from_value
        }

        template<nt::integer... Args> requires (sizeof...(Args) == SIZE)
        [[nodiscard]] NOA_HD static constexpr Strides from_values(Args... values) noexcept {
            return {vector_type::from_values(values...)};
        }

        template<nt::integer U, usize AR>
        [[nodiscard]] NOA_HD static constexpr Strides from_vec(const Vec<U, SIZE, AR>& vector) noexcept {
            return {vector_type::from_vec(vector)};
        }

        template<nt::integer U, usize AR>
        [[nodiscard]] NOA_HD static constexpr Strides from_strides(const Strides<U, SIZE, AR>& strides) noexcept {
            return from_vec(strides.vec);
        }

        template<nt::integer U>
        [[nodiscard]] NOA_HD static constexpr Strides from_pointer(const U* values) noexcept {
            return {vector_type::from_pointer(values)};
        }

    public:
        // Allow explicit conversion constructor (while still being an aggregate)
        // and add support for static_cast<Strides<U>>(Strides<T>{}).
        template<typename U, usize AR>
        [[nodiscard]] NOA_HD constexpr explicit operator Strides<U, SIZE, AR>() const noexcept {
            return Strides<U, SIZE, AR>::from_strides(*this);
        }

        // Allow implicit conversion from a strides with a different alignment.
        template<usize AR> requires (A != AR)
        [[nodiscard]] NOA_HD constexpr /*implicit*/ operator Strides<value_type, SIZE, AR>() const noexcept {
            return Strides<value_type, SIZE, AR>::from_strides(*this);
        }

    public: // Accessor operators and functions
        [[nodiscard]] NOA_HD constexpr auto operator[](nt::integer auto i) noexcept -> value_type& requires (SIZE > 0) { return vec[i]; }
        [[nodiscard]] NOA_HD constexpr auto operator[](nt::integer auto i) const noexcept -> const value_type& requires (SIZE > 0) { return vec[i]; }

        [[nodiscard]] NOA_HD constexpr auto batch()  noexcept       -> value_type&       requires (SIZE == 4) { return vec[0]; }
        [[nodiscard]] NOA_HD constexpr auto batch()  const noexcept -> const value_type& requires (SIZE == 4) { return vec[0]; }
        [[nodiscard]] NOA_HD constexpr auto depth()  noexcept       -> value_type&       requires (SIZE >= 3) { return vec[SIZE - 3]; }
        [[nodiscard]] NOA_HD constexpr auto depth()  const noexcept -> const value_type& requires (SIZE >= 3) { return vec[SIZE - 3]; }
        [[nodiscard]] NOA_HD constexpr auto height() noexcept       -> value_type&       requires (SIZE >= 2) { return vec[SIZE - 2]; }
        [[nodiscard]] NOA_HD constexpr auto height() const noexcept -> const value_type& requires (SIZE >= 2) { return vec[SIZE - 2]; }
        [[nodiscard]] NOA_HD constexpr auto width()  noexcept       -> value_type&       requires (SIZE >= 1) { return vec[SIZE - 1]; }
        [[nodiscard]] NOA_HD constexpr auto width()  const noexcept -> const value_type& requires (SIZE >= 1) { return vec[SIZE - 1]; }

        // Structure binding support.
        template<int I> [[nodiscard]] NOA_HD constexpr const value_type& get() const noexcept { return vec[I]; }
        template<int I> [[nodiscard]] NOA_HD constexpr       value_type& get() noexcept { return vec[I]; }

        [[nodiscard]] NOA_HD constexpr auto data()   const noexcept -> const value_type* { return vec.data(); }
        [[nodiscard]] NOA_HD constexpr auto data()         noexcept -> value_type* { return vec.data(); }
        [[nodiscard]] NOA_HD static constexpr auto size()  noexcept -> usize { return SIZE; }
        [[nodiscard]] NOA_HD static constexpr auto ssize() noexcept -> isize { return SSIZE; }

    public: // Iterators -- support for range loops
        [[nodiscard]] NOA_HD constexpr       value_type* begin()        noexcept { return vec.begin(); }
        [[nodiscard]] NOA_HD constexpr const value_type* begin()  const noexcept { return vec.begin(); }
        [[nodiscard]] NOA_HD constexpr const value_type* cbegin() const noexcept { return vec.cbegin(); }
        [[nodiscard]] NOA_HD constexpr       value_type* end()          noexcept { return vec.end(); }
        [[nodiscard]] NOA_HD constexpr const value_type* end()    const noexcept { return vec.end(); }
        [[nodiscard]] NOA_HD constexpr const value_type* cend()   const noexcept { return vec.cend(); }

    public: // Assignment operators
        NOA_FHD constexpr Strides& operator=(value_type size) noexcept {
            *this = Strides::filled_with(size);
            return *this;
        }

        NOA_FHD constexpr auto operator+=(const Strides& shape) noexcept -> Strides& { *this = *this + shape; return *this; }
        NOA_FHD constexpr auto operator-=(const Strides& shape) noexcept -> Strides& { *this = *this - shape; return *this; }
        NOA_FHD constexpr auto operator*=(const Strides& shape) noexcept -> Strides& { *this = *this * shape; return *this; }
        NOA_FHD constexpr auto operator/=(const Strides& shape) noexcept -> Strides& { *this = *this / shape; return *this; }

        NOA_FHD constexpr auto operator+=(value_type value) noexcept -> Strides& { *this = *this + value; return *this; }
        NOA_FHD constexpr auto operator-=(value_type value) noexcept -> Strides& { *this = *this - value; return *this; }
        NOA_FHD constexpr auto operator*=(value_type value) noexcept -> Strides& { *this = *this * value; return *this; }
        NOA_FHD constexpr auto operator/=(value_type value) noexcept -> Strides& { *this = *this / value; return *this; }

    public: // Non-member functions
        [[nodiscard]] NOA_FHD friend constexpr Strides operator+(const Strides& strides) noexcept {
            return strides;
        }

        [[nodiscard]] NOA_FHD friend constexpr Strides operator-(Strides strides) noexcept {
            return {-strides.vec};
        }

        [[nodiscard]] NOA_FHD friend constexpr Strides operator+(Strides lhs, Strides rhs) noexcept { return {lhs.vec + rhs.vec}; }
        [[nodiscard]] NOA_FHD friend constexpr Strides operator-(Strides lhs, Strides rhs) noexcept { return {lhs.vec - rhs.vec}; }
        [[nodiscard]] NOA_FHD friend constexpr Strides operator*(Strides lhs, Strides rhs) noexcept { return {lhs.vec * rhs.vec}; }
        [[nodiscard]] NOA_FHD friend constexpr Strides operator/(Strides lhs, Strides rhs) noexcept { return {lhs.vec / rhs.vec}; }

        [[nodiscard]] NOA_FHD friend constexpr Strides operator+(const Strides& lhs, value_type rhs) noexcept { return lhs + Strides::filled_with(rhs); }
        [[nodiscard]] NOA_FHD friend constexpr Strides operator-(const Strides& lhs, value_type rhs) noexcept { return lhs - Strides::filled_with(rhs); }
        [[nodiscard]] NOA_FHD friend constexpr Strides operator*(const Strides& lhs, value_type rhs) noexcept { return lhs * Strides::filled_with(rhs); }
        [[nodiscard]] NOA_FHD friend constexpr Strides operator/(const Strides& lhs, value_type rhs) noexcept { return lhs / Strides::filled_with(rhs); }

        [[nodiscard]] NOA_FHD friend constexpr Strides operator+(value_type lhs, const Strides& rhs) noexcept { return Strides::filled_with(lhs) + rhs; }
        [[nodiscard]] NOA_FHD friend constexpr Strides operator-(value_type lhs, const Strides& rhs) noexcept { return Strides::filled_with(lhs) - rhs; }
        [[nodiscard]] NOA_FHD friend constexpr Strides operator*(value_type lhs, const Strides& rhs) noexcept { return Strides::filled_with(lhs) * rhs; }
        [[nodiscard]] NOA_FHD friend constexpr Strides operator/(value_type lhs, const Strides& rhs) noexcept { return Strides::filled_with(lhs) / rhs; }

    public: // comparison operators
        [[nodiscard]] NOA_FHD friend constexpr bool operator==(const Strides& lhs, const Strides& rhs) noexcept { return nd::vec_op_bool<Equal>(lhs.vec, rhs.vec); }
        [[nodiscard]] NOA_FHD friend constexpr bool operator!=(const Strides& lhs, const Strides& rhs) noexcept { return nd::vec_op_bool<NotEqual>(lhs.vec, rhs.vec); }
        [[nodiscard]] NOA_FHD friend constexpr bool operator<=(const Strides& lhs, const Strides& rhs) noexcept { return nd::vec_op_bool<LessEqual>(lhs.vec, rhs.vec); }
        [[nodiscard]] NOA_FHD friend constexpr bool operator>=(const Strides& lhs, const Strides& rhs) noexcept { return nd::vec_op_bool<GreaterEqual>(lhs.vec, rhs.vec); }
        [[nodiscard]] NOA_FHD friend constexpr bool operator<(const Strides& lhs, const Strides& rhs) noexcept { return nd::vec_op_bool<Less>(lhs.vec, rhs.vec); }
        [[nodiscard]] NOA_FHD friend constexpr bool operator>(const Strides& lhs, const Strides& rhs) noexcept { return nd::vec_op_bool<Greater>(lhs.vec, rhs.vec); }

        [[nodiscard]] NOA_FHD friend constexpr bool operator==(const Strides& lhs, const value_type& rhs) noexcept { return lhs == Strides::filled_with(rhs); }
        [[nodiscard]] NOA_FHD friend constexpr bool operator!=(const Strides& lhs, const value_type& rhs) noexcept { return lhs != Strides::filled_with(rhs); }
        [[nodiscard]] NOA_FHD friend constexpr bool operator<=(const Strides& lhs, const value_type& rhs) noexcept { return lhs <= Strides::filled_with(rhs); }
        [[nodiscard]] NOA_FHD friend constexpr bool operator>=(const Strides& lhs, const value_type& rhs) noexcept { return lhs >= Strides::filled_with(rhs); }
        [[nodiscard]] NOA_FHD friend constexpr bool operator<(const Strides& lhs, const value_type& rhs) noexcept { return lhs < Strides::filled_with(rhs); }
        [[nodiscard]] NOA_FHD friend constexpr bool operator>(const Strides& lhs, const value_type& rhs) noexcept { return lhs > Strides::filled_with(rhs); }

        [[nodiscard]] NOA_FHD friend constexpr bool operator==(const value_type& lhs, const Strides& rhs) noexcept { return Strides::filled_with(lhs) == rhs; }
        [[nodiscard]] NOA_FHD friend constexpr bool operator!=(const value_type& lhs, const Strides& rhs) noexcept { return Strides::filled_with(lhs) != rhs; }
        [[nodiscard]] NOA_FHD friend constexpr bool operator<=(const value_type& lhs, const Strides& rhs) noexcept { return Strides::filled_with(lhs) <= rhs; }
        [[nodiscard]] NOA_FHD friend constexpr bool operator>=(const value_type& lhs, const Strides& rhs) noexcept { return Strides::filled_with(lhs) >= rhs; }
        [[nodiscard]] NOA_FHD friend constexpr bool operator<(const value_type& lhs, const Strides& rhs) noexcept { return Strides::filled_with(lhs) < rhs; }
        [[nodiscard]] NOA_FHD friend constexpr bool operator>(const value_type& lhs, const Strides& rhs) noexcept { return Strides::filled_with(lhs) > rhs; }

    public: // element-wise comparison
        [[nodiscard]] NOA_FHD constexpr auto cmp_eq(const Strides& rhs) const noexcept { return nd::vec_cmp<Equal>(vec, rhs.vec); }
        [[nodiscard]] NOA_FHD constexpr auto cmp_ne(const Strides& rhs) const noexcept { return nd::vec_cmp<NotEqual>(vec, rhs.vec); }
        [[nodiscard]] NOA_FHD constexpr auto cmp_le(const Strides& rhs) const noexcept { return nd::vec_cmp<LessEqual>(vec, rhs.vec); }
        [[nodiscard]] NOA_FHD constexpr auto cmp_ge(const Strides& rhs) const noexcept { return nd::vec_cmp<GreaterEqual>(vec, rhs.vec); }
        [[nodiscard]] NOA_FHD constexpr auto cmp_lt(const Strides& rhs) const noexcept { return nd::vec_cmp<Less>(vec, rhs.vec); }
        [[nodiscard]] NOA_FHD constexpr auto cmp_gt(const Strides& rhs) const noexcept { return nd::vec_cmp<Greater>(vec, rhs.vec); }

        [[nodiscard]] NOA_FHD constexpr auto cmp_eq(const value_type& rhs) const noexcept { return nd::vec_cmp<Equal>(vec, vector_type::filled_with(rhs)); }
        [[nodiscard]] NOA_FHD constexpr auto cmp_ne(const value_type& rhs) const noexcept { return nd::vec_cmp<NotEqual>(vec, vector_type::filled_with(rhs)); }
        [[nodiscard]] NOA_FHD constexpr auto cmp_le(const value_type& rhs) const noexcept { return nd::vec_cmp<LessEqual>(vec, vector_type::filled_with(rhs)); }
        [[nodiscard]] NOA_FHD constexpr auto cmp_ge(const value_type& rhs) const noexcept { return nd::vec_cmp<GreaterEqual>(vec, vector_type::filled_with(rhs)); }
        [[nodiscard]] NOA_FHD constexpr auto cmp_lt(const value_type& rhs) const noexcept { return nd::vec_cmp<Less>(vec, vector_type::filled_with(rhs)); }
        [[nodiscard]] NOA_FHD constexpr auto cmp_gt(const value_type& rhs) const noexcept { return nd::vec_cmp<Greater>(vec, vector_type::filled_with(rhs)); }

        [[nodiscard]] NOA_FHD constexpr bool any_eq(const Strides& rhs) const noexcept { return vec.any_eq(rhs.vec); }
        [[nodiscard]] NOA_FHD constexpr bool any_ne(const Strides& rhs) const noexcept { return vec.any_ne(rhs.vec); }
        [[nodiscard]] NOA_FHD constexpr bool any_le(const Strides& rhs) const noexcept { return vec.any_le(rhs.vec); }
        [[nodiscard]] NOA_FHD constexpr bool any_ge(const Strides& rhs) const noexcept { return vec.any_ge(rhs.vec); }
        [[nodiscard]] NOA_FHD constexpr bool any_lt(const Strides& rhs) const noexcept { return vec.any_lt(rhs.vec); }
        [[nodiscard]] NOA_FHD constexpr bool any_gt(const Strides& rhs) const noexcept { return vec.any_gt(rhs.vec); }

        [[nodiscard]] NOA_FHD constexpr bool any_eq(const value_type& rhs) const noexcept { return vec.any_eq(rhs); }
        [[nodiscard]] NOA_FHD constexpr bool any_ne(const value_type& rhs) const noexcept { return vec.any_ne(rhs); }
        [[nodiscard]] NOA_FHD constexpr bool any_le(const value_type& rhs) const noexcept { return vec.any_le(rhs); }
        [[nodiscard]] NOA_FHD constexpr bool any_ge(const value_type& rhs) const noexcept { return vec.any_ge(rhs); }
        [[nodiscard]] NOA_FHD constexpr bool any_lt(const value_type& rhs) const noexcept { return vec.any_lt(rhs); }
        [[nodiscard]] NOA_FHD constexpr bool any_gt(const value_type& rhs) const noexcept { return vec.any_gt(rhs); }

    public: // Type casts
        template<nt::integer U, usize AR = 0>
        [[nodiscard]] NOA_HD constexpr auto as() const noexcept {
            return static_cast<Strides<U, SIZE, AR>>(*this);
        }

        template<nt::integer U, usize AR = 0>
        [[nodiscard]] NOA_HD constexpr auto as_clamp() const noexcept {
            return clamp_cast<Strides<U, SIZE, AR>>(*this);
        }

        template<nt::integer U, usize AR = 0>
        [[nodiscard]] constexpr auto as_safe() const {
            return safe_cast<Strides<U, SIZE, AR>>(*this);
        }

    public:
        template<usize S = 1, usize AR = 0> requires (SIZE >= S)
        [[nodiscard]] NOA_HD constexpr auto pop_front() const noexcept {
            return Strides<value_type, SIZE - S, AR>::from_pointer(data() + S);
        }

        template<usize S = 1, usize AR = 0> requires (SIZE >= S)
        [[nodiscard]] NOA_HD constexpr auto pop_back() const noexcept {
            return Strides<value_type, SIZE - S, AR>::from_pointer(data());
        }

        template<usize S = 1, usize AR = 0>
        [[nodiscard]] NOA_HD constexpr auto push_front(value_type value) const noexcept {
            return Strides<value_type, SIZE + S, AR>{vec.template push_front<S, AR>(value)};
        }

        template<usize S = 1, usize AR = 0>
        [[nodiscard]] NOA_HD constexpr auto push_back(value_type value) const noexcept {
            return Strides<value_type, SIZE + S, AR>{vec.template push_back<S, AR>(value)};
        }

        template<usize AR = 0, usize S, usize AR0>
        [[nodiscard]] NOA_HD constexpr auto push_front(const Vec<value_type, S, AR0>& vector) const noexcept {
            constexpr usize NEW_SIZE = SIZE + S;
            return Strides<value_type, NEW_SIZE, AR>{vec.template push_front<AR>(vector)};
        }

        template<usize AR = 0, usize S, usize AR0>
        [[nodiscard]] NOA_HD constexpr auto push_back(const Vec<value_type, S, AR0>& vector) const noexcept {
            constexpr usize NEW_SIZE = SIZE + S;
            return Strides<value_type, NEW_SIZE, AR>{vec.template push_back<AR>(vector)};
        }

        template<nt::integer... U>
        [[nodiscard]] NOA_HD constexpr auto filter(U... ts) const noexcept {
            return Strides<value_type, sizeof...(U)>{(*this)[ts]...};
        }

        template<usize S> requires (S < N and N == 4)
        [[nodiscard]] constexpr auto filter_nd() const noexcept {
            if constexpr (S == 1)
                return filter(0, 3);
            else if constexpr (S == 2)
                return filter(0, 2, 3);
            else if constexpr (S == 3)
                return *this;
            else
                static_assert(nt::always_false<T>);
        }

        [[nodiscard]] NOA_HD constexpr Strides flip() const noexcept {
            return {vec.flip()};
        }

        template<nt::integer I = value_type, usize AR>
        [[nodiscard]] NOA_HD constexpr Strides reorder(const Vec<I, SIZE, AR>& order) const noexcept {
            return {vec.reorder(order)};
        }

        [[nodiscard]] NOA_HD constexpr Strides circular_shift(isize count) const noexcept {
            return {vec.circular_shift(count)};
        }

        [[nodiscard]] NOA_HD constexpr Strides copy() const noexcept {
            return *this;
        }

        template<usize INDEX> requires (INDEX < SIZE)
        [[nodiscard]] NOA_HD constexpr Strides set(value_type value) const noexcept {
            auto output = *this;
            output[INDEX] = value;
            return output;
        }

    public:
        /// Whether there's at least one dimension equal to 0.
        [[nodiscard]] NOA_HD constexpr bool is_broadcast() const noexcept {
            for (usize i{}; i < SIZE; ++i)
                if (vec[i] == 0)
                    return true;
            return false;
        }

        /// Whether the strides are in the rightmost order.
        /// Rightmost order is when the innermost stride (i.e. the dimension with the smallest stride)
        /// is on the right, and strides increase right-to-left.
        [[nodiscard]] NOA_HD constexpr bool is_rightmost() const noexcept requires (SIZE > 0) {
            for (usize i{}; i < SIZE - 1; ++i)
                if (vec[i] < vec[i + 1])
                    return false;
            return true;
        }

        /// Computes the physical layout (the actual memory footprint) encoded in these strides.
        /// Note that the left-most size is not-encoded in the strides, and therefore cannot be recovered.
        template<char ORDER = 'C'> requires (SIZE >= 2)
        [[nodiscard]] NOA_HD constexpr auto physical_shape() const noexcept {
            NOA_ASSERT(not is_broadcast() and "Cannot recover the physical shape from broadcast strides");
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
                static_assert(nt::always_false<value_type>);
            }
        }

        [[nodiscard]] NOA_HD constexpr auto split_batch() const noexcept
        -> Pair<value_type, Strides<value_type, 3>> requires (SIZE == 4) {
            return {batch(), pop_front()};
        }
    };

    /// Deduction guide.
    template<nt::integer T, usize N, usize A>
    Strides(Vec<T, N, A>) -> Strides<T, N, A>;

    template<nt::integer T, nt::same_as<T>... U>
    Strides(T, U...) -> Strides<T, 1 + sizeof...(U)>;
}

// Support for structure bindings:
namespace std {
    template<typename T, noa::usize N, noa::usize A>
    struct tuple_size<noa::Shape<T, N, A>> : std::integral_constant<noa::usize, N> {};

    template<noa::usize I, noa::usize N, noa::usize A, typename T>
    struct tuple_element<I, noa::Shape<T, N, A>> { using type = T; };

    template<typename T, noa::usize N, noa::usize A>
    struct tuple_size<noa::Strides<T, N, A>> : std::integral_constant<noa::usize, N> {};

    template<noa::usize I, noa::usize N, noa::usize A, typename T>
    struct tuple_element<I, noa::Strides<T, N, A>> { using type = T; };

    template<typename T, noa::usize N, noa::usize A>
    struct tuple_size<const noa::Shape<T, N, A>> : std::integral_constant<noa::usize, N> {};

    template<noa::usize I, noa::usize N, noa::usize A, typename T>
    struct tuple_element<I, const noa::Shape<T, N, A>> { using type = const T; };

    template<typename T, noa::usize N, noa::usize A>
    struct tuple_size<const noa::Strides<T, N, A>> : std::integral_constant<noa::usize, N> {};

    template<noa::usize I, noa::usize N, noa::usize A, typename T>
    struct tuple_element<I, const noa::Strides<T, N, A>> { using type = const T; };
}

// Type aliases:
namespace noa::inline types {
    using Shape1 = Shape<isize, 1>;
    using Shape2 = Shape<isize, 2>;
    using Shape3 = Shape<isize, 3>;
    using Shape4 = Shape<isize, 4>;

    using Strides1 = Strides<isize, 1>;
    using Strides2 = Strides<isize, 2>;
    using Strides3 = Strides<isize, 3>;
    using Strides4 = Strides<isize, 4>;
}

namespace noa::traits {
    template<typename T, usize N, usize A> struct proclaim_is_shape<noa::Shape<T, N, A>> : std::true_type {};
    template<typename V1, usize N, usize A, typename V2> struct proclaim_is_shape_of_type<noa::Shape<V1, N, A>, V2> : std::bool_constant<std::is_same_v<V1, V2>> {};
    template<typename V, usize N1, usize A, usize N2> struct proclaim_is_shape_of_size<noa::Shape<V, N1, A>, N2> : std::bool_constant<N1 == N2> {};

    template<typename T, usize N, usize A> struct proclaim_is_strides<noa::Strides<T, N, A>> : std::true_type {};
    template<typename V1, usize N, usize A, typename V2> struct proclaim_is_strides_of_type<noa::Strides<V1, N, A>, V2> : std::bool_constant<std::is_same_v<V1, V2>> {};
    template<typename V, usize N1, usize A, usize N2> struct proclaim_is_strides_of_size<noa::Strides<V, N1, A>, N2> : std::bool_constant<N1 == N2> {};
}

namespace noa::inline types {
    // -- Modulo Operator --
    template<nt::shape_or_strides T> requires (T::SIZE > 0)
    [[nodiscard]] NOA_HD constexpr T operator%(T lhs, const T& rhs) noexcept {
        for (usize i{}; i < T::SIZE; ++i)
            lhs[i] %= rhs[i];
        return lhs;
    }

    template<nt::integer T, usize N, usize A> requires (N > 0)
    [[nodiscard]] NOA_HD constexpr auto operator%(const Shape<T, N, A>& lhs, std::type_identity_t<T> rhs) noexcept {
        return lhs % Shape<T, N, A>::filled_with(rhs);
    }

    template<nt::integer T, usize N, usize A> requires (N > 0)
    [[nodiscard]] NOA_HD constexpr auto operator%(std::type_identity_t<T> lhs, const Shape<T, N, A>& rhs) noexcept {
        return Shape<T, N, A>::filled_with(lhs) % rhs;
    }

    template<nt::integer T, usize N, usize A> requires (N > 0)
    [[nodiscard]] NOA_HD constexpr auto operator%(const Strides<T, N, A>& lhs, std::type_identity_t<T> rhs) noexcept {
        return lhs % Strides<T, N, A>::filled_with(rhs);
    }

    template<nt::integer T, usize N, usize A> requires (N > 0)
    [[nodiscard]] NOA_HD constexpr auto operator%(std::type_identity_t<T> lhs, const Strides<T, N, A>& rhs) noexcept {
        return Strides<T, N, A>::filled_with(lhs) % rhs;
    }
}

namespace noa {
    // Cast Shape->Shape
    template<typename To, typename T, usize N, usize A> requires nt::shape_of_size<To, N>
    [[nodiscard]] NOA_FHD constexpr bool is_safe_cast(const Shape<T, N, A>& src) noexcept {
        return is_safe_cast<typename To::vector_type>(src.vec);
    }

    template<typename To, typename T, usize N, usize A> requires nt::shape_of_size<To, N>
    [[nodiscard]] NOA_FHD constexpr To clamp_cast(const Shape<T, N, A>& src) noexcept {
        return To{clamp_cast<typename To::vector_type>(src.vec)};
    }

    // Cast Strides->Strides
    template<typename To, typename T, usize N, usize A> requires nt::strides_of_size<To, N>
    [[nodiscard]] NOA_FHD constexpr bool is_safe_cast(const Strides<T, N, A>& src) noexcept {
        return is_safe_cast<typename To::vector_type>(src.vec);
    }

    template<typename To, typename T, usize N, usize A> requires nt::strides_of_size<To, N>
    [[nodiscard]] NOA_FHD constexpr To clamp_cast(const Strides<T, N, A>& src) noexcept {
        return To{clamp_cast<typename To::vector_type>(src.vec)};
    }

    template<nt::shape_or_strides T>
    [[nodiscard]] NOA_FHD constexpr T abs(T shape) noexcept {
        return {abs(shape.vec)};
    }

    template<nt::shape_or_strides T> requires (T::SIZE > 0)
    [[nodiscard]] NOA_FHD constexpr auto sum(const T& shape) noexcept {
        return sum(shape.vec);
    }

    template<nt::shape_or_strides T> requires (T::SIZE > 0)
    [[nodiscard]] NOA_FHD constexpr auto product(const T& shape) noexcept {
        return product(shape.vec);
    }

    template<nt::shape_or_strides T> requires (T::SIZE > 0)
    [[nodiscard]] NOA_FHD constexpr auto min(const T& shape) noexcept {
        return min(shape.vec);
    }

    template<nt::shape_or_strides T>
    [[nodiscard]] NOA_FHD constexpr T min(const T& lhs, const T& rhs) noexcept {
        return {min(lhs.vec, rhs.vec)};
    }

    template<nt::shape_or_strides T, nt::same_as_value_type_of<T> I>
    [[nodiscard]] NOA_FHD constexpr auto min(const T& lhs, I rhs) noexcept {
        return min(lhs, T::filled_with(rhs));
    }

    template<nt::shape_or_strides T, nt::same_as_value_type_of<T> I>
    [[nodiscard]] NOA_FHD constexpr auto min(I lhs, const T& rhs) noexcept {
        return min(T::filled_with(lhs), rhs);
    }

    template<nt::shape_or_strides T>
    [[nodiscard]] NOA_FHD constexpr auto max(const T& shape) noexcept {
        return max(shape.vec);
    }

    template<nt::shape_or_strides T> requires (T::SIZE > 0)
    [[nodiscard]] NOA_FHD constexpr T max(const T& lhs, const T& rhs) noexcept {
        return {max(lhs.vec, rhs.vec)};
    }

    template<nt::shape_or_strides T, nt::same_as_value_type_of<T> U>
    [[nodiscard]] NOA_FHD constexpr auto max(const T& lhs, U rhs) noexcept {
        return max(lhs, T::filled_with(rhs));
    }

    template<nt::shape_or_strides T, nt::same_as_value_type_of<T> U>
    [[nodiscard]] NOA_FHD constexpr auto max(U lhs, const T& rhs) noexcept {
        return max(T::filled_with(lhs), rhs);
    }

    template<nt::shape_or_strides T>
    [[nodiscard]] NOA_FHD constexpr auto clamp(const T& lhs, const T& low, const T& high) noexcept {
        return min(max(lhs, low), high);
    }

    template<nt::shape_or_strides T, nt::same_as_value_type_of<T> U>
    [[nodiscard]] NOA_FHD constexpr auto clamp(const T& lhs, U low, U high) noexcept {
        return min(max(lhs, low), high);
    }

    template<nt::shape_or_strides T, typename Op = Less>
    [[nodiscard]] NOA_IHD constexpr auto stable_sort(T shape, Op&& comp = {}) noexcept {
        small_stable_sort<T::SIZE>(shape.data(), std::forward<Op>(comp));
        return shape;
    }

    template<nt::shape_or_strides T, typename Op = Less>
    [[nodiscard]] NOA_IHD constexpr auto sort(T shape, Op&& comp = {}) noexcept {
        small_stable_sort<T::SIZE>(shape.data(), std::forward<Op>(comp));
        return shape;
    }
}

namespace noa::inline types {
    template<typename T, usize N>
    std::ostream& operator<<(std::ostream& os, const Shape<T, N>& v) {
        os << fmt::format("{}", v.vec);
        return os;
    }

    template<typename T, usize N>
    std::ostream& operator<<(std::ostream& os, const Strides<T, N>& v) {
        os << fmt::format("{}", v.vec);
        return os;
    }
}

namespace noa::details {
    template<typename T, usize N, usize A>
    struct Stringify<Shape<T, N, A>> {
        static auto get() -> std::string {
            if constexpr (A == 0)
                return fmt::format("Shape<{},{}>", stringify<T>(), N);
            else
                return fmt::format("Shape<{},{},{}>", stringify<T>(), N, A);
        }
    };
    template<typename T, usize N, usize A>
    struct Stringify<Strides<T, N, A>> {
        static auto get() -> std::string {
            if constexpr (A == 0)
                return fmt::format("Strides<{},{}>", stringify<T>(), N);
            else
                return fmt::format("Strides<{},{},{}>", stringify<T>(), N, A);
        }
    };
}
