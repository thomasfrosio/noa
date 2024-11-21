#pragma once

#include "noa/core/Traits.hpp"
#include "noa/core/types/Vec.hpp"
#include "noa/core/types/Tuple.hpp"

namespace noa::inline types {
    template<typename Int, size_t N, size_t A>
    class Strides;
}

namespace noa::inline types {
    template<typename T, size_t N, size_t A = 0>
    class Shape {
    public:
        static_assert(nt::integer<T> and N <= 4);
        using vector_type = Vec<T, N, A>;
        using value_type = vector_type::value_type;
        using mutable_value_type = value_type;
        static constexpr i64 SSIZE = N;
        static constexpr size_t SIZE = N;

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

        template<nt::integer U, size_t AR>
        [[nodiscard]] NOA_HD static constexpr Shape from_vec(const Vec<U, SIZE, AR>& vector) noexcept {
            return {vector_type::from_vec(vector)};
        }

        template<typename U, size_t AR>
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
        template<typename U, size_t AR>
        [[nodiscard]] NOA_HD constexpr explicit operator Shape<U, SIZE, AR>() const noexcept {
            return Shape<U, SIZE, AR>::from_shape(*this);
        }

        // Allow implicit conversion from a shape with a different alignment.
        template<size_t AR> requires (A != AR)
        [[nodiscard]] NOA_HD constexpr /*implicit*/ operator Shape<value_type, SIZE, AR>() const noexcept {
            return Shape<value_type, SIZE, AR>::from_vec(*this);
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
        [[nodiscard]] NOA_HD static constexpr auto size()  noexcept -> size_t { return SIZE; };
        [[nodiscard]] NOA_HD static constexpr auto ssize() noexcept -> i64 { return SSIZE; };

    public: // Iterators -- support for range loops
        [[nodiscard]] NOA_HD constexpr       value_type* begin()        noexcept { return vec.begin(); }
        [[nodiscard]] NOA_HD constexpr const value_type* begin()  const noexcept { return vec.begin(); }
        [[nodiscard]] NOA_HD constexpr const value_type* cbegin() const noexcept { return vec.cbegin(); }
        [[nodiscard]] NOA_HD constexpr       value_type* end()          noexcept { return vec.end(); }
        [[nodiscard]] NOA_HD constexpr const value_type* end()    const noexcept { return vec.end(); }
        [[nodiscard]] NOA_HD constexpr const value_type* cend()   const noexcept { return vec.cend(); }

    public: // Assignment operators
        NOA_HD constexpr Shape& operator=(value_type size) noexcept {
            *this = Shape::filled_with(size);
            return *this;
        }

        #define NOA_SHAPE_ASSIGN_(name, op)                                 \
        NOA_HD constexpr name& operator op##=(const name& shape) noexcept { \
            *this = *this op shape;                                         \
            return *this;                                                   \
        }                                                                   \
        NOA_HD constexpr name& operator op##=(value_type value) noexcept {  \
            *this = *this op value;                                         \
            return *this;                                                   \
        }
        NOA_SHAPE_ASSIGN_(Shape, +)
        NOA_SHAPE_ASSIGN_(Shape, -)
        NOA_SHAPE_ASSIGN_(Shape, *)
        NOA_SHAPE_ASSIGN_(Shape, /)

    public: // Non-member functions
        [[nodiscard]] friend NOA_HD constexpr Shape operator+(const Shape& shape) noexcept {
            return shape;
        }

        [[nodiscard]] friend NOA_HD constexpr Shape operator-(Shape shape) noexcept {
            return {-shape.vec};
        }

        #define NOA_SHAPE_ARITH_(name, op)                                                                  \
        [[nodiscard]] friend NOA_HD constexpr name operator op(name lhs, name rhs) noexcept {               \
            return {lhs.vec op rhs.vec};                                                                    \
        }                                                                                                   \
        [[nodiscard]] friend NOA_HD constexpr name operator op(const name& lhs, value_type rhs) noexcept {  \
            return lhs op name::filled_with(rhs);                                                           \
        }                                                                                                   \
        [[nodiscard]] friend NOA_HD constexpr name operator op(value_type lhs, const name& rhs) noexcept {  \
            return name::filled_with(lhs) op rhs;                                                           \
        }
        NOA_SHAPE_ARITH_(Shape, +)
        NOA_SHAPE_ARITH_(Shape, -)
        NOA_SHAPE_ARITH_(Shape, *)
        NOA_SHAPE_ARITH_(Shape, /)

        #define NOA_SHAPE_COMP_(name, op)                                                                   \
        [[nodiscard]] friend NOA_HD constexpr auto operator op(name lhs, name rhs) noexcept {               \
            return lhs.vec op rhs.vec;                                                                      \
        }                                                                                                   \
        [[nodiscard]] friend NOA_HD constexpr auto operator op(const name& lhs, value_type rhs) noexcept {  \
            return lhs op name::filled_with(rhs);                                                           \
        }                                                                                                   \
        [[nodiscard]] friend NOA_HD constexpr auto operator op(value_type lhs, const name& rhs) noexcept {  \
            return name::filled_with(lhs) op rhs;                                                           \
        }
        NOA_SHAPE_COMP_(Shape, >)
        NOA_SHAPE_COMP_(Shape, <)
        NOA_SHAPE_COMP_(Shape, >=)
        NOA_SHAPE_COMP_(Shape, <=)
        NOA_SHAPE_COMP_(Shape, ==)
        NOA_SHAPE_COMP_(Shape, !=)

    public: // Type casts
        template<nt::integer U, size_t AR = 0>
        [[nodiscard]] NOA_HD constexpr auto as() const noexcept {
            return static_cast<Shape<U, SIZE, AR>>(*this);
        }

        template<nt::integer U, size_t AR = 0>
        [[nodiscard]] NOA_HD constexpr auto as_clamp() const noexcept {
            return clamp_cast<Shape<U, SIZE, AR>>(*this);
        }

        template<nt::integer U, size_t AR = 0>
        [[nodiscard]] constexpr auto as_safe() const {
            return safe_cast<Shape<U, SIZE, AR>>(*this);
        }

    public:
        template<size_t S = 1, size_t AR = 0> requires (SIZE >= S)
        [[nodiscard]] NOA_HD constexpr auto pop_front() const noexcept {
            return Shape<value_type, SIZE - S, AR>::from_pointer(data() + S);
        }

        template<size_t S = 1, size_t AR = 0> requires (SIZE >= S)
        [[nodiscard]] NOA_HD constexpr auto pop_back() const noexcept {
            return Shape<value_type, SIZE - S, AR>::from_pointer(data());
        }

        template<size_t S = 1, size_t AR = 0>
        [[nodiscard]] NOA_HD constexpr auto push_front(value_type value) const noexcept {
            return Shape<value_type, SIZE + S, AR>{vec.template push_front<S, AR>(value)};
        }

        template<size_t S = 1, size_t AR = 0>
        [[nodiscard]] NOA_HD constexpr auto push_back(value_type value) const noexcept {
            return Shape<value_type, SIZE + S, AR>{vec.template push_back<S, AR>(value)};
        }

        template<size_t AR = 0, size_t S, size_t AR0>
        [[nodiscard]] NOA_HD constexpr auto push_front(const Vec<value_type, S, AR0>& vector) const noexcept {
            constexpr size_t NEW_SIZE = SIZE + S;
            return Shape<value_type, NEW_SIZE, AR>{vec.template push_front<AR>(vector)};
        }

        template<size_t AR = 0, size_t S, size_t AR0>
        [[nodiscard]] NOA_HD constexpr auto push_back(const Vec<value_type, S, AR0>& vector) const noexcept {
            constexpr size_t NEW_SIZE = SIZE + S;
            return Shape<value_type, NEW_SIZE, AR>{vec.template push_back<AR>(vector)};
        }

        template<nt::integer... U>
        [[nodiscard]] NOA_HD constexpr auto filter(U... ts) const noexcept {
            return Shape<value_type, sizeof...(U)>{(*this)[ts]...};
        }

        template<size_t S> requires (S < N and N == 4)
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

        template<nt::integer I = value_type, size_t AR>
        [[nodiscard]] NOA_HD constexpr Shape reorder(const Vec<I, SIZE, AR>& order) const noexcept {
            return {vec.reorder(order)};
        }

        [[nodiscard]] NOA_HD constexpr Shape circular_shift(i64 count) const noexcept {
            return {vec.circular_shift(count)};
        }

        [[nodiscard]] NOA_HD constexpr Shape copy() const noexcept {
            return *this;
        }

        template<size_t INDEX> requires (INDEX < SIZE)
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
                for (size_t i{1}; i < SIZE; ++i)
                    output *= vec[i];
                return output;
            }
        }

        /// Whether the shape has at least one dimension equal to 0.
        [[nodiscard]] NOA_HD constexpr bool is_empty() const noexcept {
            for (size_t i{}; i < SIZE; ++i)
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
                return vec[0] > 1 and vec[1] > 1 ? 2 : 1;
            } else if constexpr (SIZE == 3) {
                return vec[0] > 1 ? 3 :
                       vec[1] > 1 and vec[2] > 1 ? 2 : 1;
            } else {
                return vec[1] > 1 ? 3 :
                       vec[2] > 1 and vec[3] > 1 ? 2 : 1;
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
            for (size_t i{}; i < SIZE; ++i) {
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
            for (size_t i{}; i < SIZE; ++i) {
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
    template<nt::integer T, size_t N, size_t A>
    Shape(Vec<T, N, A>) -> Shape<T, N, A>;

    template<nt::integer T, nt::same_as<T>... U>
    Shape(T, U...) -> Shape<T, 1 + sizeof...(U)>;

    template<typename T, size_t N, size_t A = 0>
    class Strides {
    public:
        static_assert(nt::integer<T> and N <= 4);
        using vector_type = Vec<T, N, A>;
        using value_type = vector_type::value_type;
        using mutable_value_type = value_type;
        static constexpr i64 SSIZE = N;
        static constexpr size_t SIZE = N;

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

        template<nt::integer U, size_t AR>
        [[nodiscard]] NOA_HD static constexpr Strides from_vec(const Vec<U, SIZE, AR>& vector) noexcept {
            return {vector_type::from_vec(vector)};
        }

        template<nt::integer U, size_t AR>
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
        template<typename U, size_t AR>
        [[nodiscard]] NOA_HD constexpr explicit operator Strides<U, SIZE, AR>() const noexcept {
            return Strides<U, SIZE, AR>::from_strides(*this);
        }

        // Allow implicit conversion from a strides with a different alignment.
        template<size_t AR> requires (A != AR)
        [[nodiscard]] NOA_HD constexpr /*implicit*/ operator Strides<value_type, SIZE, AR>() const noexcept {
            return Strides<value_type, SIZE, AR>::from_vec(*this);
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
        [[nodiscard]] NOA_HD static constexpr auto size()  noexcept -> size_t { return SIZE; }
        [[nodiscard]] NOA_HD static constexpr auto ssize() noexcept -> i64 { return SSIZE; }

    public: // Iterators -- support for range loops
        [[nodiscard]] NOA_HD constexpr       value_type* begin()        noexcept { return vec.begin(); }
        [[nodiscard]] NOA_HD constexpr const value_type* begin()  const noexcept { return vec.begin(); }
        [[nodiscard]] NOA_HD constexpr const value_type* cbegin() const noexcept { return vec.cbegin(); }
        [[nodiscard]] NOA_HD constexpr       value_type* end()          noexcept { return vec.end(); }
        [[nodiscard]] NOA_HD constexpr const value_type* end()    const noexcept { return vec.end(); }
        [[nodiscard]] NOA_HD constexpr const value_type* cend()   const noexcept { return vec.cend(); }

    public: // Assignment operators
        NOA_HD constexpr Strides& operator=(value_type size) noexcept {
            *this = Strides::filled_with(size);
            return *this;
        }

        NOA_SHAPE_ASSIGN_(Strides, +)
        NOA_SHAPE_ASSIGN_(Strides, -)
        NOA_SHAPE_ASSIGN_(Strides, *)
        NOA_SHAPE_ASSIGN_(Strides, /)
        #undef NOA_SHAPE_ASSIGN_

    public: // Non-member functions
        [[nodiscard]] friend NOA_HD constexpr Strides operator+(const Strides& strides) noexcept {
            return strides;
        }

        [[nodiscard]] friend NOA_HD constexpr Strides operator-(Strides strides) noexcept {
            return {-strides.vec};
        }

        NOA_SHAPE_ARITH_(Strides, +)
        NOA_SHAPE_ARITH_(Strides, -)
        NOA_SHAPE_ARITH_(Strides, *)
        NOA_SHAPE_ARITH_(Strides, /)
        #undef NOA_SHAPE_ARITH_

        NOA_SHAPE_COMP_(Strides, >)
        NOA_SHAPE_COMP_(Strides, <)
        NOA_SHAPE_COMP_(Strides, >=)
        NOA_SHAPE_COMP_(Strides, <=)
        NOA_SHAPE_COMP_(Strides, ==)
        NOA_SHAPE_COMP_(Strides, !=)
        #undef NOA_SHAPE_COMP_

    public: // Type casts
        template<nt::integer U, size_t AR = 0>
        [[nodiscard]] NOA_HD constexpr auto as() const noexcept {
            return static_cast<Strides<U, SIZE, AR>>(*this);
        }

        template<nt::integer U, size_t AR = 0>
        [[nodiscard]] NOA_HD constexpr auto as_clamp() const noexcept {
            return clamp_cast<Strides<U, SIZE, AR>>(*this);
        }

        template<nt::integer U, size_t AR = 0>
        [[nodiscard]] constexpr auto as_safe() const {
            return safe_cast<Strides<U, SIZE, AR>>(*this);
        }

    public:
        template<size_t S = 1, size_t AR = 0> requires (SIZE >= S)
        [[nodiscard]] NOA_HD constexpr auto pop_front() const noexcept {
            return Strides<value_type, SIZE - S, AR>::from_pointer(data() + S);
        }

        template<size_t S = 1, size_t AR = 0> requires (SIZE >= S)
        [[nodiscard]] NOA_HD constexpr auto pop_back() const noexcept {
            return Strides<value_type, SIZE - S, AR>::from_pointer(data());
        }

        template<size_t S = 1, size_t AR = 0>
        [[nodiscard]] NOA_HD constexpr auto push_front(value_type value) const noexcept {
            return Strides<value_type, SIZE + S, AR>{vec.template push_front<S, AR>(value)};
        }

        template<size_t S = 1, size_t AR = 0>
        [[nodiscard]] NOA_HD constexpr auto push_back(value_type value) const noexcept {
            return Strides<value_type, SIZE + S, AR>{vec.template push_back<S, AR>(value)};
        }

        template<size_t AR = 0, size_t S, size_t AR0>
        [[nodiscard]] NOA_HD constexpr auto push_front(const Vec<value_type, S, AR0>& vector) const noexcept {
            constexpr size_t NEW_SIZE = SIZE + S;
            return Strides<value_type, NEW_SIZE, AR>{vec.template push_front<AR>(vector)};
        }

        template<size_t AR = 0, size_t S, size_t AR0>
        [[nodiscard]] NOA_HD constexpr auto push_back(const Vec<value_type, S, AR0>& vector) const noexcept {
            constexpr size_t NEW_SIZE = SIZE + S;
            return Strides<value_type, NEW_SIZE, AR>{vec.template push_back<AR>(vector)};
        }

        template<nt::integer... U>
        [[nodiscard]] NOA_HD constexpr auto filter(U... ts) const noexcept {
            return Strides<value_type, sizeof...(U)>{(*this)[ts]...};
        }

        template<size_t S> requires (S < N and N == 4)
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

        template<nt::integer I = value_type, size_t AR>
        [[nodiscard]] NOA_HD constexpr Strides reorder(const Vec<I, SIZE, AR>& order) const noexcept {
            return {vec.reorder(order)};
        }

        [[nodiscard]] NOA_HD constexpr Strides circular_shift(i64 count) const noexcept {
            return {vec.circular_shift(count)};
        }

        [[nodiscard]] NOA_HD constexpr Strides copy() const noexcept {
            return *this;
        }

        template<size_t INDEX> requires (INDEX < SIZE)
        [[nodiscard]] NOA_HD constexpr Strides set(value_type value) const noexcept {
            auto output = *this;
            output[INDEX] = value;
            return output;
        }

    public:
        /// Whether there's at least one dimension equal to 0.
        [[nodiscard]] NOA_HD constexpr bool is_broadcast() const noexcept {
            for (size_t i{}; i < SIZE; ++i)
                if (vec[i] == 0)
                    return true;
            return false;
        }

        /// Whether the strides are in the rightmost order.
        /// Rightmost order is when the innermost stride (i.e. the dimension with the smallest stride)
        /// is on the right, and strides increase right-to-left.
        [[nodiscard]] NOA_HD constexpr bool is_rightmost() const noexcept requires (SIZE > 0) {
            for (size_t i{}; i < SIZE - 1; ++i)
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
    template<nt::integer T, size_t N, size_t A>
    Strides(Vec<T, N, A>) -> Strides<T, N, A>;

    template<nt::integer T, nt::same_as<T>... U>
    Strides(T, U...) -> Strides<T, 1 + sizeof...(U)>;
}

// Support for structure bindings:
namespace std {
    template<typename T, size_t N, size_t A>
    struct tuple_size<noa::Shape<T, N, A>> : std::integral_constant<size_t, N> {};

    template<size_t I, size_t N, size_t A, typename T>
    struct tuple_element<I, noa::Shape<T, N, A>> { using type = T; };

    template<typename T, size_t N, size_t A>
    struct tuple_size<noa::Strides<T, N, A>> : std::integral_constant<size_t, N> {};

    template<size_t I, size_t N, size_t A, typename T>
    struct tuple_element<I, noa::Strides<T, N, A>> { using type = T; };

    template<typename T, size_t N, size_t A>
    struct tuple_size<const noa::Shape<T, N, A>> : std::integral_constant<size_t, N> {};

    template<size_t I, size_t N, size_t A, typename T>
    struct tuple_element<I, const noa::Shape<T, N, A>> { using type = const T; };

    template<typename T, size_t N, size_t A>
    struct tuple_size<const noa::Strides<T, N, A>> : std::integral_constant<size_t, N> {};

    template<size_t I, size_t N, size_t A, typename T>
    struct tuple_element<I, const noa::Strides<T, N, A>> { using type = const T; };
}

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
    template<typename T, size_t N, size_t A> struct proclaim_is_shape<noa::Shape<T, N, A>> : std::true_type {};
    template<typename V1, size_t N, size_t A, typename V2> struct proclaim_is_shape_of_type<noa::Shape<V1, N, A>, V2> : std::bool_constant<std::is_same_v<V1, V2>> {};
    template<typename V, size_t N1, size_t A, size_t N2> struct proclaim_is_shape_of_size<noa::Shape<V, N1, A>, N2> : std::bool_constant<N1 == N2> {};

    template<typename T, size_t N, size_t A> struct proclaim_is_strides<noa::Strides<T, N, A>> : std::true_type {};
    template<typename V1, size_t N, size_t A, typename V2> struct proclaim_is_strides_of_type<noa::Strides<V1, N, A>, V2> : std::bool_constant<std::is_same_v<V1, V2>> {};
    template<typename V, size_t N1, size_t A, size_t N2> struct proclaim_is_strides_of_size<noa::Strides<V, N1, A>, N2> : std::bool_constant<N1 == N2> {};
}

namespace noa::inline types {
    // -- Modulo Operator --
    template<nt::shape_or_strides T> requires (T::SIZE > 0)
    [[nodiscard]] NOA_HD constexpr T operator%(T lhs, const T& rhs) noexcept {
        for (size_t i{}; i < T::SIZE; ++i)
            lhs[i] %= rhs[i];
        return lhs;
    }

    template<nt::integer T, size_t N, size_t A> requires (N > 0)
    [[nodiscard]] NOA_HD constexpr auto operator%(const Shape<T, N, A>& lhs, std::type_identity_t<T> rhs) noexcept {
        return lhs % Shape<T, N, A>::filled_with(rhs);
    }

    template<nt::integer T, size_t N, size_t A> requires (N > 0)
    [[nodiscard]] NOA_HD constexpr auto operator%(std::type_identity_t<T> lhs, const Shape<T, N, A>& rhs) noexcept {
        return Shape<T, N, A>::filled_with(lhs) % rhs;
    }

    template<nt::integer T, size_t N, size_t A> requires (N > 0)
    [[nodiscard]] NOA_HD constexpr auto operator%(const Strides<T, N, A>& lhs, std::type_identity_t<T> rhs) noexcept {
        return lhs % Strides<T, N, A>::filled_with(rhs);
    }

    template<nt::integer T, size_t N, size_t A> requires (N > 0)
    [[nodiscard]] NOA_HD constexpr auto operator%(std::type_identity_t<T> lhs, const Strides<T, N, A>& rhs) noexcept {
        return Strides<T, N, A>::filled_with(lhs) % rhs;
    }
}

namespace noa {
    // Cast Shape->Shape
    template<typename To, typename T, size_t N, size_t A> requires nt::shape_of_size<To, N>
    [[nodiscard]] NOA_FHD constexpr bool is_safe_cast(const Shape<T, N, A>& src) noexcept {
        return is_safe_cast<typename To::vector_type>(src.vec);
    }

    template<typename To, typename T, size_t N, size_t A> requires nt::shape_of_size<To, N>
    [[nodiscard]] NOA_FHD constexpr To clamp_cast(const Shape<T, N, A>& src) noexcept {
        return To{clamp_cast<typename To::vector_type>(src.vec)};
    }

    // Cast Strides->Strides
    template<typename To, typename T, size_t N, size_t A> requires nt::strides_of_size<To, N>
    [[nodiscard]] NOA_FHD constexpr bool is_safe_cast(const Strides<T, N, A>& src) noexcept {
        return is_safe_cast<typename To::vector_type>(src.vec);
    }

    template<typename To, typename T, size_t N, size_t A> requires nt::strides_of_size<To, N>
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
    template<typename T, size_t N>
    std::ostream& operator<<(std::ostream& os, const Shape<T, N>& v) {
        os << fmt::format("{}", v.vec);
        return os;
    }

    template<typename T, size_t N>
    std::ostream& operator<<(std::ostream& os, const Strides<T, N>& v) {
        os << fmt::format("{}", v.vec);
        return os;
    }
}

namespace noa::string {
    template<typename T, size_t N, size_t A>
    struct Stringify<Shape<T, N, A>> {
        static auto get() -> std::string {
            if constexpr (A == 0)
                return fmt::format("Shape<{},{}>", stringify<T>(), N);
            else
                return fmt::format("Shape<{},{},{}>", stringify<T>(), N, A);
        }
    };
    template<typename T, size_t N, size_t A>
    struct Stringify<Strides<T, N, A>> {
        static auto get() -> std::string {
            if constexpr (A == 0)
                return fmt::format("Strides<{},{}>", stringify<T>(), N);
            else
                return fmt::format("Strides<{},{},{}>", stringify<T>(), N, A);
        }
    };
}
