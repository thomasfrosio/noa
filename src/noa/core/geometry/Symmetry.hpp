#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/Exception.hpp"
#include "noa/core/string/Format.hpp"
#include "noa/core/string/Parse.hpp"
#include "noa/core/geometry/Transform.hpp"
#include "noa/core/types/Mat22.hpp"
#include "noa/core/types/Mat33.hpp"
#include "noa/core/types/Span.hpp"

#ifdef NOA_IS_OFFLINE
#include <string_view>
#include <algorithm> // std::copy
#include <memory>
#include <optional>

// TODO This is a first draft, more symmetries should be added...

namespace noa::geometry {
    /// Symmetry symbol.
    struct SymmetrySymbol {
        int32_t order; // O = 0, C|D = X, I1 = 1, I2 = 2
        char type; // C, D, O, I

        /// Formats the symmetry symbol to its string representation.
        [[nodiscard]] std::string to_string() const {
            if (order) // CX, DX, I1, I2
                return fmt::format("{}{}", type, order);
            else
                return {type}; // O
        }
    };

    /// Symmetry utility class to parse a symmetry symbol and converts it to rotation-matrices.
    /// \details This class can parse a string containing a symmetry symbol and converts this symbol in a set of
    ///          rotation matrices that can then be applied to an array to enforce the desired symmetry.
    /// \note Supported symbols: TODO WIP...
    ///     - CX, with X being a non-zero positive number.
    /// \note The identity matrix is NOT generated, since performing an interpolation for the identity is quite
    ///       overkill compared to a simple copy of the input array, which is faster and more accurate. For instance,
    ///       a C6 symmetry returns 5 rotation matrices.
    template<typename Real, size_t N>
    class Symmetry {
    public:
        static_assert(nt::is_real_v<Real> and (N == 2 or N == 3));
        using value_type = Real;
        using matrix_type = std::conditional_t<N == 2, Mat22<value_type>, Mat33<value_type>>;

    public: // Static functions
        /// Parses the input string into a valid symmetry symbol.
        static auto parse(std::string_view symbol) -> std::optional<SymmetrySymbol> {
            std::optional<SymmetrySymbol> out = parse_symbol_(symbol);
            if (out.has_value() and out.value().type == 'C' and out.value().order > 0)
                return out;
            return std::nullopt;
        }

    public: // Constructors
        /// Creates an empty instance.
        constexpr Symmetry() = default;

        /// Parses the symmetry symbol and sets the underlying symmetry matrices.
        explicit Symmetry(std::string_view symmetry) { parse_and_set_matrices_(symmetry); }

    public:
        [[nodiscard]] auto symbol() const -> SymmetrySymbol { return m_symbol; }
        [[nodiscard]] auto count() const -> i64 { return m_count; }
        [[nodiscard]] auto get() const -> const matrix_type* { return m_data.get(); }
        [[nodiscard]] auto span() const -> Span<const matrix_type> { return {get(), count()}; }
        [[nodiscard]] auto share() const -> const std::shared_ptr<matrix_type[]>& { return m_data; }

    private:
        // Parses the symbol but doesn't check if it is recognized.
        static auto parse_symbol_(std::string_view symbol) -> std::optional<SymmetrySymbol> {
            symbol = ns::trim(symbol);
            if (symbol.empty())
                return std::nullopt;

            SymmetrySymbol out{};
            out.type = static_cast<char>(std::toupper(static_cast<unsigned char>(symbol[0])));

            if (symbol.size() > 1) {
                const std::string number(symbol, 1, symbol.length()); // offset by 1
                int error{};
                out.order = ns::parse<int32_t>(number, error);
                if (error)
                    return std::nullopt;
            } else {
                out.order = 0;
            }
            return out;
        }

        // Supported are CX, DX, O, I1, I2. X is a non-zero positive integer.
        // The string should be left trimmed.
        void parse_and_set_matrices_(std::string_view symbol) {
            auto parsed_symbol = parse_symbol_(symbol);
            check(parsed_symbol.has_value(), "Failed to parse \"{}\" to a valid symmetry", symbol);
            m_symbol = parsed_symbol.value();

            check(m_symbol.type == 'C' and m_symbol.order > 0, "{} symmetry is not supported", m_symbol.to_string());
            m_count = m_symbol.order - 1; // remove the identity from the matrices
            m_data = std::make_shared<matrix_type[]>(static_cast<size_t>(m_count));
            set_cx_matrices_(m_data.get(), m_symbol.order);
        }

        // Axial on Z.
        constexpr void set_cx_matrices_(matrix_type* rotation_matrices, i64 order) {
            NOA_ASSERT(order > 0);
            const auto angle = noa::Constant<f64>::PI * 2 / static_cast<f64>(order);
            for (i64 i = 1; i < order; ++i) { // skip the identity
                const auto i_angle = static_cast<f64>(i) * angle;
                if constexpr (N == 2)
                    rotation_matrices[i - 1] = noa::geometry::rotate(i_angle).as<value_type>();
                else
                    rotation_matrices[i - 1] = noa::geometry::rotate_z(i_angle).as<value_type>();
            }
        }

    private:
        std::shared_ptr<matrix_type[]> m_data{};
        i64 m_count{};
        SymmetrySymbol m_symbol{};
    };
}

namespace noa::geometry {
    /// 3d or 4d iwise operator used to symmetrize 2d or 3d array(s).
    ///  * Can apply an affine transformation before and after the symmetry.
    ///  * The symmetry is applied around a specified center.
    template<size_t N,
            typename Index, typename SymmetryMatrix,
            typename Interpolator, typename OutputAccessor,
            typename PreAffineMatrix = Empty,
            typename PostAffineMatrix = Empty>
    requires ((N == 2 || N == 3) and
              nt::is_int<Index>::value and
              nt::is_interpolator_nd<Interpolator, N>::value and
              nt::is_accessor_pure_nd<OutputAccessor, N + 1>::value)
    class Symmetrize {
    public:
        using index_type = Index;
        using interpolator_type = Interpolator;
        using accessor_type = OutputAccessor;
        using input_value_type = interpolator_type::mutable_value_type;
        using input_real_type = nt::value_type_t<input_value_type>;
        using output_value_type = accessor_type::mutable_value_type;

        using symmetry_matrix_type = SymmetryMatrix;
        using symmetry_span_type = Span<symmetry_matrix_type, -1, index_type>;
        using coord_type = symmetry_matrix_type::coord_type;
        using vec_type = Vec<coord_type, N>;
        static_assert((N == 2 and std::is_same_v<symmetry_matrix_type, Mat22<coord_type>>) or
                      (N == 3 and std::is_same_v<symmetry_matrix_type, Mat33<coord_type>>));

        // Expect the truncated affine with the same precision as symmetry matrices.
        using pre_inverse_affine = PreAffineMatrix;
        using post_inverse_affine = PostAffineMatrix;
        static constexpr bool has_pre_transform = not std::is_empty_v<pre_inverse_affine>;
        static constexpr bool has_post_transform = not std::is_empty_v<post_inverse_affine>;
        using expected_truncated_type = std::conditional_t<N == 2, Mat23<coord_type>, Mat34<coord_type>>;
        static_assert(nt::is_any_v<pre_inverse_affine, Empty, expected_truncated_type>);
        static_assert(nt::is_any_v<post_inverse_affine, Empty, expected_truncated_type>);

    public:
        Symmetrize(
                const interpolator_type& input,
                const accessor_type& output,
                symmetry_span_type symmetry_inverse_rotation_matrices,
                const vec_type& symmetry_center,
                input_real_type symmetry_scaling,
                const pre_inverse_affine& pre_inverse_affine_matrix = {},
                const post_inverse_affine& post_inverse_affine_matrix = {}
        ) : m_input(input), m_output(output),
            m_symmetry_matrices(symmetry_inverse_rotation_matrices),
            m_symmetry_center(symmetry_center),
            m_symmetry_scaling(symmetry_scaling),
            m_pre_matrix(pre_inverse_affine_matrix),
            m_post_matrix(post_inverse_affine_matrix) {}

        NOA_HD constexpr void operator()(index_type batch, index_type y, index_type x) const requires (N == 2) {
            m_output(batch, y, x) = static_cast<output_value_type>(compute_(batch, y, x));
        }

        NOA_HD constexpr void operator()(index_type batch, index_type z, index_type y, index_type x) const requires (N == 3) {
            m_output(batch, z, y, x) = static_cast<output_value_type>(compute_(batch, z, y, x));
        }

    private:
        NOA_HD constexpr auto compute_(index_type batch, auto... indices) {
            auto coordinates = vec_type::from_values(indices...);

            input_value_type value;
            if constexpr (has_post_transform) {
                coordinates = m_post_matrix * coordinates;
                value = m_input(coordinates, batch);
            } else {
                value = m_input.at(batch, indices...); // skip interpolation if possible
            }

            coordinates -= m_symmetry_center;
            for (const auto& symmetry_matrix: m_symmetry_matrices) {
                auto i_coord = symmetry_matrix * coordinates + m_symmetry_center;
                if constexpr (has_pre_transform)
                    i_coord = m_pre_matrix * i_coord;
                value += m_input(i_coord, batch);
            }
            return value * m_symmetry_scaling;
        }

    private:
        interpolator_type m_input;
        accessor_type m_output;
        symmetry_span_type m_symmetry_matrices;
        vec_type m_symmetry_center;
        input_real_type m_symmetry_scaling;
        NOA_NO_UNIQUE_ADDRESS pre_inverse_affine m_pre_matrix;
        NOA_NO_UNIQUE_ADDRESS post_inverse_affine m_post_matrix;
    };
}
#endif
