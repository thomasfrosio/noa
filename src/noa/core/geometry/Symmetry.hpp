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

#if defined(NOA_IS_OFFLINE)
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
        static_assert(nt::is_real_v<Real> && (N == 2 || N == 3));
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
            m_data = std::make_unique<matrix_type[]>(static_cast<size_t>(m_count)); // TODO C++20 make_shared
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
#endif
