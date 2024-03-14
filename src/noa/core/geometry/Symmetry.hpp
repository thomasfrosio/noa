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

namespace noa::geometry {
    /// Symmetry code.
    struct SymmetryCode {
        i32 order{}; // O = 0, C|D = X, I1 = 1, I2 = 2
        char type{}; // C, D, O, I

        [[nodiscard]] static auto from_string(std::string_view symmetry) -> std::optional<SymmetryCode> {
            symmetry = ns::trim(symmetry);
            if (symmetry.empty())
                return std::nullopt;

            SymmetryCode out{};
            out.type = static_cast<char>(std::toupper(static_cast<unsigned char>(symmetry[0])));

            if (symmetry.size() > 1) {
                const std::string number(symmetry, 1, symmetry.length()); // offset by 1
                int error{};
                out.order = ns::parse<i32>(number, error);
                if (error)
                    return std::nullopt;
            } else {
                out.order = 0;
            }
            return out;
        }

        [[nodiscard]] std::string to_string() const {
            if (order) // CX, DX, I1, I2
                return fmt::format("{}{}", type, order);
            else
                return {type}; // O
        }
    };

    /// Sets the 2d rotation matrices for the CX symmetry.
    /// \param[out] matrices Rotation matrices, excluding the identity. The order (i.e. X) is matrices.size() + 1.
    template<typename T> requires (nt::is_mat22_v<T> or nt::is_mat33_v<T>)
    constexpr void set_cx_symmetry_matrices(Span<T> matrices) {
        using value_t = nt::value_type_t<T>;
        i64 order = matrices.ssize() + 1;
        const auto angle = noa::Constant<f64>::PI * 2 / static_cast<f64>(order);
        for (i64 i = 1; i < order; ++i) { // skip the identity
            const auto i_angle = static_cast<f64>(i) * angle;
            if constexpr (nt::is_mat22_v<T>)
                matrices[i - 1] = rotate(i_angle).as<value_t>();
            else
                matrices[i - 1] = rotate_z(i_angle).as<value_t>();
        }
    }
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
