#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/utils/Strings.hpp"
#include "noa/core/geometry/Transform.hpp"
#include "noa/core/types/Mat.hpp"
#include "noa/core/types/Span.hpp"

#ifdef NOA_IS_OFFLINE
#include <optional>

namespace noa::geometry {
    /// Symmetry code.
    struct SymmetryCode {
        i32 order{};
        char type{};

        [[nodiscard]] static auto from_string(std::string_view symmetry) -> std::optional<SymmetryCode> {
            symmetry = ns::trim(symmetry);
            if (symmetry.empty())
                return std::nullopt;

            SymmetryCode out{};
            out.type = static_cast<char>(std::toupper(static_cast<unsigned char>(symmetry[0])));

            if (symmetry.size() > 1) {
                auto opt = ns::parse<i32>(std::string(symmetry, 1, symmetry.length())); // offset by 1
                if (not opt)
                    return std::nullopt;
                out.order = opt.value();
            } else {
                out.order = 0;
            }
            return out;
        }

        [[nodiscard]] std::string to_string() const {
            if (order)
                return fmt::format("{}{}", type, order);
            return {type};
        }
    };
}

namespace noa::geometry::guts {
    /// Sets the {2|3}d rotation matrices for the CX symmetry.
    /// \param[out] matrices Rotation matrices, excluding the identity. The order X is matrices.size() + 1.
    template<typename T, typename I, StridesTraits S> requires (nt::mat22<T> or nt::mat33<T>)
    constexpr void set_cx_symmetry_matrices(Span<T, 1, I, S> matrices) {
        using value_t = nt::value_type_t<T>;
        i64 order = matrices.ssize() + 1;
        const auto angle = Constant<f64>::PI * 2 / static_cast<f64>(order);
        for (i64 i = 1; i < order; ++i) { // skip the identity
            const auto i_angle = static_cast<f64>(i) * angle;
            if constexpr (nt::mat22<T>)
                matrices[i - 1] = rotate(i_angle).as<value_t>();
            else
                matrices[i - 1] = rotate_z(i_angle).as<value_t>();
        }
    }

    /// 3d or 4d iwise operator used to symmetrize 2d or 3d array(s).
    ///  * Can apply a per batch affine transformation before and after the symmetry.
    ///  * The symmetry is applied around a specified center.
    template<size_t N,
            nt::integer Index,
            nt::span_contiguous_nd<1> SymmetryMatrices,
            nt::interpolator_nd<N> Input,
            nt::writable_nd<N + 1> Output,
            nt::batched_parameter PreInvAffine,
            nt::batched_parameter PostInvAffine>
    requires (N == 2 or N == 3)
    class Symmetrize {
    public:
        using index_type = Index;
        using symmetry_matrices_type = SymmetryMatrices;
        using input_type = Input;
        using output_type = Output;
        using batched_pre_inverse_affine_type = PreInvAffine;
        using batched_post_inverse_affine_type = PostInvAffine;

        using input_value_type = nt::mutable_value_type_t<input_type>;
        using input_real_type = nt::value_type_t<input_value_type>;
        using output_value_type = nt::value_type_t<output_type>;

        using symmetry_matrix_type = nt::value_type_t<symmetry_matrices_type>;
        static_assert(nt::mat_of_shape<symmetry_matrix_type, N, N>);
        using coord_type = nt::value_type_t<symmetry_matrix_type>;
        using vec_type = Vec<coord_type, N>;

        // Expect the (truncated) affine with the same precision as symmetry matrices.
        using pre_inverse_affine_type = nt::value_type_t<batched_pre_inverse_affine_type>;
        using post_inverse_affine_type = nt::value_type_t<batched_post_inverse_affine_type>;
        static_assert(nt::empty<pre_inverse_affine_type> or
                      (nt::same_as<coord_type, nt::value_type_t<pre_inverse_affine_type>> and
                       (nt::mat_of_shape<pre_inverse_affine_type, N, N + 1> or
                        nt::mat_of_shape<pre_inverse_affine_type, N + 1, N + 1>)));
        static_assert(nt::empty<post_inverse_affine_type> or
                      (nt::same_as<coord_type, nt::value_type_t<post_inverse_affine_type>> and
                       (nt::mat_of_shape<post_inverse_affine_type, N, N + 1> or
                        nt::mat_of_shape<post_inverse_affine_type, N + 1, N + 1>)));

    public:
        constexpr Symmetrize(
            const input_type& input,
            const output_type& output,
            symmetry_matrices_type symmetry_inverse_rotation_matrices,
            const vec_type& symmetry_center,
            input_real_type symmetry_scaling,
            const batched_pre_inverse_affine_type& pre_inverse_affine_matrices,
            const batched_post_inverse_affine_type& post_inverse_affine_matrices
        ) noexcept :
            m_input(input), m_output(output),
            m_symmetry_matrices(symmetry_inverse_rotation_matrices),
            m_symmetry_center(symmetry_center),
            m_symmetry_scaling(symmetry_scaling),
            m_pre_inverse_affine_matrices(pre_inverse_affine_matrices),
            m_post_inverse_affine_matrices(post_inverse_affine_matrices) {}

        template<nt::same_as<index_type>... I> requires (sizeof...(I) == N)
        NOA_HD constexpr void operator()(index_type batch, I... indices) const {
            auto coordinates = vec_type::from_values(indices...);

            input_value_type value;
            if constexpr (nt::empty<post_inverse_affine_type> or nt::readable_nd<input_type, N + 1>) {
                value = m_input(batch, indices...); // skip interpolation if possible
            } else {
                coordinates = transform_vector(m_post_inverse_affine_matrices[batch], coordinates);
                value = m_input.interpolate_at(coordinates, batch);
            }

            coordinates -= m_symmetry_center;
            for (const auto& symmetry_matrix: m_symmetry_matrices) {
                auto i_coord = symmetry_matrix * coordinates + m_symmetry_center;
                if constexpr (not nt::empty<pre_inverse_affine_type>)
                    i_coord = transform_vector(m_pre_inverse_affine_matrices[batch], i_coord);
                value += m_input.interpolate_at(i_coord, batch);
            }
            value *= m_symmetry_scaling;

            m_output(batch, indices...) = static_cast<output_value_type>(value);
        }

    private:
        input_type m_input;
        output_type m_output;
        symmetry_matrices_type m_symmetry_matrices;
        vec_type m_symmetry_center;
        input_real_type m_symmetry_scaling;
        NOA_NO_UNIQUE_ADDRESS batched_pre_inverse_affine_type m_pre_inverse_affine_matrices;
        NOA_NO_UNIQUE_ADDRESS batched_post_inverse_affine_type m_post_inverse_affine_matrices;
    };
}
#endif
