#pragma once

#include "noa/core/types/Shape.hpp"
#include "noa/core/Interpolation.hpp"
#include "noa/core/utils/Atomic.hpp"
#include "noa/core/geometry/Transform.hpp"

namespace noa::geometry::guts {
    template<nt::any_of<f32, f64> T, typename X>
    constexpr auto project_vector(
        const X& xform,
        const Vec<T, 4>& vector
    ) -> Vec<T, 2> {
        if constexpr (nt::mat_of_shape<X, 4, 4>) {
            return Vec{
                dot(xform[1], vector),
                dot(xform[2], vector),
            };
        } else if constexpr (nt::mat_of_shape<X, 2, 4>) {
            return xform * vector;
        } else {
            static_assert(nt::always_false<X>);
        }
    }

    // When doing the projection, we need to ensure that the z is large enough to cover the entire volume
    // in any direction. To do so, we set the z equal to the longest diagonal of the volume, i.e sqrt(3) *
    // max(volume_shape). However, this assumes no shifts (along the project axis) and that the center of
    // the volume is equal to the rotation center. To account for these cases, we shift the "projection window",
    // along the normal of the projected plane, back to the center of the volume.
    template<typename T>
    constexpr auto distance_from_center_along_normal(
        const Vec<T, 3>& center,
        const Mat<T, 3, 4>& affine
    ) -> Vec<T, 3> {
        const auto normal = affine.col(0); // aka projection axis
        const auto distance = center - affine.col(3);
        const auto distance_along_normal = normal * dot(normal, distance);
        return distance_along_normal;
    }

    template<nt::sinteger Index,
             nt::interpolator_nd<2> Input,
             nt::writable_nd<3> Output,
             nt::batched_parameter BatchedMatrix>
    class BackwardProject {
    public:
        using index_type = Index;
        using input_type = Input;
        using output_type = Output;
        using input_value_type = nt::mutable_value_type_t<input_type>;
        using output_value_type = nt::value_type_t<output_type>;

        using batched_matrix_type = BatchedMatrix;
        using matrix_type = nt::mutable_value_type_t<batched_matrix_type>;
        using coord_type = nt::value_type_t<matrix_type>;
        using coord_4d_type = Vec<coord_type, 4>;

        static_assert(nt::any_of<matrix_type, Mat<coord_type, 4, 4>, Mat<coord_type, 2, 4>>);

    public:
        constexpr BackwardProject(
            const input_type& input,
            const output_type& output,
            const batched_matrix_type& batched_inverse_matrices,
            index_type n_inputs,
            bool add_to_output
        ) :
            m_input(input),
            m_output(output),
            m_batched_inverse_matrices(batched_inverse_matrices),
            m_n_inputs(n_inputs),
            m_add_to_output(add_to_output) {}

        constexpr void operator()(index_type z, index_type y, index_type x) const {
            const auto output_coordinates = coord_4d_type::from_values(z, y, x, 1);

            input_value_type value{};
            for (index_type i{}; i < m_n_inputs; ++i) {
                const auto input_coordinates = project_vector(m_batched_inverse_matrices[i], output_coordinates);
                value += static_cast<output_value_type>(m_input.interpolate_at(input_coordinates, i));
            }
            auto& output = m_output(z, y, x);
            output = m_add_to_output ? output + value : value;
        }

        // Alternative implementation to benchmark. This exposes the number of images to backproject
        // so to have more work to distribute, but requires the output to be already set and to write
        // atomically...
        constexpr void operator()(index_type i, index_type z, index_type y, index_type x) const {
            const auto output_coordinates = coord_4d_type::from_values(z, y, x, 1);
            const auto input_coordinates = project_vector(m_batched_inverse_matrices[i], output_coordinates);
            const auto value = static_cast<output_value_type>(m_input.interpolate_at(input_coordinates, i));
            ng::atomic_add(m_output, value, z, y, x);
        }

    private:
        input_type m_input;
        output_type m_output;
        batched_matrix_type m_batched_inverse_matrices;
        index_type m_n_inputs;
        bool m_add_to_output;
    };

    template<nt::sinteger Index,
             nt::interpolator_nd<3> Input,
             nt::atomic_addable_nd<3> Output,
             nt::batched_parameter BatchedMatrix>
    class ForwardProject {
    public:
        using index_type = Index;
        using input_type = Input;
        using output_type = Output;
        using output_value_type = nt::value_type_t<output_type>;

        using batched_matrix_type = BatchedMatrix;
        using matrix_type = nt::mutable_value_type_t<batched_matrix_type>;
        using coord_type = nt::value_type_t<matrix_type>;
        using coord_3d_type = Vec<coord_type, 3>;

        static_assert(nt::any_of<matrix_type, Mat<coord_type, 4, 4>, Mat<coord_type, 3, 4>>);

    public:
        constexpr ForwardProject(
            const input_type& input,
            const output_type& output,
            const batched_matrix_type& batched_forward_matrices
        ) :
            m_input(input),
            m_output(output),
            m_batched_forward_matrices(batched_forward_matrices) {}

        constexpr auto set_volume_shape(const Shape<index_type, 3>& volume_shape) -> index_type{
            // Distance of the projection window to cover the volume in any direction.
            auto longest_diagonal = static_cast<index_type>(ceil(sqrt(3.f) * static_cast<f32>(max(volume_shape))));
            m_longest_diagonal_radius = (longest_diagonal + 1) / 2;

            // Point at the center of the projection window.
            m_volume_center = (volume_shape.vec / 2).template as<coord_type>();

            return longest_diagonal;
        }

        // For every pixel (y,x) of the forward projected output image (i is the batch).
        // z is the extra dimension for the projection window (the longest diagonal).
        constexpr void operator()(index_type i, index_type z, index_type y, index_type x) const {
            const auto affine = m_batched_forward_matrices[i].filter(0, 1, 2); // truncated
            const auto output_coordinates = coord_3d_type::from_values(z - (m_longest_diagonal_radius + 1) / 2, y, x);
            auto input_coordinates = transform_vector(affine, output_coordinates);
            input_coordinates += distance_from_center_along_normal(m_volume_center, affine);

            const auto value = static_cast<output_value_type>(m_input.interpolate_at(input_coordinates, i));
            ng::atomic_add(m_output, value, i, y, x); // sum along z
        }

    private:
        input_type m_input;
        output_type m_output;
        batched_matrix_type m_batched_forward_matrices;
        coord_3d_type m_volume_center{};
        index_type m_longest_diagonal_radius{};
    };

    template<nt::sinteger Index,
             nt::interpolator_nd<2> Input,
             nt::atomic_addable_nd<3> Output,
             nt::batched_parameter BatchedInputMatrix,
             nt::batched_parameter BatchedOutputMatrix>
    class BackwardForwardProject {
    public:
        using index_type = Index;
        using input_type = Input;
        using output_type = Output;
        using input_value_type = nt::mutable_value_type_t<input_type>;
        using output_value_type = nt::value_type_t<output_type>;

        using batched_input_matrix_type = BatchedInputMatrix;
        using batched_output_matrix_type = BatchedOutputMatrix;
        using input_matrix_type = nt::mutable_value_type_t<batched_input_matrix_type>;
        using output_matrix_type = nt::mutable_value_type_t<batched_output_matrix_type>;
        using coord_type = nt::value_type_t<input_matrix_type>;
        using coord_3d_type = Vec<coord_type, 3>;

        static_assert(nt::any_of<input_matrix_type, Mat<coord_type, 4, 4>, Mat<coord_type, 2, 4>>);
        static_assert(nt::any_of<output_matrix_type, Mat<coord_type, 4, 4>, Mat<coord_type, 3, 4>>);

    public:
        constexpr BackwardForwardProject(
            const input_type& input,
            const output_type& output,
            const batched_input_matrix_type& batched_backward_matrices,
            const batched_output_matrix_type& batched_forward_matrices,
            index_type n_inputs
        ) :
            m_input(input),
            m_output(output),
            m_batched_backward_matrices(batched_backward_matrices),
            m_batched_forward_matrices(batched_forward_matrices),
            m_n_input_images(n_inputs) {}

        constexpr auto set_volume_shape(const Shape<index_type, 3>& volume_shape) -> index_type{
            // Distance of the projection window to cover the volume in any direction.
            auto longest_diagonal = static_cast<index_type>(ceil(sqrt(3.f) * static_cast<f32>(max(volume_shape))));
            m_longest_diagonal_radius = (longest_diagonal + 1) / 2;

            // Point at the center of the projection window.
            m_volume_center = (volume_shape.vec / 2).template as<coord_type>();

            return longest_diagonal;
        }

    public:
        // For every pixel (y,x) of the forward projected output image (i is the batch).
        // z is the extra dimension for the projection window (the longest diagonal).
        constexpr void operator()(index_type i, index_type z, index_type y, index_type x) const {
            const auto affine = m_batched_forward_matrices[i].filter(0, 1, 2);
            const auto output_coordinates = coord_3d_type::from_values(z - m_longest_diagonal_radius, y, x);
            auto volume_coordinates = transform_vector(affine, output_coordinates);
            volume_coordinates += distance_from_center_along_normal(m_volume_center, affine);

            // Sample the virtual volume (backprojection)
            input_value_type value{};
            for (index_type j{}; j < m_n_input_images; ++j) {
                const auto input_coordinates = project_vector(
                    m_batched_backward_matrices[j], volume_coordinates.push_back(1));
                value += static_cast<output_value_type>(m_input.interpolate_at(input_coordinates, j));
            }

            ng::atomic_add(m_output, value, i, y, x); // sum along z
        }

    private:
        input_type m_input;
        output_type m_output;
        batched_input_matrix_type m_batched_backward_matrices;
        batched_output_matrix_type m_batched_forward_matrices;
        coord_3d_type m_volume_center{};
        index_type m_longest_diagonal_radius{};
        index_type m_n_input_images;
    };
}
