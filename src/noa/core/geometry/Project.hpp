#pragma once

#include "noa/core/types/Shape.hpp"
#include "noa/core/Interpolation.hpp"
#include "noa/core/utils/Atomic.hpp"

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

    template<typename T>
    auto forward_projection_transform_vector(
        const Vec<T, 3>& image_coordinates,
        const Vec<T, 3>& projection_window_center,
        const Mat<T, 3, 4>& projection_matrix
    ) -> Vec<T, 3> {
        const auto projection_axis = projection_matrix.col(0);
        const auto projection_matrix_no_z = projection_matrix.filter_columns(1, 2, 3);

        // Retrieve the base plane (z, y, or x) that was used to compute the projection window.
        i32 i{};
        {
            T distance{std::numeric_limits<T>::max()};
            for (i32 j{}; j < 3; ++j) {
                if (abs(projection_axis[j]) > 0) {
                    auto tmp = abs(projection_window_center[j] / projection_axis[j]);
                    if (tmp < distance) {
                        i = j;
                        distance = tmp;
                    }
                }
            }
        }

        // Transform from image space to volume space.
        // Since we need the transformed yz-plane, extract the 0yx transformed vector from the matrix-vector product.
        Vec<T, 3> plane_0yx = projection_matrix_no_z * image_coordinates.pop_front().push_back(1);
        auto volume_coordinates = plane_0yx + projection_axis * image_coordinates[0];

        // Compute and add the distance along the projection axis,
        // from the yx-plane to the selected plane centered on the window center.
        // This is from https://en.m.wikipedia.org/wiki/Line-plane_intersection
        const T distance = ((projection_window_center[i] - plane_0yx[i]) * projection_axis[i]) / projection_axis[i];
        volume_coordinates += projection_axis * distance;

        return volume_coordinates;
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
        using shape_3d_type = Shape<index_type, 3>;

        static_assert(nt::any_of<matrix_type, Mat<coord_type, 4, 4>, Mat<coord_type, 3, 4>>);
        static_assert(input_type::BORDER == Border::ZERO);

    public:
        constexpr ForwardProject(
            const input_type& input,
            const output_type& output,
            const shape_3d_type& volume_shape,
            const batched_matrix_type& batched_forward_matrices,
            index_type projection_window_size
        ) :
            m_input(input),
            m_output(output),
            m_batched_forward_matrices(batched_forward_matrices),
            m_volume_shape(volume_shape),
            m_volume_center((volume_shape.vec / 2).template as<coord_type>()),
            m_projection_window_radius(projection_window_size / 2) {}

        // For every pixel (y,x) of the forward projected output image (i is the batch).
        // z is the extra dimension for the projection window (the longest diagonal).
        constexpr void operator()(index_type i, index_type z, index_type y, index_type x) const {
            const auto affine = m_batched_forward_matrices[i].filter_rows(0, 1, 2); // truncated
            const auto image_coordinates = coord_3d_type::from_values(z - m_projection_window_radius, y, x);
            const auto volume_coordinates = forward_projection_transform_vector(
                image_coordinates, m_volume_center, affine);

            // The interpolator handles OOB coordinates using Border::ZERO, so we could skip that.
            // However, we do expect a significant number of cases where the volume_coordinates are OOB,
            // so try to shortcut here directly.
            if (not is_within_interpolation_window<input_type::INTERP, Border::ZERO>(volume_coordinates, m_volume_shape)) // FIXME make sure this works
                return;

            const auto value = static_cast<output_value_type>(m_input.interpolate_at(volume_coordinates, i));
            ng::atomic_add(m_output, value, i, y, x); // sum along z
        }

    private:
        input_type m_input;
        output_type m_output;
        batched_matrix_type m_batched_forward_matrices;
        shape_3d_type m_volume_shape{};
        coord_3d_type m_volume_center{};
        index_type m_projection_window_radius{};
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
        using output_real_type = nt::value_type_t<output_value_type>;

        using batched_input_matrix_type = BatchedInputMatrix;
        using batched_output_matrix_type = BatchedOutputMatrix;
        using input_matrix_type = nt::mutable_value_type_t<batched_input_matrix_type>;
        using output_matrix_type = nt::mutable_value_type_t<batched_output_matrix_type>;
        using coord_type = nt::value_type_t<input_matrix_type>;
        using coord_3d_type = Vec<coord_type, 3>;
        using shape_3d_type = Shape<index_type, 3>;

        static_assert(nt::any_of<input_matrix_type, Mat<coord_type, 4, 4>, Mat<coord_type, 2, 4>>);
        static_assert(nt::any_of<output_matrix_type, Mat<coord_type, 4, 4>, Mat<coord_type, 3, 4>>);

    public:
        constexpr BackwardForwardProject(
            const input_type& input,
            const output_type& output,
            const shape_3d_type& volume_shape,
            const batched_input_matrix_type& batched_backward_matrices,
            const batched_output_matrix_type& batched_forward_matrices,
            index_type projection_window_size,
            index_type n_inputs
        ) :
            m_input(input),
            m_output(output),
            m_batched_backward_matrices(batched_backward_matrices),
            m_batched_forward_matrices(batched_forward_matrices),
            m_volume_shape(volume_shape.vec.template as<coord_type>()),
            m_volume_center((volume_shape.vec / 2).template as<coord_type>()),
            m_projection_window_radius(projection_window_size / 2),
            m_n_input_images(n_inputs) {}

    public:
        // For every pixel (y,x) of the forward projected output image (i is the batch).
        // z is the extra dimension for the projection window (the longest diagonal).
        constexpr void operator()(index_type i, index_type z, index_type y, index_type x) const {
            const auto affine = m_batched_forward_matrices[i].filter_rows(0, 1, 2);
            const auto image_coordinates = coord_3d_type::from_values(z - m_projection_window_radius, y, x);
            const auto volume_coordinates = forward_projection_transform_vector(
                image_coordinates, m_volume_center, affine);

            // The interpolator handles OOB coordinates on the 2d plane (yx) using Border::ZERO, so we could skip that.
            // However, along the z, we need to stop the virtual volume somewhere. We could stop exactly at the volume
            // edge, but that creates a pixel aliasing at the edges in certain orientations. While this is probably
            // fine, instead, do a linear antialiasing for the edges by sampling one more than necessary on each side
            // and do a lerp to smooth it out correctly.
            if (volume_coordinates[0] < -1 or volume_coordinates[0] > m_volume_shape[0] or
                volume_coordinates[1] < -1 or volume_coordinates[1] > m_volume_shape[1] or
                volume_coordinates[2] < -1 or volume_coordinates[2] > m_volume_shape[2])
                return;

            // Sample the virtual volume (backprojection).
            input_value_type value{};
            for (index_type j{}; j < m_n_input_images; ++j) {
                const auto input_coordinates = project_vector(
                    m_batched_backward_matrices[j], volume_coordinates.push_back(1));
                value += static_cast<output_value_type>(m_input.interpolate_at(input_coordinates, j));
            }

            // Smooth the volume edges using a linear weighting.
            for (size_t j{}; j < 3; ++j) {
                if (volume_coordinates[j] < 0) {
                    const auto fraction = volume_coordinates[j] + 1;
                    value = value * static_cast<output_real_type>(fraction);

                } else if (volume_coordinates[j] > m_volume_shape[j] - 1) {
                    const auto fraction = volume_coordinates[j] - m_volume_shape[j] + 1;
                    value = value * static_cast<output_real_type>(1 - fraction);
                }
            }

            ng::atomic_add(m_output, value, i, y, x); // sum along z
        }

    private:
        input_type m_input;
        output_type m_output;
        batched_input_matrix_type m_batched_backward_matrices;
        batched_output_matrix_type m_batched_forward_matrices;
        coord_3d_type m_volume_shape{};
        coord_3d_type m_volume_center{};
        index_type m_projection_window_radius{};
        index_type m_n_input_images;
    };
}
