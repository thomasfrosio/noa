#pragma once

#include "noa/common/Functors.h"
#include "noa/unified/Array.h"

namespace noa::signal::details {
    using namespace ::noa::fft;
    template<int32_t NDIM, typename Value, typename Matrix, typename Functor>
    constexpr bool is_valid_shape_v =
            traits::is_any_v<Value, float, cfloat_t, double, cdouble_t> &&
            traits::is_any_v<Functor, noa::math::multiply_t, noa::math::plus_t> &&
            (NDIM == 2 && traits::is_any_v<Matrix, float22_t, float23_t, float33_t> ||
             NDIM == 3 && traits::is_any_v<Matrix, float33_t, float34_t, float44_t>);
}

namespace noa::signal {
    /// Returns or applies an elliptical mask.
    /// \tparam Value       float, double, cfloat_t, cdouble_t.
    /// \tparam Matrix      float33_t, float34_t, float44_t.
    /// \tparam Functor     math::multiply_t, math::plus_t.
    /// \param[in] input    2D or 3D array(s) to mask. If empty, write the mask in \p output.
    /// \param[out] output  Masked array(s). Can be equal to \p input.
    /// \param center       DHW center of the ellipse.
    /// \param radius       DHW radius, in elements, of the ellipse.
    /// \param edge_size    Width, in elements, of the cosine edge, including the zero.
    /// \param inv_matrix   Inverse DHW (affine) matrix to apply on the ellipse.
    ///                     For non-affine matrices, the rotation center is located at \p center.
    /// \param functor      Operator defining how to apply the mask onto \p input. This is ignored if \p input is empty.
    /// \param invert       Whether the mask should be inverted. If true, everything within the ellipse is removed.
    template<typename Value, typename Matrix = float33_t, typename Functor = noa::math::multiply_t,
             typename = std::enable_if_t<details::is_valid_shape_v<3, Value, Matrix, Functor>>>
    void ellipse(const Array<Value>& input, const Array<Value>& output,
                 float3_t center, float3_t radius, float edge_size,
                 Matrix inv_matrix = {}, Functor functor = {}, bool invert = false);

    /// Returns or applies an 2D elliptical mask.
    /// \tparam Value       float, double, cfloat_t, cdouble_t.
    /// \tparam Matrix      float22_t, float23_t, float33_t.
    /// \param[in] input    2D array(s) to mask. If empty, write the mask in \p output.
    /// \param[out] output  Masked array(s). Can be equal to \p input.
    /// \param center       HW center of the ellipse.
    /// \param radius       HW radius, in elements, of the ellipse.
    /// \param edge_size    Width, in elements, of the cosine edge, including the zero.
    /// \param inv_matrix   Inverse HW (affine) matrix to apply on the ellipse.
    ///                     For non-affine matrices, the rotation center is located at \p center.
    /// \param functor      Operator defining how to apply the mask onto \p input. This is ignored if \p input is empty.
    /// \param invert       Whether the mask should be inverted. If true, everything within the ellipse is removed.
    template<typename Value, typename Matrix = float22_t, typename Functor = noa::math::multiply_t,
             typename = std::enable_if_t<details::is_valid_shape_v<2, Value, Matrix, Functor>>>
    void ellipse(const Array<Value>& input, const Array<Value>& output,
                 float2_t center, float2_t radius, float edge_size,
                 Matrix inv_matrix = {}, Functor functor = {}, bool invert = false);

    /// Returns or applies a spherical mask.
    /// \tparam Value       float, double, cfloat_t, cdouble_t.
    /// \tparam Matrix      float33_t, float34_t, float44_t.
    /// \param[in] input    2D or 3D array(s) to mask. If empty, write the mask in \p output.
    /// \param[out] output  Masked array(s). Can be equal to \p input.
    /// \param center       DHW center of the sphere.
    /// \param radius       Radius, in elements, of the sphere.
    /// \param edge_size    Width, in elements, of the cosine edge, including the zero.
    /// \param inv_matrix   Inverse DHW (affine) matrix to apply on the ellipse.
    ///                     For non-affine matrices, the rotation center is located at \p center.
    /// \param functor      Operator defining how to apply the mask onto \p input. This is ignored if \p input is empty.
    /// \param invert       Whether the mask should be inverted. If true, everything within the sphere is removed.
    template<typename Value, typename Matrix = float33_t, typename Functor = noa::math::multiply_t,
             typename = std::enable_if_t<details::is_valid_shape_v<3, Value, Matrix, Functor>>>
    void sphere(const Array<Value>& input, const Array<Value>& output,
                float3_t center, float radius, float edge_size,
                Matrix inv_matrix = {}, Functor functor = {}, bool invert = false);

    /// Returns or applies a 2D spherical mask.
    /// \tparam Value       float, double, cfloat_t, cdouble_t.
    /// \tparam Matrix      float22_t, float23_t, float33_t.
    /// \param[in] input    2D array(s) to mask. If empty, write the mask in \p output.
    /// \param[out] output  Masked array(s). Can be equal to \p input.
    /// \param center       HW center of the sphere.
    /// \param radius       Radius, in elements, of the sphere.
    /// \param edge_size    Width, in elements, of the cosine edge, including the zero.
    /// \param inv_matrix   Inverse HW (affine) matrix to apply on the ellipse.
    ///                     For non-affine matrices, the rotation center is located at \p center.
    /// \param functor      Operator defining how to apply the mask onto \p input. This is ignored if \p input is empty.
    /// \param invert       Whether the mask should be inverted. If true, everything within the sphere is removed.
    template<typename Value, typename Matrix = float22_t, typename Functor = noa::math::multiply_t,
             typename = std::enable_if_t<details::is_valid_shape_v<2, Value, Matrix, Functor>>>
    void sphere(const Array<Value>& input, const Array<Value>& output,
                float2_t center, float radius, float edge_size,
                Matrix inv_matrix = {}, Functor functor = {}, bool invert = false);

    /// Returns or applies a rectangular mask.
    /// \tparam Value       float, double, cfloat_t, cdouble_t.
    /// \tparam Matrix      float33_t, float34_t, float44_t.
    /// \param[in] input    2D or 3D array(s) to mask. If empty, write the mask in \p output.
    /// \param[out] output  Masked array(s). Can be equal to \p input.
    /// \param center       DHW center of the rectangle.
    /// \param radius       DHW radius, in elements, of the rectangle.
    /// \param edge_size    Width, in elements, of the cosine edge, including the zero.
    /// \param inv_matrix   Inverse DHW (affine) matrix to apply on the ellipse.
    ///                     For non-affine matrices, the rotation center is located at \p center.
    /// \param functor      Operator defining how to apply the mask onto \p input. This is ignored if \p input is empty.
    /// \param invert       Whether the mask should be inverted. If true, everything within the sphere is removed.
    template<typename Value, typename Matrix = float33_t, typename Functor = noa::math::multiply_t,
             typename = std::enable_if_t<details::is_valid_shape_v<3, Value, Matrix, Functor>>>
    void rectangle(const Array<Value>& input, const Array<Value>& output,
                   float3_t center, float3_t radius, float edge_size,
                   Matrix inv_matrix = {}, Functor functor = {}, bool invert = false);

    /// Returns or applies a 2D rectangular mask.
    /// \tparam Value       float, double, cfloat_t, cdouble_t.
    /// \tparam Matrix      float22_t, float23_t, float33_t.
    /// \param[in] input    2D array(s) to mask. If empty, write the mask in \p output.
    /// \param[out] output  Masked array(s). Can be equal to \p input.
    /// \param center       HW center of the rectangle.
    /// \param radius       HW radius, in elements, of the rectangle.
    /// \param edge_size    Width, in elements, of the cosine edge, including the zero.
    /// \param inv_matrix   Inverse HW (affine) matrix to apply on the ellipse.
    ///                     For non-affine matrices, the rotation center is located at \p center.
    /// \param functor      Operator defining how to apply the mask onto \p input. This is ignored if \p input is empty.
    /// \param invert       Whether the mask should be inverted. If true, everything within the sphere is removed.
    template<typename Value, typename Matrix = float22_t, typename Functor = noa::math::multiply_t,
             typename = std::enable_if_t<details::is_valid_shape_v<2, Value, Matrix, Functor>>>
    void rectangle(const Array<Value>& input, const Array<Value>& output,
                   float2_t center, float2_t radius, float edge_size,
                   Matrix inv_matrix = {}, Functor functor = {}, bool invert = false);

    /// Returns or applies a cylindrical mask.
    /// \tparam Value       float, double, cfloat_t, cdouble_t.
    /// \tparam Matrix      float33_t, float34_t, float44_t.
    /// \param[in] input    2D or 3D array(s) to mask. If empty, write the mask in \p output.
    /// \param[out] output  Masked array(s). Can be equal to \p input.
    /// \param center       DHW center of the cylinder, in \p T elements.
    /// \param radius       Radius of the cylinder.
    /// \param length       Length of the cylinder along the depth dimension.
    /// \param edge_size    Width, in elements, of the cosine edge, including the zero.
    /// \param inv_matrix   Inverse DHW (affine) matrix to apply on the ellipse.
    ///                     For non-affine matrices, the rotation center is located at \p center.
    /// \param functor      Operator defining how to apply the mask onto \p input. This is ignored if \p input is empty.
    /// \param invert       Whether the mask should be inverted. If true, everything within the sphere is removed.
    template<typename Value, typename Matrix = float33_t, typename Functor = noa::math::multiply_t,
             typename = std::enable_if_t<details::is_valid_shape_v<3, Value, Matrix, Functor>>>
    void cylinder(const Array<Value>& input, const Array<Value>& output,
                  float3_t center, float radius, float length, float edge_size,
                  Matrix inv_matrix = {}, Functor functor = {}, bool invert = false);
}

#define NOA_UNIFIED_SHAPE_
#include "noa/unified/signal/Shape.inl"
#undef NOA_UNIFIED_SHAPE_
