#pragma once

#include "noa/unified/Array.h"

namespace noa::signal::fft::details {
    using namespace ::noa::fft;
    template<Remap REMAP, typename Value>
    constexpr bool is_valid_shape_v =
            traits::is_any_v<Value, float, cfloat_t, double, cdouble_t> &&
            (REMAP == F2F || REMAP == FC2FC || REMAP == F2FC || REMAP == FC2F);
}

namespace noa::signal::fft {
    using Remap = ::noa::fft::Remap;

    /// Returns or applies an elliptical mask.
    /// \tparam REMAP       Layouts of \p input and \p output. Only F2F, FC2FC, F2FC and FC2F are supported.
    /// \tparam Value       float, double, cfloat_t, cdouble_t.
    /// \param[in] input    2D or 3D array(s) to mask. If empty, write the mask in \p output.
    /// \param[out] output  Masked array(s). Can be equal to \p input if there's no remapping.
    /// \param center       DHW center of the ellipse.
    /// \param radius       DHW radius, in elements, of the ellipse.
    /// \param edge_size    Width, in elements, of the raised-cosine, including the first zero.
    /// \param inv_matrix   Inverse HW matrix to apply on the ellipse. The rotation center is located at \p center.
    /// \param invert       Whether the mask should be inverted. If true, everything within the ellipse is removed.
    template<fft::Remap REMAP, typename Value,
             typename = std::enable_if_t<details::is_valid_shape_v<REMAP, Value>>>
    void ellipse(const Array<Value>& input, const Array<Value>& output,
                 float3_t center, float3_t radius, float edge_size,
                 float33_t inv_matrix = {}, bool invert = false);

    /// Returns or applies an 2D elliptical mask.
    /// \tparam REMAP       Layouts of \p input and \p output. Only F2F, FC2FC, F2FC and FC2F are supported.
    /// \tparam Value       float, double, cfloat_t, cdouble_t.
    /// \param[in] input    2D array(s) to mask. If empty, write the mask in \p output.
    /// \param[out] output  Masked array(s). Can be equal to \p input if there's no remapping.
    /// \param center       HW center of the ellipse.
    /// \param radius       HW radius, in elements, of the ellipse.
    /// \param edge_size    Width, in elements, of the raised-cosine, including the first zero.
    /// \param inv_matrix   Inverse HW matrix to apply on the ellipse. The rotation center is located at \p center.
    /// \param invert       Whether the mask should be inverted. If true, everything within the ellipse is removed.
    template<fft::Remap REMAP, typename Value,
             typename = std::enable_if_t<details::is_valid_shape_v<REMAP, Value>>>
    void ellipse(const Array<Value>& input, const Array<Value>& output,
                 float2_t center, float2_t radius, float edge_size,
                 float22_t inv_matrix = {}, bool invert = false);

    /// Returns or applies a spherical mask.
    /// \tparam REMAP       Layouts of \p input and \p output. Only F2F, FC2FC, F2FC and FC2F are supported.
    /// \tparam Value       float, double, cfloat_t, cdouble_t.
    /// \param[in] input    2D or 3D array(s) to mask. If empty, write the mask in \p output.
    /// \param[out] output  Masked array(s). Can be equal to \p input if there's no remapping.
    /// \param center       DHW center of the sphere.
    /// \param radius       Radius, in elements, of the sphere.
    /// \param edge_size    Width, in elements, of the raised-cosine, including the first zero.
    /// \param inv_matrix   Inverse HW matrix to apply on the sphere. The rotation center is located at \p center.
    /// \param invert       Whether the mask should be inverted. If true, everything within the sphere is removed.
    template<fft::Remap REMAP, typename Value,
             typename = std::enable_if_t<details::is_valid_shape_v<REMAP, Value>>>
    void sphere(const Array<Value>& input, const Array<Value>& output,
                float3_t center, float radius, float edge_size,
                float33_t inv_matrix = {}, bool invert = false);

    /// Returns or applies a 2D spherical mask.
    /// \tparam REMAP       Layouts of \p input and \p output. Only F2F, FC2FC, F2FC and FC2F are supported.
    /// \tparam Value       float, double, cfloat_t, cdouble_t.
    /// \param[in] input    2D array(s) to mask. If empty, write the mask in \p output.
    /// \param[out] output  Masked array(s). Can be equal to \p input if there's no remapping.
    /// \param center       HW center of the sphere.
    /// \param radius       Radius, in elements, of the sphere.
    /// \param edge_size    Width, in elements, of the raised-cosine, including the first zero.
    /// \param inv_matrix   Inverse HW matrix to apply on the sphere. The rotation center is located at \p center.
    /// \param invert       Whether the mask should be inverted. If true, everything within the sphere is removed.
    template<fft::Remap REMAP, typename Value,
             typename = std::enable_if_t<details::is_valid_shape_v<REMAP, Value>>>
    void sphere(const Array<Value>& input, const Array<Value>& output,
                float2_t center, float radius, float edge_size,
                float22_t inv_matrix = {}, bool invert = false);

    /// Returns or applies a rectangular mask.
    /// \tparam REMAP       Layouts of \p input and \p output. Only F2F, FC2FC, F2FC and FC2F are supported.
    /// \tparam Value       float, double, cfloat_t, cdouble_t.
    /// \param[in] input    2D or 3D array(s) to mask. If empty, write the mask in \p output.
    /// \param[out] output  Masked array(s). Can be equal to \p input if there's no remapping.
    /// \param center       DHW center of the rectangle.
    /// \param radius       DHW radius, in elements, of the rectangle.
    /// \param edge_size    Width, in elements, of the raised-cosine, including the first zero.
    /// \param inv_matrix   Inverse HW matrix to apply on the rectangle. The rotation center is located at \p center.
    /// \param invert       Whether the mask should be inverted. If true, everything within the sphere is removed.
    template<fft::Remap REMAP, typename Value,
             typename = std::enable_if_t<details::is_valid_shape_v<REMAP, Value>>>
    void rectangle(const Array<Value>& input, const Array<Value>& output,
                   float3_t center, float3_t radius, float edge_size,
                   float33_t inv_matrix = {}, bool invert = false);

    /// Returns or applies a 2D rectangular mask.
    /// \tparam REMAP       Layouts of \p input and \p output. Only F2F, FC2FC, F2FC and FC2F are supported.
    /// \tparam Value       float, double, cfloat_t, cdouble_t.
    /// \param[in] input    2D array(s) to mask. If empty, write the mask in \p output.
    /// \param[out] output  Masked array(s). Can be equal to \p input if there's no remapping.
    /// \param center       HW center of the rectangle.
    /// \param radius       HW radius, in elements, of the rectangle.
    /// \param edge_size    Width, in elements, of the raised-cosine, including the first zero.
    /// \param inv_matrix   Inverse HW matrix to apply on the rectangle. The rotation center is located at \p center.
    /// \param invert       Whether the mask should be inverted. If true, everything within the sphere is removed.
    template<fft::Remap REMAP, typename Value,
             typename = std::enable_if_t<details::is_valid_shape_v<REMAP, Value>>>
    void rectangle(const Array<Value>& input, const Array<Value>& output,
                   float2_t center, float2_t radius, float edge_size,
                   float22_t inv_matrix = {}, bool invert = false);

    /// Returns or applies a cylindrical mask.
    /// \tparam REMAP       Layouts of \p input and \p output. Only F2F, FC2FC, F2FC and FC2F are supported.
    /// \tparam Value       float, double, cfloat_t, cdouble_t.
    /// \param[in] input    2D or 3D array(s) to mask. If empty, write the mask in \p output.
    /// \param[out] output  Masked array(s). Can be equal to \p input if there's no remapping.
    /// \param center       DHW center of the cylinder, in \p T elements.
    /// \param radius       Radius of the cylinder.
    /// \param length       Length of the cylinder along the depth dimension.
    /// \param edge_size    Width, in elements, of the raised-cosine, including the first zero.
    /// \param inv_matrix   Inverse HW matrix to apply on the cylinder. The rotation center is located at \p center.
    /// \param invert       Whether the mask should be inverted. If true, everything within the sphere is removed.
    template<fft::Remap REMAP, typename Value,
             typename = std::enable_if_t<details::is_valid_shape_v<REMAP, Value>>>
    void cylinder(const Array<Value>& input, const Array<Value>& output,
                  float3_t center, float radius, float length, float edge_size,
                  float33_t inv_matrix = {}, bool invert = false);
}

#define NOA_UNIFIED_SHAPE_
#include "noa/unified/signal/fft/Shape.inl"
#undef NOA_UNIFIED_SHAPE_
