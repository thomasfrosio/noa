
// -- Using texture - for centered Fourier transforms -- //
namespace noa::cuda::transform {
    /// Applies one or multiple 2D rotations to a non-redundant centered Fourier transform.
    /// \tparam T               float or cfloat_t. With cfloat_t, \a texture should have its descriptor set to float2.
    /// \tparam REMAP           Whether or not \p output should be remapped to a non-centered layout (i.e. ifftshift)
    ///                         so that it can be passed directly to the c2r routines.
    /// \param texture          Non-redundant and centered transform.
    ///                         Should use the INTERP_NEAREST, INTERP_COSINE or INTERP_LINEAR filtering mode.
    ///                         Should use the BORDER_ZERO filtering mode (although it doesn't effect on \p output).
    ///                         Un-normalized coordinates should be used.
    ///                         It's center of rotation is expected to be at index `(0, shape/2)`.
    /// \param outputs          On the \b device. Non-redundant rotated, using bilinear interpolation, Fourier transform.
    ///                         One per rotation. Output arrays can be centered or non-centered depending on \p REMAP.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical shape of \p texture and \p outputs (one per rotation).
    ///                         All dimensions should therefore have the same logical size.
    /// \param rotations        On the \b host. Rotation angles, in radians. One per rotation.
    /// \param nb_rotations     Number of rotations.
    /// \param freq_cutoff      Frequency cutoff. From 0 to 0.5 (0.5 being the Nyquist frequency).
    /// \param stream           Stream on which to enqueue this function.
    ///
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<bool REMAP = false, typename T>
    NOA_HOST void rotate2DFT(cudaTextureObject_t texture, T* output, size_t output_pitch, size_t shape,
                             const float* rotations, uint nb_rotations, float freq_cutoff, Stream& stream);

    /// Applies a single 2D rotation to a non-redundant centered Fourier transform.
    /// \see This function has the same features and limitations than the overload above.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<bool REMAP = false, typename T>
    NOA_HOST void rotate2DFT(cudaTextureObject_t texture, T* output, size_t output_pitch, size_t shape,
                             float rotation, float freq_cutoff, Stream& stream);

    /// Applies one or multiple 3D rotations to a non-redundant centered Fourier transform.
    /// \tparam T               float or cfloat_t. With cfloat_t, \a texture should have its descriptor set to float2.
    /// \tparam REMAP           Whether or not \p output should be remapped to a non-centered layout (i.e. ifftshift)
    ///                         so that it can be passed directly to the c2r routines.
    /// \param texture          Non-redundant and centered transform.
    ///                         Should use the INTERP_NEAREST, INTERP_COSINE or INTERP_LINEAR filtering mode.
    ///                         Should use the BORDER_ZERO filtering mode (although it doesn't effect on \p output).
    ///                         Un-normalized coordinates should be used.
    ///                         It's center of rotation is expected to be at index `(0, shape/2, shape/2)`.
    /// \param outputs          On the \b device. Non-redundant rotated, using bilinear interpolation, Fourier transform.
    ///                         One per rotation. Output arrays can be centered or non-centered depending on \p REMAP.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical shape of \p texture and \p outputs (one per rotation).
    ///                         All dimensions should therefore have the same logical size.
    /// \param rotations        On the \b host. ZYZ Euler angles, in radians. One trio per rotation.
    /// \param nb_rotations     Number of rotations.
    /// \param freq_cutoff      Frequency cutoff. From 0 to 0.5 (0.5 being the Nyquist frequency).
    /// \param stream           Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    template<bool REMAP = false, typename T>
    NOA_HOST void rotate3DFT(cudaTextureObject_t texture, T* outputs, size_t output_pitch, size_t shape,
                             const float3_t* rotations, uint nb_rotations, float freq_cutoff, Stream& stream);

    /// Applies a single 3D rotation to a non-redundant centered Fourier transform.
    /// \see This function has the same features and limitations than the overload above. The only difference is that
    ///      since it computes a single rotation, it doesn't need to allocate a temporary array to store the rotation
    ///      matrices. As such, this function is asynchronous relative to the host and may return before completion.
    template<bool REMAP = false, typename T>
    NOA_HOST void rotate3DFT(cudaTextureObject_t texture, T* output, size_t output_pitch, size_t shape,
                             float3_t rotation, float freq_cutoff, Stream& stream);
}

// -- Using arrays - for centered Fourier transforms -- //
namespace noa::cuda::transform {
    /// Rotates a 2D non-redundant centered Fourier transform, either inplace or out-of-place.
    /// \tparam T               float or cfloat_t. With cfloat_t, \a texture should have its descriptor set to float2.
    /// \tparam REMAP           Whether or not \p outputs should be remapped to a non-centered layout (i.e. ifftshift)
    ///                         so that it can be passed directly to the c2r routines.
    /// \param input            On the \b device. Non-redundant and centered Fourier transform.
    ///                         It's center of rotation is expected to be at index `(0, shape/2)`.
    /// \param input_pitch      Pitch, in elements, of \p input.
    /// \param outputs          On the \b device. Non-redundant rotated, using bilinear interpolation, Fourier transform.
    ///                         One per rotation. Can be equal to \p input.
    ///                         Output arrays can be centered or non-centered depending on \p REMAP.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical shape of \p input and \p outputs (one per rotation).
    ///                         All dimensions should therefore have the same logical size.
    /// \param rotations        Rotation angles, in radians. One per rotation.
    /// \param nb_rotations     Number of rotations.
    /// \param freq_cutoff      Frequency cutoff. From 0 to 0.5 (0.5 being the Nyquist frequency).
    /// \param stream           Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    template<bool REMAP = false, typename T>
    NOA_HOST void rotate2DFT(const T* input, size_t input_pitch, T* outputs, size_t output_pitch, size_t shape,
                             const float* rotations, uint nb_rotations, float freq_cutoff, Stream& stream);

    /// Applies a single 2D rotation to a non-redundant centered Fourier transform.
    template<bool REMAP = false, typename T>
    NOA_HOST void rotate2DFT(const T* input, size_t input_pitch, T* output, size_t output_pitch, size_t shape,
                             float rotation, float freq_cutoff, Stream& stream);

    /// Rotates a 3D non-redundant centered Fourier transform, either inplace or out-of-place.
    /// \tparam T               float or cfloat_t. With cfloat_t, \a texture should have its descriptor set to float2.
    /// \tparam REMAP           Whether or not \p outputs should be remapped to a non-centered layout (i.e. ifftshift)
    ///                         so that it can be passed directly to the c2r routines.
    /// \param input            On the \b device. Non-redundant and centered Fourier transform.
    ///                         It's center of rotation is expected to be at index `(0, shape/2, shape/2)`.
    /// \param input_pitch      Pitch, in elements, of \p input.
    /// \param outputs          On the \b device. Non-redundant rotated, using bilinear interpolation, Fourier transform.
    ///                         One per rotation. Can be equal to \p input.
    ///                         Output arrays can be centered or non-centered depending on \p REMAP.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical shape of \p input and \p outputs (one per rotation).
    ///                         All dimensions should therefore have the same logical size.
    /// \param rotations        ZYZ Euler angles, in radians. One trio per rotation.
    /// \param nb_rotations     Number of rotations.
    /// \param freq_cutoff      Frequency cutoff. From 0 to 0.5 (0.5 being the Nyquist frequency).
    /// \param stream           Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    template<bool REMAP = false, typename T>
    NOA_HOST void rotate3DFT(const T* input, size_t input_pitch, T* outputs, size_t output_pitch, size_t shape,
                             const float3_t* rotations, uint nb_rotations, float freq_cutoff, Stream& stream);

    /// Applies a single 3D rotation to a non-redundant centered Fourier transform.
    template<bool REMAP = false, typename T>
    NOA_HOST void rotate3DFT(const T* input, size_t input_pitch, T* output, size_t output_pitch, size_t shape,
                             float3_t rotation, float freq_cutoff, Stream& stream);
}
