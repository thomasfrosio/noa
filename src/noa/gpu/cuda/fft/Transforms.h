/// \file noa/gpu/cuda/fft/Transforms.h
/// \brief Fast Fourier Transforms.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include <cufft.h>

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"
#include "noa/gpu/cuda/fft/Plan.h"
#include "noa/gpu/cuda/fft/Exception.h"

namespace noa::cuda::fft {
    enum Sign : int { FORWARD = CUFFT_FORWARD, BACKWARD = CUFFT_INVERSE };

    /// Computes the R2C transform (i.e forward transform) using the \a plan.
    /// \param[in] input    On the \b device. Should match the layout (shape, etc.) used to create \a plan.
    /// \param[out] output  On the \b device. Should match the layout (shape, etc.) used to create \a plan.
    /// \param plan         Existing plan.
    /// \note This functions is asynchronous with respect to the host.
    ///       All operations are enqueued to the stream associated with the \a plan.
    NOA_IH void r2c(float* input, cfloat_t* output, const Plan<float>& plan) {
        NOA_THROW_IF(cufftExecR2C(plan.get(), input, reinterpret_cast<cufftComplex*>(output)));
    }
    NOA_IH void r2c(double* input, cdouble_t* output, const Plan<double>& plan) {
        NOA_THROW_IF(cufftExecD2Z(plan.get(), input, reinterpret_cast<cufftDoubleComplex*>(output)));
    }

    /// Computes the C2R transform (i.e backward transform) using the \a plan.
    /// \param[in] input    On the \b device. Should match the output used to create \a plan.
    /// \param[out] output  On the \b device. Should match the output used to create \a plan.
    /// \param plan         Existing plan.
    /// \note This functions is asynchronous with respect to the host.
    ///       All operations are enqueued to the stream associated with the \a plan.
    NOA_IH void c2r(cfloat_t* input, float* output, const Plan<float>& plan) {
        NOA_THROW_IF(cufftExecC2R(plan.get(), reinterpret_cast<cufftComplex*>(input), output));
    }
    NOA_IH void c2r(cdouble_t* input, double* output, const Plan<double>& plan) {
        NOA_THROW_IF(cufftExecZ2D(plan.get(), reinterpret_cast<cufftDoubleComplex*>(input), output));
    }

    /// Computes the C2C transform using the \a plan.
    /// \param[in] input    On the \b device. Should match the output used to create \a plan.
    /// \param[out] output  On the \b device. Should match the output used to create \a plan.
    /// \param plan         Existing plan.
    /// \param sign         Sign of the exponent in the formula that defines the Fourier transform.
    ///                     It can be −1 (\c FORWARD) or +1 (\c BACKWARD).
    /// \note This functions is asynchronous with respect to the host.
    ///       All operations are enqueued to the stream associated with the \a plan.
    NOA_IH void c2c(cfloat_t* input, cfloat_t* output, const Plan<float>& plan, Sign sign) {
        NOA_THROW_IF(cufftExecC2C(plan.get(),
                                  reinterpret_cast<cufftComplex*>(input),
                                  reinterpret_cast<cufftComplex*>(output), sign));
    }
    NOA_IH void c2c(cdouble_t* input, cdouble_t* output, const Plan<double>& plan, Sign sign) {
        NOA_THROW_IF(cufftExecZ2Z(plan.get(),
                                  reinterpret_cast<cufftDoubleComplex*>(input),
                                  reinterpret_cast<cufftDoubleComplex*>(output), sign));
    }
}

// -- "One time" transforms -- //
namespace noa::cuda::fft {
    /// Computes the R2C transform (i.e forward transform).
    /// \param[in] inputs       On the \b device. Real space array.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b device. Non-redundant, non-centered transform. Can be equal to \p inputs.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches          Batch size, in number of contiguous batches.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \see cuda::fft::Plan<float> for more details.
    NOA_IH void r2c(float* inputs, size_t input_pitch, cfloat_t* outputs, size_t output_pitch,
                    size3_t shape, uint batches, Stream& stream) {
        Plan<float> fast_plan(shape, batches, input_pitch, output_pitch, fft::R2C, stream);
        r2c(inputs, outputs, fast_plan);
    }
    NOA_IH void r2c(double* inputs,  size_t input_pitch, cdouble_t* outputs, size_t output_pitch,
                    size3_t shape, uint batches, Stream& stream) {
        Plan<double> fast_plan(shape, batches, input_pitch, output_pitch, fft::R2C, stream);
        r2c(inputs, outputs, fast_plan);
    }

    /// Computes the R2C transform (i.e forward transform).
    /// \param[in] inputs       On the \b device. Real space array.
    /// \param[out] outputs     On the \b device. Non-redundant, non-centered transform. Can be equal to \p inputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches          Batch size, in number of contiguous batches.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \see cuda::fft::Plan<float> for more details.
    NOA_IH void r2c(float* inputs, cfloat_t* outputs, size3_t shape, uint batches, Stream& stream) {
        Plan<float> fast_plan(shape, batches, fft::R2C, stream);
        r2c(inputs, outputs, fast_plan);
    }
    NOA_IH void r2c(double* inputs, cdouble_t* outputs, size3_t shape, uint batches, Stream& stream) {
        Plan<double> fast_plan(shape, batches, fft::R2C, stream);
        r2c(inputs, outputs, fast_plan);
    }

    /// Computes the C2R transform (i.e backward transform).
    /// \param[in] inputs       On the \b device. Non-redundant, non-centered transform.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b device. Real space array. Can be equal to \p inputs.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches          Batch size, in number of contiguous batches.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \see cuda::fft::Plan<float> for more details.
    NOA_IH void c2r(cfloat_t* inputs, size_t input_pitch, float* outputs, size_t output_pitch,
                    size3_t shape, uint batches, Stream& stream) {
        Plan<float> fast_plan(shape, batches, input_pitch, output_pitch, fft::C2R, stream);
        fast_plan.setStream(stream);
        c2r(inputs, outputs, fast_plan);
    }
    NOA_IH void c2r(cdouble_t* inputs, size_t input_pitch, double* outputs, size_t output_pitch,
                    size3_t shape, uint batches, Stream& stream) {
        Plan<double> fast_plan(shape, batches, input_pitch, output_pitch, fft::C2R, stream);
        fast_plan.setStream(stream);
        c2r(inputs, outputs, fast_plan);
    }

    /// Computes the C2R transform (i.e backward transform).
    /// \param[in] inputs       On the \b device. Non-redundant, non-centered transform.
    /// \param[out] outputs     On the \b device. Real space array. Can be equal to \p inputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches          Batch size, in number of contiguous batches.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \see cuda::fft::Plan<float> for more details.
    NOA_IH void c2r(cfloat_t* inputs, float* outputs, size3_t shape, uint batches, Stream& stream) {
        Plan<float> fast_plan(shape, batches, fft::C2R, stream);
        fast_plan.setStream(stream);
        c2r(inputs, outputs, fast_plan);
    }
    NOA_IH void c2r(cdouble_t* inputs, double* outputs, size3_t shape, uint batches, Stream& stream) {
        Plan<double> fast_plan(shape, batches, fft::C2R, stream);
        fast_plan.setStream(stream);
        c2r(inputs, outputs, fast_plan);
    }

    /// Computes the C2C transform, either forward or backward depending on \a sign.
    /// \param[in] inputs       On the \b device.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b device. Can be equal to \p inputs.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches          Batch size, in number of contiguous batches.
    /// \param sign             Sign of the exponent in the formula that defines the Fourier transform.
    ///                         It can be −1 (\c FORWARD) or +1 (\c BACKWARD).
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \see cuda::fft::Plan<float> for more details.
    NOA_IH void c2c(cfloat_t* inputs, size_t input_pitch,
                    cfloat_t* outputs, size_t output_pitch,
                    size3_t shape, uint batches, Sign sign, Stream& stream) {
        Plan<float> fast_plan(shape, batches, input_pitch, output_pitch, fft::C2C, stream);
        c2c(inputs, outputs, fast_plan, sign);
    }
    NOA_IH void c2c(cdouble_t* inputs, size_t input_pitch,
                    cdouble_t* outputs, size_t output_pitch,
                    size3_t shape, uint batches, Sign sign, Stream& stream) {
        Plan<double> fast_plan(shape, batches, input_pitch, output_pitch, fft::C2C, stream);
        c2c(inputs, outputs, fast_plan, sign);
    }

    /// Computes the C2C transform, either forward or backward depending on \a sign.
    /// \param[in] inputs       On the \b device.
    /// \param[out] outputs     On the \b device. Can be equal to \p inputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches          Batch size, in number of contiguous batches.
    /// \param sign             Sign of the exponent in the formula that defines the Fourier transform.
    ///                         It can be −1 (\c FORWARD) or +1 (\c BACKWARD).
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \see cuda::fft::Plan<float> for more details.
    NOA_IH void c2c(cfloat_t* inputs, cfloat_t* outputs, size3_t shape, uint batches, Sign sign, Stream& stream) {
        Plan<float> fast_plan(shape, batches, fft::C2C, stream);
        c2c(inputs, outputs, fast_plan, sign);
    }
    NOA_IH void c2c(cdouble_t* inputs, cdouble_t* outputs, size3_t shape, uint batches, Sign sign, Stream& stream) {
        Plan<double> fast_plan(shape, batches, fft::C2C, stream);
        c2c(inputs, outputs, fast_plan, sign);
    }
}
