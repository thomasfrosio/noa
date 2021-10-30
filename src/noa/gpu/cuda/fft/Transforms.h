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

    /// Computes the forward R2C transform encoded in \p plan.
    /// \param[in] input    On the \b device. Should match the config encoded in \p plan.
    /// \param[out] output  On the \b device. Should match the config encoded in \p plan.
    /// \param[in] plan     Existing plan.
    /// \note This functions is asynchronous with respect to the host and may return before completion.
    ///       All operations are enqueued to the stream associated with the \p plan.
    NOA_IH void r2c(float* input, cfloat_t* output, const Plan<float>& plan) {
        NOA_THROW_IF(cufftExecR2C(plan.get(), input, reinterpret_cast<cufftComplex*>(output)));
    }
    NOA_IH void r2c(double* input, cdouble_t* output, const Plan<double>& plan) {
        NOA_THROW_IF(cufftExecD2Z(plan.get(), input, reinterpret_cast<cufftDoubleComplex*>(output)));
    }

    /// Computes the backward C2R transform encoded in \p plan.
    /// \param[in] input    On the \b device. Should match the config encoded in \p plan.
    /// \param[out] output  On the \b device. Should match the config encoded in \p plan.
    /// \param[in] plan     Existing plan.
    /// \note This functions is asynchronous with respect to the host and may return before completion.
    ///       All operations are enqueued to the stream associated with the \p plan.
    NOA_IH void c2r(cfloat_t* input, float* output, const Plan<float>& plan) {
        NOA_THROW_IF(cufftExecC2R(plan.get(), reinterpret_cast<cufftComplex*>(input), output));
    }
    NOA_IH void c2r(cdouble_t* input, double* output, const Plan<double>& plan) {
        NOA_THROW_IF(cufftExecZ2D(plan.get(), reinterpret_cast<cufftDoubleComplex*>(input), output));
    }

    /// Computes the C2C transform using the \p plan.
    /// \param[in] input    On the \b device. Should match the config encoded in \p plan.
    /// \param[out] output  On the \b device. Should match the config encoded in \p plan.
    /// \param[in] plan     Existing plan.
    /// \param sign         Sign of the exponent in the formula that defines the Fourier transform.
    ///                     It can be −1 (\c fft::FORWARD) or +1 (\c fft::BACKWARD).
    /// \note This functions is asynchronous with respect to the host and may return before completion.
    ///       All operations are enqueued to the stream associated with the \p plan.
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
    /// Computes the forward R2C transform.
    /// \param[in] inputs       On the \b device. Real space array(s).
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b device. Non-redundant non-centered FFT(s).
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches          Number of contiguous batches.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note \p outputs can be equal to \p inputs. See \c fft::Plan<float> for more details.
    /// \note This function is asynchronous relative to the host and may return before completion.
    NOA_IH void r2c(float* inputs, size_t input_pitch, cfloat_t* outputs, size_t output_pitch,
                    size3_t shape, size_t batches, Stream& stream) {
        Plan<float> fast_plan(fft::R2C, shape, batches, input_pitch, output_pitch, stream);
        r2c(inputs, outputs, fast_plan);
    }
    NOA_IH void r2c(double* inputs,  size_t input_pitch, cdouble_t* outputs, size_t output_pitch,
                    size3_t shape, size_t batches, Stream& stream) {
        Plan<double> fast_plan(fft::R2C, shape, batches, input_pitch, output_pitch, stream);
        r2c(inputs, outputs, fast_plan);
    }

    /// Computes the forward R2C transform.
    /// \param[in] inputs       On the \b device. Real space array(s).
    /// \param[out] outputs     On the \b device. Non-redundant non-centered FFT.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches          Number of contiguous batches.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note \p outputs can be equal to \p inputs. See \c fft::Plan<float> for more details.
    /// \note This function is asynchronous relative to the host and may return before completion.
    NOA_IH void r2c(float* inputs, cfloat_t* outputs, size3_t shape, size_t batches, Stream& stream) {
        Plan<float> fast_plan(fft::R2C, shape, batches, stream);
        r2c(inputs, outputs, fast_plan);
    }
    NOA_IH void r2c(double* inputs, cdouble_t* outputs, size3_t shape, size_t batches, Stream& stream) {
        Plan<double> fast_plan(fft::R2C, shape, batches, stream);
        r2c(inputs, outputs, fast_plan);
    }

    /// Computes the backward C2R transform.
    /// \param[in] inputs       On the \b device. Non-redundant non-centered FFT(s).
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b device. Real space array(s).
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches          Number of contiguous batches.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note \p outputs can be equal to \p inputs. See \c fft::Plan<float> for more details.
    /// \note This function is asynchronous relative to the host and may return before completion.
    NOA_IH void c2r(cfloat_t* inputs, size_t input_pitch, float* outputs, size_t output_pitch,
                    size3_t shape, size_t batches, Stream& stream) {
        Plan<float> fast_plan(fft::C2R, shape, batches, input_pitch, output_pitch, stream);
        fast_plan.setStream(stream);
        c2r(inputs, outputs, fast_plan);
    }
    NOA_IH void c2r(cdouble_t* inputs, size_t input_pitch, double* outputs, size_t output_pitch,
                    size3_t shape, size_t batches, Stream& stream) {
        Plan<double> fast_plan(fft::C2R, shape, batches, input_pitch, output_pitch, stream);
        fast_plan.setStream(stream);
        c2r(inputs, outputs, fast_plan);
    }

    /// Computes the backward C2R transform.
    /// \param[in] inputs       On the \b device. Non-redundant non-centered FFT.
    /// \param[out] outputs     On the \b device. Real space array(s).
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches          Number of contiguous batches.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note \p outputs can be equal to \p inputs. See \c fft::Plan<float> for more details.
    /// \note This function is asynchronous relative to the host and may return before completion.
    NOA_IH void c2r(cfloat_t* inputs, float* outputs, size3_t shape, size_t batches, Stream& stream) {
        Plan<float> fast_plan(fft::C2R, shape, batches, stream);
        fast_plan.setStream(stream);
        c2r(inputs, outputs, fast_plan);
    }
    NOA_IH void c2r(cdouble_t* inputs, double* outputs, size3_t shape, size_t batches, Stream& stream) {
        Plan<double> fast_plan(fft::C2R, shape, batches, stream);
        fast_plan.setStream(stream);
        c2r(inputs, outputs, fast_plan);
    }

    /// Computes the C2C transform.
    /// \param[in] inputs       On the \b device.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b device.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches          Number of contiguous batches.
    /// \param sign             Sign of the exponent in the formula that defines the Fourier transform.
    ///                         It can be −1 (\c fft::FORWARD) or +1 (\c fft::BACKWARD).
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note \p outputs can be equal to \p inputs. See \c fft::Plan<float> for more details.
    /// \note This function is asynchronous relative to the host and may return before completion.
    NOA_IH void c2c(cfloat_t* inputs, size_t input_pitch,
                    cfloat_t* outputs, size_t output_pitch,
                    size3_t shape, size_t batches, Sign sign, Stream& stream) {
        Plan<float> fast_plan(fft::C2C, shape, batches, input_pitch, output_pitch, stream);
        c2c(inputs, outputs, fast_plan, sign);
    }
    NOA_IH void c2c(cdouble_t* inputs, size_t input_pitch,
                    cdouble_t* outputs, size_t output_pitch,
                    size3_t shape, size_t batches, Sign sign, Stream& stream) {
        Plan<double> fast_plan(fft::C2C, shape, batches, input_pitch, output_pitch, stream);
        c2c(inputs, outputs, fast_plan, sign);
    }

    /// Computes the C2C transform.
    /// \param[in] inputs       On the \b device.
    /// \param[out] outputs     On the \b device.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches          Number of contiguous batches.
    /// \param sign             Sign of the exponent in the formula that defines the Fourier transform.
    ///                         It can be −1 (\c fft::FORWARD) or +1 (\c fft::BACKWARD).
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note \p outputs can be equal to \p inputs. See \c fft::Plan<float> for more details.
    /// \note This function is asynchronous relative to the host and may return before completion.
    NOA_IH void c2c(cfloat_t* inputs, cfloat_t* outputs, size3_t shape, size_t batches, Sign sign, Stream& stream) {
        Plan<float> fast_plan(fft::C2C, shape, batches, stream);
        c2c(inputs, outputs, fast_plan, sign);
    }
    NOA_IH void c2c(cdouble_t* inputs, cdouble_t* outputs, size3_t shape, size_t batches, Sign sign, Stream& stream) {
        Plan<double> fast_plan(fft::C2C, shape, batches, stream);
        c2c(inputs, outputs, fast_plan, sign);
    }
}
