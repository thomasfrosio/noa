/// \file noa/gpu/cuda/fft/Exception.h
/// \brief Expansion of noa/gpu/cuda/Exception.h for cufft.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include <cufft.h>
#include <exception>

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Exception.h"

namespace noa::cuda {
    /// Formats the cufft \a result to a human readable string.
    NOA_IH std::string toString(cufftResult_t result) {
        std::string out;
        switch (result) {
            case cufftResult_t::CUFFT_SUCCESS:
                out = "ERROR::cufft_success";
                break;
            case cufftResult_t::CUFFT_INVALID_PLAN:
                out = "ERROR::cufft_invalid_plan";
                break;
            case cufftResult_t::CUFFT_ALLOC_FAILED:
                out = "ERROR::cufft_alloc_failed";
                break;
            case cufftResult_t::CUFFT_INVALID_TYPE:
                out = "ERROR::cufft_invalid_type";
                break;
            case cufftResult_t::CUFFT_INVALID_VALUE:
                out = "ERROR::cufft_invalid_value";
                break;
            case cufftResult_t::CUFFT_INTERNAL_ERROR:
                out = "ERROR::cufft_internal_error";
                break;
            case cufftResult_t::CUFFT_EXEC_FAILED:
                out = "ERROR::cufft_exec_failed";
                break;
            case cufftResult_t::CUFFT_SETUP_FAILED:
                out = "ERROR::cufft_setup_failed";
                break;
            case cufftResult_t::CUFFT_INVALID_SIZE:
                out = "ERROR::cufft_invalid_size";
                break;
            case cufftResult_t::CUFFT_UNALIGNED_DATA:
                out = "ERROR::cufft_unaligned_data";
                break;
            case cufftResult_t::CUFFT_INCOMPLETE_PARAMETER_LIST:
                out = "ERROR::cufft_incomplete_parameter_list";
                break;
            case cufftResult_t::CUFFT_INVALID_DEVICE:
                out = "ERROR::cufft_invalid_device";
                break;
            case cufftResult_t::CUFFT_PARSE_ERROR:
                out = "ERROR::cufft_parse_error";
                break;
            case cufftResult_t::CUFFT_NO_WORKSPACE:
                out = "ERROR::cufft_no_workspace";
                break;
            case cufftResult_t::CUFFT_NOT_IMPLEMENTED:
                out = "ERROR::cufft_not_implemented";
                break;
            case cufftResult_t::CUFFT_LICENSE_ERROR:
                out = "ERROR::cufft_licence_error";
                break;
            case cufftResult_t::CUFFT_NOT_SUPPORTED:
                out = "ERROR::cufft_not_supported";
                break;
            default:
                out = "ERROR::cufft_unknown";
        }
        return out;
    }

    NOA_IH void throwIf(cufftResult_t result, const char* file, const char* function, int line) {
        if (result != cufftResult_t::CUFFT_SUCCESS)
        std::throw_with_nested(noa::Exception(file, function, line, toString(result)));
    }
}
