#pragma once

#include <cufft.h>
#include <exception>

#include "noa/Definitions.h"
#include "noa/gpu/cuda/Exception.h"

namespace Noa::CUDA {
    /// Formats the cufft @a result to a human readable string.
    NOA_IH std::string toString(cufftResult_t result) {
        std::string out;
        switch (result) {
            case cufftResult_t::CUFFT_SUCCESS:
                out = "Errno::cufft_success";
                break;
            case cufftResult_t::CUFFT_INVALID_PLAN:
                out = "Errno::cufft_invalid_plan";
                break;
            case cufftResult_t::CUFFT_ALLOC_FAILED:
                out = "Errno::cufft_alloc_failed";
                break;
            case cufftResult_t::CUFFT_INVALID_TYPE:
                out = "Errno::cufft_invalid_type";
                break;
            case cufftResult_t::CUFFT_INVALID_VALUE:
                out = "Errno::cufft_invalid_value";
                break;
            case cufftResult_t::CUFFT_INTERNAL_ERROR:
                out = "Errno::cufft_internal_error";
                break;
            case cufftResult_t::CUFFT_EXEC_FAILED:
                out = "Errno::cufft_exec_failed";
                break;
            case cufftResult_t::CUFFT_SETUP_FAILED:
                out = "Errno::cufft_setup_failed";
                break;
            case cufftResult_t::CUFFT_INVALID_SIZE:
                out = "Errno::cufft_invalid_size";
                break;
            case cufftResult_t::CUFFT_UNALIGNED_DATA:
                out = "Errno::cufft_unaligned_data";
                break;
            case cufftResult_t::CUFFT_INCOMPLETE_PARAMETER_LIST:
                out = "Errno::cufft_incomplete_parameter_list";
                break;
            case cufftResult_t::CUFFT_INVALID_DEVICE:
                out = "Errno::cufft_invalid_device";
                break;
            case cufftResult_t::CUFFT_PARSE_ERROR:
                out = "Errno::cufft_parse_error";
                break;
            case cufftResult_t::CUFFT_NO_WORKSPACE:
                out = "Errno::cufft_no_workspace";
                break;
            case cufftResult_t::CUFFT_NOT_IMPLEMENTED:
                out = "Errno::cufft_not_implemented";
                break;
            case cufftResult_t::CUFFT_LICENSE_ERROR:
                out = "Errno::cufft_licence_error";
                break;
            case cufftResult_t::CUFFT_NOT_SUPPORTED:
                out = "Errno::cufft_not_supported";
                break;
            default:
                out = "Errno::cufft_unknown";
        }
        return out;
    }

    NOA_IH void throwIf(cufftResult_t result, const char* file, const char* function, int line) {
        if (result != cufftResult_t::CUFFT_SUCCESS)
        std::throw_with_nested(Noa::Exception(file, function, line, toString(result)));
    }
}
