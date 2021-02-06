#pragma once

#include <cufft.h>

#include <exception>

#include "noa/Define.h"
#include "noa/Exception.h"
#include "noa/util/string/Format.h"

namespace Noa::CUDA::FFT {
    NOA_IH const char* toString(cufftResult_t result) {
        switch (result) {
            case cufftResult_t::CUFFT_SUCCESS:
                return "Errno::cufft::success";
            case cufftResult_t::CUFFT_INVALID_PLAN:
                return "Errno::cufft_invalid_plan";
            case cufftResult_t::CUFFT_ALLOC_FAILED:
                return "Errno::cufft_alloc_failed";
            case cufftResult_t::CUFFT_INVALID_TYPE:
                return "Errno::cufft_invalid_type";
            case cufftResult_t::CUFFT_INVALID_VALUE:
                return "Errno::cufft_invalid_value";
            case cufftResult_t::CUFFT_INTERNAL_ERROR:
                return "Errno::cufft_internal_error";
            case cufftResult_t::CUFFT_EXEC_FAILED:
                return "Errno::cufft_exec_failed";
            case cufftResult_t::CUFFT_SETUP_FAILED:
                return "Errno::cufft_setup_failed";
            case cufftResult_t::CUFFT_INVALID_SIZE:
                return "Errno::cufft_invalid_size";
            case cufftResult_t::CUFFT_UNALIGNED_DATA:
                return "Errno::cufft_unaligned_data";
            case cufftResult_t::CUFFT_INCOMPLETE_PARAMETER_LIST:
                return "Errno::cufft_incomplete_parameter_list";
            case cufftResult_t::CUFFT_INVALID_DEVICE:
                return "Errno::cufft_invalid_device";
            case cufftResult_t::CUFFT_PARSE_ERROR:
                return "Errno::cufft_parse_error";
            case cufftResult_t::CUFFT_NO_WORKSPACE:
                return "Errno::cufft_no_workspace";
            case cufftResult_t::CUFFT_NOT_IMPLEMENTED:
                return "Errno::cufft_not_implemented";
            case cufftResult_t::CUFFT_LICENSE_ERROR:
                return "Errno::cufft_licence_error";
            case cufftResult_t::CUFFT_NOT_SUPPORTED:
                return "Errno::cufft_not_supported";
            default:
                return "Errno::cufft::?";
        }
    }

    /**
     * Throw a nested @c Noa::Exception if cudaError_t =! cudaSuccess.
     * @note    As the result of this function being defined in NOA::CUDA, the macro NOA_THROW_IF, defined in
     *          noa/Exception.h, will now call this function when used within NOA::CUDA.
     */
    NOA_IH void throwIf(cufftResult_t result, const char* file, const char* function, const int line) {
        if (result != cufftResult_t::CUFFT_SUCCESS)
            std::throw_with_nested(Noa::Exception(file, function, line, toString(result)));
    }
}
