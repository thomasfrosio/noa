#pragma once

#include <cufft.h>

#include <exception>

#include "noa/Definitions.h"
#include "noa/Exception.h"

namespace Noa {
    NOA_IH std::ostream& operator<<(std::ostream& os, cufftResult_t result) {
        switch (result) {
            case cufftResult_t::CUFFT_SUCCESS:
                os << "Errno::cufft::success";
                break;
            case cufftResult_t::CUFFT_INVALID_PLAN:
                os << "Errno::cufft_invalid_plan";
                break;
            case cufftResult_t::CUFFT_ALLOC_FAILED:
                os << "Errno::cufft_alloc_failed";
                break;
            case cufftResult_t::CUFFT_INVALID_TYPE:
                os << "Errno::cufft_invalid_type";
                break;
            case cufftResult_t::CUFFT_INVALID_VALUE:
                os << "Errno::cufft_invalid_value";
                break;
            case cufftResult_t::CUFFT_INTERNAL_ERROR:
                os << "Errno::cufft_internal_error";
                break;
            case cufftResult_t::CUFFT_EXEC_FAILED:
                os << "Errno::cufft_exec_failed";
                break;
            case cufftResult_t::CUFFT_SETUP_FAILED:
                os << "Errno::cufft_setup_failed";
                break;
            case cufftResult_t::CUFFT_INVALID_SIZE:
                os << "Errno::cufft_invalid_size";
                break;
            case cufftResult_t::CUFFT_UNALIGNED_DATA:
                os << "Errno::cufft_unaligned_data";
                break;
            case cufftResult_t::CUFFT_INCOMPLETE_PARAMETER_LIST:
                os << "Errno::cufft_incomplete_parameter_list";
                break;
            case cufftResult_t::CUFFT_INVALID_DEVICE:
                os << "Errno::cufft_invalid_device";
                break;
            case cufftResult_t::CUFFT_PARSE_ERROR:
                os << "Errno::cufft_parse_error";
                break;
            case cufftResult_t::CUFFT_NO_WORKSPACE:
                os << "Errno::cufft_no_workspace";
                break;
            case cufftResult_t::CUFFT_NOT_IMPLEMENTED:
                os << "Errno::cufft_not_implemented";
                break;
            case cufftResult_t::CUFFT_LICENSE_ERROR:
                os << "Errno::cufft_licence_error";
                break;
            case cufftResult_t::CUFFT_NOT_SUPPORTED:
                os << "Errno::cufft_not_supported";
                break;
            default:
                os << "Errno::cufft_unknown";
        }
        return os;
    }

    /**
     * Throw a nested @c Noa::Exception if cudaError_t =! cudaSuccess.
     * @note    As the result of this function being defined in NOA::CUDA::Fourier, the macro NOA_THROW_IF, defined in
     *          noa/Exception.h, will now call this function when used within NOA::CUDA.
     */
    template<>
    NOA_IH void throwIf<cufftResult_t>(cufftResult_t result, const char* file, const char* function, const int line) {
        if (result != cufftResult_t::CUFFT_SUCCESS)
            std::throw_with_nested(Noa::Exception(file, function, line, result));
    }
}
