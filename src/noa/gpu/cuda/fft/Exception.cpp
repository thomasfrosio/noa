#include "noa/gpu/cuda/fft/Exception.h"

namespace noa::cuda {
    std::string toString(cufftResult_t result) {
        switch (result) {
            case cufftResult_t::CUFFT_SUCCESS:
                return "ERROR::CUFFT_SUCCESS";
            case cufftResult_t::CUFFT_INVALID_PLAN:
                return "ERROR::CUFFT_INVALID_PLAN";
            case cufftResult_t::CUFFT_ALLOC_FAILED:
                return "ERROR::CUFFT_ALLOC_FAILED";
            case cufftResult_t::CUFFT_INVALID_TYPE:
                return "ERROR::CUFFT_INVALID_TYPE";
            case cufftResult_t::CUFFT_INVALID_VALUE:
                return "ERROR::CUFFT_INVALID_VALUE";
            case cufftResult_t::CUFFT_INTERNAL_ERROR:
                return "ERROR::CUFFT_INTERNAL_ERROR";
            case cufftResult_t::CUFFT_EXEC_FAILED:
                return "ERROR::CUFFT_EXEC_FAILED";
            case cufftResult_t::CUFFT_SETUP_FAILED:
                return "ERROR::CUFFT_SETUP_FAILED";
            case cufftResult_t::CUFFT_INVALID_SIZE:
                return "ERROR::CUFFT_INVALID_SIZE";
            case cufftResult_t::CUFFT_UNALIGNED_DATA:
                return "ERROR::CUFFT_UNALIGNED_DATA";
            case cufftResult_t::CUFFT_INCOMPLETE_PARAMETER_LIST:
                return "ERROR::CUFFT_INCOMPLETE_PARAMETER_LIST";
            case cufftResult_t::CUFFT_INVALID_DEVICE:
                return "ERROR::CUFFT_INVALID_DEVICE";
            case cufftResult_t::CUFFT_PARSE_ERROR:
                return "ERROR::CUFFT_PARSE_ERROR";
            case cufftResult_t::CUFFT_NO_WORKSPACE:
                return "ERROR::CUFFT_NO_WORKSPACE";
            case cufftResult_t::CUFFT_NOT_IMPLEMENTED:
                return "ERROR::CUFFT_NOT_IMPLEMENTED";
            case cufftResult_t::CUFFT_LICENSE_ERROR:
                return "ERROR::CUFFT_LICENSE_ERROR";
            case cufftResult_t::CUFFT_NOT_SUPPORTED:
                return "ERROR::CUFFT_NOT_SUPPORTED";
            default:
                return "ERROR::CUFFT_UNKNOWN";
        }
    }
}
