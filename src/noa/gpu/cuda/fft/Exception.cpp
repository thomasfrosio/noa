#include "noa/gpu/cuda/fft/Exception.hpp"

namespace noa::cuda {
    std::string to_string(cufftResult_t result) {
        switch (result) {
            case cufftResult_t::CUFFT_SUCCESS:
                return "CUFFT_SUCCESS";
            case cufftResult_t::CUFFT_INVALID_PLAN:
                return "CUFFT_INVALID_PLAN";
            case cufftResult_t::CUFFT_ALLOC_FAILED:
                return "CUFFT_ALLOC_FAILED";
            case cufftResult_t::CUFFT_INVALID_TYPE:
                return "CUFFT_INVALID_TYPE";
            case cufftResult_t::CUFFT_INVALID_VALUE:
                return "CUFFT_INVALID_VALUE";
            case cufftResult_t::CUFFT_INTERNAL_ERROR:
                return "CUFFT_INTERNAL_ERROR";
            case cufftResult_t::CUFFT_EXEC_FAILED:
                return "CUFFT_EXEC_FAILED";
            case cufftResult_t::CUFFT_SETUP_FAILED:
                return "CUFFT_SETUP_FAILED";
            case cufftResult_t::CUFFT_INVALID_SIZE:
                return "CUFFT_INVALID_SIZE";
            case cufftResult_t::CUFFT_UNALIGNED_DATA:
                return "CUFFT_UNALIGNED_DATA";
            case cufftResult_t::CUFFT_INCOMPLETE_PARAMETER_LIST:
                return "CUFFT_INCOMPLETE_PARAMETER_LIST";
            case cufftResult_t::CUFFT_INVALID_DEVICE:
                return "CUFFT_INVALID_DEVICE";
            case cufftResult_t::CUFFT_PARSE_ERROR:
                return "CUFFT_PARSE_ERROR";
            case cufftResult_t::CUFFT_NO_WORKSPACE:
                return "CUFFT_NO_WORKSPACE";
            case cufftResult_t::CUFFT_NOT_IMPLEMENTED:
                return "CUFFT_NOT_IMPLEMENTED";
            case cufftResult_t::CUFFT_LICENSE_ERROR:
                return "CUFFT_LICENSE_ERROR";
            case cufftResult_t::CUFFT_NOT_SUPPORTED:
                return "CUFFT_NOT_SUPPORTED";
        }
        return {};
    }
}
