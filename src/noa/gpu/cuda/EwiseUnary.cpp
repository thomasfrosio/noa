#include "noa/gpu/cuda/EwiseUnary.hpp"

#include <cuda-rtc/jitify2.hpp>

// Declare the automatically generated serialized preprocessed-program:
namespace noa {
    extern const unsigned char gpu_cuda_EwiseUnary_jit[];
    extern const unsigned long long gpu_cuda_EwiseUnary_jit_size;
}

namespace {
    void deserialize_preprocessed_source_() {
        std::string_view buffer{
                reinterpret_cast<const char*>(noa::gpu_cuda_EwiseUnary_jit),
                noa::gpu_cuda_EwiseUnary_jit_size};

        const jitify2::PreprocessedProgram preprocessed_program =
                jitify2::PreprocessedProgram::deserialize(buffer);
    }
}

namespace noa::cuda {
    void ewise_unary_launch(
            std::string_view kernel,
            LaunchConfig config,
            void** arguments,
            Stream& stream
    ) {

    }

    // additional headers can be easily added at the session level!
    // add the name and preprocessed source to our map of headers
    // and pass the header name to `--pre-include` as an additional preincluded header!
    // This can be done at startup by the application, and very cheap too.

    // If new headers are added, we need the session to tell everyone to reset the cache?
    // Not even sure since the thing is compiled with specific types...
    // Anyway, regardless, the cache can also be easily reset at any point and lazyly.
}
