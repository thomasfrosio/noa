#pragma once

#include "noa/base/Error.hpp"
#include "noa/base/Strings.hpp"
#include "noa/runtime/gpu/Runtime.hpp"

namespace noa::gpu {
    using noa::check; // bring the check functions to this namespace.

    #if defined(NOA_ENABLE_CUDA)
    inline auto error2string(cudaError_t result) -> std::string {
        return fmt::format("{}: {}", cudaGetErrorName(result), cudaGetErrorString(result));
    }
    template<typename T> requires std::same_as<std::decay_t<T>, cudaError_t>
    constexpr void check(T&& result, const std::source_location& location = std::source_location::current()) {
        if (result == cudaSuccess) {
            /*do nothing*/
        } else {
            panic_at_location(location, "{}: {}", cudaGetErrorName(result), cudaGetErrorString(result));
        }
    }
    #elif defined(NOA_ENABLE_HIP)
    inline auto error2string(hipError_t result) -> std::string {
        return fmt::format("{}: {}", hipGetErrorName(result), hipGetErrorString(result));
    }
    template<typename T> requires std::same_as<std::decay_t<T>, hipError_t>
    constexpr void check(T&& result, const std::source_location& location = std::source_location::current()) {
        if (result == hipSuccess) {
            /*do nothing*/
        } else {
            panic_at_location(location, "{}: {}", cudaGetErrorName(result), cudaGetErrorString(result));
        }
    }
    #elif defined(NOA_ENABLE_SYCL)
    namespace details {
        // Handle for queue.wait_and_throw(), throw_asynchronous(), and dtor.
        // Without a handler, std::terminate is used.
        inline const auto sycl_async_error_handler = [](sycl::exception_list errors) {
            for (const std::exception_ptr& p : errors) {
                try { std::rethrow_exception(p); }
                catch (const sycl::exception& e) {
                    panic("{}: {}", e.code().message(), e.what());
                }
            }
        };
    }
    #endif
}
