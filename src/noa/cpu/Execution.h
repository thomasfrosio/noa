/// \file noa/cpu/Execution.h
/// \brief Execution policies.
/// \author Thomas - ffyr2w
/// \date 24 Jun 2021

#pragma once

#include "noa/Definitions.h"
#include "noa/util/traits/BaseTypes.h"

#if NOA_BUILD_STL_EXECUTION
    #include <execution>
#endif

// This is very much like https://en.cppreference.com/w/cpp/header/execution
// and simply acts as an interface that can be compiled by all compilers.

namespace noa::execution {

    struct SEQ {
        #if NOA_BUILD_STL_EXECUTION
        static constexpr std::execution::sequenced_policy std_policy;
        #endif
    };

    class PAR {
        #if NOA_BUILD_STL_EXECUTION
        static constexpr std::execution::parallel_policy std_policy;
        #endif
    };

    class PAR_VEC {
        #if NOA_BUILD_STL_EXECUTION
        static constexpr std::execution::parallel_unsequenced_policy std_policy;
        #endif
    };

    class VEC {
        #if NOA_BUILD_STL_EXECUTION
        static constexpr std::execution::unsequenced_policy std_policy;
        #endif
    };
}

namespace noa::traits {
    template<typename> struct proclaim_is_execution_policy : std::false_type {};

#if NOA_BUILD_STL_EXECUTION
    template<> struct proclaim_is_execution_policy<std::execution::sequenced_policy> : std::true_type {};
    template<> struct proclaim_is_execution_policy<std::execution::parallel_policy> : std::true_type {};
    template<> struct proclaim_is_execution_policy<std::execution::parallel_unsequenced_policy> : std::true_type {};
    template<> struct proclaim_is_execution_policy<std::execution::unsequenced_policy> : std::true_type {};
#endif

    template<> struct proclaim_is_execution_policy<noa::execution::SEQ> : std::true_type {};
    template<> struct proclaim_is_execution_policy<noa::execution::PAR> : std::true_type {};
    template<> struct proclaim_is_execution_policy<noa::execution::PAR_VEC> : std::true_type {};
    template<> struct proclaim_is_execution_policy<noa::execution::VEC> : std::true_type {};

    template<typename T>
    using is_execution_policy = std::bool_constant<proclaim_is_execution_policy<noa::traits::remove_ref_cv_t<T>>::value>;

    /// Whether or not T is an execution policy.
    template<typename T>
    constexpr bool is_execution_policy_v = is_execution_policy<T>::value;

}
