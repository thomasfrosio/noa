/// \file noa/cpu/Execution.h
/// \brief Execution policies.
/// \author Thomas - ffyr2w
/// \date 24 Jun 2021

#pragma once

#include "noa/Definitions.h"
#include "noa/util/traits/BaseTypes.h"

// !! This header is currently not use and is just an anticipation to SIMD, oneTBB and OpenMP support. !!
// This policies are compile time flags, i.e. functions can select which implementation to run given
// these input flags without runtime overhead.

namespace noa::execution {
    struct Policy {};

    // Generic policies, ala STL
    struct SEQ : public Policy {};
    struct PAR : public Policy {};
    struct PAR_VEC : public Policy {};
    struct VEC : public Policy {};

    // Multithreading libraries
    struct TBB : public Policy {};
    struct OpenMP : public Policy {};
}

namespace noa::traits {
    template<typename> struct proclaim_is_execution_policy : std::false_type {};
    template<> struct proclaim_is_execution_policy<noa::execution::Policy> : std::true_type {};

    template<typename T>
    using is_execution_policy = std::bool_constant<proclaim_is_execution_policy<noa::traits::remove_ref_cv_t<T>>::value>;

    /// Whether or not T is an execution policy.
    template<typename T>
    constexpr bool is_execution_policy_v = is_execution_policy<T>::value;
}
