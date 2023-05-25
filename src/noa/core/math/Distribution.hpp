#pragma once

#include "noa/core/string/Format.hpp"

namespace noa::math {
    struct distribution_t {};
    struct uniform_t : public distribution_t {};
    struct normal_t : public distribution_t {};
    struct log_normal_t : public distribution_t {};
    struct poisson_t : public distribution_t {};

    inline std::ostream& operator<<(std::ostream& os, uniform_t) { return os << "distribution::uniform"; }
    inline std::ostream& operator<<(std::ostream& os, normal_t) { return os << "distribution::normal"; }
    inline std::ostream& operator<<(std::ostream& os, log_normal_t) { return os << "distribution::log-normal"; }
    inline std::ostream& operator<<(std::ostream& os, poisson_t) { return os << "distribution::poisson"; }
}

// fmt 9.1.0 fix (Disabled automatic std::ostream insertion operator (operator<<))
namespace fmt {
    template<> struct formatter<noa::math::uniform_t> : ostream_formatter {};
    template<> struct formatter<noa::math::normal_t> : ostream_formatter {};
    template<> struct formatter<noa::math::log_normal_t> : ostream_formatter {};
    template<> struct formatter<noa::math::poisson_t> : ostream_formatter {};
}
