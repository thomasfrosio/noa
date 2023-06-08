#pragma once

#include <type_traits>
#include "noa/core/Definitions.hpp"

namespace noa::math {
    template<typename Value>
    struct ArangeOperator {
        ArangeOperator(Value start_, Value step_) : start(start_), step(step_) {}

        template<typename Index, typename = std::enable_if_t<std::is_integral_v<Index>>>
        [[nodiscard]] NOA_HD constexpr Value operator()(Index index) const noexcept {
            return start + static_cast<Value>(index) * step;
        }

    public:
        Value start;
        Value step;
    };

    template<typename Value, typename Index>
    struct LinspaceOperator {
        LinspaceOperator(Value start_, Value stop_, const Index& size_, bool endpoint_)
                : index_end(size_ - 1),
                  start(start_),
                  stop(stop_),
                  endpoint(endpoint_) {
            const auto count = size_ - static_cast<Index>(endpoint);
            const auto delta = stop - start;
            step = delta / static_cast<Value>(count);
        }

        [[nodiscard]] NOA_HD constexpr Value operator()(Index i) const noexcept {
            return endpoint && i == index_end ? stop : start + static_cast<Value>(i) * step;
        }

    public:
        Index index_end;
        Value start;
        Value step;
        Value stop;
        bool endpoint;
    };
}
