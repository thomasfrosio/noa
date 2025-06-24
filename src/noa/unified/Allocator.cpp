#include "noa/unified/Allocator.hpp"

namespace noa::inline types {
    std::ostream& operator<<(std::ostream& os, Allocator::Enum allocator) {
        switch (allocator) {
            case Allocator::NONE:
                return os << "Allocator::NONE";
            case Allocator::DEFAULT:
                return os << "Allocator::DEFAULT";
            case Allocator::ASYNC:
                return os << "Allocator::ASYNC";
            case Allocator::PITCHED:
                return os << "Allocator::PITCHED";
            case Allocator::PITCHED_UNIFIED:
                return os << "Allocator::PITCHED_UNIFIED";
            case Allocator::PINNED:
                return os << "Allocator::PINNED";
            case Allocator::UNIFIED:
                return os << "Allocator::UNIFIED";
            case Allocator::UNIFIED_GLOBAL:
                return os << "Allocator::UNIFIED_GLOBAL";
            case Allocator::CUDA_ARRAY:
                return os << "Allocator::CUDA_ARRAY";
        }
        return os;
    }

    Allocator::Enum Allocator::parse_(std::string_view name) {
        std::string str_ = ns::to_lower(ns::trim(name));
        std::ranges::replace(str_, '-', '_');

        if (str_ == "default") {
            return DEFAULT;
        } else if (str_ == "default_async" or str_ == "async") {
            return DEFAULT_ASYNC;
        } else if (str_ == "pitched") {
            return PITCHED;
        } else if (str_ == "pitched_managed" or str_ == "pitched_unified") {
            return MANAGED_GLOBAL;
        } else if (str_ == "pinned") {
            return PINNED;
        } else if (str_ == "managed" or str_ == "unified") {
            return MANAGED;
        } else if (str_ == "managed_global" or str_ == "unified_global") {
            return MANAGED_GLOBAL;
        } else if (str_ == "cuda_array") {
            return CUDA_ARRAY;
        } else if (str_ == "none" or str_.empty()) {
            return NONE;
        } else {
            panic("{} is not a valid allocator", str_);
        }
    }
}
