#include "noa/unified/Allocator.hpp"
#ifdef NOA_ENABLE_CUDA
#   include "noa/gpu/cuda/Pointers.hpp"
#endif

namespace noa::inline types {
    void Allocator::validate(const void* ptr, const Device& device) {
        check(value != CUDA_ARRAY, "CUDA arrays are not supported by the Array class. Use a Texture instead");
        check(value != NONE or ptr == nullptr, "{} is for nullptr only", NONE);

        if (device.is_cpu()) {
            if (not Device::is_any_gpu())
                return; // Everything is allocated using AllocatorHeap
            #ifdef NOA_ENABLE_CUDA
            const cudaPointerAttributes attr = noa::cuda::pointer_attributes(ptr);
            switch (attr.type) {
                case cudaMemoryTypeUnregistered:
                    check(is_any(DEFAULT, DEFAULT_ASYNC, PITCHED),
                          "Attempting to create a CPU array with {} from a CPU-only (CUDA unregistered) memory region",
                          value);
                    break;
                case cudaMemoryTypeHost:
                    check(value == PINNED,
                          "Attempting to create a CPU array with {} from a pinned memory region",
                          value);
                    break;
                case cudaMemoryTypeDevice:
                    panic("Attempting to create an CPU array that points to a GPU-only memory region");
                case cudaMemoryTypeManaged:
                    check(is_any(DEFAULT, DEFAULT_ASYNC, PITCHED, MANAGED, MANAGED_GLOBAL, PITCHED_MANAGED),
                          "Attempting to create an CPU array with {} from a (CUDA) managed pointer",
                          value);
                    break;
            }
            #endif
        } else {
            #ifdef NOA_ENABLE_CUDA
            const cudaPointerAttributes attr = noa::cuda::pointer_attributes(ptr);
            switch (attr.type) {
                case cudaMemoryTypeUnregistered:
                    panic("Attempting to create GPU array from a CPU-only (CUDA unregistered) memory region");
                case cudaMemoryTypeHost:
                    check(value == PINNED,
                          "Attempting to create a GPU array with {} from a pinned memory region",
                          value);
                    break;
                case cudaMemoryTypeDevice:
                    check(attr.device == device.id(),
                          "Attempting to create a GPU array with a device ID of {} from a memory region "
                          "located on another device (ID={})", device.id(), attr.device);
                    break;
                case cudaMemoryTypeManaged:
                    check(is_any(DEFAULT, DEFAULT_ASYNC, PITCHED, MANAGED, MANAGED_GLOBAL, PITCHED_MANAGED),
                          "Attempting to create a GPU array with {} from a (CUDA) managed pointer",
                          value);
                    break;
            }
            #endif
        }
    }

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
        std::string str_ = nd::to_lower(nd::trim(name));
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
