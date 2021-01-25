/**
 * @file Memory.h
 * @brief Memory related function for CUDA.
 * @author Thomas - ffyr2w
 * @date 05 Jan 2021
 */
#pragma once

/** Memory related functions. */
namespace Noa::CUDA::Memory {
    enum class Resource { host, pinned, device };

    inline void* alloc(size_t bytes) noexcept {

    }

    inline void* allocPinned(size_t bytes) noexcept {

    }

    inline void free(void* pointer) noexcept {

    }

    inline void freePinned(void* pointer) noexcept {

    }

    void* copyDeviceToDevice(void* pointer, size_t bytes) {
        return nullptr;
    }

    void* copyDeviceToPinned(void* pointer, size_t bytes) {
        return nullptr;
    }

    void* copyDeviceToHost(void* pointer, size_t bytes) {
        return nullptr;
    }

    void* copyHostToDevice(void* pointer, size_t bytes) {
        return nullptr;
    }

    void* copyHostToPinned(void* pointer, size_t bytes) {
        return nullptr;
    }

    void* copyPinnedToDevice(void* pointer, size_t bytes) {
        return nullptr;
    }

    void* copyPinnedToPinned(void* pointer, size_t bytes) {
        return nullptr;
    }

    void* copyPinnedToHost(void* pointer, size_t bytes) {
        return nullptr;
    }
}
