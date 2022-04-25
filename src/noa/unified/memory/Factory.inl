#pragma once

#ifndef NOA_UNIFIED_FACTORY_
#error "This is an internal header"
#endif

#include "noa/cpu/memory/Arange.h"
#include "noa/cpu/memory/Linspace.h"
#include "noa/cpu/memory/Set.h"

#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/memory/Arange.h"
#include "noa/gpu/cuda/memory/Linspace.h"
#include "noa/gpu/cuda/memory/Set.h"
#endif

namespace noa::memory {
    template<typename T>
    void fill(const Array<T>& output, T value) {
        NOA_PROFILE_FUNCTION();
        const Device device{output.device()};
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::memory::set(output.share(), output.stride(), output.shape(), value, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::memory::set(output.share(), output.stride(), output.shape(), value, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T>
    Array<T> fill(size4_t shape, T value, ArrayOption option) {
        NOA_PROFILE_FUNCTION();
        Array<T> out{shape, option};
        fill(out, value);
        return out;
    }

    template<typename T>
    Array<T> zeros(size4_t shape, ArrayOption option) {
        return fill(shape, T{0}, option); // TODO add calloc
    }

    template<typename T>
    Array<T> ones(size4_t shape, ArrayOption option) {
        return fill(shape, T{1}, option);
    }

    template<typename T>
    Array<T> empty(size4_t shape, ArrayOption option) {
        NOA_PROFILE_FUNCTION();
        return Array<T>{shape, option};
    }
}

namespace noa::memory {
    template<typename T>
    void arange(const Array<T>& output, T start, T step) {
        NOA_PROFILE_FUNCTION();
        const Device device{output.device()};
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::memory::arange(output.share(), output.stride(), output.shape(), start, step, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::memory::arange(output.share(), output.stride(), output.shape(), start, step, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T>
    Array<T> arange(size4_t shape, T start, T step, ArrayOption option) {
        NOA_PROFILE_FUNCTION();
        Array<T> out{shape, option};
        arange(out, start, step);
        return out;
    }

    template<typename T>
    Array<T> arange(size_t elements, T start, T step, ArrayOption option) {
        NOA_PROFILE_FUNCTION();
        Array<T> out{elements, option};
        arange(out, start, step);
        return out;
    }
}

namespace noa::memory {
    template<typename T>
    void linspace(const Array<T>& output, T start, T stop, bool endpoint) {
        NOA_PROFILE_FUNCTION();
        const Device device{output.device()};
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::memory::linspace(output.share(), output.stride(), output.shape(),
                                  start, stop, endpoint, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::memory::linspace(output.share(), output.stride(), output.shape(),
                                   start, stop, endpoint, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T>
    Array<T> linspace(size4_t shape, T start, T stop, bool endpoint, ArrayOption option) {
        NOA_PROFILE_FUNCTION();
        Array<T> out{shape, option};
        linspace(out, start, stop, endpoint);
        return out;
    }

    template<typename T>
    Array<T> linspace(size_t elements, T start, T stop, bool endpoint, ArrayOption option) {
        NOA_PROFILE_FUNCTION();
        Array<T> out{elements, option};
        linspace(out, start, stop, endpoint);
        return out;
    }
}
