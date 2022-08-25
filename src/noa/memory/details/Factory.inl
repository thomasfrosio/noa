#pragma once

#ifndef NOA_UNIFIED_FACTORY_
#error "This is an internal header. Include the corresponding .h file instead"
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
        const Device device{output.device()};
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::memory::set(output.share(), output.strides(), output.shape(), value, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::memory::set(output.share(), output.strides(), output.shape(), value, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T>
    Array<T> fill(size4_t shape, T value, ArrayOption option) {
        using namespace ::noa::traits;
        if constexpr (is_data_v<T> || is_boolX_v<T> || is_intX_v<T> || is_floatX_v<T> || is_floatXX_v<T>) {
            if (value == T{0} && option.device().cpu() &&
                (!Device::any(Device::GPU) || (option.allocator() == Allocator::DEFAULT ||
                                               option.allocator() == Allocator::DEFAULT_ASYNC ||
                                               option.allocator() == Allocator::PITCHED))) {
                shared_t<T[]> ptr = cpu::memory::PtrHost<T>::calloc(shape.elements());
                return Array<T>(ptr, shape, shape.strides(), option);
            }
        }
        Array<T> out(shape, option);
        fill(out, value);
        return out;
    }

    template<typename T>
    Array<T> zeros(size4_t shape, ArrayOption option) {
        return fill(shape, T{0}, option);
    }

    template<typename T>
    Array<T> ones(size4_t shape, ArrayOption option) {
        return fill(shape, T{1}, option);
    }

    template<typename T>
    Array<T> empty(size4_t shape, ArrayOption option) {
        return Array<T>(shape, option);
    }

    template<typename T>
    Array<T> like(const Array<T>& array) {
        return Array<T>(array.shape(), array.options());
    }
}

namespace noa::memory {
    template<typename T>
    void arange(const Array<T>& output, T start, T step) {
        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::memory::arange(output.share(), output.strides(), output.shape(), start, step, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::memory::arange(output.share(), output.strides(), output.shape(), start, step, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T>
    Array<T> arange(size4_t shape, T start, T step, ArrayOption option) {
        Array<T> out(shape, option);
        arange(out, start, step);
        return out;
    }

    template<typename T>
    Array<T> arange(size_t elements, T start, T step, ArrayOption option) {
        Array<T> out(elements, option);
        arange(out, start, step);
        return out;
    }
}

namespace noa::memory {
    template<typename T>
    T linspace(const Array<T>& output, T start, T stop, bool endpoint) {
        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            return cpu::memory::linspace(output.share(), output.strides(), output.shape(),
                                         start, stop, endpoint, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::memory::linspace(output.share(), output.strides(), output.shape(),
                                          start, stop, endpoint, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T>
    Array<T> linspace(size4_t shape, T start, T stop, bool endpoint, ArrayOption option) {
        Array<T> out(shape, option);
        linspace(out, start, stop, endpoint);
        return out;
    }

    template<typename T>
    Array<T> linspace(size_t elements, T start, T stop, bool endpoint, ArrayOption option) {
        Array<T> out(elements, option);
        linspace(out, start, stop, endpoint);
        return out;
    }
}
