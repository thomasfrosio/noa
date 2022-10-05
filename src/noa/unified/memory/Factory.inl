#pragma once

#ifndef NOA_UNIFIED_FACTORY_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

#include "noa/cpu/memory/Arange.h"
#include "noa/cpu/memory/Linspace.h"
#include "noa/cpu/memory/Iota.h"
#include "noa/cpu/memory/Set.h"

#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/memory/Arange.h"
#include "noa/gpu/cuda/memory/Linspace.h"
#include "noa/gpu/cuda/memory/Iota.h"
#include "noa/gpu/cuda/memory/Set.h"
#endif

namespace noa::memory {
    template<typename T>
    void fill(const Array<T>& output, T value) {
        NOA_CHECK(!output.empty(), "Empty array detected");

        const Device device{output.device()};
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::memory::set(output.share(), output.strides(), output.shape(), value, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::memory::details::is_valid_set_v<T>) {
                cuda::memory::set(output.share(), output.strides(), output.shape(), value, stream.cuda());
            } else {
                NOA_THROW("The CUDA backend does not support this type ({})", string::human<T>());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T>
    Array<T> fill(dim4_t shape, T value, ArrayOption option) {
        using namespace ::noa::traits;
        if constexpr (is_data_v<T> || is_boolX_v<T> || is_intX_v<T> || is_floatX_v<T> || is_floatXX_v<T>) {
            if (value == T{0} && option.device().cpu() &&
                (!Device::any(Device::GPU) || (option.allocator() == Allocator::DEFAULT ||
                                               option.allocator() == Allocator::DEFAULT_ASYNC ||
                                               option.allocator() == Allocator::PITCHED))) {
                return Array<T>(cpu::memory::PtrHost<T>::calloc(shape.elements()),
                                shape, shape.strides(), option);
            }
        }
        Array<T> out(shape, option);
        fill(out, value);
        return out;
    }

    template<typename T>
    Array<T> zeros(dim4_t shape, ArrayOption option) {
        return fill(shape, T{0}, option);
    }

    template<typename T>
    Array<T> ones(dim4_t shape, ArrayOption option) {
        return fill(shape, T{1}, option);
    }

    template<typename T>
    Array<T> empty(dim4_t shape, ArrayOption option) {
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
        NOA_CHECK(!output.empty(), "Empty array detected");

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::memory::arange(output.share(), output.strides(), output.shape(), start, step, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (traits::is_restricted_data_v<T> && !traits::is_bool_v<T>) {
                cuda::memory::arange(output.share(), output.strides(), output.shape(), start, step, stream.cuda());
            } else {
                NOA_THROW("The CUDA backend does not support this type ({})", string::human<T>());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T>
    Array<T> arange(dim4_t shape, T start, T step, ArrayOption option) {
        Array<T> out(shape, option);
        arange(out, start, step);
        return out;
    }

    template<typename T>
    Array<T> arange(dim_t elements, T start, T step, ArrayOption option) {
        Array<T> out(elements, option);
        arange(out, start, step);
        return out;
    }
}

namespace noa::memory {
    template<typename T>
    T linspace(const Array<T>& output, T start, T stop, bool endpoint) {
        NOA_CHECK(!output.empty(), "Empty array detected");

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            return cpu::memory::linspace(output.share(), output.strides(), output.shape(),
                                         start, stop, endpoint, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (traits::is_restricted_data_v<T> && !traits::is_bool_v<T>) {
                return cuda::memory::linspace(output.share(), output.strides(), output.shape(),
                                              start, stop, endpoint, stream.cuda());
            } else {
                NOA_THROW("The CUDA backend does not support this type ({})", string::human<T>());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T>
    Array<T> linspace(dim4_t shape, T start, T stop, bool endpoint, ArrayOption option) {
        Array<T> out(shape, option);
        linspace(out, start, stop, endpoint);
        return out;
    }

    template<typename T>
    Array<T> linspace(dim_t elements, T start, T stop, bool endpoint, ArrayOption option) {
        Array<T> out(elements, option);
        linspace(out, start, stop, endpoint);
        return out;
    }
}

namespace noa::memory {
    template<typename T>
    void iota(const Array<T>& output, dim4_t tile) {
        NOA_CHECK(!output.empty(), "Empty array detected");

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            return cpu::memory::iota(output.share(), output.strides(), output.shape(),
                                     tile, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (traits::is_restricted_scalar_v<T>) {
                return cuda::memory::iota(output.share(), output.strides(), output.shape(),
                                          tile, stream.cuda());
            } else {
                NOA_THROW("The CUDA backend does not support this type ({})", string::human<T>());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T>
    Array<T> iota(dim4_t shape, dim4_t tile, ArrayOption option) {
        Array<T> out(shape, option);
        iota(out, tile);
        return out;
    }

    template<typename T>
    Array<T> iota(dim_t elements, dim_t tile, ArrayOption option) {
        Array<T> out(elements, option);
        iota(out, dim4_t{1, 1, 1, tile});
        return out;
    }
}
