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
    template<typename Value>
    void fill(const Array<Value>& output, Value value) {
        NOA_CHECK(!output.empty(), "Empty array detected");

        const Device device{output.device()};
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::memory::set(output.share(), output.strides(), output.shape(), value, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::memory::details::is_valid_set_v<Value>) {
                cuda::memory::set(output.share(), output.strides(), output.shape(), value, stream.cuda());
            } else {
                NOA_THROW("The CUDA backend does not support this type ({})", string::human<Value>());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename Value>
    [[nodiscard]] Array<Value> fill(dim4_t shape, Value value, ArrayOption option) {
        using namespace ::noa::traits;
        if constexpr (is_data_v<Value> || is_boolX_v<Value> ||
                      is_intX_v<Value> || is_floatX_v<Value> ||
                      is_floatXX_v<Value>) {
            if (value == Value{0} && option.device().cpu() &&
                (!Device::any(Device::GPU) || (option.allocator() == Allocator::DEFAULT ||
                                               option.allocator() == Allocator::DEFAULT_ASYNC ||
                                               option.allocator() == Allocator::PITCHED))) {
                return Array<Value>(cpu::memory::PtrHost<Value>::calloc(shape.elements()),
                                    shape, shape.strides(), option);
            }
        }
        Array<Value> out(shape, option);
        fill(out, value);
        return out;
    }

    template<typename Value>
    [[nodiscard]] Array<Value> zeros(dim4_t shape, ArrayOption option) {
        return fill(shape, Value{0}, option);
    }

    template<typename Value>
    [[nodiscard]] Array<Value> ones(dim4_t shape, ArrayOption option) {
        return fill(shape, Value{1}, option);
    }

    template<typename Value>
    [[nodiscard]] Array<Value> empty(dim4_t shape, ArrayOption option) {
        return Array<Value>(shape, option);
    }

    template<typename Value>
    [[nodiscard]] Array<Value> like(const Array<Value>& array) {
        return Array<Value>(array.shape(), array.options());
    }
}

namespace noa::memory {
    template<typename Value>
    void arange(const Array<Value>& output, Value start, Value step) {
        NOA_CHECK(!output.empty(), "Empty array detected");

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::memory::arange(output.share(), output.strides(), output.shape(), start, step, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (traits::is_restricted_data_v<Value> && !traits::is_bool_v<Value>) {
                cuda::memory::arange(output.share(), output.strides(), output.shape(), start, step, stream.cuda());
            } else {
                NOA_THROW("The CUDA backend does not support this type ({})", string::human<Value>());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename Value>
    [[nodiscard]] Array<Value> arange(dim4_t shape, Value start, Value step, ArrayOption option) {
        Array<Value> out(shape, option);
        arange(out, start, step);
        return out;
    }

    template<typename Value>
    [[nodiscard]] Array<Value> arange(dim_t elements, Value start, Value step, ArrayOption option) {
        Array<Value> out(elements, option);
        arange(out, start, step);
        return out;
    }
}

namespace noa::memory {
    template<typename Value>
    Value linspace(const Array<Value>& output, Value start, Value stop, bool endpoint) {
        NOA_CHECK(!output.empty(), "Empty array detected");

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            return cpu::memory::linspace(
                    output.share(), output.strides(), output.shape(),
                    start, stop, endpoint, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (traits::is_restricted_data_v<Value> && !traits::is_bool_v<Value>) {
                return cuda::memory::linspace(
                        output.share(), output.strides(), output.shape(),
                        start, stop, endpoint, stream.cuda());
            } else {
                NOA_THROW("The CUDA backend does not support this type ({})", string::human<Value>());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename Value>
    [[nodiscard]] Array<Value> linspace(dim4_t shape, Value start, Value stop, bool endpoint, ArrayOption option) {
        Array<Value> out(shape, option);
        linspace(out, start, stop, endpoint);
        return out;
    }

    template<typename Value>
    [[nodiscard]] Array<Value> linspace(dim_t elements, Value start, Value stop, bool endpoint, ArrayOption option) {
        Array<Value> out(elements, option);
        linspace(out, start, stop, endpoint);
        return out;
    }
}

namespace noa::memory {
    template<typename Value>
    void iota(const Array<Value>& output, dim4_t tile) {
        NOA_CHECK(!output.empty(), "Empty array detected");

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            return cpu::memory::iota(
                    output.share(), output.strides(), output.shape(),
                    tile, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (traits::is_restricted_scalar_v<Value>) {
                return cuda::memory::iota(
                        output.share(), output.strides(), output.shape(),
                        tile, stream.cuda());
            } else {
                NOA_THROW("The CUDA backend does not support this type ({})", string::human<Value>());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename Value>
    [[nodiscard]] Array<Value> iota(dim4_t shape, dim4_t tile, ArrayOption option) {
        Array<Value> out(shape, option);
        iota(out, tile);
        return out;
    }

    template<typename Value>
    [[nodiscard]] Array<Value> iota(dim_t elements, dim_t tile, ArrayOption option) {
        Array<Value> out(elements, option);
        iota(out, dim4_t{1, 1, 1, tile});
        return out;
    }
}
