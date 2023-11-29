#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/unified/Stream.hpp"

#include "noa/cpu/Iwise.hpp"

namespace noa {
    template<typename IwiseOp>
    void iwise_4d(const Shape4<i64>& shape, const Device& device, IwiseOp&& iwise_op) {
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            cpu_stream.enqueue(
                    noa::cpu::iwise_4d,
                    shape, std::forward<IwiseOp>(iwise_op),
                    cpu_stream.thread_limit());
        } else {

        }
    }
}
