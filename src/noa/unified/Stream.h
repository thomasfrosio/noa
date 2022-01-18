#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/types/Constants.h"

namespace noa::details {
    class StreamImp {
    public:
        NOA_HOST cudaStream_t get() const noexcept { return m_stream; }
        NOA_HOST cudaStream_t id() const noexcept { return m_stream; }
        NOA_HOST Device device() const noexcept { return m_device; }
        NOA_HOST void synchronize() const { synchronize(*this); };
        NOA_HOST bool hasCompleted() const { return hasCompleted(*this); };

    };
}

/*
 * Session::backend(Backend::CUDA);
 * Device::setCurrent(0);
 *
 * auto resource = Resource::DEVICE;
 * Stream stream(resource);
 * Array<T> a({128,128}, resource);
 * memory::set(a, T(0), stream):
 * memory::transpose(a, {0, 1, 2}, stream);
 * stream.synchronize();
 *
 */
namespace noa {


    class Stream {

    };


}
