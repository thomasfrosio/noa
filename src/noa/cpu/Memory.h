#pragma once

#include "noa/Exception.h"
#include "noa/cpu/PtrHost.h"

namespace Noa::Memory {
    /** Copies the underlying data of @a src into @a dst. */
    template<class T>
    void copy(PtrHost<T>& dst, const PtrHost<T>& src) {
        if (dst.size() < src.size())
            NOA_THROW("Cannot copy {} bytes into an allocated region of {} bytes.", src.size(), dst.size());
        std::memcpy(dst.get(), src.get(), src.bytes());
    }
}
