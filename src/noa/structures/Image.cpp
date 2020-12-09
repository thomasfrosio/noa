/**
 * @file Image.h
 * @brief Image
 * @author Thomas - ffyr2w
 * @date 02 Dec 2020
 */
#pragma once

#include "noa/Base.h"
#include "noa/util/Traits.h"


namespace Noa {
    template<typename T = int32_t, typename = std::enable_if_t<Traits::is_int_v<T>>>

    class Image {
    public:
        T a;
    private:

    };
}
