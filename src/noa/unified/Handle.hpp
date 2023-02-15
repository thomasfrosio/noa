#pragma once

namespace noa {
    template<typename T> class View;
    template<typename T> class Array;
}

namespace noa::details {
    template<typename T>
    struct ViewHandle {
        T* handle;
        constexpr T* get() const noexcept { return handle; }
    };

    template<typename T> auto get_handle(const Array<T>& array) { return array.share(); }
    template<typename T> auto get_handle(const View<T>& array) { return ViewHandle<T>{array.get()}; }
}
