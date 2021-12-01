

namespace noa {
    enum Padding {
        CONTIGUOUS,
        PADDED_FFTW
        PADDED
    };

    enum Resource {
        HOST,
        PINNED,
        DEVICE,
        IMAGE
    };

    template<typename T>
    class Array {
        // (de)allocation
    };
}
