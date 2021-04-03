#include "noa/cpu/fourier/Plan.h"

namespace Noa::Fourier::Details {
    std::mutex Mutex::mutex;

    int getThreads(size3_t shape, uint batches, int rank) {
        double geom_size;
        if (rank == 1)
            geom_size = (Math::sqrt(static_cast<double>(shape.x) * batches) + batches) / 2.;
        else
            geom_size = Math::pow(static_cast<double>(getElements(shape)), 1. / rank);
        int threads = static_cast<int>((Math::log(geom_size) / Math::log(2.) - 5.95) * 2.);
        return Math::clamp(threads, 1, getNiceShape(shape) == shape ? 8 : 4);
    }
}

namespace Noa::Fourier {
    bool Plan<float>::is_initialized{false};
    int Plan<float>::max_threads{};
    bool Plan<double>::is_initialized{false};
    int Plan<double>::max_threads{};
}
