#include "noa/cpu/fourier/Plan.h"

namespace Noa::Fourier::Details {
    std::mutex Mutex::mutex;
}

namespace Noa::Fourier {
    bool Plan<float>::is_initialized{false};
    int Plan<float>::max_threads{};
    bool Plan<double>::is_initialized{false};
    int Plan<double>::max_threads{};
}
