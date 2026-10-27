#pragma once

#include "noa/runtime/core/Border.hpp"

namespace noa::signal {
    struct CompileOptions {
        /// Border modes to generate code for.
        BorderSet borders{BorderSet::all()};

        /// Whether CPU code should be generated.
        bool generate_cpu{true};

        /// Whether GPU code should be generated.
        bool generate_gpu{true};
    };
}
