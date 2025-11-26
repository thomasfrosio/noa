#pragma once

namespace noa {
    inline namespace types {}

    namespace details {}
    namespace traits {}
    namespace string {}
    namespace indexing {}
    namespace geometry {}
    namespace signal {}
    namespace linalg {}
    namespace fft {}

    // Internal aliases
    namespace nt = ::noa::traits;
    namespace nd = ::noa::details;
    namespace ni = ::noa::indexing;
    namespace nf = ::noa::fft;
    namespace ns = ::noa::signal;
    namespace ng = ::noa::geometry;
}
