#pragma once

namespace noa {
    // This is to import core types while still referring to them
    // as if they were in namespace noa.
    inline namespace types {}

    namespace traits {}
    namespace string {}
    namespace indexing {}
    namespace geometry {}
    namespace signal {}
    namespace linalg {}
    namespace fft {}

    // Internal aliases
    namespace nt = ::noa::traits;
    namespace ni = ::noa::indexing;
    namespace ns = ::noa::string;
}
