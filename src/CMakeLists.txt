# Collect library files.
set(NOA_HEADERS
#    noa/Base.hpp
#    noa/Core.hpp
#    noa/FFT.hpp
#    noa/IO.hpp
#    noa/Signal.hpp
#    noa/Types.hpp
#    noa/Utils.hpp
#    noa/Xform.hpp
    noa/Version.hpp
)
set(NOA_SOURCES)

include(${PROJECT_SOURCE_DIR}/src/noa/base/Append.cmake)
include(${PROJECT_SOURCE_DIR}/src/noa/runtime/Append.cmake)
include(${PROJECT_SOURCE_DIR}/src/noa/fft/Append.cmake)
include(${PROJECT_SOURCE_DIR}/src/noa/io/Append.cmake)
#include(${PROJECT_SOURCE_DIR}/src/noa/signal/Append.cmake)
#include(${PROJECT_SOURCE_DIR}/src/noa/xform/Append.cmake)

# Create the target.
include(${PROJECT_SOURCE_DIR}/cmake/targets/noa.cmake)
