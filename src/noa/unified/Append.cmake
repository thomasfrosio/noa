# Included files for noa/unified:
set(NOA_UNIFIED_HEADERS
    unified/Allocator.hpp
    unified/Array.hpp
    unified/ArrayOption.hpp
    unified/Cast.hpp
    unified/Complex.hpp
    unified/CopyBatches.hpp
    unified/Device.hpp
    unified/Ewise.hpp
    unified/Indexing.hpp
    unified/Iwise.hpp
    unified/Random.hpp
    unified/Reduce.hpp
    unified/ReduceAxesEwise.hpp
    unified/ReduceAxesIwise.hpp
    unified/ReduceEwise.hpp
    unified/ReduceIwise.hpp
    unified/Resize.hpp
    unified/Session.hpp
    unified/Sort.hpp
    unified/Stream.hpp
    unified/Subregion.hpp
    unified/Texture.hpp
    unified/Traits.hpp
    unified/Utilities.hpp
    unified/View.hpp

    # noa::io
    unified/io/ImageFile.hpp

    # noa::fft
    unified/fft/Factory.hpp
    unified/fft/Remap.hpp
    unified/fft/Resize.hpp
    unified/fft/Transform.hpp

    # noa::signal
    unified/signal/Bandpass.hpp
    unified/signal/Correlate.hpp
    unified/signal/CTF.hpp
    unified/signal/FSC.hpp
    unified/signal/PhaseShift.hpp
    unified/signal/Standardize.hpp
    unified/signal/Convolve.hpp
    unified/signal/Median.hpp
    unified/signal/Windows.hpp

#    # noa::geometry
#    unified/geometry/fft/Polar.hpp
#    unified/geometry/fft/Project.hpp
#    unified/geometry/Polar.hpp
    unified/geometry/Prefilter.hpp
    unified/geometry/Transform.hpp
    unified/geometry/Symmetry.hpp
    unified/geometry/Shape.hpp
    )

set(NOA_UNIFIED_SOURCES
    unified/Device.cpp
    unified/Session.cpp
    unified/Stream.cpp
    )

set(NOA_HEADERS ${NOA_HEADERS} ${NOA_UNIFIED_HEADERS})
set(NOA_SOURCES ${NOA_SOURCES} ${NOA_UNIFIED_SOURCES})
