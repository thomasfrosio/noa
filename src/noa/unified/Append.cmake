# Included files for noa/unified:
set(NOA_UNIFIED_HEADERS
    unified/Allocator.hpp
    unified/Array.hpp
    unified/ArrayOption.hpp
    unified/Device.hpp
    unified/Ewise.hpp
    unified/Find.hpp
    unified/Indexing.hpp
    unified/Reduce.hpp
    unified/Sort.hpp
    unified/Stream.hpp
    unified/Texture.hpp
    unified/View.hpp

    # noa::io
    unified/io/ImageFile.hpp

    # noa::fft
    unified/fft/Factory.hpp
    unified/fft/Remap.hpp
    unified/fft/Resize.hpp
    unified/fft/Transform.hpp

    # noa::signal
    unified/signal/fft/Bandpass.hpp
    unified/signal/fft/Correlate.hpp
    unified/signal/fft/CTF.hpp
    unified/signal/fft/FSC.hpp
    unified/signal/fft/PhaseShift.hpp
    unified/signal/fft/Standardize.hpp
    unified/signal/Convolve.hpp
    unified/signal/Median.hpp
    unified/signal/Windows.hpp

    # noa::geometry
    unified/geometry/fft/Polar.hpp
    unified/geometry/fft/Project.hpp
    unified/geometry/fft/Shape.hpp
    unified/geometry/fft/Transform.hpp
    unified/geometry/Polar.hpp
    unified/geometry/Prefilter.hpp
    unified/geometry/Transform.hpp
    unified/geometry/Shape.hpp

    # noa::memory
    unified/memory/Cast.hpp
    unified/memory/Copy.hpp
    unified/memory/CopyBatches.hpp
    unified/memory/Factory.hpp
    unified/memory/Index.hpp
    unified/memory/Permute.hpp
    unified/memory/Resize.hpp
    unified/memory/Subregion.hpp

    # noa::math
    unified/math/Blas.hpp
    unified/math/Complex.hpp
    unified/math/LinAlg.hpp
    unified/math/Random.hpp
    unified/math/Reduce.hpp
    )

set(NOA_UNIFIED_SOURCES
    unified/Device.cpp
    unified/Stream.cpp
    )

set(NOA_HEADERS ${NOA_HEADERS} ${NOA_UNIFIED_HEADERS})
set(NOA_SOURCES ${NOA_SOURCES} ${NOA_UNIFIED_SOURCES})
