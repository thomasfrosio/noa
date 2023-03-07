# Included files for noa/unified:
if (NOT NOA_ENABLE_UNIFIED)
    return()
endif ()

set(NOA_UNIFIED_HEADERS
        unified/Allocator.hpp
        unified/Array.hpp
        unified/ArrayOption.hpp
        unified/Device.hpp
        unified/Indexing.hpp
        unified/Stream.hpp
        unified/Texture.hpp
        unified/View.hpp
        unified/Ewise.hpp
        unified/Sort.hpp
        unified/Find.hpp

        # noa::io
        unified/io/ImageFile.hpp

        # noa::fft
        unified/fft/Factory.h
        unified/fft/Remap.h
        unified/fft/Resize.h
        unified/fft/Transform.h

#        # noa::signal
#        unified/signal/Convolve.h
#        unified/signal/Convolve.inl
#        unified/signal/fft/Bandpass.h
#        unified/signal/fft/Bandpass.inl
#        unified/signal/fft/Correlate.h
#        unified/signal/fft/FSC.h
#        unified/signal/fft/Shape.h
#        unified/signal/fft/Shape.inl
#        unified/signal/fft/Shift.h
#        unified/signal/fft/Shift.inl
#        unified/signal/fft/Standardize.h
#        unified/signal/fft/Standardize.inl
#        unified/signal/Median.h
#        unified/signal/Median.inl
#        unified/signal/Shape.h
#        unified/signal/Shape.inl

        # noa::geometry
        unified/geometry/fft/Polar.h
        unified/geometry/fft/Project.h
        unified/geometry/fft/Transform.h
        unified/geometry/Polar.h
        unified/geometry/Prefilter.h
        unified/geometry/Transform.h

        # noa::memory
        unified/memory/Cast.hpp
        unified/memory/Copy.hpp
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
