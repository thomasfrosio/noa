if (NOT NOA_ENABLE_UNIFIED)
    return()
endif()

set(TEST_UNIFIED_SOURCES
        noa/unified/TestUnifiedArray.cpp
        noa/unified/TestUnifiedDevice.cpp
        noa/unified/TestUnifiedStream.cpp

        noa/unified/signal/fft/TestUnifiedAlignment.cpp
        noa/unified/signal/fft/TestUnifiedCorrelate.cpp
        noa/unified/signal/fft/TestUnifiedMellin.cpp
        noa/unified/signal/fft/TestUnifiedShape.cpp
        noa/unified/signal/fft/TestUnifiedStandardize.cpp
        noa/unified/signal/TestUnifiedSignal.cpp

        noa/unified/math/TestUnifiedBlas.cpp
        noa/unified/math/TestUnifiedEwise.cpp

        noa/unified/memory/TestUnifiedCast.cpp
        noa/unified/memory/TestUnifiedCopy.cpp
        noa/unified/memory/TestUnifiedFactory.cpp
        noa/unified/memory/TestUnifiedIndex.cpp
        noa/unified/memory/TestUnifiedPermute.cpp
        noa/unified/memory/TestUnifiedResize.cpp

        noa/unified/fft/TestUnifiedFFT.cpp
        noa/unified/fft/TestUnifiedResize.cpp

#        noa/unified/geometry/fft/TestUnifiedReconstruction.cpp
#        noa/unified/geometry/fft/TestUnifiedTransform.cpp
        noa/unified/geometry/fft/TestUnifiedProject.cpp
#        noa/unified/geometry/TestUnifiedBackproject.cpp
#        noa/unified/geometry/TestUnifiedCosineStretch.cpp
        noa/unified/geometry/TestUnifiedTransform.cpp
        )

set(TEST_SOURCES ${TEST_SOURCES} ${TEST_UNIFIED_SOURCES})
