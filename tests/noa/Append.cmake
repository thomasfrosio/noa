if (NOT NOA_ENABLE_UNIFIED)
    return()
endif()

set(TEST_UNIFIED_SOURCES
        noa/TestUnifiedArray.cpp
        noa/TestUnifiedDevice.cpp
        noa/TestUnifiedStream.cpp

        noa/signal/fft/TestUnifiedMellin.cpp
        noa/signal/fft/TestUnifiedStandardize.cpp
        noa/signal/TestUnifiedSignal.cpp

        noa/math/TestUnifiedBlas.cpp
        noa/math/TestUnifiedEwise.cpp

        noa/memory/TestUnifiedCast.cpp
        noa/memory/TestUnifiedCopy.cpp
        noa/memory/TestUnifiedFactory.cpp
        noa/memory/TestUnifiedIndex.cpp
        noa/memory/TestUnifiedPermute.cpp
        noa/memory/TestUnifiedResize.cpp

        noa/fft/TestUnifiedFFT.cpp
        noa/fft/TestUnifiedResize.cpp

        noa/geometry/TestUnifiedBackproject.cpp
        noa/geometry/TestUnifiedTransform.cpp
        noa/geometry/TestUnifiedCosineStretch.cpp
        )

set(TEST_SOURCES ${TEST_SOURCES} ${TEST_UNIFIED_SOURCES})
