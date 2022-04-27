if (NOT NOA_ENABLE_UNIFIED)
    return()
endif()

set(TEST_UNIFIED_SOURCES
        noa/unified/TestUnifiedArray.cpp
        noa/unified/TestUnifiedDevice.cpp
        noa/unified/TestUnifiedStream.cpp

        noa/unified/math/fft/TestUnifiedStandardize.cpp
        noa/unified/math/TestUnifiedEwise.cpp

        noa/unified/memory/TestUnifiedCast.cpp
        noa/unified/memory/TestUnifiedCopy.cpp
        noa/unified/memory/TestUnifiedIndex.cpp
        noa/unified/memory/TestUnifiedFactory.cpp
        noa/unified/memory/TestUnifiedResize.cpp
        noa/unified/memory/TestUnifiedTranspose.cpp
        )

set(TEST_SOURCES ${TEST_SOURCES} ${TEST_UNIFIED_SOURCES})
