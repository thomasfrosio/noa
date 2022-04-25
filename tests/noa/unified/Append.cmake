if (NOT NOA_ENABLE_UNIFIED)
    return()
endif()

set(TEST_UNIFIED_SOURCES
        noa/unified/TestUnifiedArray.cpp
        noa/unified/TestUnifiedDevice.cpp
        noa/unified/TestUnifiedStream.cpp

#        noa/unified/fft/TestUnifiedFilters.cpp
#        noa/unified/fft/TestUnifiedRemap.cpp
#        noa/unified/fft/TestUnifiedResize.cpp
#        noa/unified/fft/TestUnifiedTransform.cpp
#
#        noa/unified/filter/TestUnifiedConvolve.cpp
#        noa/unified/filter/TestUnifiedMedian.cpp
#        noa/unified/filter/TestUnifiedShape.cpp
#
#        noa/unified/geometry/fft/TestUnifiedShift.cpp
#        noa/unified/geometry/fft/TestUnifiedTransform.cpp
#        noa/unified/geometry/TestUnifiedSymmetry.cpp
#        noa/unified/geometry/TestUnifiedTransform.cpp

        noa/unified/math/TestUnifiedEwise.cpp
        noa/unified/math/TestUnifiedReduce.cpp

        noa/unified/memory/TestUnifiedCast.cpp
        noa/unified/memory/TestUnifiedCopy.cpp
        noa/unified/memory/TestUnifiedIndex.cpp
        noa/unified/memory/TestUnifiedFactory.cpp
        noa/unified/memory/TestUnifiedResize.cpp
        noa/unified/memory/TestUnifiedTranspose.cpp
        )

set(TEST_SOURCES ${TEST_SOURCES} ${TEST_UNIFIED_SOURCES})
