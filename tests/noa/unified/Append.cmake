set(TEST_UNIFIED_SOURCES
    noa/unified/TestUnifiedArray.cpp
    noa/unified/TestUnifiedDevice.cpp
    noa/unified/TestUnifiedStream.cpp
    noa/unified/TestUnifiedView.cpp

    noa/unified/TestUnifiedEwise.cpp
    noa/unified/TestUnifiedFind.cpp
    noa/unified/TestUnifiedReduce.cpp
    noa/unified/TestUnifiedSort.cpp

    #        noa/unified/signal/fft/TestUnifiedAlignment.cpp
    #        noa/unified/signal/fft/TestUnifiedMellin.cpp

    noa/unified/signal/fft/TestUnifiedCorrelate.cpp
    noa/unified/signal/fft/TestUnifiedCTF.cpp
    noa/unified/signal/fft/TestUnifiedBandpass.cpp
    noa/unified/signal/fft/TestUnifiedPhaseShift.cpp
    noa/unified/signal/fft/TestUnifiedStandardize.cpp
    noa/unified/signal/TestUnifiedMedian.cpp
    noa/unified/signal/TestUnifiedConvolve.cpp

    noa/unified/math/TestUnifiedBlas.cpp
    noa/unified/math/TestUnifiedComplex.cpp
    noa/unified/math/TestUnifiedLinalg.cpp
    noa/unified/math/TestUnifiedRandom.cpp
    noa/unified/math/TestUnifiedReduce.cpp

    noa/unified/memory/TestUnifiedCast.cpp
    noa/unified/memory/TestUnifiedCopy.cpp
    noa/unified/memory/TestUnifiedFactory.cpp
    noa/unified/memory/TestUnifiedIndex.cpp
    noa/unified/memory/TestUnifiedPermute.cpp
    noa/unified/memory/TestUnifiedResize.cpp
    noa/unified/memory/TestUnifiedSubregion.cpp

    noa/unified/fft/TestUnifiedFFT.cpp
    noa/unified/fft/TestUnifiedRemap.cpp
    noa/unified/fft/TestUnifiedResize.cpp

    #        noa/unified/geometry/fft/TestUnifiedReconstruction.cpp
    noa/unified/geometry/fft/TestUnifiedPolar.cpp
    noa/unified/geometry/fft/TestUnifiedProject.cpp
    noa/unified/geometry/fft/TestUnifiedShape.cpp
    noa/unified/geometry/fft/TestUnifiedTransform.cpp
    noa/unified/geometry/TestUnifiedPolar.cpp
    noa/unified/geometry/TestUnifiedShape.cpp
    noa/unified/geometry/TestUnifiedSymmetry.cpp
    noa/unified/geometry/TestUnifiedTransform.cpp
#
#    noa/unified/geometry/TestUnifiedTileExtraction.cpp

    )

set(TEST_SOURCES ${TEST_SOURCES} ${TEST_UNIFIED_SOURCES})
