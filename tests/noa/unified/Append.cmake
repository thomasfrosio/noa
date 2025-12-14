set(TEST_UNIFIED_SOURCES
    noa/unified/TestUnifiedArray.cpp
    noa/unified/TestUnifiedDevice.cpp
    noa/unified/TestUnifiedStream.cpp
    noa/unified/TestUnifiedView.cpp
    noa/unified/TestUnifiedFactory.cpp

    noa/unified/TestUnifiedBlas.cpp
    noa/unified/TestUnifiedCast.cpp
    noa/unified/TestUnifiedComplex.cpp
    noa/unified/TestUnifiedCopy.cpp
    noa/unified/TestUnifiedEwise.cpp
    noa/unified/TestUnifiedImageFile.cpp
    noa/unified/TestUnifiedIwise.cpp
    noa/unified/TestUnifiedPermute.cpp
    noa/unified/TestUnifiedRandom.cpp
    noa/unified/TestUnifiedReduce.cpp
    noa/unified/TestUnifiedReduceAxes.cpp
    noa/unified/TestUnifiedReduceBatch.cpp
    noa/unified/TestUnifiedReduceEwise.cpp
    noa/unified/TestUnifiedReduceIwise.cpp
    noa/unified/TestUnifiedResize.cpp
    noa/unified/TestUnifiedSort.cpp
    noa/unified/TestUnifiedSubregion.cpp

    noa/unified/fft/TestUnifiedFFT.cpp
    noa/unified/fft/TestUnifiedRemap.cpp
    noa/unified/fft/TestUnifiedResize.cpp

    noa/unified/signal/TestUnifiedBandpass.cpp
    noa/unified/signal/TestUnifiedCTF.cpp
    noa/unified/signal/TestUnifiedConvolve.cpp
    noa/unified/signal/TestUnifiedCorrelate.cpp
    noa/unified/signal/TestUnifiedMedian.cpp
    noa/unified/signal/TestUnifiedPhaseShift.cpp
    noa/unified/signal/TestUnifiedStandardize.cpp

    noa/unified/geometry/TestUnifiedFourierExtract.cpp
    noa/unified/geometry/TestUnifiedFourierInsertInterpolate.cpp
    noa/unified/geometry/TestUnifiedFourierInsertInterpolateExtract.cpp
    noa/unified/geometry/TestUnifiedFourierInsertRasterize.cpp
    noa/unified/geometry/TestUnifiedPolar.cpp
    noa/unified/geometry/TestUnifiedPolarSpectrum.cpp
    noa/unified/geometry/TestUnifiedProject.cpp
    noa/unified/geometry/TestUnifiedShape.cpp
    noa/unified/geometry/TestUnifiedSymmetry.cpp
    noa/unified/geometry/TestUnifiedTransform2d.cpp
    noa/unified/geometry/TestUnifiedTransform3d.cpp
    noa/unified/geometry/TestUnifiedTransformSpectrum.cpp

#    noa/unified/test.cpp
    )

set(TEST_SOURCES ${TEST_SOURCES} ${TEST_UNIFIED_SOURCES})
