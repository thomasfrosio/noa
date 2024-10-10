# Included files for noa/unified:
set(NOA_UNIFIED_HEADERS
    unified/Allocator.hpp
    unified/Array.hpp
    unified/ArrayOption.hpp
    unified/Blas.hpp
    unified/Cast.hpp
    unified/Complex.hpp
    unified/CopyBatches.hpp
    unified/Device.hpp
    unified/Ewise.hpp
    unified/Factory.hpp
    unified/Indexing.hpp
    unified/Interpolation.hpp
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
    unified/signal/Convolve.hpp
    unified/signal/Correlate.hpp
    unified/signal/CTF.hpp
    unified/signal/FSC.hpp
    unified/signal/MedianFilter.hpp
    unified/signal/PhaseShift.hpp
    unified/signal/Standardize.hpp
    unified/signal/Windows.hpp

    # noa::geometry
    unified/geometry/CubicBSplinePrefilter.hpp
    unified/geometry/DrawShape.hpp
    unified/geometry/FourierProject.hpp
    unified/geometry/PolarTransform.hpp
    unified/geometry/PolarTransformSpectrum.hpp
    unified/geometry/RotationalAverage.hpp
    unified/geometry/Symmetry.hpp
    unified/geometry/Transform.hpp
    unified/geometry/TransformSpectrum.hpp
    )

set(NOA_UNIFIED_SOURCES
    unified/Allocator.cpp
    unified/Device.cpp
    unified/Session.cpp
    unified/Stream.cpp
    )

set(NOA_HEADERS ${NOA_HEADERS} ${NOA_UNIFIED_HEADERS})
set(NOA_SOURCES ${NOA_SOURCES} ${NOA_UNIFIED_SOURCES})
