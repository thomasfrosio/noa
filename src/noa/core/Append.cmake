set(NOA_CORE_HEADERS
    core/Config.hpp
    core/Enums.hpp
    core/Ewise.hpp
    core/Error.hpp
    core/Interfaces.hpp
    core/Interpolation.hpp
    core/Iwise.hpp
    core/Namespace.hpp
    core/Reduce.hpp
    core/Traits.hpp

    core/fft/FourierRemap.hpp
    core/fft/FourierResize.hpp
    core/fft/Frequency.hpp

    core/indexing/Bounds.hpp
    core/indexing/Layout.hpp
    core/indexing/Offset.hpp
    core/indexing/Subregion.hpp

    core/io/BinaryFile.hpp
    core/io/IO.hpp
    core/io/OS.hpp
    core/io/Stats.hpp
    core/io/TextFile.hpp
    core/io/MRCFile.hpp
    core/io/TIFFFile.hpp

    core/math/Comparison.hpp
    core/math/Constant.hpp
    core/math/Distribution.hpp
    core/math/Generic.hpp
    core/math/LeastSquare.hpp

    core/Types.hpp
    core/types/Accessor.hpp
    core/types/Complex.hpp
    core/types/Half.hpp
    core/types/Mat.hpp
    core/types/Pair.hpp
    core/types/Shape.hpp
    core/types/Span.hpp
    core/types/Tuple.hpp
    core/types/Vec.hpp

    core/utils/Adaptor.hpp
    core/utils/Atomic.hpp
    core/utils/BatchedParameter.hpp
    core/utils/ClampCast.hpp
    core/utils/Irange.hpp
    core/utils/Misc.hpp
    core/utils/SafeCast.hpp
    core/utils/ShareHandles.hpp
    core/utils/Sort.hpp
    core/utils/Threadpool.hpp
    core/utils/Timer.hpp
    core/utils/Strings.hpp
    core/utils/Zip.hpp

    core/signal/Bandpass.hpp
    core/signal/Correlation.hpp
    core/signal/CTF.hpp
    core/signal/FilterSpectrum.hpp
    core/signal/FSC.hpp
    core/signal/PhaseShift.hpp
    core/signal/StandardizeIFFT.hpp
    core/signal/Windows.hpp

    core/geometry/Project.hpp
    core/geometry/DrawShape.hpp
    core/geometry/Euler.hpp
    core/geometry/FourierExtract.hpp
    core/geometry/FourierGriddingCorrection.hpp
    core/geometry/FourierInsertExtract.hpp
    core/geometry/FourierInsertInterpolate.hpp
    core/geometry/FourierInsertRasterize.hpp
    core/geometry/FourierUtilities.hpp
    core/geometry/Polar.hpp
    core/geometry/PolarTransform.hpp
    core/geometry/PolarTransformSpectrum.hpp
    core/geometry/Prefilter.hpp
    core/geometry/Quaternion.hpp
    core/geometry/RotationalAverage.hpp
    core/geometry/Symmetry.hpp
    core/geometry/Transform.hpp
    core/geometry/TransformSpectrum.hpp
    )

set(NOA_CORE_SOURCES
    core/Enums.cpp
    core/Error.cpp
    core/geometry/Euler.cpp
    core/io/IO.cpp
    core/io/BinaryFile.cpp
    core/io/MRCFile.cpp
    core/io/TIFFFile.cpp
    )
