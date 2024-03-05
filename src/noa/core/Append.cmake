set(NOA_CORE_HEADERS
    core/Arange.hpp
    core/Config.hpp
    core/Enums.hpp
    core/Exception.hpp
    core/Iota.hpp
    core/Linspace.hpp
    core/Namespace.hpp
    core/Operators.hpp
    core/Resize.hpp
    core/Subregion.hpp
    core/Traits.hpp

    core/fft/FourierRemap.hpp
    core/fft/FourierResize.hpp
    core/fft/Frequency.hpp
    core/fft/RemapInterface.hpp

    core/geometry/Euler.hpp
    core/geometry/FourierExtract.hpp
    core/geometry/FourierGriddingCorrection.hpp
    core/geometry/FourierInsertExtract.hpp
    core/geometry/FourierInsertInterpolate.hpp
    core/geometry/FourierInsertRasterize.hpp
    core/geometry/FourierPolar.hpp
    core/geometry/FourierTransform.hpp
    core/geometry/FourierUtilities.hpp
    core/geometry/Interpolate.hpp
    core/geometry/Interpolator.hpp
    core/geometry/Polar.hpp
    core/geometry/Prefilter.hpp
    core/geometry/Quaternion.hpp
    core/geometry/RotationalAverage.hpp
    core/geometry/Shape.hpp
    core/geometry/Symmetry.hpp
    core/geometry/Transform.hpp

    core/Indexing.hpp
    core/indexing/Offset.hpp
    core/indexing/Layout.hpp
    core/indexing/Subregion.hpp

    core/io/BinaryFile.hpp
    core/io/ImageFile.hpp
    core/io/IO.hpp
    core/io/MRCFile.hpp
    core/io/OS.hpp
    core/io/Stats.hpp
    core/io/TextFile.hpp
    core/io/TIFFFile.hpp

    core/Math.hpp
    core/math/Comparison.hpp
    core/math/Constant.hpp
    core/math/Distribution.hpp
    core/math/Generic.hpp
    core/math/LeastSquare.hpp

    core/signal/Bandpass.hpp
    core/signal/CorrelationPeak.hpp
    core/signal/CTF.hpp
    core/signal/FSC.hpp
    core/signal/PhaseShift.hpp
    core/signal/Windows.hpp

    core/String.hpp
    core/string/Format.hpp
    core/string/Parse.hpp
    core/string/Reflect.hpp
    core/string/Split.hpp

    core/Types.hpp
    core/types/Accessor.hpp
    core/types/Complex.hpp
    core/types/Half.hpp
    core/types/Mat.hpp
    core/types/Mat22.hpp
    core/types/Mat23.hpp
    core/types/Mat33.hpp
    core/types/Mat34.hpp
    core/types/Mat44.hpp
    core/types/Pair.hpp
    core/types/Shape.hpp
    core/types/Span.hpp
    core/types/Tuple.hpp
    core/types/Vec.hpp

    core/utils/Adaptor.hpp
    core/utils/Atomic.hpp
    core/utils/ClampCast.hpp
    core/utils/Interfaces.hpp
    core/utils/Irange.hpp
    core/utils/Misc.hpp
    core/utils/SafeCast.hpp
    core/utils/ShareHandles.hpp
    core/utils/Sort.hpp
    core/utils/Threadpool.hpp
    core/utils/Timer.hpp
    )

set(NOA_CORE_SOURCES
    core/Exception.cpp
    core/geometry/Euler.cpp
    core/io/IO.cpp
    core/io/BinaryFile.cpp
    core/io/MRCFile.cpp
    core/io/TIFFFile.cpp
    )
