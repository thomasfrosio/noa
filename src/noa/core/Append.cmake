# Included files for noa/core:

set(NOA_COMMON_HEADERS
        core/Assert.hpp
        core/Definitions.hpp
        core/Exception.hpp
        core/Logger.hpp
        core/OS.hpp
        core/Session.hpp

        core/Math.hpp
        core/math/Comparison.hpp
        core/math/Constant.hpp
        core/math/Generic.hpp
        core/math/LeastSquare.hpp

        core/utils/Any.hpp
        core/utils/Atomic.hpp
        core/utils/ClampCast.hpp
        core/utils/Indexing.hpp
        core/utils/Irange.hpp
        core/utils/Pair.hpp
        core/utils/SafeCast.hpp
        core/utils/Sort.hpp
        core/utils/Threadpool.hpp
        core/utils/Timer.hpp

        core/string/Format.hpp
        core/string/Parse.hpp
        core/string/Parse.inl
        core/string/Split.hpp

        core/Traits.hpp
        core/traits/MatrixTypes.hpp
        core/traits/Numerics.hpp
        core/traits/Shape.hpp
        core/traits/String.hpp
        core/traits/Utilities.hpp
        core/traits/VecTypes.hpp

        core/Types.hpp
        core/types/Accessor.hpp
        core/types/Complex.hpp
        core/types/Constants.hpp
        core/types/Functors.hpp
        core/types/Half.hpp
        core/types/Mat.hpp
        core/types/Mat22.hpp
        core/types/Mat23.hpp
        core/types/Mat33.hpp
        core/types/Mat34.hpp
        core/types/Mat44.hpp
        core/types/Shape.hpp
        core/types/Vec.hpp

        core/geometry/Euler.hpp
        core/geometry/Interpolate.hpp
        core/geometry/Interpolator.hpp
        core/geometry/InterpolatorValue.hpp
        core/geometry/Polar.hpp
        core/geometry/Shape.hpp
        core/geometry/Symmetry.hpp
        core/geometry/Transform.hpp
        core/geometry/Windows.hpp

        core/io/BinaryFile.hpp
        core/io/ImageFile.hpp
        core/io/IO.hpp
        core/io/IO.inl
        core/io/MRCFile.hpp
        core/io/Stats.hpp
        core/io/TextFile.hpp
        core/io/TIFFFile.hpp

#        core/memory/details/ExtractInsert.hpp
#        core/geometry/details/FourierProjections.hpp
#        core/geometry/details/LinearTransform2D.hpp
#        core/geometry/details/LinearTransform2DFourier.hpp
#        core/geometry/details/LinearTransform3D.hpp
#        core/geometry/details/LinearTransform3DFourier.hpp
#        core/geometry/details/PolarTransform.hpp
#        core/geometry/details/PolarTransformFourier.hpp
#        core/geometry/details/Utilities.hpp

#        core/signal/details/Shape.hpp
#        core/signal/details/FourierCorrelationPeak.hpp
#        core/signal/details/FSC.hpp

        )

set(NOA_COMMON_SOURCES
        core/Exception.cpp
        core/Logger.cpp
        core/Session.cpp

        core/geometry/Euler.cpp
        core/geometry/Symmetry.cpp

        core/io/IO.cpp
        core/io/BinaryFile.cpp
        core/io/MRCFile.cpp
        core/io/TIFFFile.cpp

        core/types/Constants.cpp
        )

set(NOA_HEADERS ${NOA_HEADERS} ${NOA_COMMON_HEADERS})
set(NOA_SOURCES ${NOA_SOURCES} ${NOA_COMMON_SOURCES})
