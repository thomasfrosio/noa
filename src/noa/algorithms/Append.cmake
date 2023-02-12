# Included files for noa/algorithms:

set(NOA_ALGO_HEADERS
        algorithms/memory/Arange.hpp
        algorithms/memory/ExtractInsert.hpp
        algorithms/memory/Iota.hpp
        algorithms/memory/Linspace.hpp
        algorithms/memory/Resize.hpp

        algorithms/math/AccurateSum.hpp

#        algorithms/geometry/FourierProjections.hpp
#        algorithms/geometry/LinearTransform2D.hpp
#        algorithms/geometry/LinearTransform2DFourier.hpp
#        algorithms/geometry/LinearTransform3D.hpp
#        algorithms/geometry/LinearTransform3DFourier.hpp
#        algorithms/geometry/PolarTransform.hpp
#        algorithms/geometry/PolarTransformFourier.hpp
#        algorithms/geometry/Utilities.hpp

#        algorithms/signal/Shape.hpp
#        algorithms/signal/FourierCorrelationPeak.hpp
#        algorithms/signal/FSC.hpp

        )


set(NOA_HEADERS ${NOA_HEADERS} ${NOA_ALGO_HEADERS})
