# Included files for noa/algorithms:

set(NOA_ALGO_HEADERS
        algorithms/memory/Arange.hpp
        algorithms/memory/ExtractInsert.hpp
        algorithms/memory/Iota.hpp
        algorithms/memory/Linspace.hpp
        algorithms/memory/Resize.hpp

        algorithms/math/AccurateSum.hpp

        algorithms/fft/Remap.hpp
        algorithms/fft/Resize.hpp

        algorithms/geometry/PolarTransform.hpp
        algorithms/geometry/PolarTransformRFFT.hpp
        algorithms/geometry/ProjectionsFFT.hpp
        algorithms/geometry/Transform.hpp
        algorithms/geometry/TransformRFFT.hpp
        algorithms/geometry/Utilities.hpp

#        algorithms/signal/Shape.hpp
#        algorithms/signal/FourierCorrelationPeak.hpp
#        algorithms/signal/FSC.hpp

        )


set(NOA_HEADERS ${NOA_HEADERS} ${NOA_ALGO_HEADERS})
