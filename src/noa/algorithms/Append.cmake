# Included files for noa/algorithms:

set(NOA_ALGO_HEADERS
    algorithms/Algorithms.hpp

    algorithms/memory/Arange.hpp
    algorithms/memory/Subregion.hpp
    algorithms/memory/Iota.hpp
    algorithms/memory/Linspace.hpp
    algorithms/memory/Resize.hpp

    algorithms/fft/Remap.hpp
    algorithms/fft/Resize.hpp

    algorithms/geometry/FourierExtract.hpp
    algorithms/geometry/FourierInsertExtract.hpp
    algorithms/geometry/FourierInsertInterpolate.hpp
    algorithms/geometry/FourierInsertRasterize.hpp
    algorithms/geometry/FourierUtilities.hpp
    algorithms/geometry/PolarTransform.hpp
    algorithms/geometry/PolarTransformRFFT.hpp
    algorithms/geometry/RotationalAverage.hpp
    algorithms/geometry/Shape.hpp
    algorithms/geometry/Transform.hpp
    algorithms/geometry/TransformRFFT.hpp

    algorithms/signal/Bandpass.hpp
    algorithms/signal/CorrelationPeak.hpp
    algorithms/signal/CTF.hpp
    algorithms/signal/FSC.hpp
    algorithms/signal/PhaseShift.hpp

    )


set(NOA_HEADERS ${NOA_HEADERS} ${NOA_ALGO_HEADERS})
