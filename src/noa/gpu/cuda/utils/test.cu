#include "noa/core/Definitions.hpp"
#include "noa/core/math/Comparison.hpp"
#include "noa/core/math/Constant.hpp"
#include "noa/core/math/Generic.hpp"
#include "noa/core/math/LeastSquare.hpp"

#include "noa/core/types/Half.hpp"
#include "noa/core/types/Complex.hpp"
#include "noa/gpu/cuda/utils/Warp.cuh"
#include "noa/gpu/cuda/utils/Block.cuh"
//
//#include "noa/gpu/cuda/utils/EwiseUnary.cuh"
//#include "noa/gpu/cuda/utils/EwiseBinary.cuh"
//#include "noa/gpu/cuda/utils/EwiseTrinary.cuh"
//#include "noa/gpu/cuda/utils/Iwise.cuh"
//
//#include "noa/gpu/cuda/utils/ReduceUnary.cuh"
#include "noa/gpu/cuda/utils/ReduceBinary.cuh"

void test(int32_t a) {

    static_assert(noa::cuda::utils::details::is_valid_suffle_v<float>);
}
