#include "noa/common/Exception.h"
#include "noa/common/Math.h"

#include "noa/gpu/cuda/math/Ewise.h"
#include "noa/gpu/cuda/math/Reduce.h"
#include "noa/gpu/cuda/memory/Copy.h"

#include "noa/gpu/cuda/util/Block.cuh"
#include "noa/gpu/cuda/util/ReduceUnary.cuh"

namespace {
    using namespace ::noa;
    using namespace ::noa::cuda;

    constexpr uint32_t ELEMENTS_PER_THREAD = 4;
    constexpr uint32_t BLOCK_SIZE = 512;

    // For the variance, it is more difficult since we need to first reduce to get the mean, and then reduce
    // once again using that same mean to get the variance (one mean per element in the reduced axis). This kernel
    // does exactly that...
    template<typename T, typename U, bool STDDEV, int32_t BLOCK_DIM_X>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void reduceVarianceRows_(AccessorRestrict<const T, 4, uint32_t> input, uint2_t shape /* YX */,
                             AccessorRestrict<U, 3, uint32_t> output, U inv_count) {
        const uint32_t tid = threadIdx.y * BLOCK_DIM_X + threadIdx.x;
        const int4_t gid{blockIdx.z,
                         blockIdx.y,
                         blockIdx.x * blockDim.y + threadIdx.y,
                         threadIdx.x};
        const bool is_valid_row = gid[2] < shape[0];

        const auto input_row = input[gid[0]][gid[1]][gid[2]];

        // Get the mean:
        T reduced = T(0);
        for (uint32_t tidx = gid[3]; tidx < shape[1] && is_valid_row; tidx += BLOCK_DIM_X) // compute entire row
            reduced += input_row[tidx]; // TODO Save input values to shared memory?

        // Each row in shared memory is reduced to one value.
        T* s_data = util::block::dynamicSharedResource<T>(); // BLOCK_SIZE elements.
        s_data[tid] = reduced;
        util::block::synchronize();
        T mean = util::block::reduceShared1D<BLOCK_DIM_X>(
                s_data + BLOCK_DIM_X * threadIdx.y, gid[3], noa::math::plus_t{});

        // Share the mean of the row to all threads within that row:
        if (gid[3] == 0)
            s_data[threadIdx.y] = mean * inv_count;
        util::block::synchronize();
        mean = s_data[threadIdx.y]; // bank-conflict...

        // Now get the variance:
        U reduced_dist2 = U(0);
        for (uint32_t tidx = gid[3]; tidx < shape[1] && is_valid_row; tidx += BLOCK_DIM_X) {  // compute entire row
            T tmp = input_row[tidx];
            if constexpr (noa::traits::is_complex_v<T>) {
                U distance = noa::math::abs(tmp - mean);
                reduced_dist2 += distance * distance;
            } else {
                U distance = tmp - mean;
                reduced_dist2 += distance * distance;
            }
        }
        U* s_data_real = reinterpret_cast<U*>(s_data);
        s_data_real[tid] = reduced_dist2;
        util::block::synchronize();
        U var = util::block::reduceShared1D<BLOCK_DIM_X>(
                s_data_real + BLOCK_DIM_X * threadIdx.y, gid[3], noa::math::plus_t{});
        if (gid[3] == 0 && is_valid_row) {
            var *= inv_count;
            if constexpr (STDDEV)
                var = noa::math::sqrt(var);
            output(gid[0], gid[1], gid[2]) = var;
        }
    }

    // Keep X to one warp to have memory coalescing, even though a half-warp should be OK as well.
    // The Y dimension of the block is where the reduction happens.
    constexpr dim3 BLOCK_SIZE_2D(32, BLOCK_SIZE / 32);
    constexpr dim3 BLOCK_WORK_SIZE_2D(BLOCK_SIZE_2D.x, BLOCK_SIZE_2D.y * ELEMENTS_PER_THREAD);

    template<typename T, typename U, bool STDDEV>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void reduceVarianceDim_(const T* __restrict__ input, uint4_t input_strides, uint2_t shape,
                            U* __restrict__ output, uint4_t output_strides, U inv_count) {
        const uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
        const int4_t gid{blockIdx.z,
                         blockIdx.y,
                         threadIdx.y, // one block in the dimension to reduce
                         blockIdx.x * BLOCK_SIZE_2D.x + threadIdx.x};
        const bool is_valid_column = gid[3] < shape[1];

        input += indexing::at(gid[0], gid[1], input_strides) + gid[3] * input_strides[3];

        // Get the sum:
        T reduced_sum = T(0);
        for (uint32_t tidy = gid[2]; tidy < shape[0] && is_valid_column; tidy += BLOCK_SIZE_2D.y) // compute entire row
            reduced_sum += input[tidy * input_strides[2]]; // TODO Save input values to shared memory?

        T* s_data = util::block::dynamicSharedResource<T>(); // BLOCK_SIZE elements.
        T* s_data_tid = s_data + tid;
        *s_data_tid = reduced_sum;
        util::block::synchronize();

        #pragma unroll
        for (uint32_t SIZE = BLOCK_SIZE_2D.y; SIZE >= 2; SIZE /= 2) {
            if (gid[2] < SIZE / 2)
                *s_data_tid += s_data_tid[BLOCK_SIZE_2D.x * SIZE / 2];
            util::block::synchronize();
        }
        T mean = s_data[threadIdx.x];
        mean *= inv_count;

        // Get the variance:
        U reduced_dist2 = U(0);
        for (uint32_t tidy = gid[2]; tidy < shape[0] && is_valid_column; tidy += BLOCK_SIZE_2D.y) {
            T tmp = input[tidy * input_strides[2]];
            if constexpr (noa::traits::is_complex_v<T>) {
                U distance = noa::math::abs(tmp - mean);
                reduced_dist2 += distance * distance;
            } else {
                U distance = tmp - mean;
                reduced_dist2 += distance * distance;
            }
        }
        U* s_data_real = reinterpret_cast<U*>(s_data);
        U* s_data_real_tid = s_data_real + tid;
        *s_data_real_tid = reduced_dist2;
        util::block::synchronize();

        #pragma unroll
        for (uint32_t SIZE = BLOCK_SIZE_2D.y; SIZE >= 2; SIZE /= 2) {
            if (gid[2] < SIZE / 2)
                *s_data_real_tid += s_data_real_tid[BLOCK_SIZE_2D.x * SIZE / 2];
            util::block::synchronize();
        }

        if (gid[2] == 0 && is_valid_column) {
            U var = *s_data_real_tid; // s_data[threadIdx.x]
            var *= inv_count;
            if constexpr (STDDEV)
                var = noa::math::sqrt(var);
            output[indexing::at(gid[0], gid[1], output_strides) + gid[3] * output_strides[3]] = var;
        }
    }

    template<bool STDDEV, typename T, typename U>
    void reduceVarianceAxis_(const char* name,
                             const T* input, dim4_t input_strides, dim4_t input_shape,
                             U* output, dim4_t output_strides, dim4_t output_shape,
                             bool4_t mask, int32_t ddof, Stream& stream) {
        if (noa::math::sum(int4_t(mask)) > 1) {
            NOA_THROW_FUNC(name, "Reducing more than one axis at a time is only supported if the reduction results in "
                                 "one value per batch, i.e. the 3 innermost dimensions are shape=1 after reduction. "
                                 "Got input:{}, output:{}, reduce:{}", input_shape, output_shape, mask);
        }

        const auto input_strides_ = safe_cast<uint4_t>(input_strides);
        const auto input_shape_ = safe_cast<uint4_t>(input_shape);
        const auto output_strides_ = safe_cast<uint4_t>(output_strides);

        if (mask[3]) {
            const U inv_count = U(1) / static_cast<U>(input_shape_[3] - ddof);
            const uint32_t block_dim_x = input_shape_[3] > 512 ? 256 : 64;
            const dim3 threads(block_dim_x, BLOCK_SIZE / block_dim_x);
            const uint32_t blocks_y = noa::math::divideUp(input_shape_[2], threads.y);
            const dim3 blocks(blocks_y, input_shape_[1], input_shape_[0]);
            const LaunchConfig config{blocks, threads, BLOCK_SIZE * sizeof(T)};

            const AccessorRestrict<const T, 4, uint32_t> input_accessor(input, input_strides_);
            const AccessorRestrict<U, 3, uint32_t> output_accessor(output, output_strides_.get());
            if (threads.x == 256) {
                stream.enqueue(name, reduceVarianceRows_<T, U, STDDEV, 256>, config,
                               input_accessor, uint2_t{input_shape_[2], input_shape_[3]},
                               output_accessor, inv_count);
            } else {
                stream.enqueue(name, reduceVarianceRows_<T, U, STDDEV, 64>, config,
                               input_accessor, uint2_t{input_shape_[2], input_shape_[3]},
                               output_accessor, inv_count);
            }

        } else if (mask[2]) {
            const U inv_count = U(1) / static_cast<U>(input_shape_[2] - ddof);
            const uint32_t blocks_x = noa::math::divideUp(input_shape_[3], BLOCK_WORK_SIZE_2D.x);
            const dim3 blocks(blocks_x, input_shape_[1], input_shape_[0]);
            const LaunchConfig config{blocks, BLOCK_SIZE_2D, BLOCK_SIZE * sizeof(T)};
            const uint2_t shape{input_shape_[2], input_shape_[3]};
            stream.enqueue(name, reduceVarianceDim_<T, U, STDDEV>, config,
                           input, input_strides_, shape, output, output_strides_, inv_count);

        } else if (mask[1]) {
            const U inv_count = U(1) / static_cast<U>(input_shape_[1] - ddof);
            const uint32_t blocks_x = noa::math::divideUp(input_shape_[3], BLOCK_WORK_SIZE_2D.x);
            const dim3 blocks(blocks_x, input_shape_[2], input_shape_[0]);
            const LaunchConfig config{blocks, BLOCK_SIZE_2D, BLOCK_SIZE * sizeof(T)};
            const uint4_t i_strides{input_strides_[0], input_strides_[2], input_strides_[1], input_strides_[3]};
            const uint4_t o_strides{output_strides_[0], output_strides_[2], output_strides_[1], output_strides_[3]};
            const uint2_t shape{input_shape_[1], input_shape_[3]};
            stream.enqueue(name, reduceVarianceDim_<T, U, STDDEV>, config,
                           input, i_strides, shape, output, o_strides, inv_count);

        } else if (mask[0]) {
            const U inv_count = U(1) / static_cast<U>(input_shape_[0] - ddof);
            const uint32_t blocks_x = noa::math::divideUp(input_shape_[3], BLOCK_WORK_SIZE_2D.x);
            const dim3 blocks(blocks_x, input_shape_[2], input_shape_[1]);
            const LaunchConfig config{blocks, BLOCK_SIZE_2D, BLOCK_SIZE * sizeof(T)};
            const uint4_t i_strides{input_strides_[1], input_strides_[2], input_strides_[0], input_strides_[3]};
            const uint4_t o_strides{output_strides_[1], output_strides_[2], output_strides_[0], output_strides_[3]};
            const uint2_t shape{input_shape_[0], input_shape_[3]};
            stream.enqueue(name, reduceVarianceDim_<T, U, STDDEV>, config,
                           input, i_strides, shape, output, o_strides, inv_count);
        }
    }

    bool4_t getMask_(const char* func, dim4_t input_shape, dim4_t output_shape) {
        const bool4_t mask(input_shape != output_shape);
        if (any(mask && (output_shape != 1))) {
            NOA_THROW_FUNC(func, "Dimensions should match the input shape, or be 1, indicating the dimension should be "
                                 "reduced to one element. Got input:{}, output:{}", input_shape, output_shape);
        }
        return mask;
    }
}

namespace noa::cuda::math {
    template<typename T, typename U, typename>
    void var(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
             const shared_t<U[]>& output, dim4_t output_strides, dim4_t output_shape,
             int32_t ddof, Stream& stream) {
        const char* name = "math::var";
        const bool4_t mask = getMask_(name, input_shape, output_shape);
        const bool4_t is_or_should_reduce(output_shape == 1 || mask);
        using real_t = noa::traits::value_type_t<T>;

        if (!any(mask)) {
            if constexpr (noa::traits::is_complex_v<T>)
                math::ewise(input, input_strides, output, output_strides, output_shape, noa::math::abs_t{}, stream);
            else
                memory::copy(input, input_strides, output, output_strides, output_shape, stream);

        } else if (is_or_should_reduce[1] && is_or_should_reduce[2] && is_or_should_reduce[3]) {
            util::reduceVar<false>(name, input.get(), input_strides, input_shape,
                                   output.get(), output_strides[0], ddof, is_or_should_reduce[0], true, stream);
            stream.attach(input, output);
        } else {
            reduceVarianceAxis_<false>(name,
                                       input.get(), input_strides, input_shape,
                                       output.get(), output_strides, output_shape,
                                       mask, ddof, stream);
            stream.attach(input, output);
        }
    }

    template<typename T, typename U, typename>
    void std(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
             const shared_t<U[]>& output, dim4_t output_strides, dim4_t output_shape,
             int32_t ddof, Stream& stream) {
        const char* name = "math::std";
        const bool4_t mask = getMask_(name, input_shape, output_shape);
        const bool4_t is_or_should_reduce(output_shape == 1 || mask);
        using real_t = noa::traits::value_type_t<T>;

        if (!any(mask)) {
            if constexpr (noa::traits::is_complex_v<T>)
                math::ewise(input, input_strides, output, output_strides, output_shape, noa::math::abs_t{}, stream);
            else
                memory::copy(input, input_strides, output, output_strides, output_shape, stream);

        } else if (is_or_should_reduce[1] && is_or_should_reduce[2] && is_or_should_reduce[3]) {
            util::reduceVar<true>(name, input.get(), input_strides, input_shape,
                                  output.get(), output_strides[0], ddof, is_or_should_reduce[0], true, stream);
            stream.attach(input, output);
        } else {
            reduceVarianceAxis_<true>(name,
                                      input.get(), input_strides, input_shape,
                                      output.get(), output_strides, output_shape,
                                      mask, ddof, stream);
            stream.attach(input, output);
        }
    }

    #define NOA_INSTANTIATE_VAR_(T,U)                                                                                       \
    template void var<T,U,void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<U[]>&, dim4_t, dim4_t, int32_t, Stream&);  \
    template void std<T,U,void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<U[]>&, dim4_t, dim4_t, int32_t, Stream&)

    NOA_INSTANTIATE_VAR_(float, float);
    NOA_INSTANTIATE_VAR_(double, double);
    NOA_INSTANTIATE_VAR_(cfloat_t, float);
    NOA_INSTANTIATE_VAR_(cdouble_t, double);
}
