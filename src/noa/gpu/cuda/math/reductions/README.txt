Implementation of noa/gpu/cuda/math/Reductions.h
================================================

The reductions are very much similar to each other. See Min_Max_SumMean.cu for more details.

These reduction kernels are adapted from different sources, but mostly come from:
    https://github.com/NVIDIA/cuda-samples/tree/master/Samples/reduction
    https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

TODO We don't use __shfl_down_sync, __reduce_add_sync or cooperative_groups reduce. Although these could be explored.
TODO The reductions often use a second kernel launch as synchronization barrier.
     1) With cooperative groups, grid synchronization is possible but forces CUDA 9.0 minimum.
     2) An atomic operation could directly add the reduction of each block to global memory.
        Hardware atomicAdd for double is __CUDA_ARCH__ >= 6, otherwise it should be fine.
     3) Warp reduction: ballot? shfl_down?
