/**
 * @file Memory.h
 * @brief Memory interface.
 * @author Thomas - ffyr2w
 * @date 05 Jan 2021
 */
#pragma once

#define NOA_BACKEND_CUDA

#ifdef NOA_BACKEND_CUDA
#include "cuda/Memory.h"
#endif
