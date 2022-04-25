#pragma once
#ifndef NOA_UNIFIED_INDEX_
#error "This is an internal header"
#endif

#include "noa/cpu/memory/Index.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/memory/Index.h"
#endif

#include "noa/unified/Array.h"

namespace noa::memory {
    template<typename T>
    void extract(const Array<T>& input, const Array<T>& subregions, const Array<int4_t>& origins,
                 BorderMode border_mode, T border_value) {
        NOA_CHECK(subregions.device() == input.device(),
                  "The input and subregion arrays must be on the same device, but got input:{} and subregion:{}",
                  subregions.device(), input.device());
        NOA_CHECK(origins.shape().ndim() == 1 && origins.shape()[3] == subregions.shape()[0],
                  "The indexes should be specified as a row vector of shape {} but got {}",
                  int4_t{1, 1, 1, subregions.shape()[0]}, origins.shape());
        NOA_CHECK(subregions.get() != input.get(), "The subregion(s) and the output arrays should not overlap");

        const Device device(subregions.device());
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(origins.dereferencable(), "The origins should be accessible to the CPU");
            cpu::memory::extract<T>(input.share(), input.stride(), input.shape(),
                                    subregions.share(), subregions.stride(), subregions.shape(),
                                    origins.share(), border_mode, border_value, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::memory::extract<T>(input.share(), input.stride(), input.shape(),
                                     subregions.share(), subregions.stride(), subregions.shape(),
                                     origins.share(), border_mode, border_value, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T>
    void insert(const Array<T>& subregions, const Array<T>& output, const Array<int4_t>& origins) {
        NOA_CHECK(subregions.device() == output.device(),
                  "The output and subregion arrays must be on the same device, but got output:{} and subregion:{}",
                  subregions.device(), output.device());
        NOA_CHECK(origins.shape().ndim() == 1 && origins.shape()[3] == subregions.shape()[0],
                  "The indexes should be specified as a row vector of shape {} but got {}",
                  int4_t{1, 1, 1, subregions.shape()[0]}, origins.shape());
        NOA_CHECK(subregions.get() != output.get(), "The subregion(s) and the output arrays should not overlap");

        const Device device(subregions.device());
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(origins.dereferencable(), "The origins should be accessible to the CPU");
            cpu::memory::insert<T>(subregions.share(), subregions.stride(), subregions.shape(),
                                   output.share(), output.stride(), output.shape(),
                                   origins.share(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::memory::insert<T>(subregions.share(), subregions.stride(), subregions.shape(),
                                    output.share(), output.stride(), output.shape(),
                                    origins.share(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    size4_t atlasLayout(size4_t subregion_shape, int4_t* origins) {
        return cpu::memory::atlasLayout(subregion_shape, origins);
    }
}

namespace noa::memory {
    template<typename value_t, typename index_t, typename T, typename U, typename UnaryOp>
    Extracted<value_t, index_t> extract(const Array<T>& input, const Array<U>& lhs, UnaryOp unary_op,
                                        bool extract_values, bool extract_indexes) {
        NOA_CHECK(all(input.shape() == lhs.shape()),
                  "The input arrays should have the same shape, but got input:{} and lhs:{}",
                  input.shape(), lhs.shape());
        NOA_CHECK(input.device() == lhs.device(),
                  "The input arrays should be on the same device, but got input:{} and lhs:{}",
                  input.device(), lhs.device());

        Extracted<value_t, index_t> out;
        const Device device(input.device());
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            auto extracted = cpu::memory::extract<value_t, index_t>(
                    input.share(), input.stride(), lhs.share(), lhs.stride(), input.shape(),
                    unary_op, extract_values, extract_indexes, stream.cpu());
            out.values = Array<value_t>{extracted.values, extracted.count, ArrayOption{input.device}};
            out.indexes = Array<index_t>{extracted.indexes, extracted.count, ArrayOption{input.device}};
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::memory::details::is_valid_extract_unary_v<T, U, value_t, index_t, UnaryOp>) {
                auto extracted = cuda::memory::extract<value_t, index_t>(
                        input.share(), input.stride(), lhs.share(), lhs.stride(), input.shape(),
                        unary_op, extract_values, extract_indexes, stream.cuda());
                const ArrayOption option{input.device(), Allocator::DEFAULT_ASYNC};
                out.values = Array<value_t>{extracted.values, extracted.count, option};
                out.indexes = Array<index_t>{extracted.indexes, extracted.count, option};
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See noa::cuda::memory::extract(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
        return out;
    }

    template<typename value_t, typename index_t, typename T, typename U, typename V, typename BinaryOp>
    Extracted<value_t, index_t> extract(const Array<T>& input, const Array<U>& lhs, V rhs, BinaryOp binary_op,
                                        bool extract_values, bool extract_indexes) {
        NOA_CHECK(all(input.shape() == lhs.shape()),
                  "The input arrays should have the same shape, but got input:{} and lhs:{}",
                  input.shape(), lhs.shape());
        NOA_CHECK(input.device() == lhs.device(),
                  "The input arrays should be on the same device, but got input:{} and lhs:{}",
                  input.device(), lhs.device());

        Extracted<value_t, index_t> out;
        const Device device = lhs.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            auto extracted = cpu::memory::extract<value_t, index_t>(
                    input.share(), input.stride(), lhs.share(), lhs.stride(), rhs, lhs.shape(),
                    binary_op, extract_values, extract_indexes, stream.cpu());
            out.values = Array<value_t>{extracted.values, extracted.count, ArrayOption{input.device()}};
            out.indexes = Array<index_t>{extracted.indexes, extracted.count, ArrayOption{input.device()}};
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::memory::details::is_valid_extract_binary_v<T, U, T, value_t, index_t, BinaryOp>) {
                auto extracted = cuda::memory::extract<value_t, index_t>(
                        input.share(), input.stride(), lhs.share(), lhs.stride(), static_cast<T>(rhs), lhs.shape(),
                        binary_op, extract_values, extract_indexes, stream.cuda());
                const ArrayOption option{input.device(), Allocator::DEFAULT_ASYNC};
                out.values = Array<value_t>{extracted.values, extracted.count, option};
                out.indexes = Array<index_t>{extracted.indexes, extracted.count, option};
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See noa::cuda::memory::extract(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
        return out;
    }

    template<typename value_t, typename index_t, typename T, typename U, typename V, typename BinaryOp>
    Extracted<value_t, index_t> extract(const Array<T>& input, U lhs, const Array<V>& rhs, BinaryOp binary_op,
                                        bool extract_values, bool extract_indexes) {
        NOA_CHECK(all(input.shape() == rhs.shape()),
                  "The input arrays should have the same shape, but got input:{} and rhs:{}",
                  input.shape(), rhs.shape());
        NOA_CHECK(input.device() == rhs.device(),
                  "The input arrays should be on the same device, but got input:{} and rhs:{}",
                  input.device(), rhs.device());

        Extracted<value_t, index_t> out;
        const Device device = rhs.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            auto extracted = cpu::memory::extract<value_t, index_t>(
                    input.share(), input.stride(), lhs, rhs.share(), rhs.stride(), rhs.shape(),
                    binary_op, extract_values, extract_indexes, stream.cpu());
            out.values = Array<value_t>{extracted.values, extracted.count, ArrayOption{input.device()}};
            out.indexes = Array<index_t>{extracted.indexes, extracted.count, ArrayOption{input.device()}};
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::memory::details::is_valid_extract_binary_v<T, U, T, value_t, index_t, BinaryOp>) {
                auto extracted = cuda::memory::extract<value_t, index_t>(
                        input.share(), input.stride(), static_cast<T>(lhs), rhs.share(), rhs.stride(), rhs.shape(),
                        binary_op, extract_values, extract_indexes, stream.cuda());
                const ArrayOption option{input.device(), Allocator::DEFAULT_ASYNC};
                out.values = Array<value_t>{extracted.values, extracted.count, option};
                out.indexes = Array<index_t>{extracted.indexes, extracted.count, option};
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See noa::cuda::memory::extract(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
        return out;
    }

    template<typename value_t, typename index_t, typename T, typename U, typename V, typename BinaryOp>
    Extracted<value_t, index_t> extract(const Array<T>& input, const Array<U>& lhs, const Array<V>& rhs,
                                        BinaryOp binary_op, bool extract_values, bool extract_indexes) {
        NOA_CHECK(all(input.shape() == lhs.shape()) && all(input.shape() == rhs.shape()),
                  "The input arrays should have the same shape, but got input:{}, lhs:{} and rhs:{}",
                  input.shape(), lhs.shape(), rhs.shape());
        NOA_CHECK(input.device() == lhs.device() && input.device() == rhs.device(),
                  "The input arrays should be on the same device, but got input:{}, lhs:{} and rhs:{}",
                  input.device(), lhs.device(), rhs.device());

        Extracted<value_t, index_t> out;
        const Device device = lhs.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            auto extracted = cpu::memory::extract<value_t, index_t>(
                    input.share(), input.stride(), lhs.share(), lhs.stride(),
                    rhs.share(), rhs.stride(), rhs.shape(), binary_op,
                    extract_values, extract_indexes, stream.cpu());
            out.values = Array<value_t>{extracted.values, extracted.count, ArrayOption{input.device()}};
            out.indexes = Array<index_t>{extracted.indexes, extracted.count, ArrayOption{input.device()}};
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::memory::details::is_valid_extract_binary_v<T, U, V, value_t, index_t, BinaryOp>) {
                auto extracted = cuda::memory::extract<value_t, index_t>(
                        input.share(), input.stride(), lhs.share(), lhs.stride(),
                        rhs.share(), rhs.stride(), rhs.shape(), binary_op,
                        extract_values, extract_indexes, stream.cuda());
                const ArrayOption option{input.device(), Allocator::DEFAULT_ASYNC};
                out.values = Array<value_t>{extracted.values, extracted.count, option};
                out.indexes = Array<index_t>{extracted.indexes, extracted.count, option};
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See noa::cuda::memory::extract(...) for more details");
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
        return out;
    }

    template<typename value_t, typename index_t, typename T>
    void insert(const Extracted<value_t, index_t>& extracted, const Array<T>& output) {
        NOA_CHECK(extracted.values.device() == extracted.indexes.device() &&
                  extracted.values.device() == output.device(),
                  "The input and output arrays should be on the same device, "
                  "but got values:{}, indexes:{} and output:{}",
                  extracted.values.device(), extracted.indexes.device(), output.device());
        NOA_CHECK(all(extracted.values.shape() == extracted.indexes.shape()) &&
                  extracted.indexes.shape().ndim() == 1,
                  "The sequence of values and indexes should be two row vectors of the same size, "
                  "but got values:{} and indexes:{}", extracted.values.shape(), extracted.indexes.shape());

        const size_t elements = extracted.values.shape()[3];
        const Device device(output.device());
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::memory::Extracted<value_t, index_t> tmp{
                extracted.values.share(), extracted.indexes.share(), elements};
            cpu::memory::insert(tmp, output.share(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::memory::details::is_valid_insert_v<value_t, index_t, T>) {
                cuda::memory::Extracted<value_t, index_t> tmp{
                    extracted.values.share(), extracted.indexes.share(), elements};
                cuda::memory::insert(tmp, output.share(), stream.cuda());
            } else {
                NOA_THROW("These types of operands are not supported by the CUDA backend. "
                          "See noa::cuda::memory::extract(...) for more details");
            }

            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename value_t, typename index_t, typename T>
    void insert(const Array<value_t>& values, const Array<index_t>& indexes, const Array<T>& output) {
        insert(Extracted<value_t, index_t>{values, indexes}, output);
    }
}
