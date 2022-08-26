#pragma once
#ifndef NOA_UNIFIED_INDEX_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

#include "noa/cpu/memory/Index.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/memory/Index.h"
#endif

namespace noa::memory {
    template<typename T, typename>
    void extract(const Array<T>& input, const Array<T>& subregions, const Array<int4_t>& origins,
                 BorderMode border_mode, T border_value) {
        NOA_CHECK(subregions.device() == input.device(),
                  "The input and subregion arrays must be on the same device, but got input:{} and subregions:{}",
                  input.device(), subregions.device());
        NOA_CHECK(indexing::isVector(origins.shape()) && origins.contiguous() &&
                  origins.shape().elements() == subregions.shape()[0],
                  "The indexes should be a contiguous vector of {} elements but got shape {} and strides {}",
                  subregions.shape()[0], origins.shape(), origins.strides());
        NOA_CHECK(subregions.get() != input.get(), "The subregion(s) and the output arrays should not overlap");

        const Device device = subregions.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(origins.dereferenceable(), "The origins should be accessible to the CPU");
            cpu::memory::extract<T>(input.share(), input.strides(), input.shape(),
                                    subregions.share(), subregions.strides(), subregions.shape(),
                                    origins.share(), border_mode, border_value, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::memory::extract<T>(input.share(), input.strides(), input.shape(),
                                     subregions.share(), subregions.strides(), subregions.shape(),
                                     origins.share(), border_mode, border_value, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    void insert(const Array<T>& subregions, const Array<T>& output, const Array<int4_t>& origins) {
        NOA_CHECK(subregions.device() == output.device(),
                  "The output and subregion arrays must be on the same device, but got output:{} and subregions:{}",
                  output.device(), subregions.device());
        NOA_CHECK(indexing::isVector(origins.shape()) && origins.contiguous() &&
                  origins.shape().elements() == subregions.shape()[0],
                  "The indexes should be a contiguous vector of {} elements but got shape {} and strides {}",
                  subregions.shape()[0], origins.shape(), origins.strides());
        NOA_CHECK(subregions.get() != output.get(), "The subregion(s) and the output arrays should not overlap");

        const Device device(subregions.device());
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(origins.dereferenceable(), "The origins should be accessible to the CPU");
            cpu::memory::insert<T>(subregions.share(), subregions.strides(), subregions.shape(),
                                   output.share(), output.strides(), output.shape(),
                                   origins.share(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::memory::insert<T>(subregions.share(), subregions.strides(), subregions.shape(),
                                    output.share(), output.strides(), output.shape(),
                                    origins.share(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    size4_t atlasLayout(size4_t subregion_shape, T* origins) {
        return cpu::memory::atlasLayout(subregion_shape, origins);
    }
}

namespace noa::memory {
    template<typename value_t, typename offset_t, typename T, typename U, typename UnaryOp>
    Extracted<value_t, offset_t> extract(const Array<T>& input, const Array<U>& lhs, UnaryOp unary_op,
                                         bool extract_values, bool extract_offsets) {
        NOA_CHECK(!extract_values || !input.empty(), "The input array should not be empty");
        NOA_CHECK(input.empty() || all(input.shape() == lhs.shape()),
                  "The input arrays should have the same shape, but got input:{} and lhs:{}",
                  input.shape(), lhs.shape());
        NOA_CHECK(input.empty() || input.device() == lhs.device(),
                  "The input arrays should be on the same device, but got input:{} and lhs:{}",
                  input.device(), lhs.device());

        Extracted<value_t, offset_t> out;
        const Device device = lhs.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            auto extracted = cpu::memory::extract<value_t, offset_t>(
                    input.share(), input.strides(), lhs.share(), lhs.strides(), input.shape(),
                    unary_op, extract_values, extract_offsets, stream.cpu());
            out.values = Array<value_t>(extracted.values, extracted.count);
            out.offsets = Array<offset_t>(extracted.offsets, extracted.count);
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::memory::details::is_valid_extract_unary_v<T, U, value_t, offset_t, UnaryOp>) {
                auto extracted = cuda::memory::extract<value_t, offset_t>(
                        input.share(), input.strides(), lhs.share(), lhs.strides(), input.shape(),
                        unary_op, extract_values, extract_offsets, stream.cuda());
                const ArrayOption option(input.device(), Allocator::DEFAULT_ASYNC);
                out.values = Array<value_t>(extracted.values, extracted.count, option);
                out.offsets = Array<offset_t>(extracted.offsets, extracted.count, option);
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

    template<typename value_t, typename offset_t, typename T, typename U, typename V, typename BinaryOp, typename>
    Extracted<value_t, offset_t> extract(const Array<T>& input, const Array<U>& lhs, V rhs, BinaryOp binary_op,
                                        bool extract_values, bool extract_offsets) {
        NOA_CHECK(!extract_values || !input.empty(), "The input array should not be empty");
        NOA_CHECK(input.empty() || all(input.shape() == lhs.shape()),
                  "The input arrays should have the same shape, but got input:{} and lhs:{}",
                  input.shape(), lhs.shape());
        NOA_CHECK(input.empty() || input.device() == lhs.device(),
                  "The input arrays should be on the same device, but got input:{} and lhs:{}",
                  input.device(), lhs.device());

        Extracted<value_t, offset_t> out;
        const Device device = lhs.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            auto extracted = cpu::memory::extract<value_t, offset_t>(
                    input.share(), input.strides(), lhs.share(), lhs.strides(), rhs, lhs.shape(),
                    binary_op, extract_values, extract_offsets, stream.cpu());
            out.values = Array<value_t>(extracted.values, extracted.count);
            out.offsets = Array<offset_t>(extracted.offsets, extracted.count);
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::memory::details::is_valid_extract_binary_v<T, U, T, value_t, offset_t, BinaryOp>) {
                auto extracted = cuda::memory::extract<value_t, offset_t>(
                        input.share(), input.strides(), lhs.share(), lhs.strides(), static_cast<T>(rhs), lhs.shape(),
                        binary_op, extract_values, extract_offsets, stream.cuda());
                const ArrayOption option(input.device(), Allocator::DEFAULT_ASYNC);
                out.values = Array<value_t>(extracted.values, extracted.count, option);
                out.offsets = Array<offset_t>(extracted.offsets, extracted.count, option);
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

    template<typename value_t, typename offset_t, typename T, typename U, typename V, typename BinaryOp, typename>
    Extracted<value_t, offset_t> extract(const Array<T>& input, U lhs, const Array<V>& rhs, BinaryOp binary_op,
                                        bool extract_values, bool extract_offsets) {
        NOA_CHECK(!extract_values || !input.empty(), "The input array should not be empty");
        NOA_CHECK(input.empty() || all(input.shape() == rhs.shape()),
                  "The input arrays should have the same shape, but got input:{} and rhs:{}",
                  input.shape(), rhs.shape());
        NOA_CHECK(input.empty() || input.device() == rhs.device(),
                  "The input arrays should be on the same device, but got input:{} and rhs:{}",
                  input.device(), rhs.device());

        Extracted<value_t, offset_t> out;
        const Device device = rhs.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            auto extracted = cpu::memory::extract<value_t, offset_t>(
                    input.share(), input.strides(), lhs, rhs.share(), rhs.strides(), rhs.shape(),
                    binary_op, extract_values, extract_offsets, stream.cpu());
            out.values = Array<value_t>(extracted.values, extracted.count);
            out.offsets = Array<offset_t>(extracted.offsets, extracted.count);
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::memory::details::is_valid_extract_binary_v<T, U, T, value_t, offset_t, BinaryOp>) {
                auto extracted = cuda::memory::extract<value_t, offset_t>(
                        input.share(), input.strides(), static_cast<T>(lhs), rhs.share(), rhs.strides(), rhs.shape(),
                        binary_op, extract_values, extract_offsets, stream.cuda());
                const ArrayOption option(input.device(), Allocator::DEFAULT_ASYNC);
                out.values = Array<value_t>(extracted.values, extracted.count, option);
                out.offsets = Array<offset_t>(extracted.offsets, extracted.count, option);
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

    template<typename value_t, typename offset_t, typename T, typename U, typename V, typename BinaryOp>
    Extracted<value_t, offset_t> extract(const Array<T>& input, const Array<U>& lhs, const Array<V>& rhs,
                                        BinaryOp binary_op, bool extract_values, bool extract_offsets) {
        NOA_CHECK(!extract_values || !input.empty(), "The input array should not be empty");
        NOA_CHECK(input.empty() || all(input.shape() == lhs.shape()) && all(input.shape() == rhs.shape()),
                  "The input arrays should have the same shape, but got input:{}, lhs:{} and rhs:{}",
                  input.shape(), lhs.shape(), rhs.shape());
        NOA_CHECK(input.empty() || input.device() == lhs.device() && input.device() == rhs.device(),
                  "The input arrays should be on the same device, but got input:{}, lhs:{} and rhs:{}",
                  input.device(), lhs.device(), rhs.device());

        Extracted<value_t, offset_t> out;
        const Device device = lhs.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            auto extracted = cpu::memory::extract<value_t, offset_t>(
                    input.share(), input.strides(), lhs.share(), lhs.strides(),
                    rhs.share(), rhs.strides(), rhs.shape(), binary_op,
                    extract_values, extract_offsets, stream.cpu());
            out.values = Array<value_t>(extracted.values, extracted.count);
            out.offsets = Array<offset_t>(extracted.offsets, extracted.count);
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::memory::details::is_valid_extract_binary_v<T, U, V, value_t, offset_t, BinaryOp>) {
                auto extracted = cuda::memory::extract<value_t, offset_t>(
                        input.share(), input.strides(), lhs.share(), lhs.strides(),
                        rhs.share(), rhs.strides(), rhs.shape(), binary_op,
                        extract_values, extract_offsets, stream.cuda());
                const ArrayOption option(input.device(), Allocator::DEFAULT_ASYNC);
                out.values = Array<value_t>(extracted.values, extracted.count, option);
                out.offsets = Array<offset_t>(extracted.offsets, extracted.count, option);
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

    template<typename value_t, typename offset_t, typename T>
    void insert(const Extracted<value_t, offset_t>& extracted, const Array<T>& output) {
        NOA_CHECK(extracted.values.device() == extracted.offsets.device() &&
                  extracted.values.device() == output.device(),
                  "The input and output arrays should be on the same device, "
                  "but got values:{}, offsets:{} and output:{}",
                  extracted.values.device(), extracted.offsets.device(), output.device());
        NOA_CHECK(all(extracted.values.shape() == extracted.offsets.shape()) &&
                  indexing::isVector(extracted.offsets.shape()) &&
                  extracted.values.contiguous() && extracted.offsets.contiguous(),
                  "The sequence of values and offsets should be two contiguous vectors of the same size, "
                  "but got values:{} and offsets:{}", extracted.values.shape(), extracted.offsets.shape());

        const size_t elements = extracted.values.shape().elements();
        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::memory::Extracted<value_t, offset_t> tmp{
                extracted.values.share(), extracted.offsets.share(), elements};
            cpu::memory::insert(tmp, output.share(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::memory::details::is_valid_insert_v<value_t, offset_t, T>) {
                cuda::memory::Extracted<value_t, offset_t> tmp{
                    extracted.values.share(), extracted.offsets.share(), elements};
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

    template<typename value_t, typename offset_t, typename T>
    void insert(const Array<value_t>& values, const Array<offset_t>& offsets, const Array<T>& output) {
        insert(Extracted<value_t, offset_t>{values, offsets}, output);
    }
}
