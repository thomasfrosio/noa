#pragma once

#ifndef NOA_EWISE_INL_
#error "This is an internal header; it should not be included."
#endif

// TODO OpenMP support? I don't want to add #pragma omp parallel for everywhere since it prevents constant-evaluated
//      expression to be moved to outer-loops. Benchmarking will be necessary.

namespace noa::cpu::math {
    template<typename T, typename U, typename UnaryOp>
    void ewise(const shared_t<T[]>& input, size4_t input_strides,
               const shared_t<U[]>& output, size4_t output_strides,
               size4_t shape, UnaryOp unary_op, Stream& stream) {
        stream.enqueue([=]() mutable {
            const size4_t order = indexing::order(output_strides, shape);
            input_strides = indexing::reorder(input_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);

            const T* iptr = input.get();
            U* optr = output.get();
            for (size_t i = 0; i < shape[0]; ++i)
                for (size_t j = 0; j < shape[1]; ++j)
                    for (size_t k = 0; k < shape[2]; ++k)
                        for (size_t l = 0; l < shape[3]; ++l)
                            optr[indexing::at(i, j, k, l, output_strides)] =
                                    static_cast<U>(unary_op(iptr[indexing::at(i, j, k, l, input_strides)]));
        });
    }

    template<typename T, typename U, typename V, typename BinaryOp, std::enable_if_t<traits::is_data_v<U>, bool>>
    void ewise(const shared_t<T[]>& lhs, size4_t lhs_strides, U rhs,
               const shared_t<V[]>& output, size4_t output_strides,
               size4_t shape, BinaryOp binary_op, Stream& stream) {
        stream.enqueue([=]() mutable {
            const size4_t order = indexing::order(output_strides, shape);
            lhs_strides = indexing::reorder(lhs_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);

            const T* lptr = lhs.get();
            V* optr = output.get();
            for (size_t i = 0; i < shape[0]; ++i)
                for (size_t j = 0; j < shape[1]; ++j)
                    for (size_t k = 0; k < shape[2]; ++k)
                        for (size_t l = 0; l < shape[3]; ++l)
                            optr[indexing::at(i, j, k, l, output_strides)] =
                                    static_cast<V>(binary_op(lptr[indexing::at(i, j, k, l, lhs_strides)], rhs));
        });
    }

    template<typename T, typename U, typename V, typename BinaryOp, std::enable_if_t<traits::is_data_v<T>, bool>>
    void ewise(T lhs, const shared_t<U[]>& rhs, size4_t rhs_strides,
               const shared_t<V[]>& output, size4_t output_strides,
               size4_t shape, BinaryOp binary_op, Stream& stream) {
        stream.enqueue([=]() mutable {
            const size4_t order = indexing::order(output_strides, shape);
            rhs_strides = indexing::reorder(rhs_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);

            const U* rptr = rhs.get();
            V* optr = output.get();
            for (size_t i = 0; i < shape[0]; ++i)
                for (size_t j = 0; j < shape[1]; ++j)
                    for (size_t k = 0; k < shape[2]; ++k)
                        for (size_t l = 0; l < shape[3]; ++l)
                            optr[indexing::at(i, j, k, l, output_strides)] =
                                    static_cast<V>(binary_op(lhs, rptr[indexing::at(i, j, k, l, rhs_strides)]));
        });
    }

    template<typename T, typename U, typename V, typename BinaryOp>
    void ewise(const shared_t<T[]>& lhs, size4_t lhs_strides,
               const shared_t<U[]>& rhs, size4_t rhs_strides,
               const shared_t<V[]>& output, size4_t output_strides,
               size4_t shape, BinaryOp binary_op, Stream& stream) {
        stream.enqueue([=]() mutable {
            const size4_t order = indexing::order(output_strides, shape);
            lhs_strides = indexing::reorder(lhs_strides, order);
            rhs_strides = indexing::reorder(rhs_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);

            const T* lptr = lhs.get();
            const U* rptr = rhs.get();
            V* optr = output.get();
            for (size_t i = 0; i < shape[0]; ++i)
                for (size_t j = 0; j < shape[1]; ++j)
                    for (size_t k = 0; k < shape[2]; ++k)
                        for (size_t l = 0; l < shape[3]; ++l)
                            optr[indexing::at(i, j, k, l, output_strides)] =
                                    static_cast<V>(binary_op(lptr[indexing::at(i, j, k, l, lhs_strides)],
                                                             rptr[indexing::at(i, j, k, l, rhs_strides)]));
        });
    }

    template<typename T, typename U, typename V, typename TrinaryOp, typename>
    void ewise(const shared_t<T[]>& lhs, size4_t lhs_strides, U mhs, U rhs,
               const shared_t<V[]>& output, size4_t output_strides,
               size4_t shape, TrinaryOp trinary_op, Stream& stream) {
        stream.enqueue([=]() mutable {
            const size4_t order = indexing::order(output_strides, shape);
            lhs_strides = indexing::reorder(lhs_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);

            const T* lptr = lhs.get();
            V* optr = output.get();
                for (size_t i = 0; i < shape[0]; ++i)
                    for (size_t j = 0; j < shape[1]; ++j)
                        for (size_t k = 0; k < shape[2]; ++k)
                            for (size_t l = 0; l < shape[3]; ++l)
                                optr[indexing::at(i, j, k, l, output_strides)] =
                                        static_cast<V>(trinary_op(lptr[indexing::at(i, j, k, l, lhs_strides)],
                                                                  mhs, rhs));
        });
    }

    template<typename T, typename U, typename V, typename W, typename TrinaryOp,
             std::enable_if_t<traits::is_data_v<V>, bool>>
    void ewise(const shared_t<T[]>& lhs, size4_t lhs_strides,
               const shared_t<U[]>& mhs, size4_t mhs_strides, V rhs,
               const shared_t<W[]>& output, size4_t output_strides,
               size4_t shape, TrinaryOp trinary_op, Stream& stream) {
        stream.enqueue([=]() mutable {
            const size4_t order = indexing::order(output_strides, shape);
            lhs_strides = indexing::reorder(lhs_strides, order);
            mhs_strides = indexing::reorder(mhs_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);

            const T* lptr = lhs.get();
            const U* mptr = mhs.get();
            W* optr = output.get();
            for (size_t i = 0; i < shape[0]; ++i)
                for (size_t j = 0; j < shape[1]; ++j)
                    for (size_t k = 0; k < shape[2]; ++k)
                        for (size_t l = 0; l < shape[3]; ++l)
                            optr[indexing::at(i, j, k, l, output_strides)] =
                                    static_cast<V>(trinary_op(lptr[indexing::at(i, j, k, l, lhs_strides)],
                                                              mptr[indexing::at(i, j, k, l, mhs_strides)],
                                                              rhs));
        });
    }

    template<typename T, typename U, typename V, typename W, typename TrinaryOp,
             std::enable_if_t<traits::is_data_v<U>, bool>>
    void ewise(const shared_t<T[]>& lhs, size4_t lhs_strides, U mhs,
               const shared_t<V[]>& rhs, size4_t rhs_strides,
               const shared_t<W[]>& output, size4_t output_strides,
               size4_t shape, TrinaryOp trinary_op, Stream& stream) {
        stream.enqueue([=]() mutable {
            const size4_t order = indexing::order(output_strides, shape);
            lhs_strides = indexing::reorder(lhs_strides, order);
            rhs_strides = indexing::reorder(rhs_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);

            const T* lptr = lhs.get();
            const V* rptr = rhs.get();
            W* optr = output.get();
            for (size_t i = 0; i < shape[0]; ++i)
                for (size_t j = 0; j < shape[1]; ++j)
                    for (size_t k = 0; k < shape[2]; ++k)
                        for (size_t l = 0; l < shape[3]; ++l)
                            optr[indexing::at(i, j, k, l, output_strides)] =
                                    static_cast<V>(trinary_op(lptr[indexing::at(i, j, k, l, lhs_strides)],
                                                              mhs,
                                                              rptr[indexing::at(i, j, k, l, rhs_strides)]));
        });
    }

    template<typename T, typename U, typename V, typename W, typename TrinaryOp>
    void ewise(const shared_t<T[]>& lhs, size4_t lhs_strides,
               const shared_t<U[]>& mhs, size4_t mhs_strides,
               const shared_t<V[]>& rhs, size4_t rhs_strides,
               const shared_t<W[]>& output, size4_t output_strides,
               size4_t shape, TrinaryOp trinary_op, Stream& stream) {
        stream.enqueue([=]() mutable {
            const size4_t order = indexing::order(output_strides, shape);
            lhs_strides = indexing::reorder(lhs_strides, order);
            mhs_strides = indexing::reorder(mhs_strides, order);
            rhs_strides = indexing::reorder(rhs_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);

            const T* lptr = lhs.get();
            const U* mptr = mhs.get();
            const V* rptr = rhs.get();
            W* optr = output.get();
            for (size_t i = 0; i < shape[0]; ++i)
                for (size_t j = 0; j < shape[1]; ++j)
                    for (size_t k = 0; k < shape[2]; ++k)
                        for (size_t l = 0; l < shape[3]; ++l)
                            optr[indexing::at(i, j, k, l, output_strides)] =
                                    static_cast<W>(trinary_op(lptr[indexing::at(i, j, k, l, lhs_strides)],
                                                              mptr[indexing::at(i, j, k, l, mhs_strides)],
                                                              rptr[indexing::at(i, j, k, l, rhs_strides)]));
        });
    }
}
