#pragma once

#ifndef NOA_EWISE_INL_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

// TODO This is a naive implementation, without explicit SIMD support... Check for contiguity?
//      OpenMP support? I don't want to add #pragma omp parallel for everywhere since it prevents constant-evaluated
//      expression to be moved to outer-loops. Benchmarking will be necessary.

namespace noa::cpu::math {
    template<typename In, typename Out, typename UnaryOp>
    void ewise(const shared_t<In[]>& input, dim4_t input_strides,
               const shared_t<Out[]>& output, dim4_t output_strides,
               dim4_t shape, UnaryOp&& unary_op, Stream& stream) {
        NOA_ASSERT(input && output && all(shape > 0));
        stream.enqueue([=, functor = std::forward<UnaryOp>(unary_op)]() mutable {
            const dim4_t order = indexing::order(output_strides, shape);
            input_strides = indexing::reorder(input_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);

            const In* iptr = input.get();
            Out* optr = output.get();
            for (dim_t i = 0; i < shape[0]; ++i)
                for (dim_t j = 0; j < shape[1]; ++j)
                    for (dim_t k = 0; k < shape[2]; ++k)
                        for (dim_t l = 0; l < shape[3]; ++l)
                            optr[indexing::at(i, j, k, l, output_strides)] =
                                    static_cast<Out>(functor(iptr[indexing::at(i, j, k, l, input_strides)]));
        });
    }

    template<typename Lhs, typename Rhs, typename Out, typename BinaryOp, typename>
    void ewise(const shared_t<Lhs[]>& lhs, dim4_t lhs_strides,
               Rhs rhs,
               const shared_t<Out[]>& output, dim4_t output_strides,
               dim4_t shape, BinaryOp&& binary_op, Stream& stream) {
        NOA_ASSERT(lhs && output && all(shape > 0));
        stream.enqueue([=, functor = std::forward<BinaryOp>(binary_op)]() mutable {
            const dim4_t order = indexing::order(output_strides, shape);
            lhs_strides = indexing::reorder(lhs_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);

            const Lhs* lptr = lhs.get();
            Out* optr = output.get();
            for (dim_t i = 0; i < shape[0]; ++i)
                for (dim_t j = 0; j < shape[1]; ++j)
                    for (dim_t k = 0; k < shape[2]; ++k)
                        for (dim_t l = 0; l < shape[3]; ++l)
                            optr[indexing::at(i, j, k, l, output_strides)] =
                                    static_cast<Out>(functor(lptr[indexing::at(i, j, k, l, lhs_strides)], rhs));
        });
    }

    template<typename Lhs, typename Rhs, typename Out, typename BinaryOp, typename>
    void ewise(Lhs lhs, const shared_t<Rhs[]>& rhs, dim4_t rhs_strides,
               const shared_t<Out[]>& output, dim4_t output_strides,
               dim4_t shape, BinaryOp&& binary_op, Stream& stream) {
        NOA_ASSERT(rhs && output && all(shape > 0));
        stream.enqueue([=, functor = std::forward<BinaryOp>(binary_op)]() mutable {
            const dim4_t order = indexing::order(output_strides, shape);
            rhs_strides = indexing::reorder(rhs_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);

            const Rhs* rptr = rhs.get();
            Out* optr = output.get();
            for (dim_t i = 0; i < shape[0]; ++i)
                for (dim_t j = 0; j < shape[1]; ++j)
                    for (dim_t k = 0; k < shape[2]; ++k)
                        for (dim_t l = 0; l < shape[3]; ++l)
                            optr[indexing::at(i, j, k, l, output_strides)] =
                                    static_cast<Out>(functor(lhs, rptr[indexing::at(i, j, k, l, rhs_strides)]));
        });
    }

    template<typename Lhs, typename Rhs, typename Out, typename BinaryOp>
    void ewise(const shared_t<Lhs[]>& lhs, dim4_t lhs_strides,
               const shared_t<Rhs[]>& rhs, dim4_t rhs_strides,
               const shared_t<Out[]>& output, dim4_t output_strides,
               dim4_t shape, BinaryOp&& binary_op, Stream& stream) {
        NOA_ASSERT(lhs && rhs && output && all(shape > 0));
        stream.enqueue([=, functor = std::forward<BinaryOp>(binary_op)]() mutable {
            const dim4_t order = indexing::order(output_strides, shape);
            lhs_strides = indexing::reorder(lhs_strides, order);
            rhs_strides = indexing::reorder(rhs_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);

            const Lhs* lptr = lhs.get();
            const Rhs* rptr = rhs.get();
            Out* optr = output.get();
            for (dim_t i = 0; i < shape[0]; ++i)
                for (dim_t j = 0; j < shape[1]; ++j)
                    for (dim_t k = 0; k < shape[2]; ++k)
                        for (dim_t l = 0; l < shape[3]; ++l)
                            optr[indexing::at(i, j, k, l, output_strides)] =
                                    static_cast<Out>(functor(lptr[indexing::at(i, j, k, l, lhs_strides)],
                                                             rptr[indexing::at(i, j, k, l, rhs_strides)]));
        });
    }

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp, typename>
    void ewise(const shared_t<Lhs[]>& lhs, dim4_t lhs_strides,
               Mhs mhs,
               Rhs rhs,
               const shared_t<Out[]>& output, dim4_t output_strides,
               dim4_t shape, TrinaryOp&& trinary_op, Stream& stream) {
        NOA_ASSERT(lhs && output && all(shape > 0));
        stream.enqueue([=, functor = std::forward<TrinaryOp>(trinary_op)]() mutable {
            const dim4_t order = indexing::order(output_strides, shape);
            lhs_strides = indexing::reorder(lhs_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);

            const Lhs* lptr = lhs.get();
            Out* optr = output.get();
                for (dim_t i = 0; i < shape[0]; ++i)
                    for (dim_t j = 0; j < shape[1]; ++j)
                        for (dim_t k = 0; k < shape[2]; ++k)
                            for (dim_t l = 0; l < shape[3]; ++l)
                                optr[indexing::at(i, j, k, l, output_strides)] =
                                        static_cast<Out>(functor(lptr[indexing::at(i, j, k, l, lhs_strides)],
                                                                 mhs, rhs));
        });
    }

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp, typename>
    void ewise(const shared_t<Lhs[]>& lhs, dim4_t lhs_strides,
               const shared_t<Mhs[]>& mhs, dim4_t mhs_strides,
               Rhs rhs,
               const shared_t<Out[]>& output, dim4_t output_strides,
               dim4_t shape, TrinaryOp&& trinary_op, Stream& stream) {
        NOA_ASSERT(lhs && mhs && output && all(shape > 0));
        stream.enqueue([=, functor = std::forward<TrinaryOp>(trinary_op)]() mutable {
            const dim4_t order = indexing::order(output_strides, shape);
            lhs_strides = indexing::reorder(lhs_strides, order);
            mhs_strides = indexing::reorder(mhs_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);

            const Lhs* lptr = lhs.get();
            const Mhs* mptr = mhs.get();
            Out* optr = output.get();
            for (dim_t i = 0; i < shape[0]; ++i)
                for (dim_t j = 0; j < shape[1]; ++j)
                    for (dim_t k = 0; k < shape[2]; ++k)
                        for (dim_t l = 0; l < shape[3]; ++l)
                            optr[indexing::at(i, j, k, l, output_strides)] =
                                    static_cast<Out>(functor(lptr[indexing::at(i, j, k, l, lhs_strides)],
                                                             mptr[indexing::at(i, j, k, l, mhs_strides)],
                                                             rhs));
        });
    }

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp, typename>
    void ewise(const shared_t<Lhs[]>& lhs, dim4_t lhs_strides,
               Mhs mhs,
               const shared_t<Rhs[]>& rhs, dim4_t rhs_strides,
               const shared_t<Out[]>& output, dim4_t output_strides,
               dim4_t shape, TrinaryOp&& trinary_op, Stream& stream) {
        NOA_ASSERT(lhs && rhs && output && all(shape > 0));
        stream.enqueue([=, functor = std::forward<TrinaryOp>(trinary_op)]() mutable {
            const dim4_t order = indexing::order(output_strides, shape);
            lhs_strides = indexing::reorder(lhs_strides, order);
            rhs_strides = indexing::reorder(rhs_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);

            const Lhs* lptr = lhs.get();
            const Rhs* rptr = rhs.get();
            Out* optr = output.get();
            for (dim_t i = 0; i < shape[0]; ++i)
                for (dim_t j = 0; j < shape[1]; ++j)
                    for (dim_t k = 0; k < shape[2]; ++k)
                        for (dim_t l = 0; l < shape[3]; ++l)
                            optr[indexing::at(i, j, k, l, output_strides)] =
                                    static_cast<Out>(functor(lptr[indexing::at(i, j, k, l, lhs_strides)],
                                                             mhs,
                                                             rptr[indexing::at(i, j, k, l, rhs_strides)]));
        });
    }

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp>
    void ewise(const shared_t<Lhs[]>& lhs, dim4_t lhs_strides,
               const shared_t<Mhs[]>& mhs, dim4_t mhs_strides,
               const shared_t<Rhs[]>& rhs, dim4_t rhs_strides,
               const shared_t<Out[]>& output, dim4_t output_strides,
               dim4_t shape, TrinaryOp&& trinary_op, Stream& stream) {
        NOA_ASSERT(lhs && mhs && rhs && output && all(shape > 0));

        stream.enqueue([=, functor = std::forward<TrinaryOp>(trinary_op)]() mutable {
            const dim4_t order = indexing::order(output_strides, shape);
            lhs_strides = indexing::reorder(lhs_strides, order);
            mhs_strides = indexing::reorder(mhs_strides, order);
            rhs_strides = indexing::reorder(rhs_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);

            const Lhs* lptr = lhs.get();
            const Mhs* mptr = mhs.get();
            const Rhs* rptr = rhs.get();
            Out* optr = output.get();
            for (dim_t i = 0; i < shape[0]; ++i)
                for (dim_t j = 0; j < shape[1]; ++j)
                    for (dim_t k = 0; k < shape[2]; ++k)
                        for (dim_t l = 0; l < shape[3]; ++l)
                            optr[indexing::at(i, j, k, l, output_strides)] =
                                    static_cast<Out>(functor(lptr[indexing::at(i, j, k, l, lhs_strides)],
                                                             mptr[indexing::at(i, j, k, l, mhs_strides)],
                                                             rptr[indexing::at(i, j, k, l, rhs_strides)]));
        });
    }
}
