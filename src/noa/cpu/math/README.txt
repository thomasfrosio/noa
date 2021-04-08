Extension of noa/Math.h
============================

These headers extends the Math namespace for CPU arrays. These contains new functions and overloads of the
"per-elements" functions declared in noa/Math.h for integer, floating-point and complex types.

     Arithmetics.h:          Functions emulating the basic + - * / operators.
     ArithmeticsComposite.h: Composite operations, e.g. squaredDistance*() or the fused multiply-add.
     Booleans.h:             Boolean logics, e.g. isLess, isGreater, isWithin, logicNOT.
     Generics.h:             Mostly overload of the "per-element" functions declared in noa/Math.h
                             for integer and floating-point types and in noa/util/Complex.h.
     Indexes.h:              Find indexes, e.g. firstMin or firstMax.
     Reductions.h:           Reduction operations, e.g. sum, mean, min, max, variance, etc.

STL execution policies
======================

Some functions use the STL execution policies. These policies are mostly allowing (or disabling) parallelization
(vectorization, migration across threads, etc.). If the implementation cannot parallelize or vectorize (e.g. due to
lack of resources), all standard execution policies can fall back to sequential execution. As of March 2021, only
gcc (linking with TBB), MSVC and Intel (although through non-standard headers) support them.

For now, the CPU backend is not focused on performance, so if the code is not compiled with gcc, then execution
will always be sequential, which is OK. This might changed in the future (OpenMP?)...

See: https://en.cppreference.com/w/cpp/compiler_support
See: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0024r2.html

TODO Cmake: Link to TBB if GCC.
