#pragma once

#include "noa/cpu/fourier/Plan.h"
#include "noa/cpu/fourier/Transforms.h"
#include "noa/cpu/fourier/Resize.h"
#include "noa/cpu/fourier/Remap.h"

/* With R = float|double, with C = cfloat_t|cdouble_t.
 *
 * Plans.h
 * =======
 *
 * Fourier::Plan<R> plan(C, R, shape_t, Fourier:Flag);  // plan a r2c transform
 * Fourier::Plan<R> plan(R, C, shape_t, Fourier:Flag);  // plan a c2r transform
 *
 * Transforms.h
 * ============
 *
 * Fourier::transform(Fourier::Plan<R>);          // execute the plan (c2r or r2c).
 * Fourier::r2c(C, R, Fourier::Plan<R>);          // execute the plan (r2c) on new-arrays
 * Fourier::c2r(R, C, Fourier::Plan<R>);          // execute the plan (c2r) on new-arrays
 * Fourier::r2c(C, R, shape_t);                   // compute a one time r2c transform.
 * Fourier::c2r(R, C, shape_t);                   // compute a one time c2r transform.
 *
 * Resize.h
 * ========
 *
 * Fourier::resize(out, shape_out, in, shape_in, Fourier::Plan<R>, batch);
 * Fourier::crop(out, shape_out, in, shape_in, Fourier::Plan<R>, batch);
 * Fourier::pad(out, shape_out, in, shape_in, Fourier::Plan<R>, batch);
 *
 * Remap.h
 * =======
 *
 * Fourier::h2f()
 * Fourier::f2h()
 * Fourier::hc2f()
 * Fourier::fc2h()
 * Fourier::h2fc()
 * Fourier::f2hc()
 * Fourier::hc2fc()
 * Fourier::fc2hc()
 *
 * Note
 * ====
 *
 * - Only the basic FFTW interface is currently available (which is enough for contiguous data).
 */
