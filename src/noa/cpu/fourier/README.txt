Fourier
=======

Plan.h:         Creates and stores plans for C2R, R2C and C2C transforms. It is recommended to reuse these plans if
                possible. Functions performing FFTs will often accept existing plans as inputs.

Transforms.h:   Executes C2R, R2C and C2C transforms. It contains functions to execute plans, execute plans on
                new (but similar) arrays and to compute "one time" transforms. In these cases, a new quick plan
                is created using the ESTIMATE flag, then thrown once the transform is done.

Resize.h:       Pads or crops non-centered Fourier transforms.

Wrap.h:         Wraps Fourier transforms to different layouts:
                h2f:  "half to full",             i.e. non-centered, non-redundant to non-centered, redundant.
                h2fc: "half to full centered",    i.e. non-centered, non-redundant to centered,     redundant.
                h2hc: "half to half centered",    i.e. non-centered, non-redundant to centered,     non-redundant.
                fc2h: "full centered to half",    i.e. centered,     redundant     to non-centered, non-redundant.
