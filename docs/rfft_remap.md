## `rfft` &harr; `fft`

Regarding the remapping from/to the `rfft`/`fft` layout, with even-sized dimensions.

Let us use the example of a 2d array with shape `{10,10}` (and `rfft` shape `{10,6}`). The non-redundant frequencies (along the width of `rfft`) are `{0,1,2,3,4,5}`, where `5` is the Nyquist frequency. The `fft` frequencies would be `{0,1,2,3,4,-5,-4,-3,-2,-1}`, where `-5` is the Nyquist frequency. Because of the hermitian symmetry (1), `f(-5)=conj(f(5))`, and because these are the real Nyquist frequencies (2), it ends up being `f(-5)=f(5)`. Another interesting property (3) of the `rfft` is that every frequency `(u,v)` with `u` equal to Nyquist (so in our example, equal to 5), `f(u,v)=f(u,-v)`. And similarly, every frequency `(u,v)` with `v` equal to Nyquist, `f(u,v)=f(-u,v)`. One can easily see this in the last column of the `rfft`.

The last _column_ of the `rfft` is:
`A={(5,0),(5,1),(5,2),(5,3),(5,4),(5,-5),(5,-4),(5,-3),(5,-2),(5,-1)}`. These do not exist in the `fft` since the Nyquist there is `-5`. Instead, the column is:
`B={(-5,0),(-5,1),(-5,2),(-5,3),(-5,4),(-5,-5),(-5,-4),(-5,-3),(-5,-2),(-5,-1)}`. Taking the example of one frequency pair in the `rfft`, `(5,2)` and `(5,-2)`. We know that `f(5,2)=f(5,-2)` from (3), adding the hermitian symmetry (1) and the fact that these are real (2), we have `f(5,2)=f(5,-2)=f(-5,-2)=f(-5,2)`.

These properties allow us to extend the `rfft` to the `fft`, and to crop the `fft` to the `rfft` very easily.

One issue arises when generating an anisotropic function with an anisotropic angle that is not a multiple of `pi/2`. In this case, the third property is broken: `f(5,2)!=f(5,-2)` and the remapping breaks for the frequencies at Nyquist except the ones on the cartesian axes (with all but one frequency at 0) (see property 3). At the time of writing, the only functions that can create this scenario are geometric scaling factors and a simulated CTF with astigmatic defocus. Note that this is only true if the scaling or astigmatism is not along the cartesian axes (i.e. if the anisotropy angle is _not_ a multiple of `pi/2`).

While this could be fixed for the remapping `fft->rfft`, it cannot be fixed for the opposite direction `rfft->fft`. This isn't such a big deal. 1) these remapping should only be done for debugging and visualization anyway, 2) if the remap is done directly after a dft, it will work since the field has to be isotropic in this case, and 3) the problematic frequencies are past the Nyquist, so lowpass filtering to Nyquist (fftfreq=0.5) fixes this issue.
