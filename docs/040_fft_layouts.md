## `FFT layouts`

As it is often done in signal and image processing, the library computes Discrete Fourier
Transforms (DFTs) using a family of algorithms called Fast Fourier Transforms (FFTs).

The DFT of real-valued inputs are Hermitian, i.e. the component at frequency x is the
complex conjugate of the component at frequency −x, which means that for real-valued
inputs there is no information in the negative frequency components that is not already
available from the positive frequency components. FFT libraries often have special implementations
designed to operate on real inputs, and exploit this symmetry by computing only
the positive frequency components, up to and including the Nyquist frequency. The resulting
“half-transforms” are often called Real Fast Fourier Transforms (rFFTs) or non-redundant
transforms (as opposed to the full-transforms which store both x and −x). Furthermore,
because complex elements are twice as large as the real-valued ones, the half-transforms are
still large enough to store the real-valued inputs. FFT libraries have used this property to
develop in-place rFFTs.

FFT algorithms ingest and output spectra with frequencies organized in a certain order. Given the 1d spectrum A of size n and using NumPy’s array notation, `A[0]` contains the zero-frequency term, then `A[1:n/2]` contains the positive-frequency terms, and `A[n/2+1:]` contains the negative-frequency terms, in order of decreasingly negative frequency. For instance, if `n=6`, the layout is `[0,1,2,-3,-2,-1]`, and if `n=7`, the layout is `[0,1,2,3,-3,-2,-1]`. For half-transforms, if `n=6` or `n=7`, the layout is `[0,1,2,3]`. To facilitate certain operations or to visualize the transform, a circular-shift can be applied to shift the zero-frequency to the center of the spectrum. We refer to this operation as “centering”, and libraries often provide the `fftshift`/`ifftshift` functions to perform this centering. We provide `noa::fft::remap`, a function used to remap FFTs to half/full and centred/non-centred layouts. It supports all the possible combinations of layouts and can operate in-place for some layouts and shapes.

Importantly, functions operating on FFTs always take a remap operation to keep track of what the input layout currently is and what the output layout should be. Note that in this case only a subset of layouts is usually supported. This remap operation is specified as the first template parameter of these functions, so everything is encoded at compile time and usually have a zero runtime cost too. The library denotes half-transforms with a `H`, full-transforms with a `F` and the centred layout with a `C`. The layout of the input and outputs, i.e. the remap operation, is encoded such as `H|F(C)2H|F(C)`, where the left side (relative to the `2`) refers to the input layout and the right side refers to the output layout. For instance, `H2HC` refers to a remap going from a non-redundant non-centred transform to a non-redundant centred transform (this is sometimes called a `rfftshift`). If the layout of the inputs and outputs is the same, the output layout can be omitted. For instance, `HC2HC` is equivalent to `HC` (or `hc`).

```c++
using namespace ::noa::types; // import f32, Array
namespace nf = ::noa::fft;
namespace ns = ::noa::signal;

// This is a special factory function to generate arrays for
// in-place rffts. "images" is real, "images_rfft" is complex,
// both point to the same memory.
auto [images, images_rfft] = nf::empty<f32>({1, 1, 64, 64});

 // ...initialise images...

// In-place bandpass filtering.
// - The Fourier transform functions always take the non-
//   centred transforms.
// - Other functions operating in Fourier space need the
//   Remap argument, which can be entered as a string or
//   using the noa::Remap enum.
// - Note that bandpass, as many other functions capable
//   of operating on rffts, needs the logical-shape of
//   the transform, i.e. the shape of the real array.
nf::r2c(images, images_rfft);
ns::bandpass<"h">( // no remap
    images_rfft, images_rfft, images.shape(),
    {.highpass_cutoff=0.05, .highpass_width=0.04,
     .lowpass_cutoff=0.4, .lowpass_width=0.1}
);
nf::c2r(images_rfft, images);

// Another example, with out-of-place filtering and FFT remapping:

// In-place rfft, then in-place centering, i.e. fftshift.
nf::r2c(images, images_rfft);
nf::remap(nf::Remap::H2HC, images_rfft, images_rfft, images.shape());
// or equivalently: nf::remap("h2hc", images_rfft, images_rfft, images.shape());

// ...do something that needs the centered images_rfft...

// Then do the out-of-place filtering.
// Notice the on-the-fly remapping hc2h, aka irfftshift,
// anticipating the c2r transform.
const auto filtered_rfft = noa::like(images_rfft);
ns::lowpass<"hc2h">(
    images_rfft, filtered_rfft, images.shape(),
    {.cutoff=0.5, .width=0.});
nf::c2r(filtered_rfft, images);
```


## `Problematic rfft <-> fft conversion`

Regarding the remapping from/to the `rfft`/`fft` layout, with even-sized dimensions.

Let us use the example of a 2d array with shape `{10,10}` (and `rfft` shape `{10,6}`). The non-redundant frequencies (along the width of `rfft`) are `{0,1,2,3,4,5}`, where `5` is the Nyquist frequency. The `fft` frequencies would be `{0,1,2,3,4,-5,-4,-3,-2,-1}`, where `-5` is the Nyquist frequency. Because of (1) the hermitian symmetry, `f(-5)=conj(f(5))`, and (2) because the Nyquist frequency is real, it ends up being `f(-5)=f(5)`. Another interesting property of the `rfft` is that (3) for every frequency `(u,v)` with `u` equal to Nyquist (so in our example, equal to 5), `f(u,v)=f(u,-v)`. And similarly, for every frequency `(u,v)` with `v` equal to Nyquist, `f(u,v)=f(-u,v)`. One can easily see this in the last column of the `rfft`.

The last _column_ of the `rfft` is:
`A={(5,0),(5,1),(5,2),(5,3),(5,4),(5,-5),(5,-4),(5,-3),(5,-2),(5,-1)}`. These do not exist in the `fft` since the Nyquist there is `-5`. Instead, the column is: `B={(-5,0),(-5,1),(-5,2),(-5,3),(-5,4),(-5,-5),(-5,-4),(-5,-3),(-5,-2),(-5,-1)}`. Taking the example of one frequency pair in the `rfft`, `(5,2)` and `(5,-2)`. We know that `f(5,2)=f(5,-2)` from (3), adding the hermitian symmetry (1) and the fact that these are real (2), we have `f(5,2)=f(5,-2)=f(-5,-2)=f(-5,2)`.

These properties allow us to extend the `rfft` to the `fft`, and to crop the `fft` to the `rfft` very easily.

One issue arises when generating an anisotropic function with an angle that is not a multiple of `pi/2`. In this case, the third property is broken: `f(5,2)!=f(5,-2)` and the remapping breaks for the frequencies at Nyquist except the ones on the cartesian axes (with all but one frequency at 0) (see property 3). At the time of writing, the only functions that can create this scenario are transforms with scaling factors and simulated CTFs with astigmatic defoci. Note that this is only true if the scaling or astigmatism is not along the cartesian axes (i.e. if the anisotropy angle is _not_ a multiple of `pi/2`).

While this could be fixed for the remapping `fft->rfft`, it cannot be fixed for the opposite direction `rfft->fft`. This isn't such a big deal. 1) These remapping should only be done for debugging and visualization anyway, 2) if the remap is done directly after a dft, it will work since the field is isotropic in this case, and 3) the problematic frequencies are past the Nyquist, so lowpass filtering to Nyquist (fftfreq=0.5) fixes this issue.
