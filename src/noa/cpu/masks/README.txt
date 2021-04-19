Mask namespace
==============

Sphere.h:       2D-3D sphere, with or without raised-cosine soft-edge. Apply to existing array or save the mask.
Rectangle.h:    2D-3D rectangle, with or without raised-cosine soft-edge. Apply to existing array or save the mask.
Cylinder.h:     3D cylinder, with or without raised-cosine soft-edge. Apply to existing array or save the mask.

ExtractMap.h:   Extract the (linear) indexes allowed by a given mask. These indexes are referred to as a map.
Remap.h:        Apply a map to an array, i.e. remove the masked out regions from an array.
RemapReverse.h: Replace the extracted elements to the original unmasked array, effectively reverse remap().
