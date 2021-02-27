#include "noa/cpu/fourier/Resize.h"

using namespace Noa;

void Fourier::crop(const cfloat_t* in, cfloat_t* out, size3_t shape_in, size3_t shape_out) {
    size_t out_x_bytes = (shape_out.x / 2 + 1) * sizeof(cfloat_t);

    // Copies each cropped row.
    size_t in_z, in_y;
    for (size_t out_z{0}; out_z < shape_out.z; ++out_z) {
        // first new half vs second new half (offset to skip cropped planes).
        in_z = (out_z < shape_out.z / 2 + 1) ? out_z : out_z - shape_out.z + shape_out.z;
        for (size_t out_y{0}; out_y < shape_out.z; ++out_y) {
            // first new half vs second new half (offset to skip cropped rows).
            in_y = (out_y < shape_out.y / 2 + 1) ? out_y : out_y - shape_out.y + shape_out.y;

            std::memcpy(out + (out_z * shape_out.y + out_y) * shape_out.x,
                        in + (in_z * shape_in.y + in_y) * shape_in.x,
                        out_x_bytes);
        }
    }
}
