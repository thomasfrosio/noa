#include "ImageFile.h"

// Must be included in the source file, as opposed to the header file,
// since MRCFile, etc. inherits from ImageFile.
#include "MRCFile.h"

using namespace ::Noa;

std::unique_ptr<ImageFile> ImageFile::get(const std::string& extension) {
    if (extension == ".mrc" || extension == ".st" || extension == ".rec" || extension == ".mrcs")
        return std::make_unique<MRCFile>();
    else
        NOA_THROW("Could not deduce the extension of the file. "
                  "Should be either \".mrc\", \".mrcs\", \".st\", or \".rec\", got \"{}\"", extension);
}

void ImageFile::save(const path_t& filename, const float* data, size3_t shape, IO::DataType dtype, float3_t ps) {
    auto file = get(filename.extension().string());
    file->open(filename, IO::WRITE);
    file->setDataType(dtype);
    file->setShape(shape);
    file->setPixelSize(ps);
    file->writeAll(data);
    file->close();
}

void ImageFile::save(const path_t& filename, const cfloat_t* data, size3_t shape, IO::DataType dtype, float3_t ps) {
    auto file = get(filename.extension().string());
    file->open(filename, IO::WRITE);
    file->setDataType(dtype);
    file->setShape(shape);
    file->setPixelSize(ps);
    file->writeAll(data);
    file->close();
}
