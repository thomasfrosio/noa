#include "noa/common/files/ImageFile.h"
#include "noa/common/Profiler.h"

// Must be included in the source file, as opposed to the header file,
// since MRCFile, etc. inherits from ImageFile.
#include "noa/common/files/MRCFile.h"

using namespace ::noa;

std::unique_ptr<ImageFile> ImageFile::get(const std::string& extension) {
    if (extension == ".mrc" || extension == ".st" || extension == ".rec" || extension == ".mrcs")
        return std::make_unique<MRCFile>();
    else
        NOA_THROW("Could not deduce the extension of the file. "
                  "Should be either \".mrc\", \".mrcs\", \".st\", or \".rec\", got \"{}\"", extension);
}

void ImageFile::save(const path_t& filename, const float* data, size3_t shape, io::DataType dtype, float3_t ps) {
    NOA_PROFILE_FUNCTION();
    auto file = get(filename.extension().string());
    file->open(filename, io::WRITE);
    file->setDataType(dtype);
    file->setShape(shape);
    file->setPixelSize(ps);
    file->writeAll(data);
    file->close();
}

void ImageFile::save(const path_t& filename, const cfloat_t* data, size3_t shape, io::DataType dtype, float3_t ps) {
    NOA_PROFILE_FUNCTION();
    auto file = get(filename.extension().string());
    file->open(filename, io::WRITE);
    file->setDataType(dtype);
    file->setShape(shape);
    file->setPixelSize(ps);
    file->writeAll(data);
    file->close();
}
