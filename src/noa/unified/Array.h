#include "noa/common/Types.h"

namespace noa {
    enum class Backend { // backend will be set on the Session
        CPU,
        CUDA,
        VULKAN,
        METAL
    };

    enum class Resource {
        HOST,
        PINNED,
        DEVICE,
    };

    /// Encodes the dimensions of an array.
    /// \details The "physical shape" corresponds to the actual memory footprint of the original (parent) array.
    ///          Within that contiguous region of memory, the data is stored in the {width:x, height:y, depth:z, page:w}
    ///          order. Note that the last dimension is often referred to as the batch dimension. This dimension has
    ///          some specificities and limitations.
    ///          The "logical shape" encodes the active region the Array is referring to. It can be different from the
    ///          physical shape for multiple reasons. This system also makes it very easy to work with Arrays that
    ///          refer to a subregion of another larger Array. In other words, arrays can be simple views of other
    ///          arrays.
    class ArrayShape {
    public:
        // 32-bits is enough to encode the shape. However, indexes should be 65-bits.
        using dim_t = uint32_t;
        using dim4_t = Int4<dim_t>;
        dim4_t m_shape_logical;
        dim4_t m_shape_physical;
        size_t m_offset;
    };

    /// Multi-dimensional (3D + batch) array located on a user-specified memory resource.
    /// \tparam T           The type of the elements. It must be copy assignable and copy constructible.
    /// \tparam Allocator   An allocator that is used to acquire/release memory and to construct/destroy
    ///                     the elements in that memory.
    template<typename T, typename Allocator>
    class Array { // Array is NOT attached to a Stream. It just holds data (which can be owned by the Array or simply viewed)
    public:
        /// Creates a contiguous array.
        /// \param shape    Physical and logical shape of the array.
        /// \param resource Memory resource to allocate from.
        Array(size4_t shape, Resource resource);

        /// Creates an array.
        /// \param logical_shape    Logical shape of the array.
        /// \param physical_shape   Physical shape of the array.
        ///                         If all elements are 0, the allocator will compute the best physical shape to hold
        ///                         and manipulate an 3D array with the specified \p logical_shape.
        /// \param resource         Memory resource to allocate from.
        /// \note Specifying the offsets can be useful for multiple reasons.
        ///       1) Allocates enough memory to support in-place r2c/r2c FFTs. In this case, for a logical shape of
        ///          {x,y,z,w}, the physical shape should be {x+n,y,z,w}, where n is 1 or 2 if x is odd or even,
        ///          respectively.
        ///       2) Allocates enough memory to hold the elements of a non-redundant FFT. In this case, for a logical
        ///          shape of {x,y,z,w}, the physical shape should be {x/2+1,y,z,w}.
        ///       3) Allocates the optimum shape to hold and manipulate an 3D array with the specified \p logical_shape.
        ///          For instance, if \p resource is Resource::DEVICE, it will trigger an allocation with cudaMalloc3D.
        ///          To trigger this, \p physical_shape should be set to 0.
        Array(size4_t logical_shape, size4_t physical_shape, Resource resource);

        /// Creates an array from an existing allocated memory region.
        /// \param[in,out] data Array to encapsulate.
        /// \param shape        WHDP shape of \p data.
        /// \param offset       WHD offsets of \p data.
        /// \param resource     Location of \p data.
        /// \param own          Whether the array should own \p data.
        Array(T* data, size4_t shape, size3_t offset, Resource resource, bool own);

    private: // typedefs


    private:
        T* m_data; //
        ArrayShape m_shape;
        uint32_t m_flags;
    };
}
