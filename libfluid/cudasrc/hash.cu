
#include "hash.cuh"

PointHashOperator::PointHashOperator(uint *_cellOcc)
    : cellOcc(_cellOcc)
{
}

/// The operator functor. Should be called with thrust::transform and a zip_iterator
__device__ uint PointHashOperator::operator()(const float3 &pt)
{
    // Note that finding the grid coordinates are much simpler if the grid is over the range [0,1] in
    // each dimension and the points are also in the same space.
    int3 grid = grid_from_point(pt);

    // Compute the hash for this grid cell
    uint hash = cell_from_grid(grid);

    // Calculate the cell occupancy counter here to save on an extra kernel launch (won't trigger if out of bounds)
    if (hash != NULL_HASH)
    {
        atomicAdd(&cellOcc[hash], 1);
    }

    // Return the cell idx (NULL_HASH if out of bounds)
    return hash;
}
