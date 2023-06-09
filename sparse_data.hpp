#ifndef SPARSE_H
#define SPARSE_H

#ifdef __CUDACC__
#define CUDA_DEV __device__
#define CUDA_GLO __global__
#define CUDA_HOS __host__
#else
#define CUDA_DEV
#define CUDA_GLO
#define CUDA_HOS
#endif

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


struct SparseData
{
    int index;
    float value;
};

CUDA_HOS thrust::device_vector<SparseData> buildSparesData();

#endif