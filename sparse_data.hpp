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
#include <string>
#include <fstream>
#include <sstream>
#include <cstring>

struct SparseEntry
{
    int index;
    float value;
};

struct SparseDataset
{
    thrust::host_vector<SparseEntry>* patterns;
    int *labels;
};

CUDA_HOS SparseDataset buildSparseData(std::string path, uint num_patterns, uint num_features);

#endif