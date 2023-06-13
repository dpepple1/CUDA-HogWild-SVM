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

#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>


// Compressed Sparse Row Matrix Format
struct CSR_Data
{
    int *labels;
    int *rowIdx;    // The column offset for the rows
    int *colIdx;    // The column for each values
    float *values;  // THe values in the sparse matrix
};

CUDA_HOS CSR_Data buildSparseData(std::string path, uint num_patterns, uint num_features);

#endif