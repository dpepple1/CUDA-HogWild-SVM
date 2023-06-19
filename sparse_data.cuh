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
    int *sparsity;  // The number of non-zero features for each row
    int *labels;    // The labels of all of the patterns 
    long *rowIdx;   // The column offset for the rows
    int *colIdx;    // The column for each values
    float *values;  // THe values in the sparse matrix

    // Some useful meta data about the dataset
    uint numPairs;
    unsigned long numObservations;
};

CUDA_HOS CSR_Data buildSparseData(std::string path, uint num_patterns, uint num_features);
CUDA_HOS CSR_Data *CSRToGPU(CSR_Data data);
CUDA_HOS void freeCSRGPU(CSR_Data *data);
CUDA_HOS void freeCSRHost(CSR_Data data);

#endif