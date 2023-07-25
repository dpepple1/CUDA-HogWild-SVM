/*
A C++ support vector machine that uses stochastic gradient descent optimization
implemented using the HOGWILD! algorithm for parrallization on GPUs.

by Derek Pepple
*/
#ifndef HOGSVM_H
#define HOGSVM_H

#ifdef __CUDACC__
#define CUDA_DEV __device__
#define CUDA_GLO __global__
#define CUDA_HOS __host__
#else
#define CUDA_DEV
#define CUDA_GLO
#define CUDA_HOS
#endif

#include <iostream>
#include <curand_kernel.h>
#include <chrono>
#include "sparse_data_managed.cuh"
#include "../include/newton_raphson.hpp"


// extern float betaVals[] = {0, 0.5, 0.61803, 0.68232, 0.72449, 0.75487, 0.77808, 0.79654, 0.81165, 0.82430, 0.83507,
// 0.84439, 0.85255, 0.85975, 0.86618, 0.87195, 0.87716, 0.88191, 
// };

struct timing_t
{
  long kernelTime;
  long mallocTime;  
};

class HOGSVM
{
    public:
        // Constructor/Destructor pair
        HOGSVM(float lambda, float learningRate, uint epochsPerCore);
        ~HOGSVM();

        // Public methods
        CUDA_HOS timing_t fit(CSR_Data *data, uint features, uint numPairs,
                            int blocks, int threadsPerBlock, int tau);
        CUDA_HOS float test(CSR_Data data);
        CUDA_HOS float* getWeights();
        CUDA_HOS float getBias();

    private:
        float learningRate;
        float stepDecay;
        uint epochsPerCore;

        uint features;
        uint numPairs;

        float bias;
        float *weights;

        CUDA_HOS void initWeights(uint features);
        CUDA_HOS int *setupGPULabels(int *labels, uint numPairs);
        CUDA_HOS float *setupGPUPatterns(float *patterns, uint features, uint numPairs);
        CUDA_HOS void freeTrainingData(float *d_patterns, int *d_labels);
};

CUDA_GLO void SGDKernel(uint threadCount, curandState_t *states, CSR_Data *d_data,
                            float *activeWeights, float *snapshotWeights, float *activeBias, float *snapshotBias,
                            uint features, uint numPairs, uint epochs, float learningRate,
                            float stepDecay, int *token, int* sync, float beta, float lambda, int blocks, int tau, int *iterBuf);
CUDA_HOS CUDA_DEV int predict(float *d_weights, float bias, float *d_pattern, uint features );       
CUDA_DEV float setGradient(float *wGrad, int trueLabel, int decision, float *row, uint features);
CUDA_DEV float setGradientSIMT(float *wGrad, int trueLabel, int decision, float *row, uint features);
CUDA_DEV void updateSparseModel(float *d_weights, float *bias, float *wGrad, float bGrad, float learningRate, int *colIdxs, int sparsity);
CUDA_HOS CUDA_DEV float testAccuracy(CSR_Data data, uint features, 
                            float *weights, float bias, uint numPairs);

#endif