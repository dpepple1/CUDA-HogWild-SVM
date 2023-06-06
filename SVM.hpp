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

class HOGSVM
{
    public:
        // Constructor/Destructor pair
        HOGSVM(float lambda, float learningRate, uint iterationsPerCore);
        ~HOGSVM();

        // Public methods
        CUDA_HOS void fit(float *patterns, uint features, int *labels, uint numPairs,
                            int blocks, int threadsPerBlock);
        CUDA_HOS float test(float *test_patterns, int *test_labels);
        CUDA_HOS float* getWeights();

    private:
        float lambda;
        float learningRate;
        uint iterationsPerCore;

        uint features;
        uint numPairs;

        float weights[];

        CUDA_HOS void initWeights(uint features);
        CUDA_HOS int *setupGPULabels(int *labels, uint numPairs);
        CUDA_HOS float *setupGPUPatterns(float *patterns, uint features, uint numPairs);
        CUDA_HOS void freeTrainingData(float *d_patterns, int *d_labels);
};

CUDA_GLO void SGDKernel(uint threadCount, curandState_t *states, float *d_patterns, 
                            int *d_labels, uint features, uint numPairs, 
                            uint iterations, float *d_weights, float learningRate);
CUDA_HOS CUDA_DEV int predict(float *d_weights, float *d_pattern, uint features );       
CUDA_DEV void setGradient(float *grad, int trueLabel, int decision, float *row, uint features);
CUDA_DEV void updateModel(float *d_weights, float *grad, uint features, float learningRate);
CUDA_HOS CUDA_DEV float testAccuracy(float *patterns, int *labels, uint features, float *weights, uint numPairs);



#endif