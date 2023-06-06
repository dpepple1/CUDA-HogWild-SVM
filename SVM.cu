/*
A C++ support vector machine that uses stochastic gradient descent optimization
implemented using the HOGWILD! algorithm for parrallization on GPUs.

by Derek Pepple
*/

#include "SVM.hpp"
#include <stdio.h>
#include <unistd.h>


__host__ HOGSVM::HOGSVM(float lambda, float learningRate, uint iterationsPerCore)
{
    this->lambda = lambda; // Still am not using anywhere
    this->learningRate = learningRate;
    this->iterationsPerCore = iterationsPerCore;
}

__host__ HOGSVM::~HOGSVM()
{
    //Free any variables 
}

__host__ void HOGSVM::initWeights(uint features)
{
    srand((unsigned) time(NULL));
    
    for(uint i = 0; i < features; i++)
    {
        float r = (float)rand() / (float)RAND_MAX;
        weights[i] = r;
    } 
}

// Copies labels of training data to GPU memory
__host__ int *HOGSVM::setupGPULabels(int *labels, uint numPairs)
{
    int *d_labels;
    cudaMalloc(&d_labels, numPairs * sizeof(int));
    cudaMemcpy(d_labels, labels, numPairs * sizeof(int), cudaMemcpyHostToDevice);

    return d_labels;
}

// Copies patterns of training data to GPU memory
__host__ float *HOGSVM::setupGPUPatterns(float *patterns, uint features, uint numPairs)
{
    /*
        Note: patterns will be stored as a flattened array instead of
        the 2D array that they are initially entered as.
    */
    
    float *d_patterns;
    cudaMalloc(&d_patterns, features * numPairs * sizeof(float));
    cudaMemcpy(d_patterns, patterns, features * numPairs * sizeof(float), cudaMemcpyHostToDevice);
    
    return d_patterns;
}

// Free Training CUDA Memory
__host__ void HOGSVM::freeTrainingData(float *d_patterns, int *d_labels)
{
    cudaFree(d_patterns);
    cudaFree(d_labels);
}

__host__ void HOGSVM::fit(float *patterns, uint features, int *labels, uint numPairs, int blocks, int threadsPerBlock)
{
    this->features = features;
    this->numPairs = numPairs;

    // Create SVM weights and copy to GPU Memory
    weights[features] = {};
    initWeights(features);

    float *d_weights = 0;
    cudaMalloc(&d_weights, features * sizeof(float));
    cudaMemcpy(d_weights, weights, features * sizeof(float), cudaMemcpyHostToDevice);

    // Set up curand states
    curandState_t* states;
    cudaMalloc((void**)&states, blocks * threadsPerBlock * sizeof(curandState_t));

    //Allocate GPU training data
    int *d_labels = setupGPULabels(labels, numPairs);
    float *d_patterns = setupGPUPatterns(patterns, features, numPairs);

    // Spawn threads to begin SGD Process
    SGDKernel<<<blocks, threadsPerBlock>>>(blocks * threadsPerBlock, states,
                         d_patterns, d_labels, features, numPairs, iterationsPerCore, 
                         d_weights, learningRate);

    // Wait for threads to finish and collect weights
    cudaDeviceSynchronize();
    cudaMemcpy(weights, d_weights, features * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_weights);
    cudaFree(states);
    freeTrainingData(d_patterns, d_labels);
}

__host__ float HOGSVM::test(float *test_patterns, int *test_labels)
{
    float result = testAccuracy(test_patterns, test_labels, features, weights, numPairs);
    return result;
}

__host__ float* HOGSVM::getWeights()
{
    return weights;
}


// ========================
//      CUDA FUNCTIONS
// ========================

// __device__ prediction that can be used by the kernel when training
__host__ __device__ int predict(float *weights, float *pattern, uint features)
{
    // To make prediction we need to calculate dot product
    float dotProd = 0;
    for(uint dim = 0; dim < features; dim++)
    {
        dotProd += (weights[dim] * pattern[dim]);
    }

    return dotProd > 0 ? 1 : -1; // Must be either 1 or -1
}

// Sets the grad array to the gradient of the hinge loss 
__device__ void setGradient(float *grad, int trueLabel, int decision, float *row, uint features)
{
    if (1 - trueLabel * decision <= 0)
    {
        memset(grad, 0, features * sizeof(float));
    }
    else
    {
        for(int comp = 0; comp < features; comp++)
        {
            grad[comp] = -trueLabel * row[comp];
        }
    }
}

// Use the gradient to update the model vector
__device__ void updateModel(float *d_weights, float *grad, uint features, float learningRate)
{
    for(int dim = 0; dim < features; dim++)
    {
        // multiply by learning rate
        d_weights[dim] -= learningRate * grad[dim]; // Would it be += or -= here?
    }   
}

// CUDA Kernel that performs HOGWILD! SGD
__global__ void SGDKernel(uint threadCount, curandState_t *states, float *d_patterns, 
                            int *d_labels, uint features, uint numPairs, 
                            uint iterations, float *d_weights, float learningRate) {   
    // Compute Thread Index
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    // partition the dataset into chunks by moving thread specific pointer
    uint pairsPerThread = numPairs / threadCount;
    float *patternStart = d_patterns + ((i * pairsPerThread) * features);
    int *labelStart = d_labels + (i * pairsPerThread);

    // Declaring here to avoid re-allocing
    float *copyWeights = new float[features];
    float *grad = new float[features];
    curand_init(123, i, 0, &states[i]);

    // Start the training process
    for(uint iter = 0; iter < iterations; iter++)
    {
        // Get random number in range of number of patterns in the chunk
        float randNum = curand_uniform(&states[i]);
        uint randIdx = (uint) (randNum * pairsPerThread);

        // Set a pointer to the start of a random row within the chunk
        float *row = patternStart + (features * randIdx);
        int *trueLabel = labelStart + randIdx;

        // Make a copy of the current weights for this thread
        memcpy(copyWeights, d_weights, features * sizeof(float));

        // Make prediction and set gradient
        int decision = predict(copyWeights, row, features);
        setGradient(grad, *trueLabel, decision, row, features);
        
        // Update model vector
        updateModel(d_weights, grad, features, learningRate);

        // Test epochs
        if(i == 0 && iter % (iterations / 10) == 0)
        {
            float sample = testAccuracy(d_patterns, d_labels, features, copyWeights, numPairs);
            printf("Iteration %d : Accuracy %f%\n", iter, sample);
        }

    }  

    delete[] copyWeights;
    delete[] grad;
}

__host__ __device__ float testAccuracy(float *patterns, int *labels, uint features,
                                        float *weights, uint numPairs) {
    int correct = 0;
    int total = 0;

    for(uint pair = 0; pair < numPairs; pair++)
    {
        float *pattern = patterns + (pair * features);
        int label = labels[pair];

        int prediction = predict(weights, pattern, features);

        if(prediction == label)
            correct++;
        
        total++;
    }

    return (float)correct / (float)total;
}