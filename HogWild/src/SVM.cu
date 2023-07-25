/*
A C++ support vector machine that uses stochastic gradient descent optimization
implemented using the HOGWILD! algorithm for parallelization on GPUs.

by Derek Pepple
*/

#include "../include/SVM.hpp"
#include <stdio.h>
#include <unistd.h>


__host__ HOGSVM::HOGSVM(float lambda, float learningRate, uint epochsPerCore)
{
    this->lambda = lambda; // Still am not using anywhere
    this->learningRate = learningRate;
    this->epochsPerCore = epochsPerCore;
    weights = NULL;
}

__host__ HOGSVM::~HOGSVM()
{
    //Free weights array
    if(weights != NULL)
        delete[] weights;
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

__host__ timing_t HOGSVM::fit(float *patterns, uint features, int *labels, 
                            uint numPairs, int blocks, int threadsPerBlock) {

    auto mallocStart = std::chrono::steady_clock::now();
    
    this->features = features;
    this->numPairs = numPairs;

    // Create SVM weights and copy to GPU Memory
    weights = new float[features]();
    //weights[features] = {};
    initWeights(features);
    
    bias = 0;
    
    float *d_weights = 0;
    cudaMalloc(&d_weights, features * sizeof(float));
    cudaMemcpy(d_weights, weights, features * sizeof(float), cudaMemcpyHostToDevice);
    
    float *d_bias = 0;
    cudaMalloc(&d_bias, sizeof(float));
    cudaMemcpy(d_bias, &bias, sizeof(float), cudaMemcpyHostToDevice);
    
    // Set up curand states
    curandState_t* states;
    cudaMalloc((void**)&states, blocks * threadsPerBlock * sizeof(curandState_t));

    //Allocate GPU training data
    int *d_labels = setupGPULabels(labels, numPairs);
    float *d_patterns = setupGPUPatterns(patterns, features, numPairs);
    
    // Spawn threads to begin SGD Process
    auto kernelStart = std::chrono::steady_clock::now();
    SGDKernel<<<blocks, threadsPerBlock>>>(blocks * threadsPerBlock, states,
                         d_patterns, d_labels, features, numPairs, epochsPerCore, 
                         d_weights, d_bias, learningRate);

    // Wait for threads to finish and collect weights
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();

    cudaMemcpy(weights, d_weights, features * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&bias, d_bias, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_weights);
    cudaFree(states);
    freeTrainingData(d_patterns, d_labels);

    timing_t time;
    time.mallocTime = std::chrono::duration_cast<std::chrono::nanoseconds>(kernelStart - mallocStart).count();
    time.kernelTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - kernelStart).count();

    return time;
}

__host__ float HOGSVM::test(float *test_patterns, int *test_labels)
{
    float result = testAccuracy(test_patterns, test_labels, features, weights, bias, numPairs);
    return result;
}

__host__ float* HOGSVM::getWeights()
{
    return weights;
}

__host__ float HOGSVM::getBias()
{
    return bias;
}

// ========================
//      CUDA FUNCTIONS
// ========================

// __device__ prediction that can be used by the kernel when training
__host__ __device__ int predict(float *weights, float bias, float *pattern, uint features)
{
    // To make prediction we need to calculate dot product
    float dotProd = 0;
    for(uint dim = 0; dim < features; dim++)
    {
        dotProd += (weights[dim] * pattern[dim]);
    }

    dotProd += bias;

    return dotProd > 0 ? 1 : -1; // Must be either 1 or -1
}

// Sets the wGrad array to the gradient of the hinge loss and returns the bGrad
__device__ float setGradient(float *wGrad, int trueLabel, int decision, float *row, uint features)
{
    if (1 - trueLabel * decision <= 0)
    {
        memset(wGrad, 0, features * sizeof(float));
        return 0;
    }
    else
    {
        for(int comp = 0; comp < features; comp++)
        {
            wGrad[comp] = -trueLabel * row[comp];
        }
        return (float) -trueLabel;
    }
}

// Sets the wGrad array to the gradient of the hinge loss and returns the bGrad 
// ***WITHOUT USING CONTROL FLOW DEVIATIONS (IF BLOCKS)***
__device__ float setGradientSIMT(float *wGrad, int trueLabel, int decision, float *row, uint features)
{
    // Whether the prediction was right or wrong
    int classification = 1 - trueLabel * decision;
    // Will be zero if incorrect and 1 if correct
    int modifier = (int)(classification > 0);

    for(int comp = 0; comp < features; comp++)
    {
        wGrad[comp] = -trueLabel * row[comp] * modifier;
    }
    return (float) (-trueLabel) * (modifier);   
}

// Use the gradient to update the model vector and bias
__device__ void updateModel(float *d_weights, float *bias, float *wGrad, float bGrad, uint features, float learningRate)
{
    for(int dim = 0; dim < features; dim++)
    {
        // multiply by learning rate
        // d_weights[dim] -= learningRate * wGrad[dim]; // Would it be += or -= here?

        float deltaW = -1 * learningRate * wGrad[dim];
        atomicAdd(&d_weights[dim], deltaW);
    }   

    // *bias -= learningRate * bGrad;
    float deltaB = -1 * learningRate * bGrad;
    atomicAdd(bias, deltaB);

}

// CUDA Kernel that performs HOGWILD! SGD
__global__ void SGDKernel(uint threadCount, curandState_t *states, float *d_patterns, 
                            int *d_labels, uint features, uint numPairs, 
                            uint epochs, float *d_weights, float *d_bias,
                            float learningRate) {   
    // Compute Thread Index
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    // partition the dataset into chunks by moving thread specific pointer
    uint pairsPerThread = numPairs / threadCount;
    float *patternStart = d_patterns + ((i * pairsPerThread) * features);
    int *labelStart = d_labels + (i * pairsPerThread);

    // Declaring here to avoid re-allocing
    float *copyWeights = new float[features];
    float *wGrad = new float[features];
    curand_init(123, i, 0, &states[i]);

    // Start the training process
    for(uint epoch = 0; epoch < epochs; epoch++)
    {
        // bool terminalEpoch = true;

        //Each epoch theoretically goes over the whole dataset
        for(uint iter = 0; iter < pairsPerThread; iter++)
        {
             // Get random number in range of number of patterns in the chunk
            float randNum = curand_uniform(&states[i]);
            uint randIdx = (uint) (randNum * pairsPerThread);

            // Set a pointer to the start of a random row within the chunk
            float *row = patternStart + (features * randIdx);
            int *trueLabel = labelStart + randIdx;

            // Make a copy of the current weights and bias for this thread
            memcpy(copyWeights, d_weights, features * sizeof(float));

            // Make prediction and set gradient
            int decision = predict(copyWeights, *d_bias, row, features);
            float bGrad = setGradientSIMT(wGrad, *trueLabel, decision, row, features);
            
            // if(terminalEpoch && bGrad != 0)
            //     terminalEpoch = false;
            
            // Update model vector
            updateModel(d_weights, d_bias, wGrad, bGrad, features, learningRate);
        }

        // If this thread has done nothing for the last epoch, stop
        // if(terminalEpoch)
        // {
        //     float sample = testAccuracy(d_patterns, d_labels, features, copyWeights, *d_bias, numPairs);
        //     printf("Thread %d finished after epoch %d with accuracy: %2.3f\n", i, epoch, sample * 100);
        //     return;
        // }
    }

    delete[] copyWeights;
    delete[] wGrad;
}

__host__ __device__ float testAccuracy(float *patterns, int *labels, uint features,
                                        float *weights, float bias, uint numPairs) {
    int correct = 0;
    int total = 0;

    for(uint pair = 0; pair < numPairs; pair++)
    {
        float *pattern = patterns + (pair * features);
        int label = labels[pair];

        int prediction = predict(weights, bias, pattern, features);

        if(prediction == label)
            correct++;
        
        total++;
    }

    return (float)correct / (float)total;
}


// TODO:
// Figure out how the bias gradient needs to be applied
// Figure out how the multithreading interacts with the bias update. 

// Before testing speedup, do something about the print statements in thread 1