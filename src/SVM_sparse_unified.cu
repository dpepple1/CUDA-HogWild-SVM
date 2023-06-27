/*
A C++ support vector machine that uses stochastic gradient descent optimization
implemented using the HOGWILD! algorithm for parallelization on GPUs.

by Derek Pepple
*/

#include "../include/SVM_sparse_managed.cuh"
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
        cudaFreeHost(weights);
}

__host__ void HOGSVM::initWeights(uint features)
{
    cudaHostAlloc(&weights, features * sizeof(float), cudaHostAllocMapped);

    srand((unsigned) time(NULL));
    
    for(uint i = 0; i < features; i++)
    {
        float r = (float)rand() / (float)RAND_MAX;
        r = (r * 2) - 1;
        //weights[i] = r;
        weights[i] = 0;
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

__host__ long HOGSVM::fit(CSR_Data *data, uint features, uint numPairs, int blocks, int threadsPerBlock)
{
    //auto begin = std::chrono::steady_clock::now();
    
    this->features = features;
    this->numPairs = numPairs;

    // Create SVM weights and copy to GPU Memory
    weights = NULL;
    initWeights(features);
    float *d_weights = NULL;
    cudaHostGetDevicePointer((void**)&d_weights, weights, 0);    

    bias = NULL;
    cudaHostAlloc(&bias, sizeof(float), cudaHostAllocMapped);
    float *d_bias = NULL;
    cudaHostGetDevicePointer((void**)&d_bias, bias, 0);
    
    // Set up curand states
    curandState_t* states;
    cudaHostAlloc(&states, blocks * threadsPerBlock * sizeof(curandState_t), cudaHostAllocMapped);
    curandState_t* d_states;
    cudaHostGetDevicePointer((void**)&d_states, states, 0);

    printf("Starting Threads\n");
    // Spawn threads to begin SGD Process
    
    auto begin = std::chrono::steady_clock::now();
    SGDKernel<<<blocks, threadsPerBlock>>>(blocks * threadsPerBlock, d_states,
                        data, features, numPairs, epochsPerCore, d_weights,
                        d_bias, learningRate);

    // Wait for threads to finish and collect weights
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();

    cudaError_t error = cudaGetLastError();
    std::cout << "Last Error: " << error << std::endl;

    cudaFreeHost(states);

    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
}

__host__ float HOGSVM::test(CSR_Data *data)
{    
    float result = testAccuracy(data, features, weights, *bias, numPairs);
    return result;
}

__host__ float* HOGSVM::getWeights()
{
    return weights;
}

__host__ float HOGSVM::getBias()
{
    return *bias;
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

// Use the gradient to update the model vector and bias
__device__ void updateSparseModel(float *d_weights, float *bias, float *wGrad, float bGrad, float learningRate, int *colIdxs, int sparsity)
{

    //Only update the features where the gradient will not be zero
    for(int dim = 0; dim < sparsity; dim++)
    {
        //multiply by learning rate
        float deltaW = -1 * learningRate * wGrad[dim];
        int column = colIdxs[dim];
        //atomicAdd(&d_weights[column], deltaW);
        d_weights[column] += deltaW;
    }

    float deltaB = -1 * learningRate * bGrad;
    //atomicAdd(bias, deltaB); 
    *bias += deltaB;
}


// CUDA Kernel that performs HOGWILD! SGD
__global__ void SGDKernel(uint threadCount, curandState_t *states, CSR_Data *d_data,
                            uint features, uint numPairs, uint epochs, float *d_weights,
                            float *d_bias, float learningRate) {   
    // Compute Thread Index
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    // partition the dataset into chunks by moving thread specific pointer
    uint pairsPerThread = numPairs / threadCount;
    int *labelStart = d_data->labels + (i * pairsPerThread); // Pointer to first label for thread
    int *sparsityStart = d_data->sparsity + (i * pairsPerThread);
    int subsetStart = i * pairsPerThread; // Index to look in row start for row    

    // Declaring here to avoid re-allocing
    curand_init(123, i, 0, &states[i]);

    // Start the training process
    for(uint epoch = 0; epoch < epochs; epoch++)
    {

        //Each epoch theoretically goes over the whole dataset
        for(uint iter = 0; iter < pairsPerThread; iter++)
        {
            // Get random number in range of number of patterns in the chunk
            float randNum = curand_uniform(&states[i]);
            uint randIdx = (uint) (randNum * pairsPerThread);

            int sparsity = *(sparsityStart + randIdx);

            // NOTE: Moved this into the body of the loop because the full size was too large
            float *copyWeights = new float[sparsity];
            float *wGrad = new float[sparsity];
            
            // Find start of values
            int trueLabel = *(labelStart + randIdx);
            int valuesStart = d_data->rowIdx[subsetStart + randIdx];
            int colStart = valuesStart;

            // Copy over relavant weights
            for(uint idx = 0; idx < sparsity; idx++)
            {
                // For each non-sparse value copy the weight from the matching column
                // int index = d_data->colIdx[colStart + idx];
                // float temp = d_weights[index];
                // copyWeights[idx] = temp;
                copyWeights[idx] = d_weights[d_data->colIdx[colStart + idx]];
            }
            
            
            //Make a prediction and set gradient
            int decision = predict(copyWeights, *d_bias, d_data->values + valuesStart, sparsity); //Double check this 
            float bGrad = setGradient(wGrad, trueLabel, decision, d_data->values + valuesStart, sparsity);
            updateSparseModel(d_weights, d_bias, wGrad, bGrad, learningRate, d_data->colIdx + colStart, sparsity);

            delete[] copyWeights;
            delete[] wGrad;
        }

    }
}

__host__ __device__ float testAccuracy(CSR_Data *data, uint features, float *weights, float bias, uint numPairs) 
{
    int correct = 0;
    int total = 0;

    float *copyWeights = new float[features];

    for(uint pair = 0; pair < numPairs; pair++)
    {
        int valuesStart = data->rowIdx[pair];
        int colStart = valuesStart;
        float *pattern = data->values + valuesStart;
        int sparsity = data->sparsity[pair];
        int label = data->labels[pair];

        for(int idx = 0; idx < sparsity; idx++)
        {
            // For each non-sparse value copy the weight from the matching column
            copyWeights[idx] = weights[data->colIdx[colStart + idx]];
        }

        int prediction = predict(copyWeights, bias, pattern, sparsity);

        if(prediction == label)
            correct++;

        total++;
    }

    return (float)correct / (float)total;
}
