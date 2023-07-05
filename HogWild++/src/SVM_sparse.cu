/*
A C++ support vector machine that uses stochastic gradient descent optimization
implemented using the HOGWILD++ algorithm for parallelization on GPUs.

by Derek Pepple
*/

#include "../include/SVM_sparse.cuh"
#include <stdio.h>
#include <unistd.h>


__host__ HOGSVM::HOGSVM(float learningRate, float stepDecay, uint epochsPerCore)
{
    this->learningRate = learningRate;
    this->stepDecay = stepDecay;
    this->epochsPerCore = epochsPerCore;
    weights = NULL;
}

__host__ HOGSVM::~HOGSVM()
{
    //Free weights array
    if(weights != NULL)
        delete[] weights;
}

// __host__ void HOGSVM::initWeights(uint features)
// {
//     srand((unsigned) time(NULL));
    
//     for(uint i = 0; i < features; i++)
//     {
//         float r = (float)rand() / (float)RAND_MAX;
//         r = (r * 2) - 1;
//         // weights[i] = r;
//         weights[i] = 0;
//     } 
// }

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

__host__ timing_t HOGSVM::fit(CSR_Data data, uint features, uint numPairs, int blocks, int threadsPerBlock)
{
    this->features = features;
    this->numPairs = numPairs;

    // Calculate special coefficients
    // DO NOT INCLUDE IN TIMING
    float beta = newtonRaphson(0.5, blocks);
    float lambda =  1 - pow(beta, blocks - 1); 

    auto mallocStart = std::chrono::steady_clock::now();
    // Create a local copy of weights
    weights = new float[features]();
    bias = 0;
    
    // Set up curand states
    curandState_t* states;
    cudaMalloc((void**)&states, blocks * threadsPerBlock * sizeof(curandState_t));

    //Allocate GPU training data
    CSR_Data *d_data = CSRToGPU(data);

    const size_t weightSize = sizeof(float) * features;
    
    // Create a token in global GPU Memory
    int *token = NULL;
    cudaMalloc(&token, sizeof(int));
    cudaMemset(token, 0, sizeof(int));

    // Because shared memory doesnt work, use global memory (THIS IS ALOT OF MEMORY)
    float *activeWeights  = NULL;
    cudaMalloc(&activeWeights, features * sizeof(float) * blocks);
    
    float *snapshotWeights = NULL;
    cudaMalloc(&snapshotWeights, features * sizeof(float) * blocks);
    
    float *activeBias = NULL;
    cudaMalloc(&activeBias, sizeof(float) * blocks);

    float *snapshotBias = NULL;
    cudaMalloc(&snapshotBias, sizeof(float) * blocks);


    printf("Starting Threads\n");
    // Spawn threads to begin SGD Process
    auto kernelStart = std::chrono::steady_clock::now();
    SGDKernel<<<blocks, threadsPerBlock, weightSize>>>(blocks * threadsPerBlock, states, d_data, 
                        activeWeights, snapshotWeights, activeBias, snapshotBias,
                        features, numPairs, epochsPerCore, learningRate,
                        stepDecay, token, beta, lambda, blocks);

    // Wait for threads to finish and collect weights
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();

    cudaMemcpy(weights, activeWeights, sizeof(float) * features, cudaMemcpyDeviceToHost);
    cudaMemcpy(&bias, activeBias, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(states);
    freeCSRGPU(d_data);
    cudaFree(activeWeights);
    cudaFree(snapshotWeights);
    cudaFree(activeBias);
    cudaFree(snapshotBias);
    cudaFree(token);
    

    cudaError_t error = cudaGetLastError();
    std::cout << "Last Error: " << error << std::endl;

    timing_t time;
    time.mallocTime = std::chrono::duration_cast<std::chrono::nanoseconds>(kernelStart - mallocStart).count();
    time.kernelTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - kernelStart).count();

    return time;
}

__host__ float HOGSVM::test(CSR_Data data)
{
    float result = testAccuracy(data, features, weights, bias, numPairs);
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

    //return dotProd > 0 ? 1 : -1; // Must be either 1 or -1
    return ((dotProd > 0) * 2) - 1;
}

// Sets the wGrad array to the gradient of the hinge loss and returns the bGrad
__device__ float setGradient(float *wGrad, int trueLabel, int decision, float *row, uint features)
{
    if ((1 - trueLabel * decision) <= 0)
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
__device__ void updateSparseModel(float *d_weights, float *bias, float *wGrad, float bGrad, float learningRate, int *colIdxs, int sparsity)
{

    //Only update the features where the gradient will not be zero
    for(int dim = 0; dim < sparsity; dim++)
    {
        //multiply by learning rate
        float deltaW = -1 * learningRate * wGrad[dim];
        int column = colIdxs[dim];
        atomicAdd(&d_weights[column], deltaW);
        // d_weights[column] += deltaW;
    }

    float deltaB = -1 * learningRate * bGrad;
    atomicAdd(bias, deltaB); 
    //*bias += deltaB;
}


// CUDA Kernel that performs HOGWILD! SGD
__global__ void SGDKernel(uint threadCount, curandState_t *states, CSR_Data *d_data,
                            float *activeWeights, float *snapshotWeights, float *activeBias, float *snapshotBias,
                            uint features, uint numPairs, uint epochs, float learningRate,
                            float stepDecay, int *token, float beta, 
                            float lambda, int blocks) {   
    // Compute Thread Index
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Get block specific pointer to seperate weights
    float *blockLocalWeights = activeWeights + (blockIdx.x * features);
    float *blockSnapshotWeights = snapshotWeights + (blockIdx.x * features);

    float *blockLocalBias = activeBias + blockIdx.x;
    float *blockSnapshotBias = snapshotBias + blockIdx.x;

    // partition the dataset into chunks by moving thread specific pointer

    uint pairsPerThread = numPairs / threadCount;
    int *labelStart = d_data->labels + (i * pairsPerThread); // Pointer to first label for thread
    int *sparsityStart = d_data->sparsity + (i * pairsPerThread);
    int subsetStart = i * pairsPerThread; // Index to look in row start for row    

    // Declaring here to avoid re-allocing
    curand_init(123, i, 0, &states[i]);


    int sendToken = -1;
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
                copyWeights[idx] = blockLocalWeights[d_data->colIdx[colStart + idx]];
            }
            
            
            //Make a prediction and set gradient
            int decision = predict(copyWeights, *blockLocalBias, d_data->values + valuesStart, sparsity); //Double check this 
            float bGrad = setGradientSIMT(wGrad, trueLabel, decision, d_data->values + valuesStart, sparsity);
            updateSparseModel(blockLocalWeights, blockLocalBias, wGrad, bGrad, learningRate, d_data->colIdx + colStart, sparsity);

            delete[] copyWeights;
            delete[] wGrad;
            
            // After an epoch, synchronize to next block
            if(sendToken == iter)
            {
                *(token) = blockIdx.x + 1;
                if(*token >= blocks)
                    *token = 0;
                sendToken = -1;
            }
            // Check token and perform synchronization step if necessary
            if((*token) == blockIdx.x and threadIdx.x == 0) // ONLY FIRST THREAD **CHECK LATER**
            {
                // Make sure you dont update out of bounds!
                float *nextLocalWeights = NULL;
                if(blockIdx.x + 1 >= blocks)
                    nextLocalWeights = activeWeights;
                else
                    nextLocalWeights = blockLocalWeights + features;

                // Handling one component at a time in the attempt to be more efficient
                for(int dim = 0; dim < features; dim++)
                {
                    float weightDifference = blockLocalWeights[dim] - snapshotWeights[dim];
                    // Refer to HogWild++ synchronization update formula
                    blockSnapshotWeights[dim] = lambda * nextLocalWeights[dim] + (1 - lambda) * blockSnapshotWeights[dim] + beta * pow(stepDecay, epoch*iter) * weightDifference;
                    // Update next active weights
                    atomicAdd(nextLocalWeights + dim, beta * pow(stepDecay, epoch *iter) * weightDifference);
                    blockLocalWeights[dim] = blockSnapshotWeights[dim];
                }

                // Update Bias (Inferred from weights calculation)
                float *nextLocalBias = blockLocalBias + 1;
                float biasDifference = *blockLocalBias - *snapshotBias;
                *blockSnapshotBias = lambda * (*nextLocalBias) + (1-lambda) * (*blockSnapshotBias) + beta * pow(stepDecay, epoch*iter) * biasDifference;
                atomicAdd(nextLocalBias, beta * pow(stepDecay, epoch*iter) * biasDifference);
                *blockLocalBias = *blockSnapshotBias;

                sendToken = iter;
                *token = -1;
            }
        }
    }
}

__host__ __device__ float testAccuracy(CSR_Data data, uint features, float *weights, float bias, uint numPairs) 
{
    int correct = 0;
    int total = 0;

    float *copyWeights = new float[features];

    for(uint pair = 0; pair < numPairs; pair++)
    {
        int valuesStart = data.rowIdx[pair];
        int colStart = valuesStart;
        float *pattern = data.values + valuesStart;
        int sparsity = data.sparsity[pair];
        int label = data.labels[pair];

        for(int idx = 0; idx < sparsity; idx++)
        {
            // For each non-sparse value copy the weight from the matching column
            copyWeights[idx] = weights[data.colIdx[colStart + idx]];
        }

        int prediction = predict(copyWeights, bias, pattern, sparsity);
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
