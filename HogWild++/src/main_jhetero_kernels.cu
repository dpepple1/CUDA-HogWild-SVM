#include "../include/SVM_jhetero_kernels.cuh"
#include "../include/sparse_data_managed.cuh"
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

// RCV1
//#define FEATURES 47236
//#define PATTERNS 20242 //TRAIN
// #define PATTERNS 677399 //TEST
//#define DATA_PATH "../data/rcv1/rcv1_train_labeled.binary" // TRAIN
// #define DATA_PATH "../data/rcv1/rcv1_test_labeled.binary" // TEST

// WEBSPAM
#define FEATURES 254
#define PATTERNS 350000
#define DATA_PATH "../data/webspam/webspam_labeled.svm"

int main(int argc, char *argv[])
{  
    
    int blocks = 1;
    int threadsPerBlock = 32;
    float learningRate = 0.1;
    float stepDecay = 0.01;
    int epochs = 10;
    bool batchMode = false;
    
    // Read input data
    for(int arg = 1; arg < argc; arg++)
    {
        if(not strcmp(argv[arg], "-b")) // Block Count
        {
            arg++;
            blocks = std::stoi(argv[arg]);
        }
        else if (not strcmp(argv[arg], "-t")) // Threads Per Block
        {
            arg++;
            threadsPerBlock = std::stoi(argv[arg]);
        }
        else if (not strcmp(argv[arg], "-l")) // Learning Rate
        {
            arg++;
            learningRate = std::stof(argv[arg]);
        }
        else if (not strcmp(argv[arg], "-e")) // Epochs
        {
            arg++;
            epochs = std::stoi(argv[arg]);
        }
        else if (not strcmp(argv[arg], "-d")) // Step Decay
        {
            arg++;
            stepDecay = std::stof(argv[arg]);
        }
        else if (not strcmp(argv[arg], "-m")) // Run in  mode
        {
            batchMode = true;
        }
        else
        {
            std::cout << "Invalid Flag: " << argv[arg] << "!" << std::endl;
        }
    }

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	std::cout << "MultiProcessor count: " << deviceProp.multiProcessorCount << std::endl;


    // Data
    CSR_Data *data = buildSparseData(DATA_PATH, PATTERNS, FEATURES);
    HOGSVM svc(learningRate, stepDecay, epochs);
    
    // Train the model and measure time
    timing_t time = svc.fit(data, FEATURES, PATTERNS, blocks, threadsPerBlock, 64);
    
    float accuracy = svc.test(*data);
    std::cout << "Final Accuracy: " << accuracy * 100 << "%" << std::endl;
    
    // Print final weights
    float *weights = svc.getWeights();
    std::cout << "Weights: " << std::endl;
    for(int i = 0; i < FEATURES; i++)
    {
        printf("%09.5f ", weights[i]);
    }
    std::cout << std::endl;

    std::cout << "Bias: " << svc.getBias() << std::endl;

    std::cout << "Kernel Time: " << time.kernelTime << " ns" << std::endl;
    std::cout << "Total Time: " << time.kernelTime + time.mallocTime << " ns" << std::endl;

    if (batchMode)
        std::cerr << accuracy << "," <<  time.kernelTime << "," << time.mallocTime << "," << time.kernelTime + time.mallocTime << std::endl;
    

    return 0;
}

//TODO: Figure out why linearly seperable data is not converging
// Possibilities:
//      Need seperate bias term?
//      C?
