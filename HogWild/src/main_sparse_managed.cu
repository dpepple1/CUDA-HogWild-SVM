#include "../include/SVM_sparse_managed.cuh"
#include "../include/sparse_data_managed.cuh"
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

 #define FEATURES 47236
// #define PATTERNS 677399
 #define PATTERNS 20242

 #define DATA_PATH "../data/rcv1/rcv1_train_labeled.binary"

// WEBSPAM
//#define FEATURES 254
//#define	PATTERNS 350000
//#define DATA_PATH "../data/webspam/webspam_labeled.svm"

int main(int argc, char *argv[])
{  
    
    int blocks = 1;
    int threadsPerBlock = 32;
    float learningRate = 0.1;
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
        else if (not strcmp(argv[arg], "-m")) // Run in  mode
        {
            batchMode = true;
        }
        else
        {
            std::cout << "Invalid Flag: " << argv[arg] << "!" << std::endl;
        }
    }

    
    // Data
    CSR_Data *data = buildSparseData(DATA_PATH, PATTERNS, FEATURES);
    HOGSVM svc(0.000001, learningRate, epochs);
    
    // Train the model and measure time
    long elapsedTime = svc.fit(data, FEATURES, PATTERNS, blocks, threadsPerBlock);

    float accuracy = svc.test(data);
    std::cout << "Final Accuracy: " << accuracy * 100 << "%" << std::endl;

    // Print final weights
    float *weights = svc.getWeights();
    std::cout << "Weights: ";
    for(int i = 0; i < 20; i++)
    {
        std::cout << weights[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Bias: " << svc.getBias() << std::endl;

    std::cout << "Time to train: " << elapsedTime << " ns" << std::endl;

    if (batchMode)
        std::cerr << accuracy << "," <<  elapsedTime << std::endl;
    
    freeCSRGPU(data);

    return 0;
}

//TODO: Figure out why linearly seperable data is not converging
// Possibilities:
//      Need seperate bias term?
//      C?
