#include "../include/SVM.hpp"
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

#define FEATURES 10
#define PATTERNS 10000

#define DATA_PATH "f10_std100"
//#define DATA_PATH "lin_sep"

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
    float patterns[PATTERNS][FEATURES];
    int labels[PATTERNS];

    // Bring in features from CSV file 
    std::string blob_url = "../data/";
    std::ifstream feat_csv(blob_url + DATA_PATH + "/blobs.csv", std::ios_base::in);
    std::string line;
    int row = 0;
    int col = -1;
    float val;

    if(feat_csv.good())
    {
        while(std::getline(feat_csv, line))
        {
            std::stringstream ss(line);
            col = -1;
            while(ss >> val)
            {
                if(col != -1)
                    patterns[row][col] = val;
                col++;
            }
            row ++;
        }
    }

    feat_csv.close();

    // Bring in class labels from CSV file

    std::ifstream label_csv(blob_url + DATA_PATH + "/blobs_classes.csv");
    int label;
    row = 0;
    while(label_csv >> row >> label)
    {
        // Labels must be 1 or -1;
        labels[row] = (label == 1) ? 1 : -1; 
    }
    
    HOGSVM svc(0.000001, learningRate, epochs);
    
    // Train the model and measure time
    timing_t time = svc.fit((float*)patterns, FEATURES, labels, PATTERNS, blocks, threadsPerBlock);
    
    // Test the model
    float accuracy = svc.test((float*)patterns, labels);
    std::cout << "Final Accuracy: " << accuracy * 100 << "%" << std::endl;
    
    // Print final weights
    float *weights = svc.getWeights();
    std::cout << "Weights: ";
    for(int i = 0; i < FEATURES; i++)
    {
        std::cout << weights[i] << " ";
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
