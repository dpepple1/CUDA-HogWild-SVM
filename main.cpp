#include "SVM.hpp"
#include <fstream>
#include <string>

#define FEATURES 2
#define PATTERNS 1000

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
        else if(not strcmp(argv[arg], "-b")) // Run in batch mode
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
    std::ifstream feat_csv("data/lin_sep/blobs.csv", std::ios_base::in);

    int row;
    float x, y;
    while(feat_csv >> row >> x >> y)
    {
        patterns[row][0] = x;
        patterns[row][1] = y;
    }

    feat_csv.close();

    // Bring in class labels from CSV file

    std::ifstream label_csv("data/lin_sep/blobs_classes.csv");
    int label;
    while(label_csv >> row >> label)
    {
        
        // Labels must be 1 or -1;
        labels[row] = (label == 1) ? 1 : -1; 
    }

    HOGSVM svc(0.000001, learningRate, epochs);

    // Train the model and measure time
    int elapsedTime = svc.fit((float*)patterns, FEATURES, labels, PATTERNS, blocks, threadsPerBlock);

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

    std::cout << "Time to train: " << elapsedTime << " ns" << std::endl;

    if (batchMode)
        std::cerr << " " << elapsedTime << std::endl;

    return 0;
}

//TODO: Figure out why linearly seperable data is not converging
// Possibilities:
//      Need seperate bias term?
//      C?