#include "../include/sparse_data_managed.cuh"

/*
A version of sparse_data.cu that creates GPU managed memory
*/


// Returns a host vector so that it can be converted to a device_vector
// when it is convienent to do so.
__host__ CSR_Data *buildSparseData(std::string path, uint num_patterns, uint num_features)
{
    // Indeterminate number of values so these are vectors
    std::vector<int> v_colIdx;
    std::vector<float> v_values;

    // Determinate number of rows and labels so these are arrays
    long *rowIdx;
    int *labels;
    int *sparsity;

    cudaMallocManaged(&rowIdx, num_patterns * sizeof(long));
    cudaMallocManaged(&labels, num_patterns * sizeof(int));
    cudaMallocManaged(&sparsity, num_patterns * sizeof(int));

    // Declare variables for file reading
    std::ifstream csv(path, std::ios_base::in);
    std::string line;
    int row = 0;
    int col = 0;
    std::string pair;

    if(csv.good())
    {
        while(std::getline(csv, line)) // For each line
        {
            // Create a vector to put all of the entries into
            std::stringstream ss(line);
            col = 0;
            while(ss >> pair) // Read in each pair one at a time
            {
                if(col == 0)
                {
                    // This is the number of non-zero features
                    sparsity[row] = stoi(pair);
                }
                else if(col == 1)
                {
                    labels[row] = stoi(pair);
                }
                else
                {
                    if(col == 2) // starting a new row
                    {
                        rowIdx[row] = v_colIdx.size();
                        //std::cout <<row << " "<<  v_colIdx.size() << std::endl;
                    }
                    std::string store;
                    std::istringstream in(pair);
                    // Get Index
                    getline(in, store, ':');
                    int idx = stoi(store);
                    // Get Value
                    getline(in, store);
                    float val = stof(store);

                    // Push values to vectors
                    v_colIdx.push_back(idx);
                    v_values.push_back(val);
                    
                }
                col ++;
            }
            row++;
        }
    }
    else
    {
        std::cerr << "Error with file!" << std::endl;
    }   

    csv.close();

    // Build CSR_Data Struct
    int *colIdx;
    cudaMallocManaged(&colIdx, v_colIdx.size() * sizeof(int));
    std::copy(v_colIdx.begin(), v_colIdx.end(), colIdx);

    float *values;
    cudaMallocManaged(&values, v_values.size() * sizeof(float));
    std::copy(v_values.begin(), v_values.end(), values);

    CSR_Data *data;
    cudaMallocManaged(&data, sizeof(CSR_Data));
    data->sparsity = sparsity;
    data->labels = labels;
    data->rowIdx = rowIdx;
    data->colIdx = colIdx;
    data->values = values;
    data->numPairs = num_patterns;
    data->numObservations = v_values.size();

    return data;
}

// Allocates the dataset on the GPU
__host__ CSR_Data *CSRToGPU(CSR_Data data)
{
    // Copy Each of the arrays first
    int *d_sparsity;
    cudaMalloc(&d_sparsity, sizeof(int) * data.numPairs);
    cudaMemcpy(d_sparsity, data.sparsity, data.numPairs * sizeof(int), cudaMemcpyHostToDevice);

    int *d_labels;
    cudaMalloc(&d_labels, sizeof(int) * data.numPairs);
    cudaMemcpy(d_labels, data.labels, data.numPairs * sizeof(int), cudaMemcpyHostToDevice);

    long *d_rowIdx;
    cudaMalloc(&d_rowIdx, sizeof(long) * data.numPairs);
    cudaMemcpy(d_rowIdx, data.rowIdx, data.numPairs * sizeof(long), cudaMemcpyHostToDevice);

    int *d_colIdx;
    cudaMalloc(&d_colIdx, sizeof(int) * data.numObservations);
    cudaMemcpy(d_colIdx, data.colIdx, data.numObservations * sizeof(int), cudaMemcpyHostToDevice);

    float *d_values;
    cudaMalloc(&d_values, sizeof(float) * data.numObservations);
    cudaMemcpy(d_values, data.values, data.numObservations * sizeof(float), cudaMemcpyHostToDevice);

    // Build a new struct with new pointers and copy it to GPU
    CSR_Data temp = {d_sparsity, d_labels, d_rowIdx, d_colIdx, d_values, data.numPairs, data.numObservations};
    
    CSR_Data *d_data;
    cudaMalloc(&d_data, sizeof(CSR_Data));
    cudaMemcpy(d_data, &temp, sizeof(CSR_Data), cudaMemcpyHostToDevice);
    
    return d_data;
}

// Frees memory allocated for CSR Dataset
__host__ void freeCSRGPU(CSR_Data *data)
{
    // We cant access member values because the struct in on the GPU
    CSR_Data temp;
    cudaMemcpy(&temp, data, sizeof(CSR_Data), cudaMemcpyDeviceToHost);
    cudaFree(data);

    // After moving the pointers over, we can now free them
    cudaFree(temp.sparsity);
    cudaFree(temp.labels);
    cudaFree(temp.rowIdx);
    cudaFree(temp.colIdx);
    cudaFree(temp.values);
}

// deletes the arrays created with new in buildSparseData()
__host__ void freeCSRHost(CSR_Data data)
{
    delete[] data.sparsity;
    delete[] data.labels;
    delete[] data.rowIdx;
    delete[] data.colIdx;
    delete[] data.values;
}