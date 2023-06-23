#include "../include/sparse_data_unified.cuh"

/*
A version of sparse_data.cu that creates GPU managed memory
*/


// Returns a host vector so that it can be converted to a device_vector
// when it is convienent to do so.
__host__ CSR_Data buildSparseData(std::string path, uint num_patterns, uint num_features)
{
    // Indeterminate number of values so these are vectors
    std::vector<int> v_colIdx;
    std::vector<float> v_values;

    // Determinate number of rows and labels so these are arrays
    long *rowIdx;
    int *labels;
    int *sparsity;

    cudaHostAlloc(&rowIdx, num_patterns * sizeof(long), cudaHostAllocMapped);
    cudaHostAlloc(&labels, num_patterns * sizeof(int), cudaHostAllocMapped);
    cudaHostAlloc(&sparsity, num_patterns * sizeof(int), cudaHostAllocMapped);

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
    cudaHostAlloc(&colIdx, v_colIdx.size() * sizeof(int), cudaHostAllocMapped);
    std::copy(v_colIdx.begin(), v_colIdx.end(), colIdx);

    float *values;
    cudaHostAlloc(&values, v_values.size() * sizeof(float), cudaHostAllocMapped);
    std::copy(v_values.begin(), v_values.end(), values);

    CSR_Data data; = {sparsity, labels, rowIdx, colIdx, values, num_patterns, v_values.size()};
    return data;
}

// Gets the associated GPU pointers for the unified memory
__host__ CSR_Data *CSRToGPU(CSR_Data data)
{
    // Copy Each of the arrays first
    int *d_sparsity;
    cudaHostGetDevicePointer((void**)&d_sparsity, data.sparsity, 0);

    int *d_labels;
	cudaHostGetDevicePointer((void**)&d_labels, data.labels, 0);    

    long *d_rowIdx;
	cudaHostGetDevicePointer((void**)&d_rowIdx, data.rowIdx, 0);

    int *d_colIdx;
	cudaHostGetDevicePointer((void**)&d_colIdx, data.colIdx, 0);

    float *d_values;
	cudaHostGetDevicePointer((void**)&d_values, data.values, 0);

    // Build a new struct with new pointers and copy it to GPU

	CSR_Data *h_d_data = NULL;
	cudaHostAlloc(&temp, sizeof(CSR_Data), cudaHostAllocMapped);

	temp->sparsity = data.sparsity;
	temp->labels = data.labels;
	temp->rowIdx = data.rowIdx;
	temp->colIdx = data.colIdx;
	temp->values = data.values;
	temp->numPairs = data.numPairs;
	temp->numObservations = data.numObservations;
    
    return h_d_data;
}

// Frees memory allocated for CSR Dataset
__host__ void freeCSRGPU(CSR_Data *h_d_data)
{
	cudaFree(h_d_data);
}

// deletes the arrays created with new in buildSparseData()
__host__ void freeCSRHost(CSR_Data data)
{
    cudaFree(data.sparsity);
	cudaFree(data.labels);
	cudaFree(data.rowIdx);
	cudaFree(data.colIdx);
	cudaFree(data.colIdx);
	cudaFree(temp.values);
}
