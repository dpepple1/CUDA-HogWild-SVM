#include "sparse_data.cuh"


// Returns a host vector so that it can be converted to a device_vector
// when it is convienent to do so.
__host__ CSR_Data buildSparseData(std::string path, uint num_patterns, uint num_features)
{
    // Indeterminate number of values so these are vectors
    std::vector<int> v_colIdx;
    std::vector<float> v_values;

    // Determinate number of rows and labels so these are arrays
    int *rowIdx = new int[num_patterns];
    int *labels = new int[num_patterns];

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

    csv.close();

    // Build CSR_Data Struct
    int *colIdx = new int[v_colIdx.size()];
    std::copy(v_colIdx.begin(), v_colIdx.end(), colIdx);

    float *values = new float[v_values.size()];
    std::copy(v_values.begin(), v_values.end(), values);

    CSR_Data data = {labels, rowIdx, colIdx, values};

    return data;

}