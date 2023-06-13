#include "sparse_data.cuh"


// Returns a host vector so that it can be converted to a device_vector
// when it is convienent to do so.
__host__ SparseDataset buildSparseData(std::string path, uint num_patterns, uint num_features)
{
    // Create list of host_vectors to append to
    std::cout << num_patterns << std::endl;
    thrust::host_vector<SparseEntry> *patterns = new thrust::host_vector<SparseEntry>[num_patterns]();
    int labels[num_patterns];

    // Declare variables for file reading
    std::ifstream csv(path, std::ios_base::in);
    std::string line;
    int row = 0;
    int col = -1;
    std::string pair;

    if(csv.good())
    {
        while(std::getline(csv, line)) // For each line
        {
            // Create a vector to put all of the entries into
            std::stringstream ss(line);
            col = -1;
            while(ss >> pair) // Read in each pair one at a time
            {
                if(col == -1)
                {
                    labels[row] = stoi(pair);
                }
                else
                {
                    std::string store;
                    std::istringstream in(pair);
                    getline(in, store, ':');
                    int idx = stoi(store);
                    getline(in, store);
                    float val = stof(store);
                    SparseEntry entry = {idx, val};
                    patterns[row].push_back(entry);
                }
                col ++;
            }
            row++;
        }
    }   

    csv.close();

    SparseDataset dataset = {patterns, labels};
    return dataset;

}