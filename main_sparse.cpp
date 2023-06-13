#include "sparse_data.cuh"

int main()
{
    CSR_Data data = buildSparseData("data/rcv1/rcv1_train_labeled.binary",20242,47236);

    for(int i = 0; i < 10; i++)
    {
        std::cout << data.colIdx[i] << std::endl;
    }
    return 0;
}


// check that data types of arrays are large enough to handle rcv1's size