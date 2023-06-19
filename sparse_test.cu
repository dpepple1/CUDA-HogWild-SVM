#include "sparse_data.cuh"

__global__ void testKernel(CSR_Data *d_data)
{
    for(uint i = 0; i < d_data->numPairs; i++)
    {
        printf("%d\n", d_data->labels[i]);
    }
    printf("End\n");
}


int main()
{
    CSR_Data data = buildSparseData("data/rcv1/rcv1_train_labeled.binary",20242,47236);

    CSR_Data *d_data = CSRToGPU(data);

    testKernel<<<1,1>>>(d_data);
    cudaDeviceSynchronize();

    freeCSRGPU(d_data);
    freeCSRHost(data);

    return 0;
}


