NVCC = nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall -gencode arch=compute_75,code=sm_75

all: dense sparse

# Dense SVM

dense: main_dense.o SVM_dense.o
	$(NVCC) $^ -o $@

main_dense.o: main_dense.cpp SVM_dense.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

SVM_dense.o: SVM_dense.cu SVM_dense.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

#Sparse SVM

sparse: main_sparse.o SVM_sparse.o sparse_data.o
	$(NVCC) $^ -o $@

main_sparse.o: main_sparse.cpp SVM_sparse.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

SVM_sparse.o: SVM_sparse.cu SVM_sparse.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

sparse_data.o: sparse_data.cu sparse_data.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm *.o dense sparse