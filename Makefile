NVCC = nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall -gencode arch=compute_75,code=sm_75

all: main

main: main.o SVM.o
	$(NVCC) $^ -o $@

main.o: main.cpp SVM.hpp
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

SVM.o: SVM.cu SVM.hpp
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

sparse_data.o: sparse_data.cu sparse_data.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@



sparse: main_sparse.o sparse_data.o
	$(NVCC) $^ -o $@
	
main_sparse.o: main_sparse.cpp
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm main *.o