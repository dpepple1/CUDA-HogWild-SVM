PLATFORM = $(shell uname -m)

NVCC := nvcc
ifeq ($(PLATFORM), x86_64) # compiling on WSL
	NVCC_FLAGS := -g -G -Xcompiler -Wall -gencode arch=compute_75,code=sm_75
	LINK_FLAGS := -arch=sm_75
else
	NVCC_FLAGS := -g -G -Xcompiler -Wall -gencode arch=compute_53,code=sm_53
	LINK_FLAGS := -arch=sm_53
endif

INC := include
OBJ := obj
SRC := src
BIN := bin

all: main sparse managed unified multi

main: $(BIN)/main
sparse: $(BIN)/sparse
managed: $(BIN)/managed
unified: $(BIN)/unified
multi: $(BIN)/multi

# Dense SVM
$(BIN)/main: $(OBJ)/main.o $(OBJ)/SVM.o
	$(NVCC) $(LINK_FLAGS) $^ -o $@

$(OBJ)/main.o: $(SRC)/main.cpp $(INC)/SVM.hpp
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ)/SVM.o: $(SRC)/SVM.cu $(INC)/SVM.hpp
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Sparse SVM
$(BIN)/sparse: $(OBJ)/main_sparse.o $(OBJ)/sparse_data.o $(OBJ)/SVM_sparse.o
	$(NVCC) $(LINK_FLAGS) $^ -o $@
	
$(OBJ)/main_sparse.o: $(SRC)/main_sparse.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ)/SVM_sparse.o: $(SRC)/SVM_sparse.cu $(INC)/SVM_sparse.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ)/sparse_data.o: $(SRC)/sparse_data.cu $(INC)/sparse_data.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Sparse SVM with Managed Memory
$(BIN)/managed: $(OBJ)/main_sparse_managed.o $(OBJ)/sparse_data_managed.o $(OBJ)/SVM_sparse_managed.o
	$(NVCC) $(LINK_FLAGS) $^ -o $@

$(OBJ)/main_sparse_managed.o: $(SRC)/main_sparse_managed.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ)/SVM_sparse_managed.o: $(SRC)/SVM_sparse_managed.cu $(INC)/SVM_sparse_managed.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ)/sparse_data_managed.o: $(SRC)/sparse_data_managed.cu $(INC)/sparse_data_managed.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Sparse SVM with Unified Memory
$(BIN)/unified: $(OBJ)/main_sparse_unified.o $(OBJ)/sparse_data_unified.o $(OBJ)/SVM_sparse_unified.o
	$(NVCC) $(LINK_FLAGS) $^ -o $@

$(OBJ)/main_sparse_unified.o: $(SRC)/main_sparse_unified.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ)/SVM_sparse_unified.o: $(SRC)/SVM_sparse_unified.cu $(INC)/SVM_sparse_unified.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ)/sparse_data_unified.o: $(SRC)/sparse_data_unified.cu $(INC)/sparse_data_unified.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Sparse SVM with Multiple Kernels
$(BIN)/multi: $(OBJ)/main_multi_kernel.o $(OBJ)/sparse_data.o $(OBJ)/SVM_multi_kernel.o
	$(NVCC) $(LINK_FLAGS) $^ -o $@

$(OBJ)/main_multi_kernel.o: $(SRC)/main_multi_kernel.cu 
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ)/SVM_multi_kernel.o: $(SRC)/SVM_multi_kernel.cu $(INC)/SVM_multi_kernel.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm $(BIN)/* $(OBJ)/*.o
