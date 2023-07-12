PLATFORM = $(shell uname -m)

NVCC := nvcc

ifeq ($(PLATFORM), x86_64) # compiling on WSL
	NVCC_FLAGS := -g -G -Xcompiler -Wall -gencode arch=compute_75,code=sm_75
	
else
	NVCC_FLAGS := -g -G -Xcompiler -Wall -gencode arch=compute_75,code=sm_75
endif

INC := include
OBJ := obj
SRC := src
BIN := bin

all: sparse multi shared

sparse: $(BIN)/sparse
multi: $(BIN)/multi
shared: $(BIN)/shared


# Sparse SVM
$(BIN)/sparse: $(OBJ)/main_sparse.o $(OBJ)/sparse_data.o $(OBJ)/SVM_sparse.o $(OBJ)/newton_raphson.o
	$(NVCC) $^ -o $@
	
$(OBJ)/main_sparse.o: $(SRC)/main_sparse.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ)/SVM_sparse.o: $(SRC)/SVM_sparse.cu $(INC)/SVM_sparse.cuh 
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ)/sparse_data.o: $(SRC)/sparse_data.cu $(INC)/sparse_data.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@


# Sparse SVM with multi-thread update
$(BIN)/multi: $(OBJ)/main_sparse.o $(OBJ)/sparse_data.o $(OBJ)/SVM_sparse_multi.o $(OBJ)/newton_raphson.o
	$(NVCC) $^ -o $@

$(OBJ)/SVM_sparse_multi.o: $(SRC)/SVM_sparse_multi.cu $(INC)/SVM_sparse.cuh 
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

#Sparse SVM with shared memory
$(BIN)/shared: $(OBJ)/main_sparse.o $(OBJ)/sparse_data.o $(OBJ)/SVM_sparse_shared.o $(OBJ)/newton_raphson.o
	$(NVCC) $^ -o $@

$(OBJ)/SVM_sparse_shared.o: $(SRC)/SVM_sparse_shared.cu $(INC)/SVM_sparse.cuh 
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@
	

# Resources
$(OBJ)/newton_raphson.o: $(SRC)/newton_raphson.cpp $(INC)/newton_raphson.hpp
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@


clean:
	rm $(BIN)/* $(OBJ)/*.o
