NVCC := nvcc
NVCC_FLAGS := -g -G -Xcompiler -Wall -gencode arch=compute_75,code=sm_75

INC := include
OBJ := obj
SRC := src
BIN := bin

all: sparse

sparse: $(BIN)/sparse

# Sparse Shared SVM
$(BIN)/sparse: $(OBJ)/main_sparse.o $(OBJ)/sparse_data.o $(OBJ)/SVM_sparse.o $(OBJ)/newton_raphson.o
	$(NVCC) $^ -o $@
	
$(OBJ)/main_sparse.o: $(SRC)/main_sparse.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ)/SVM_sparse.o: $(SRC)/SVM_sparse.cu $(INC)/SVM_sparse.cuh 
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ)/sparse_data.o: $(SRC)/sparse_data.cu $(INC)/sparse_data.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@


# Resources
$(OBJ)/newton_raphson.o: $(SRC)/newton_raphson.cpp $(INC)/newton_raphson.hpp
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@


clean:
	rm $(BIN)/* $(OBJ)/*.o
