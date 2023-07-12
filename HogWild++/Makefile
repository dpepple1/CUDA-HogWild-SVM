NVCC := nvcc
NVCC_FLAGS := -g -G -Xcompiler -Wall -gencode arch=compute_75,code=sm_75 -dc
LINK_FLAGS := -arch=sm_75

INC := include
OBJ := obj
SRC := src
BIN := bin

all: sparse multi shared hetero

sparse: $(BIN)/sparse
multi: $(BIN)/multi
shared: $(BIN)/shared
hetero: $(BIN)/hetero


# Sparse SVM
$(BIN)/sparse: $(OBJ)/main_sparse.o $(OBJ)/sparse_data.o $(OBJ)/SVM_sparse.o $(OBJ)/newton_raphson.o
	$(NVCC) $(LINK_FLAGS) $^ -o $@
	
$(OBJ)/main_sparse.o: $(SRC)/main_sparse.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ)/SVM_sparse.o: $(SRC)/SVM_sparse.cu $(INC)/SVM_sparse.cuh 
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ)/sparse_data.o: $(SRC)/sparse_data.cu $(INC)/sparse_data.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@


# Sparse SVM with multi-thread update
$(BIN)/multi: $(OBJ)/main_sparse.o $(OBJ)/sparse_data.o $(OBJ)/SVM_sparse_multi.o $(OBJ)/newton_raphson.o
	$(NVCC) $(LINK_FLAGS) $^ -o $@

$(OBJ)/SVM_sparse_multi.o: $(SRC)/SVM_sparse_multi.cu $(INC)/SVM_sparse.cuh 
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Sparse SVM with shared memory
$(BIN)/shared: $(OBJ)/main_sparse.o $(OBJ)/sparse_data.o $(OBJ)/SVM_sparse_shared.o $(OBJ)/newton_raphson.o
	$(NVCC) $(LINK_FLAGS) $^ -o $@

$(OBJ)/SVM_sparse_shared.o: $(SRC)/SVM_sparse_shared.cu $(INC)/SVM_sparse.cuh 
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@
	
# Sparse SVM with Heterogenous updates
$(BIN)/hetero: $(OBJ)/main_sparse_hetero.o $(OBJ)/sparse_data.o $(OBJ)/SVM_sparse_hetero.o $(OBJ)/newton_raphson.o
	$(NVCC) $(LINK_FLAGS) $^ -o $@
	
$(OBJ)/main_sparse_hetero.o: $(SRC)/main_sparse_hetero.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ)/SVM_sparse_hetero.o: $(SRC)/SVM_sparse_hetero.cu $(INC)/SVM_sparse_hetero.cuh 
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@


# Resources
$(OBJ)/newton_raphson.o: $(SRC)/newton_raphson.cpp $(INC)/newton_raphson.hpp
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@


clean:
	rm $(BIN)/* $(OBJ)/*.o
