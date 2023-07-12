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

all: sparse multi shared hetero jshared

sparse: $(BIN)/sparse
multi: $(BIN)/multi
shared: $(BIN)/shared
hetero: $(BIN)/hetero
jshared: $(BIN)/jshared


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

# Sparse SVM with shared memory and unified memory (JETSON)
$(BIN)/jshared: $(OBJ)/main_jshared.o $(OBJ)/sparse_data_managed.o $(OBJ)/SVM_jshared.o $(OBJ)/newton_raphson.o
	$(NVCC) $(LINK_FLAGS) $^ -o $@
	
$(OBJ)/main_jshared.o: $(SRC)/main_jshared.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ)/SVM_jshared.o: $(SRC)/SVM_jshared.cu $(INC)/SVM_jshared.cuh 
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

#$(OBJ)/sparse_data_unified.o: $(SRC)/sparse_data_unified.cu $(INC)/sparse_data_unified.cuh
#	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ)/sparse_data_managed.o: $(SRC)/sparse_data_managed.cu $(INC)/sparse_data_managed.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@



# Resources
$(OBJ)/newton_raphson.o: $(SRC)/newton_raphson.cpp $(INC)/newton_raphson.hpp
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@


clean:
	rm $(BIN)/* $(OBJ)/*.o
