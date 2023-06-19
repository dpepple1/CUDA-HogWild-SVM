NVCC := nvcc
NVCC_FLAGS := -g -G -Xcompiler -Wall -gencode arch=compute_75,code=sm_75

INC := include
OBJ := obj
SRC := src
BIN := bin

all: main sparse

main: $(BIN)/main

sparse: $(BIN)/sparse

$(BIN)/main: $(OBJ)/main.o $(OBJ)/SVM.o
	$(NVCC) $^ -o $@

$(OBJ)/main.o: $(SRC)/main.cpp $(INC)/SVM.hpp
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ)/SVM.o: $(SRC)/SVM.cu $(INC)/SVM.hpp
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ)/sparse_data.o: $(SRC)/sparse_data.cu $(INC)/sparse_data.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@



$(BIN)/sparse: $(OBJ)/main_sparse.o $(OBJ)/sparse_data.o $(OBJ)/SVM_sparse.o
	$(NVCC) $^ -o $@
	
$(OBJ)/main_sparse.o: $(SRC)/main_sparse.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ)/SVM_sparse.o: $(SRC)/SVM_sparse.cu $(INC)/SVM_sparse.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm $(BIN)/* $(OBJ)/*.o