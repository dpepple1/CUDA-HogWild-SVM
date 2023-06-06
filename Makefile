NVCC = nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall -arch sm_75

all: main

main: main.o SVM.o
	$(NVCC) $^ -o $@

main.o: main.cpp SVM.hpp
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

SVM.o: SVM.cu SVM.hpp
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm main *.o