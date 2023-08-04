PLATFORM = $(shell uname -m)
CC = $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader|head -n 1)
NVCC := nvcc


ifeq ($(PLATFORM), x86_64) # compiling on WSL
	ifeq ($(CC), 8.9)
		NVCC_FLAGS := -g -G -Xcompiler -Wall -gencode arch=compute_89,code=sm_89
		LINK_FLAGS := -arch=sm_89
	else
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

