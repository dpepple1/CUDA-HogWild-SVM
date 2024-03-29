# SPX Summer REU 2023
**Derek Pepple**

## Introduction
The purpose of this project is to explore the potential and limitations of parallelized stochastic gradient descient when executed using an accelerator architecture. In this case, several variants of Support Vector Machines have been implemented using Nvidia CUDA for operation on a GPU. 

## Folders
### HogWild
This folder contains CUDA code to implement a SVM using the HogWild! algorithm proposed by Feng Niu et al. on an Nvidia GPU.

### HogWild++
This folder is similar to HogWild, however it uses HogWild++, an improved algorithm proprosed by Huan Zhang et al.  

## Project Layout
Both the HogWild and HogWild++ folders contain variations of a SGD SVM that are built on a similar skeleton code. In each folder, the code was duplicated several times and altered to test a different CUDA functionality. You will notice several files that share similar naming schemes.  

### HogWild!
#### Implementations
- **Dense (Generic):** Reads in data represented in a dense format. Uses cudaMallocs and cudaMemcpys to transfer data between device and host memory. 
- **Sparse:** Reads in data represented in a sparse format (col:value). Uses cudaMallocs and cudaMemcpys to transfer data between device and host memory.
- **Sparse Managed:** Reads in data represented in a sparse format (col:value). Uses "managed memory" instead of manually malloced and copied data as in previous implementations.
- **Sparse Unified:** Reads in data represented in a sparse format (col:value). Uses "Zero Copy Memory" instead of manually malloced and copied data. Meant to be tested using a NVIDIA Jetson device that has built in unified host and device memory.
- **Multi Kernel:** Reads in data represented in a sparse format (col:value). Runs two kernels in parallel both performing SGD optimization.
#### Components
- **SVM:** Contains the class definition for the Support Vector Machine as well as the device functions and kernel that perform SGD.
- **sparse_data:** Contains the code to read in sparse data from the file system. 
- **main:** Contains main function to run the program.

### HogWild++
#### Implementations
- **Sparse:** Reads in data represented in a sparse format (col:value). Uses cudaMallocs and cudaMemcpys to transfer betweeen device and host memory.
- **Multi:** Reads in data represented in a sparse format (col:value). Utilizes all threads in a block to perform synchronization instead of just the first thread.
- **Shared:** Reads in data represented in a sparse format (col: value). Operates on data stored in shared (scratchpad) memory. Block being updated moves its data from shared memory into global memory to allow synchronization, and then moves it back to shared memory. 
- **JShared:** Reads in data represented in a sparse format (col:value). Operates on data stored in shared (scratchpad) memory. Adapted to use managed memory to take advantage of physically unified memory of the Tegra architecture on Jetson.
- **JHetero:** Reads in data represented in a sparse format (col:value). An implementation of Hogwild++ where training is performed on the GPU while synchronization happens on the CPU. Intended to run on Jetson.
- **MJHetero:** Reads in data represented in a sparse format (col:value). An heterogeneous implementation of Hogwild++ where two kernels split the training workload. Intended to run on Jetson.
- **JKernels:** Reads in data represented in a sparse format (col:value). A heterogeneous implementation of Hogwild++ where the CPU spawns a kernel in a new stream to perform synchronization. Intended to run on Jetson.

#### Components
- **All components from HogWild!**
- **newton_raphson:** Contains code to calculate the value of Beta (synchronization decay) as dependent on the number of clusters. 


## Operation
Before building the files, change the Makefile to the appropriate CUDA compute capability for your device. The original tests were performed on a Jetson TX1 and an RTX 2060 (capabilities 5.3 and 7.5 respectively).

```
ifeq ($(PLATFORM), x86_64) # compiling on WSL
	NVCC_FLAGS := -g -G -Xcompiler -Wall -gencode arch=compute_75,code=sm_75
	LINK_FLAGS := -arch=sm_75
else
	NVCC_FLAGS := -g -G -Xcompiler -Wall -gencode arch=compute_53,code=sm_53
	LINK_FLAGS := -arch=sm_53
endif
```

To build executables run 
```
make all

# OR (for individual executables)
make main       #HogWild! only
make sparse
make unified    #HogWild! only
make managed    #HogWild! only
make multi      
make shared     #HogWild++ only
make jshared    #HogWild++ only
make shared     #Hogwild++ only  
make jhetero    #Hogwild++ only
make mjhetero   #Hogwild++ only 
make jkernels   #Hogwild++ only 

```
from within either the HogWild or HogWild++ folder.

## Data
### Dense Data
The data folder contains subfolders for several different datsets that can be used to test the SVM implementations. Dense formatted datasets are stored as names that indicate the dimensionality of the dataset, as well as the standard deviation. For instance f2_std70 means the each data point has 2 dimensions and the dataset has a standard deviation of 0.7. The lin_sep set is also a dense dataset, however it is linearly seperable, unlike the others. Each dense set contains a blobs.csv and a blobs_classes.csv. Classes are either 1 or -1, and the datapoints are stored as tab seperated values.

### Sparse Data
The sparse datasets tested on this SVM were the Reuters RCV1 binary classification dataset and unigram Webspam. The files necessary for this operation are not included with the repository. To download them run (from the top directory):
```
# Create RCV1 Directory
mkdir data/rcv1/
cd data/rcv1/

# Download RCV1 Files
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_test.binary.bz2

# Extract RCV1 files
bzip2 -d rcv1_train.binary.bz2
bzip2 -d rcv1_test.binary.bz2

mkdir ../webspam/
cd ../webspam/

# Download Webspam Files
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/webspam_wc_normalized_unigram.svm.xz

# Extract Webspam Files
unxz webspam_wc_normalized_unigram.svm.xz
```

Due to the nature of sparse formatted data, it each observation may have a unique number of non-zero values. To allow the program to know how much memory to allocate before reading in values, a number must be added to each line of the files indicating how many non-zero features are present. This process can be accelerated with the sparse_loaded.py utility. In data folder, run this to create the files:

```
python3 pre_processing/sparse_labeler.py rcv1/rcv1_train.binary rcv1/rcv1_train_labeled.binary
python3 pre_processing/sparse_labeler.py rcv1/rcv1_test.binary rcv1/rcv1_test_labeled.binary
python3 pre_processing/sparse_labeler.py webspam/webspam_wc_normalized_unigram.svm.xz webspam/webspam_labeled.svm
```
To test using these datsets, you must use either the sparse or managed executable. To choose which dataset to test on, check the associated main file and make sure to set the correct path as well as uncomment the correct #define statement for the number of patterns (larger number is for the test set).

## Other Notes:
### Timeout
When tested on the NVIDIA Jetson TX1, the unified sparse dataset was noted as having a much longer kernel runtime. On Linux, CUDA kernel launches are limited to 5 seconds to prevent the GUI interface of the device from hanging. To accurately test the application this timeout protection needs to be disabled. Nvidia reccomends against doing this however. Here is how to disable the timeout:
```
sudo -s 
echo N > /sys/kernel/debug/gpu.0/timeouts_enabled
```
