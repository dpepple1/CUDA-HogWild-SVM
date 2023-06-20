# SPX Summer REU 2023
**Derek Pepple**

## Introduction
The purpose of this project is to explore the potential and limitations of parallelized stochastic gradient descient when executed using an accelerator architecture. In this case, several variants of Support Vector Machines have been implemented using Nvidia CUDA for operation on a GPU. 

## Operation
To build executables run 
```
make all
```

## Data
### Dense Data
The data folder contains subfolders for several different datsets that can be used to test the SVM implementations. Dense formatted datasets are stored as names that indicate the dimensionality of the dataset, as well as the standard deviation. For instance f2_std70 means the each data point has 2 dimensions and the dataset has a standard deviation of 0.7. The lin_sep set is also a dense dataset, however it is linearly seperable, unlike the others. Each dense set contains a blobs.csv and a blobs_classes.csv. Classes are either 1 or -1, and the datapoints are stored as tab seperated values.

### Sparse Data
The sparse dataset tested on this SVM was the Reuters RCV1 binary classification dataset. The files necessary for this operation are not included with the repository. To download them run (from the top directory):
```
# Create RCV1 Directory
mkdir data/rcv1/
cd data/rcv1

# Download Files
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_test.binary.bz2

#Extract files
bzip2 -d rcv1_train.binary.bz2
bzip2 -d rcv1_test.binary.bz2
```

Due to the nature of sparse formatted data, it each observation may have a unique number of non-zero values. To allow the program to know how much memory to allocate before reading in values, a number must be added to each line of the files indicating how many non-zero features are present. This process can be accelerated with the sparse_loaded.py utility. In rcv1 folder, run this to create the files:

```
python3 sparse_labeler.py rcv1_train.binary rcv1_train_labeled.binary
python3 sparse_labeler.py rcv1_test.binary rcv1_test_labeled.binary
```
To test using these datsets, you must use either the sparse or managed executable. To choose which dataset to test on, check the associated main file and make sure to set the correct path as well as uncomment the correct #define statement for the number of patterns (larger number is for the test set).
