# Sparse Multiplication on GPU

## Introduction


## Requirements

- [Bebop Sparse Matrix Conversion Library](http://bebop.cs.berkeley.edu/smc/)
- CUDA >6.1
- CuSparse

### Environmental Variables

```
export CUDA_MATH_LIB=${CUSPARSE_LIB_PATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${BEBOP_INSTALL_DIR}/bebop_util
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${BEBOP_INSTALL_DIR}/sparse_matrix_converter/
export BEBOP_SPARSE_LIB=${BEBOP_INSTALL_DIR}/sparse_matrix_converter/
export BEBOP_UTIL_LIB=${BEBOP_INSTALL_DIR}/bebop_util/
export BEBOP_SPARSE_INCLUDE=${BEBOP_INSTALL_DIR}/sparse_matrix_converter/include/
export BEBOP_UTIL_INCLUDE=${BEBOP_INSTALL_DIR}/bebop_util/include/
```

On perlmutter `CUSPARSE_LIB_PATH` should be `/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/math_libs/11.7/lib64`

## Usage



### SpGEMM

The executable multiplies the input with itself and will run with all avaliable GPUs detected using `cudaGetDeviceCount()`. Sbatch the `test_spgemm_*gpu.sh` script to schedule a task with a set number of GPUs (Running with more GPUs than rows could cause issues). Use `spgemm_profile` for a NVTX enabled version.

```
Usage: ./spgemm <harwell-boeing-file> <algorithm:0>
Algorithm: 0 - Full Dense Accumulator
         : 1 - Full Sparse Accumulator
```

#### Examples

```
# Runs spgemm on the GD97_b dataset with a dense accumulator
./spgemm data/GD97_b/GD97_b.rb 0

# Profiles spgemm on the micromass_10NN dataset with a sparse accumulator
nsys profile -s none --trace=cuda,nvtx,osrt,cusparse --force-overwrite true \
    ./spgemm_profile data/micromass_10NN/micromass_10NN.rb 1
```

### SpMSpV

```
Usage: ./spmspv <harwell-boeing-file>
```

### SpMM

## Notes

- Bebop library has a maximum size on the input it could read which is approximately $80k \times 80k$
- Bebop library automatically sets the symmetric_type of loaded matrix as *sypmmetric(1)*, which causes mistakes in the csc_to_csr() function. Mannually set the symmetric_type  as *unsymmetric*, or the csc_to_csr() function just assign the pointers to the counterparts (so that you get a transposed matrix of the original)
