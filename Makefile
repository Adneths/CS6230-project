.PHONY: all clean

CXXFLAGS += -Wall -std=c++11

LIB_PATHS += -L$(BEBOP_UTIL_LIB) -lbebop_util
INCLUDE_PATHS += -I$(BEBOP_UTIL_INCLUDE)
LIB_PATHS += -L$(BEBOP_SPARSE_LIB) -lsparse_matrix_converter
INCLUDE_PATHS += -I$(BEBOP_SPARSE_INCLUDE)

CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INCLUDE_PATHS += -I$(CUDA_TOOLKIT)/include
LIB_PATHS += -L$(CUDA_MATH_LIB) -lcusparse -lnvToolsExt

LDFLAGS += $(LIB_PATHS)
CXXFLAGS += $(INCLUDE_PATHS)
NVCCFLAGS += $(INCLUDE_PATHS)
NVCCFLAGS += -arch=sm_61

INCLUDE_PATH = include
SRC_PATH = src
INCLUDE_PATHS += -I$(INCLUDE_PATH)

HEADERS = typedef.h timer.h SpGEMM_cuda.h SpGEMM_util.h SpGEMM_cusparse.h SpMSpV_cuda.h SpMM_cuda.h SpMSpV_bucket.h SpMM_gcoo.h SpMM_cusparse.h

SPGEMMSRC = spgemm_test.cpp SpGEMM_cuda.cu SpGEMM_cusparse.cu SpGEMM_util.cu

SPMSPVSRC = spmspv_test.cpp SpMSpV_cuda.cu SpMSpV_bucket.cu

SPMMSRC = spmm_test.cpp SpMM_cuda.cu SpMM_cusparse.cu SpMM_gcoo.cu

INCLUDE_HEADERS = ./$(INCLUDE_PATH)/$(shell echo $(HEADERS) | sed 's/ / \.\/$(INCLUDE_PATH)\//g')

SPGEMMSRC_SOURCES= ./$(SRC_PATH)/$(shell echo $(SPGEMMSRC) | sed 's/ / \.\/$(SRC_PATH)\//g')
SPMSPVSRC_SOURCES= ./$(SRC_PATH)/$(shell echo $(SPMSPVSRC) | sed 's/ / \.\/$(SRC_PATH)\//g')
SPMMSRC_SOURCES= ./$(SRC_PATH)/$(shell echo $(SPMMSRC) | sed 's/ / \.\/$(SRC_PATH)\//g')

all: spgemm spgemm_profile spmspv spmspv_profile spmm spmm_profile

spgemm: $(SPGEMMSRC_SOURCES) $(INCLUDE_HEADERS)
	nvcc $(NVCCFLAGS) $(SPGEMMSRC_SOURCES) -o $@ $(LDFLAGS)

spgemm_profile: $(SPGEMMSRC_SOURCES) $(INCLUDE_HEADERS)
	nvcc $(NVCCFLAGS) $(SPGEMMSRC_SOURCES) -D PROFILE -o $@ $(LDFLAGS)

spmspv: $(SPMSPVSRC_SOURCES) $(INCLUDE_HEADERS)
	nvcc $(NVCCFLAGS) $(SPMSPVSRC_SOURCES) -o $@ $(LDFLAGS)

spmspv_profile: $(SPMSPVSRC_SOURCES) $(INCLUDE_HEADERS)
	nvcc $(NVCCFLAGS) $(SPMSPVSRC_SOURCES) -D PROFILE -o $@ $(LDFLAGS)

spmm: $(SPMMSRC_SOURCES) $(INCLUDE_HEADERS)
	nvcc $(NVCCFLAGS) $(SPMMSRC_SOURCES) -o $@ $(LDFLAGS)

spmm_profile: $(SPMMSRC_SOURCES) $(INCLUDE_HEADERS)
	nvcc $(NVCCFLAGS) $(SPMMSRC_SOURCES) -D PROFILE -o $@ $(LDFLAGS)

clean:
	rm -f spgemm spgemm_profile spmspv spmspv_profile spmm spmm_profile *.o
