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

INCLUDE_PATH = include
SRC_PATH = src
INCLUDE_PATHS += -I$(INCLUDE_PATH)

SOURCES = main.cpp
HEADERS = typedef.h timer.h SpGEMM_cuda.h SpGEMM_cusparse.h TileSpGEMM_cuda.h
CUDA_SOURCES = SpGEMM_cuda.cu SpGEMM_cusparse.cu TileSpGEMM_cuda.cu

INCLUDE_HEADERS = ./$(INCLUDE_PATH)/$(shell echo $(HEADERS) | sed 's/ / \.\/$(INCLUDE_PATH)\//g')
SRC_SOURCES= ./$(SRC_PATH)/$(shell echo $(SOURCES) | sed 's/ / \.\/$(SRC_PATH)\//g')
SRC_CUDA_SOURCES= ./$(SRC_PATH)/$(shell echo $(CUDA_SOURCES) | sed 's/ / \.\/$(SRC_PATH)\//g')

all: spgemm spgemm_p

spgemm: $(SRC_SOURCES) $(INCLUDE_HEADERS) $(SRC_CUDA_SOURCES)
	nvcc $(NVCCFLAGS) $(SRC_SOURCES) $(SRC_CUDA_SOURCES) -o $@ $(LDFLAGS)

spgemm_p: $(SRC_SOURCES) $(INCLUDE_HEADERS) $(SRC_CUDA_SOURCES)
	nvcc $(NVCCFLAGS) $(SRC_SOURCES) $(SRC_CUDA_SOURCES) -D PROFILE -o $@ $(LDFLAGS)

clean:
	rm -f spgemm spgemm_p *.o
