.PHONY: all clean

CXXFLAGS += -Wall -std=c++11

LIB_PATHS += -L$(BEBOP_UTIL_LIB) -lbebop_util
INCLUDE_PATHS += -I$(BEBOP_UTIL_INCLUDE)
LIB_PATHS += -L$(BEBOP_SPARSE_LIB) -lsparse_matrix_converter
INCLUDE_PATHS += -I$(BEBOP_SPARSE_INCLUDE)

CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INCLUDE_PATHS += -I$(CUDA_TOOLKIT)/include
LIB_PATHS += -L$(CUDA_MATH_LIB) -lcusparse

LDFLAGS += $(LIB_PATHS)
CXXFLAGS += $(INCLUDE_PATHS)
NVCCFLAGS += $(INCLUDE_PATHS)

INCLUDE_PATH = include
SRC_PATH = src

SOURCES = main.cpp
HEADERS = typedef.h timer.h SpGEMM_cuda.h SpGEMM_cusparse.h
CUDA_SOURCES = SpGEMM_cuda.cu SpGEMM_cusparse.cu

all: main profile

main: $(SOURCES) $(HEADERS) $(CUDA_SOURCES)
#	CC $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
	nvcc $(NVCCFLAGS) $(SOURCES) $(CUDA_SOURCES) -o $@ $(LDFLAGS)

profile: $(SOURCES) $(HEADERS) $(CUDA_SOURCES)
	nvcc $(NVCCFLAGS) $(SOURCES) $(CUDA_SOURCES) -D PROFILE -o $@ $(LDFLAGS)

clean:
	rm -f main profile *.o
