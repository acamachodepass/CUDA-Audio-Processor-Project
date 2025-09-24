# Compiler and tools
NVCC = nvcc
CXX = g++

# CUDA and NPP library paths (adjust if needed)
CUDA_PATH ?= /usr/local/cuda

# --- PROJECT STRUCTURE ---
# Directories
SRC_DIR = src
OUT_DIR = output
TARGET = audio_processor

# Source files located in src/
CU_SRCS = $(wildcard $(SRC_DIR)/*.cu)
CPP_SRCS = $(wildcard $(SRC_DIR)/*.cpp)

# Object files will be placed alongside sources in src/
OBJS = $(CU_SRCS:.cu=.o) $(CPP_SRCS:.cpp=.o)

# --- COMPILER AND LINKER FLAGS ---
# Compiler flags
CXXFLAGS = -std=c++17 -O2 -Wall -I$(CUDA_PATH)/include
NVCCFLAGS = -O2 --std=c++17 --compiler-options "-Wall" -arch=sm_75

# Library flags
LDFLAGS = -L$(CUDA_PATH)/lib64
LIBS = -lcudart -lnppc -lnpps -lcufft

# Include path for sources in src/
INCLUDES = -I$(SRC_DIR)

.PHONY: all clean build run

all: clean build run

# --- TARGETS ---
build: $(OUT_DIR) $(TARGET)

# Create output directory if it doesn't exist
$(OUT_DIR):
	mkdir -p $(OUT_DIR)

# Link object files to create the final executable in the root directory
$(TARGET): $(OBJS)
	$(NVCC) $(OBJS) -o $@ $(LDFLAGS) $(LIBS)

# Compile CUDA source files from src/ into object files in src/
$(SRC_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Compile C++ source files from src/ into object files in src/
$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# **MODIFICATION**: Clean command now removes object files from src/, the executable,
# and all "modified_*.wav" files from the output/ directory.
clean:
	rm -f $(OBJS) $(TARGET)
	rm -f $(OUT_DIR)/modified_*.wav

# **MODIFICATION**: The 'run' target now executes the batch script.
run: build
	bash run.sh