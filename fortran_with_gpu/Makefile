# Compiler
FC = ftn
FCFLAGS = -fPIC -O3 -h flex_mp=intolerant
HIPCC = CC
HIPFLAGS = -xhip -munsafe-fp-atomics --offload-arch=gfx90a -O3

# ROCm libraries
ROCM_LIB = $(ROCM_PATH)/lib
ROCM_LFLAGS = -L$(ROCM_LIB) -lamdhip64 -lhiprand -lhipblas -lsci_cray

# Directories
SRC = .
BUILD = build
BIN = bin

# Source files, ensuring fortran_cuda_interfaces.f90 is compiled first
SRCS = $(SRC)/fortran_cuda_interfaces.f90 $(SRC)/soap_turbo_functions.f90 $(SRC)/soap_turbo_radial.f90 $(SRC)/soap_turbo_radial_operator.f90 $(SRC)/test_radial_coeff.f90
OBJS = $(patsubst $(SRC)/%.f90, $(BUILD)/%.o, $(SRCS))

# HIP source file
HIP_SRC = $(SRC)/cuda_wrappers.cu
HIP_OBJ = $(BUILD)/cuda_wrappers.o

# Target executable
TARGET = $(BIN)/test.exe

# Default target
all: $(TARGET)

# Build rules
$(BUILD)/%.o: $(SRC)/%.f90 | $(BUILD)
	$(FC) $(FCFLAGS) -c $< -o $@

$(HIP_OBJ): $(HIP_SRC) | $(BUILD)
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

$(TARGET): $(OBJS) $(HIP_OBJ) | $(BIN)
	$(FC) $(FCFLAGS) $(OBJS) $(HIP_OBJ) $(ROCM_LFLAGS) -o $(TARGET)

# Create directories if they don't exist
$(BUILD):
	mkdir -p $(BUILD)

$(BIN):
	mkdir -p $(BIN)

# Clean rule
clean:
	rm -rf $(BUILD)/*.o $(BUILD)/*.mod $(TARGET)

.PHONY: all clean
