# Compiler
FC = ftn
FCFLAGS = -fPIC -O3 -h flex_mp=intolerant

# Directories
SRC = .
BUILD = build
BIN = bin

# Source files
SRCS = $(SRC)/soap_turbo_functions.f90 $(SRC)/soap_turbo_radial.f90 $(SRC)/soap_turbo_radial_operator.f90 $(SRC)/test_radial_coeff.f90
OBJS = $(patsubst $(SRC)/%.f90, $(BUILD)/%.o, $(SRCS))

# Target executable
TARGET = $(BIN)/test.exe

# Default target
all: $(TARGET)

# Build rules
$(BUILD)/%.o: $(SRC)/%.f90 | $(BUILD)
	$(FC) $(FCFLAGS) -c $< -o $@

$(TARGET): $(OBJS) | $(BIN)
	$(FC) $(FCFLAGS) $(OBJS) -o $(TARGET)

# Create directories if they don't exist
$(BUILD):
	mkdir -p $(BUILD)

$(BIN):
	mkdir -p $(BIN)

# Clean rule
clean:
	rm -rf $(BUILD)/*.o $(BUILD)/*.mod $(TARGET)

.PHONY: all clean
