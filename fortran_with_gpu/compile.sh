# Clean previous build
rm *.o *.mod test.exe

# Set CUDA path (optional if already in PATH)
export CUDA_PATH=/usr/local/cuda

# Compile CUDA wrapper
nvcc -arch=sm_80 -O3  -I/users/cristian/TurboGAP/hop -I/users/cristian/TurboGAP/hop/source/hip  -DHOP_TARGET_CUDA -c cuda_wrappers.cu -o cuda_wrappers.o

# Compile Fortran modules
gfortran -fPIC -O3 -c fortran_cuda_interfaces.f90
gfortran -fPIC -O3 -c soap_turbo_functions.f90
gfortran -fPIC -O3 -c soap_turbo_radial.f90
gfortran -fPIC -O3 -c soap_turbo_radial_operator.f90
gfortran -fPIC -O3 -c test_radial_coeff.f90

# Link everything
gfortran -O3 *.o -L${CUDA_PATH}/lib64 -lcudart -lcublas -lcurand -o test.exe -lopenblas
