rm *.o *.mod test.exe;
export ROCM_PATH=/opt/rocm-6.0.3
CC -xhip -munsafe-fp-atomics --offload-arch=gfx90a -O3 -c  cuda_wrappers.cu
ftn -fPIC -O3  -h flex_mp=intolerant -c fortran_cuda_interfaces.f90
ftn -fPIC -O3  -h flex_mp=intolerant -c soap_turbo_functions.f90
ftn -fPIC -O3  -h flex_mp=intolerant -c soap_turbo_radial.f90
ftn -fPIC -O3  -h flex_mp=intolerant -c soap_turbo_radial_operator.f90
ftn -fPIC -O3  -h flex_mp=intolerant -c test_radial_coeff.f90
ftn -fPIC -O3  -h flex_mp=intolerant -L/${ROCM_PATH}/lib -lamdhip64 -lhiprand -lsci_cray -lhipblas  *.o -o  test.exe
