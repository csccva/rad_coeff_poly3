rm *.o *.mod; 

#ftn  -fPIC -O3 -h flex_mp=intolerant -lsci_cray  soap_turbo_functions.f90 soap_turbo_compress.f90 soap_turbo_angular.f90 soap_turbo_radial.f90 soap_turbo.f90  test.f90 
#nvcc --resource-usage -lineinfo -O0 -x cu -I/users/cristian/TurboGAP/hop -I/users/cristian/TurboGAP/hop/source/hip  -DHOP_TARGET_CUDA -O3 -g -G -lcublas  -arch=sm_80 --ptxas-options=-v -c cuda_wrappers.cu
#mpif90 -g -Wall -fcheck=all -lstdc++ -cpp -D _MPIF90  -g  -fPIC -O0  -ffree-line-length-none -fallow-argument-mismatch  -c fortran_cuda_interfaces.f90 -o fortran_cuda_interfaces.o
mpif90 -g -Wall -fcheck=all -fcheck=bounds -g -fcheck=all -Wall -fPIC -O3 -c  vdw.f90 -lopenblas
mpif90 -g -Wall -fcheck=all -fcheck=bounds -g -fcheck=all -Wall -fPIC -O3 test.f90 vdw.o -lopenblas

# srun --nodes=1 --account=project_462000007 --partition=standard-g --time=00:15:00 --gpus-per-node=8 --ntasks=1 ./a.out