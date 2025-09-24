rm *.o *.mod; 

ftn  -fPIC -O3 -h flex_mp=intolerant  soap_turbo_functions.f90 soap_turbo_compress.f90 soap_turbo_angular.f90 soap_turbo_radial.f90 soap_turbo.f90  test.f90 -lsci_cray

# srun --nodes=1 --account=project_462000007 --partition=standard-g --time=00:15:00 --gpus-per-node=8 --ntasks=1 ./a.out