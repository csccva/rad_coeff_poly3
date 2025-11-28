#include "hip/hip_runtime.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <cmath>

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hiprand/hiprand.h>
#include <assert.h>
#include <hip/hip_complex.h>


#define tpb 64


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true)
{
   if (code != hipSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

extern "C" void cuda_set_device( int my_rank)
{

  int  num_gpus=0;
  int mygpuid;
  gpuErrchk(hipGetDeviceCount(&num_gpus));
  gpuErrchk(hipSetDevice(my_rank%num_gpus));
  gpuErrchk(hipGetDevice(&mygpuid));
  /*gpuErrchk(hipSetDevice(0));*/
  printf("\n Seta Aset at %d %d %d %d\n", num_gpus, my_rank%num_gpus,my_rank, mygpuid);
  //exit(0);
  return;
}

extern "C" void gpu_device_sync()
{
  gpuErrchk( hipDeviceSynchronize() );
}

extern "C" void cuda_malloc_all(void **a_d, size_t Np, hipStream_t *stream )
{
  gpuErrchk(hipMallocAsync((void **) a_d,  Np ,stream[0]));
  return;
}

extern "C" void cuda_free_async(void **a_d, hipStream_t *stream )
{
  gpuErrchk(hipFreeAsync(*a_d, stream[0]));
   return;
}

extern "C" void cuda_memset_async(void *a_d, int value,  size_t Np, hipStream_t *stream )
{
  hipMemsetAsync( a_d, value , Np ,stream[0]);
}

extern "C" void gpu_check_error(){
  hipError_t code=hipDeviceSynchronize() ;
  printf("\n %s \n", hipGetErrorString(code));
  gpuErrchk( code );
}


extern "C" void create_cublas_handle(hipblasHandle_t *handle,hipStream_t *stream )
{
 	hipblasCreate(handle);
    hipStreamCreate(stream);
    hipblasSetStream(*handle, *stream);
   return;
}


extern "C" void destroy_cublas_handle(hipblasHandle_t *handle,hipStream_t *stream )
{
 	 // Destroy the handle
   hipblasDestroy(*handle);
   hipStreamDestroy(*stream);
   //printf("\n cublas handle destroyed. \n The End? \n");
   return;
}

extern "C" void cuda_cpy_htod(void *a, void *a_d, size_t N, hipStream_t *stream )
{
  gpuErrchk(hipMemcpyAsync(a_d, a, N, hipMemcpyHostToDevice,stream[0] ));
   return;
}

extern "C" void cuda_cpy_dtoh(void *a_d, void *a, size_t N, hipStream_t *stream )
{
  gpuErrchk(hipMemcpyAsync(a, a_d,  N, hipMemcpyDeviceToHost,stream[0]));
   return;
}


__global__ void cuda_soap_normalize(double *soap_d, double *sqrt_dot_d, int n_soap, int n_sites)
{ 
  int i_site=blockIdx.x;
  int tid=threadIdx.x;
  double  my_sqrt_dot_p=sqrt_dot_d[i_site];
  for(int s=tid;s<n_soap;s=s+tpb){
    double my_soap_final=soap_d[s+i_site*n_soap]/my_sqrt_dot_p;
    soap_d[s+i_site*n_soap]=my_soap_final;
  }
}
extern "C" void gpu_soap_normalize(double *soap_d, double *sqrt_dot_d, int n_soap, int n_sites, hipStream_t *stream ){
  cuda_soap_normalize<<< n_sites, tpb,0,stream[0]>>> (soap_d, sqrt_dot_d, n_soap,n_sites);
}