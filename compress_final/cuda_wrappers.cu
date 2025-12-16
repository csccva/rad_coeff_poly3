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
#define tpb_get_soap_der_one 128


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

 __global__ void cuda_get_soap_der_two_one(double *soap_d, double *sqrt_dot_p_d,
                                      double *soap_rad_der_d, double *soap_azi_der_d, double *soap_pol_der_d,
                                      double *tdotoprod_der_rad, double *tdotoprod_der_azi, double *tdotoprod_der_pol,
                                      int *k2_i_site_d, 
                                      int n_sites, int n_atom_pairs, int n_soap, int k_max, int n_max, int l_max)
{ 
  int k2=blockIdx.x;
  int tid=threadIdx.x;
  int i_site=k2_i_site_d[k2]-1;
  __shared__ double sh_soap_rad_der_dot[tpb];
  __shared__ double sh_soap_azi_der_dot[tpb];
  __shared__ double sh_soap_pol_der_dot[tpb];
  double this_dotprod_rad=0.0;double this_dotprod_azi=0.0;double this_dotprod_pol=0.0;
  
  for(int s=tid;s<n_soap;s=s+tpb){
    this_dotprod_rad+=soap_d[s+i_site*n_soap]*soap_rad_der_d[s+k2*n_soap];
    this_dotprod_azi+=soap_d[s+i_site*n_soap]*soap_azi_der_d[s+k2*n_soap];
    this_dotprod_pol+=soap_d[s+i_site*n_soap]*soap_pol_der_d[s+k2*n_soap];
  }
  sh_soap_rad_der_dot[tid]=this_dotprod_rad;
  sh_soap_azi_der_dot[tid]=this_dotprod_azi;
  sh_soap_pol_der_dot[tid]=this_dotprod_pol;
  __syncthreads();

  //reduction
  for (int s=tpb/2; s>0; s>>=1) // s=s/2
  {
    if (tid < s)
    {
      sh_soap_rad_der_dot[tid] +=sh_soap_rad_der_dot[tid + s];
      sh_soap_azi_der_dot[tid] +=sh_soap_azi_der_dot[tid + s];
      sh_soap_pol_der_dot[tid] +=sh_soap_pol_der_dot[tid + s];
    }
    __syncthreads();

  }
  for(int s=tid;s<n_soap;s=s+tpb){
    tdotoprod_der_rad[s+k2*n_soap]=sh_soap_rad_der_dot[0];
    tdotoprod_der_azi[s+k2*n_soap]=sh_soap_azi_der_dot[0];
    tdotoprod_der_pol[s+k2*n_soap]=sh_soap_pol_der_dot[0];
  }
} 

__global__ void cuda_get_soap_der_two_two(double *soap_d, double *sqrt_dot_p_d,
                                          double *soap_rad_der_d, double *soap_azi_der_d, double *soap_pol_der_d,
                                          double *tdotoprod_der_rad, double *tdotoprod_der_azi, double *tdotoprod_der_pol,
                                          int *k2_i_site_d, 
                                          int n_sites, int n_atom_pairs, int n_soap, int k_max, int n_max, int l_max)
{ 
  int k2=blockIdx.x;
  int tid=threadIdx.x;
  int i_site=k2_i_site_d[k2]-1;
  double loc_sqrt_dot_p=sqrt_dot_p_d[i_site];
  for(int s=tid;s<n_soap;s=s+tpb){
    double my_soap=soap_d[s+i_site*n_soap];

    double my_soap_rad_der=soap_rad_der_d[s+k2*n_soap];
    double my_soap_azi_der=soap_azi_der_d[s+k2*n_soap];
    double my_soap_pol_der=soap_pol_der_d[s+k2*n_soap];

    double myprod_der_rad=tdotoprod_der_rad[s+k2*n_soap];
    double myprod_der_azi=tdotoprod_der_azi[s+k2*n_soap];
    double myprod_der_pol=tdotoprod_der_pol[s+k2*n_soap];


    soap_rad_der_d[s+k2*n_soap]=my_soap_rad_der/loc_sqrt_dot_p
                               -my_soap/(loc_sqrt_dot_p*loc_sqrt_dot_p*loc_sqrt_dot_p)*myprod_der_rad;
    soap_azi_der_d[s+k2*n_soap]=my_soap_azi_der/loc_sqrt_dot_p
                               -my_soap/(loc_sqrt_dot_p*loc_sqrt_dot_p*loc_sqrt_dot_p)*myprod_der_azi;
    soap_pol_der_d[s+k2*n_soap]=my_soap_pol_der/loc_sqrt_dot_p
                               -my_soap/(loc_sqrt_dot_p*loc_sqrt_dot_p*loc_sqrt_dot_p)*myprod_der_pol;
  }

}



__global__ void cuda_get_soap_der_thr_one(double3 *soap_cart_der_d,
                                          double *soap_rad_der_d, double *soap_azi_der_d, double *soap_pol_der_d,
                                          double *thetas, double *phis, double *rjs,
                                          int *k3_index, 
                                          int n_sites, int n_atom_pairs, int n_soap, int k_max, int n_max, int l_max)
{ 
  int k2=blockIdx.x;
  int tid=threadIdx.x;
  int k3=k3_index[k2]-1;

  double my_theta=thetas[k2]; double my_phi=phis[k2]; double my_rj=rjs[k2];
  for(int s=tid;s<n_soap;s=s+tpb){  
    if(k3!=k2){
      double my_soap_rad_der=soap_rad_der_d[s+k2*n_soap];
      double my_soap_azi_der=soap_azi_der_d[s+k2*n_soap];
      double my_soap_pol_der=soap_pol_der_d[s+k2*n_soap];
      double3 my_soap_cart_der;
      my_soap_cart_der.x=sin(my_theta)*cos(my_phi)*my_soap_rad_der 
                        -cos(my_theta)*cos(my_phi)/my_rj*my_soap_pol_der
                        -sin(my_phi)/my_rj*my_soap_azi_der;
      my_soap_cart_der.y=sin(my_theta)*sin(my_phi)*my_soap_rad_der 
                        -cos(my_theta)*sin(my_phi)/my_rj*my_soap_pol_der
                        +cos(my_phi)/my_rj*my_soap_azi_der;
      my_soap_cart_der.z=cos(my_theta)*my_soap_rad_der 
                        +sin(my_theta)/my_rj*my_soap_pol_der;
      soap_cart_der_d[s+k2*n_soap]=my_soap_cart_der;
    }
  }
}

__global__ void cuda_get_soap_der_thr_two(double3 *soap_cart_der_d,
                                          double *soap_rad_der_d, double *soap_azi_der_d, double *soap_pol_der_d,
                                          double *thetas, double *phis, double *rjs,
                                          int *n_neigh_d, int *i_k2_start_d, int *k2_i_site_d, int *k3_index_d, 
                                          int n_sites, int n_atom_pairs, int n_soap, int k_max, int n_max, int l_max, int maxneigh)
{ 
  int i_site=blockIdx.x;
  int tid=threadIdx.x;
  int my_start=i_k2_start_d[i_site]-1;
  int k3=my_start;
  int my_n_neigh=n_neigh_d[i_site];
  
  for(int s=tid;s<n_soap;s=s+tpb){
    double3 loc_sum;
    loc_sum.x=0,loc_sum.y=0,loc_sum.z=0;
    int k2=my_start+1;
    for(int j=1;j<my_n_neigh; j++){
      double3 my_soap_cart_der=soap_cart_der_d[s+k2*n_soap];
      loc_sum.x-=my_soap_cart_der.x;
      loc_sum.y-=my_soap_cart_der.y;
      loc_sum.z-=my_soap_cart_der.z;
      k2++;
    }
    soap_cart_der_d[s+k3*n_soap]=loc_sum;
  }
}


__global__ void naive_transpose_soap_rad_azi_pol(double *soap_rad_der_d,
                                            double *tran_soap_rad_der_d, 
                                            int n_soap, int n_atom_pairs)
{
  int i_g = threadIdx.x+blockIdx.x*blockDim.x;
  if(i_g<n_soap*n_atom_pairs){
    double loc_soap_rad=soap_rad_der_d[i_g];
    int k2=i_g/n_soap;
    int icount=i_g%n_soap;
    int new_i_g=k2+icount*n_atom_pairs;
    tran_soap_rad_der_d[new_i_g]=loc_soap_rad;
  }

}


__global__ void naive_transpose_cnk_arrays(hipDoubleComplex *C,
                                           hipDoubleComplex *tran_C, 
                                           int k_max, int n_max, int n_sites)
{
  // in Fortran is cnk( 1:k_max, 1:n_max, 1:n_sites) --> (1:n_sites,1:k_max, 1:n_max)
  //       cnk_rad_der( 1:k_max, 1:n_max, 1:n_atom_pairs) )
  int i_g = threadIdx.x+blockIdx.x*blockDim.x;
  if(i_g<k_max*n_max*n_sites){
    hipDoubleComplex loc_C=C[i_g];  // i_g=i_k+k_max*(i_n+i_site*n_max)
    int i_k=i_g%k_max;
    int i_z=i_g/k_max;
    int i_n=i_z%n_max;
    int i_site=i_z/n_max;
    int new_i_g=i_site+n_sites*(i_k+i_n*k_max);
    tran_C[new_i_g]=loc_C;
  }
}

__global__ void cuda_get_soap_der_two_one_compress(double *soap_rad_der_d,double *soap_azi_der_d,double *soap_pol_der_d,
                                   double *unc_soap_rad_der_d,double * unc_soap_azi_der_d, double *unc_soap_pol_der_d,                                                                  
                                   int *compress_P_i, int *compress_P_j, double *compress_P_el,
                                   bool compress_soap,
                                   int n_sites, int n_atom_pairs, int n_soap, int comp_P_nz, int n_soap_uncompressed,
                                   int k_max, int n_max, int l_max)
{
  int k2 = threadIdx.x+blockIdx.x*blockDim.x;
  if (k2<n_atom_pairs){
    if(compress_soap){
      for(int ik=0; ik<comp_P_nz; ik++){
        int i=compress_P_i[ik]-1;int j=compress_P_j[ik]-1;
        int idi=k2*n_soap+i;
        int idj=k2*n_soap_uncompressed+j;
        soap_rad_der_d[idi]+=compress_P_el[ik]*unc_soap_rad_der_d[idj];
        soap_azi_der_d[idi]+=compress_P_el[ik]*unc_soap_azi_der_d[idj];
        soap_pol_der_d[idi]+=compress_P_el[ik]*unc_soap_pol_der_d[idj];

      }

    }
    else{
      for(int is=0;is<n_soap;is++){
        int idx=k2*n_soap+is;
        soap_rad_der_d[idx]=unc_soap_rad_der_d[idx];
        soap_azi_der_d[idx]=unc_soap_azi_der_d[idx];
        soap_pol_der_d[idx]=unc_soap_pol_der_d[idx];
      }
    }
  }
}


__global__ void cuda_get_soap_der_one(double *multiplicity_array_d, 
                                      double *unc_soap_rad_der_d, double *unc_soap_azi_der_d,double *unc_soap_pol_der_d,
                                      hipDoubleComplex *cnk_d, 
                                      hipDoubleComplex *cnk_rad_der_d, hipDoubleComplex *cnk_azi_der_d, hipDoubleComplex *cnk_pol_der_d,
                                      int *k2_i_site_d, bool *skip_soap_component_flattened,
                                      int n_sites, int n_atom_pairs, int n_soap,
                                      int n_soap_uncompressed, int k_max, int n_max, int l_max)
{
  int k2 = threadIdx.x+blockIdx.x*blockDim.x;
  if (k2<n_atom_pairs){
    int i_site=k2_i_site_d[k2]-1;
    int counter=0;
    int counter2=0; 
    for(int n=1;n<=n_max;n++){
      for(int np=n;np<=n_max;np++){
        for(int l=0;l<=l_max;l++){
          counter++;
          double my_soap_rad_der=0; //trans_soap_rad_der_d[k2+(counter-1)*n_atom_pairs]; //soap_rad_der_d[counter-1+k2*n_soap];
          double my_soap_azi_der=0; //trans_soap_azi_der_d[k2+(counter-1)*n_atom_pairs]; //soap_azi_der_d[counter-1+k2*n_soap];
          double my_soap_pol_der=0; //trans_soap_pol_der_d[k2+(counter-1)*n_atom_pairs]; //soap_pol_der_d[counter-1+k2*n_soap];  
          if(!skip_soap_component_flattened[counter-1]){ 
            for(int m=0;m<=l; m++){
              int k=1+l*(l+1)/2+m; 
              counter2++;
              hipDoubleComplex tmp_1_cnk_d=cnk_d[k-1+ k_max*(n-1 +i_site*n_max)]; //cnk_d[i_site+n_sites*(k-1+(n-1)*k_max)]; //trans_cnk_d[i_site+n_sites*(k-1+(n-1)*k_max)];  //cnk_d[k-1+ k_max*(n-1 +i_site*n_max)];
              hipDoubleComplex tmp_2_cnk_d=cnk_d[k-1+k_max*(np-1+i_site*n_max)]; //cnk_d[i_site+n_sites*(k-1+(np-1)*k_max)]; //trans_cnk_d[i_site+n_sites*(k-1+(np-1)*k_max)]; //cnk_d[k-1+k_max*(np-1+i_site*n_max)];
              hipDoubleComplex tmp_1_cnk_rad_d=cnk_rad_der_d[k-1+k_max*(n-1 +k2*n_max)]; //cnk_rad_der_d[k2+n_atom_pairs*(k-1+(n-1)*k_max) ]; //trans_cnk_rad_der_d[k2+n_atom_pairs*(k-1+(n-1)*k_max) ]; // cnk_rad_der_d[k-1+k_max*(n-1 +k2*n_max)];
              hipDoubleComplex tmp_2_cnk_rad_d=cnk_rad_der_d[k-1+k_max*(np-1+k2*n_max)]; //cnk_rad_der_d[k2+n_atom_pairs*(k-1+(np-1)*k_max)]; //trans_cnk_rad_der_d[k2+n_atom_pairs*(k-1+(np-1)*k_max)]; // cnk_rad_der_d[k-1+k_max*(np-1+k2*n_max)];
              hipDoubleComplex tmp_1_cnk_azi_d=cnk_azi_der_d[k-1+k_max*(n-1 +k2*n_max)]; //cnk_azi_der_d[k2+n_atom_pairs*(k-1+(n-1)*k_max) ]; //trans_cnk_azi_der_d[k2+n_atom_pairs*(k-1+(n-1)*k_max) ]; //cnk_azi_der_d[k-1+k_max*(n-1 +k2*n_max)];
              hipDoubleComplex tmp_2_cnk_azi_d=cnk_azi_der_d[k-1+k_max*(np-1+k2*n_max)]; //cnk_azi_der_d[k2+n_atom_pairs*(k-1+(np-1)*k_max)]; //trans_cnk_azi_der_d[k2+n_atom_pairs*(k-1+(np-1)*k_max)]; //cnk_azi_der_d[k-1+k_max*(np-1+k2*n_max)];
              hipDoubleComplex tmp_1_cnk_pol_d=cnk_pol_der_d[k-1+k_max*(n-1 +k2*n_max)]; //cnk_pol_der_d[k2+n_atom_pairs*(k-1+(n-1)*k_max) ]; //trans_cnk_pol_der_d[k2+n_atom_pairs*(k-1+(n-1)*k_max) ]; //cnk_pol_der_d[k-1+k_max*(n-1 +k2*n_max)];
              hipDoubleComplex tmp_2_cnk_pol_d=cnk_pol_der_d[k-1+k_max*(np-1+k2*n_max)]; //cnk_pol_der_d[k2+n_atom_pairs*(k-1+(np-1)*k_max)]; //trans_cnk_pol_der_d[k2+n_atom_pairs*(k-1+(np-1)*k_max)]; //cnk_pol_der_d[k-1+k_max*(np-1+k2*n_max)];
              my_soap_rad_der+=multiplicity_array_d[counter2-1]*(tmp_1_cnk_rad_d.x*tmp_2_cnk_d.x+tmp_1_cnk_rad_d.y*tmp_2_cnk_d.y+
                                                                 tmp_1_cnk_d.x*tmp_2_cnk_rad_d.x+tmp_1_cnk_d.y*tmp_2_cnk_rad_d.y);
              my_soap_azi_der+=multiplicity_array_d[counter2-1]*(tmp_1_cnk_azi_d.x*tmp_2_cnk_d.x+tmp_1_cnk_azi_d.y*tmp_2_cnk_d.y+
                                                                 tmp_1_cnk_d.x*tmp_2_cnk_azi_d.x+tmp_1_cnk_d.y*tmp_2_cnk_azi_d.y);
              my_soap_pol_der+=multiplicity_array_d[counter2-1]*(tmp_1_cnk_pol_d.x*tmp_2_cnk_d.x+tmp_1_cnk_pol_d.y*tmp_2_cnk_d.y+
                                                                 tmp_1_cnk_d.x*tmp_2_cnk_pol_d.x+tmp_1_cnk_d.y*tmp_2_cnk_pol_d.y);

            } 
          }
          unc_soap_rad_der_d[counter-1+k2*n_soap_uncompressed]=my_soap_rad_der; //trans_soap_rad_der_d[k2+(counter-1)*n_atom_pairs]=my_soap_rad_der; //soap_rad_der_d[counter-1+k2*n_soap]=my_soap_rad_der;
          unc_soap_azi_der_d[counter-1+k2*n_soap_uncompressed]=my_soap_azi_der; //trans_soap_azi_der_d[k2+(counter-1)*n_atom_pairs]=my_soap_azi_der; //soap_azi_der_d[counter-1+k2*n_soap]=my_soap_azi_der;
          unc_soap_pol_der_d[counter-1+k2*n_soap_uncompressed]=my_soap_pol_der; //trans_soap_pol_der_d[k2+(counter-1)*n_atom_pairs]=my_soap_pol_der; //soap_pol_der_d[counter-1+k2*n_soap]=my_soap_pol_der;       
        }
      }
    }
  }
}


extern "C" void gpu_get_soap_der(double *soap_d, double *sqrt_dot_d, double3 *soap_cart_der_d, 
                                 double *soap_rad_der_d, double *soap_azi_der_d, double *soap_pol_der_d, 
                                 double *unc_soap_rad_der_d, double *unc_soap_azi_der_d, double *unc_soap_pol_der_d, 
                                 double *tdotoprod_der_azi,double *tdotoprod_der_rad,double *tdotoprod_der_pol,
                                 double *trans_soap_azi_der_d,double *trans_soap_rad_der_d,double *trans_soap_pol_der_d,
                                 double *thetas_d, double *phis_d, double *rjs_d, 
                                 double *multiplicity_array_d,
                                 hipDoubleComplex *cnk_d, 
                                 hipDoubleComplex *cnk_rad_der_d, hipDoubleComplex *cnk_azi_der_d, hipDoubleComplex *cnk_pol_der_d, 
                                 int *n_neigh_d, int *i_k2_start_d, int *k2_i_site_d, int *k3_index_d,
                                 bool *skip_soap_component_flattened,bool compress_soap,
                                 int *compress_P_i, int *compress_P_j, double *compress_P_el,
                                 int n_sites, int n_atom_pairs, int n_soap, int comp_P_nz, int n_soap_uncompressed,
                                 int k_max, int n_max, int l_max, int maxneigh, hipStream_t *stream )
{
  dim3 nblocks=dim3((n_atom_pairs-1+tpb)/tpb,1,1);
  dim3 nthreads=dim3(tpb,1,1);

  dim3 nblocks_get_soap_der_one=dim3((n_atom_pairs-1+tpb_get_soap_der_one)/tpb_get_soap_der_one,1,1);
  dim3 nthreads_get_soap_der_one=dim3(tpb_get_soap_der_one,1,1);
  // printf("\n \n \n n_soap_uncompressed %d \n\n", n_soap_uncompressed);
  // exit(0);                                        
  cuda_get_soap_der_one<<< nblocks_get_soap_der_one, nthreads_get_soap_der_one,0, stream[0]>>>(multiplicity_array_d, 
                                               unc_soap_rad_der_d, unc_soap_azi_der_d, unc_soap_pol_der_d,   
                                               cnk_d, cnk_rad_der_d, cnk_azi_der_d, cnk_pol_der_d,
                                               k2_i_site_d, skip_soap_component_flattened,
                                               n_sites,  n_atom_pairs, n_soap, 
                                               n_soap_uncompressed, k_max, n_max, l_max);
                                 
  
  /*naive_transpose_soap_rad_azi_pol<<< (n_soap*n_atom_pairs+tpb-1)/tpb, tpb,0, stream[0]>>>(trans_soap_rad_der_d,
                                            soap_rad_der_d, 
                                            n_atom_pairs,n_soap);

  naive_transpose_soap_rad_azi_pol<<< (n_soap*n_atom_pairs+tpb-1)/tpb, tpb,0, stream[0]>>>(trans_soap_azi_der_d,
                                            soap_azi_der_d, 
                                            n_atom_pairs,n_soap);
                                            
  naive_transpose_soap_rad_azi_pol<<<(n_soap*n_atom_pairs+tpb-1)/tpb, tpb,0,  stream[0]>>>(trans_soap_pol_der_d,
                                            soap_pol_der_d, 
                                            n_atom_pairs,n_soap);          */                           
  //gpuErrchk( hipDeviceSynchronize() );
  cuda_get_soap_der_two_one_compress<<<nblocks_get_soap_der_one, nthreads_get_soap_der_one,0, stream[0]>>>(soap_rad_der_d,soap_azi_der_d, soap_pol_der_d,
                                               unc_soap_rad_der_d, unc_soap_azi_der_d, unc_soap_pol_der_d,                                            
                                               compress_P_i, compress_P_j, compress_P_el,
                                               compress_soap,
                                               n_sites,  n_atom_pairs, 
                                               n_soap, comp_P_nz,n_soap_uncompressed,k_max, n_max, l_max);
  
  cuda_get_soap_der_two_one<<<n_atom_pairs, nthreads,0, stream[0]>>>(soap_d,sqrt_dot_d, 
                                               soap_rad_der_d,soap_azi_der_d, soap_pol_der_d,  
                                               tdotoprod_der_rad, tdotoprod_der_azi, tdotoprod_der_pol,                                            
                                               k2_i_site_d, 
                                               n_sites,  n_atom_pairs, n_soap,  k_max, n_max, l_max);
  //printf("%d %d  %d  %d  %d  %d  %d \n", n_sites, n_atom_pairs, n_soap, k_max, n_max, l_max, maxneigh);
  //gpuErrchk( hipDeviceSynchronize() );

  cuda_get_soap_der_two_two<<<n_atom_pairs, nthreads,0, stream[0]>>>(soap_d, sqrt_dot_d,
                                               soap_rad_der_d,soap_azi_der_d, soap_pol_der_d,
                                               tdotoprod_der_rad, tdotoprod_der_azi, tdotoprod_der_pol,                                               
                                               k2_i_site_d, 
                                               n_sites,  n_atom_pairs, n_soap,  k_max, n_max, l_max);

 cuda_get_soap_der_thr_one<<<n_atom_pairs, nthreads,0, stream[0]>>>(soap_cart_der_d,  
                                                         soap_rad_der_d,soap_azi_der_d, soap_pol_der_d, 
                                                         thetas_d, phis_d, rjs_d,
                                                         k3_index_d, 
                                                         n_sites,  n_atom_pairs, n_soap,  k_max, n_max, l_max);
  cuda_get_soap_der_thr_two<<<n_sites, nthreads,0, stream[0]>>>(soap_cart_der_d,  
                                                         soap_rad_der_d,soap_azi_der_d, soap_pol_der_d, 
                                                         thetas_d, phis_d, rjs_d,
                                                         n_neigh_d, i_k2_start_d, k2_i_site_d, k3_index_d, 
                                                         n_sites,  n_atom_pairs, n_soap,  k_max, n_max, l_max, maxneigh); 
  
  return;
}

__global__ void cuda_get_sqrt_dot_p(double *soap_d, double *sqrt_dot_p_d, int n_sites, int n_soap)
{
   int i_site = threadIdx.x+blockIdx.x*blockDim.x;
   double my_sqrt_dot_p=0.0;
   if (i_site<n_sites){ 
    for(int is=0;is<n_soap;is++){
      double my_soap=soap_d[is+i_site*n_soap];
      my_sqrt_dot_p+=my_soap*my_soap;
    }
    my_sqrt_dot_p=sqrt(my_sqrt_dot_p);
    if(my_sqrt_dot_p<1.0e-5){
      my_sqrt_dot_p=1.0;
    }
    sqrt_dot_p_d[i_site]=my_sqrt_dot_p;
 }
}


__global__ void cuda_get_soap(double *soap_d, double *unc_soap_d, bool compress_soap,
                                    int *compress_P_i, int *compress_P_j, double *compress_P_el,
                                    int n_sites, int n_soap, int comp_P_nz, int n_soap_uncompressed)
{
   int i_site = threadIdx.x+blockIdx.x*blockDim.x;
   if (i_site<n_sites){ 
    if(compress_soap){
      for(int ik=0; ik<comp_P_nz;ik++){
        int i=compress_P_i[ik]-1;int j=compress_P_j[ik]-1;
        int idi=i_site*n_soap+i;
        int idj=i_site*n_soap_uncompressed+j;
        soap_d[idi]+=compress_P_el[ik]*unc_soap_d[idj];
      }
    }
    else{
      for(int is=0;is<n_soap;is++){
          int idx=i_site*n_soap+is;
          soap_d[idx]=unc_soap_d[idx];
        }
      }
   }
}


__global__ void cuda_get_unc_soap(double *unc_soap_d, double *multiplicity_array_d, 
                           hipDoubleComplex *cnk_d, bool *skip_soap_component_flattened,
                           int n_sites, int n_soap_uncompressed,
                           int n_max, int l_max)
{
   int i_site = threadIdx.x+blockIdx.x*blockDim.x;
   int k_max=1+l_max*(l_max+1)/2+l_max;
   if (i_site<n_sites){ 
    int counter=0;
    int counter2=0; 
    for(int n=1;n<=n_max;n++){
      for(int np=n;np<=n_max;np++){
        for(int l=0;l<=l_max;l++){
           if(!skip_soap_component_flattened[counter-1]){ 
            counter++;
            double my_soap=0.0;
            for(int m=0;m<=l; m++){
              int k=1+l*(l+1)/2+m; //k = 1 + l*(l+1)/2 + m
              counter2++;
              hipDoubleComplex tmp_1_cnk_d=cnk_d[k-1+k_max*(n-1 +i_site*n_max)]; //cnk_d[i_site+n_sites*((k-1)+(n-1)*k_max)];  //cnk_d[k-1+k_max*(n-1 +i_site*n_max)];
              hipDoubleComplex tmp_2_cnk_d=cnk_d[k-1+k_max*(np-1+i_site*n_max)]; //cnk_d[i_site+n_sites*((k-1)+(np-1)*k_max)]; //cnk_d[k-1+k_max*(np-1+i_site*n_max)];
              my_soap+=multiplicity_array_d[counter2-1]*(tmp_1_cnk_d.x*tmp_2_cnk_d.x+tmp_1_cnk_d.y*tmp_2_cnk_d.y); 
            }
            unc_soap_d[counter-1+i_site*n_soap_uncompressed]=my_soap;
          }
        }
      }
    }
 }
}


extern "C" void gpu_get_sqrt_dot_p(double *sqrt_dot_d, double *soap_d, double *unc_soap_d, double *multiplicity_array_d, 
                                   hipDoubleComplex *cnk_d, bool *skip_soap_component_flattened_d, bool compress_soap,
                                   int *compress_P_i, int *compress_P_j, double *compress_P_el,
                                   int n_sites, int n_soap, int comp_P_nz, int n_soap_uncompressed,
                                   int n_max, int l_max, hipStream_t *stream )
{
  dim3 nblocks=dim3((n_sites-1+tpb)/tpb,1,1);
  dim3 nthreads=dim3(tpb,1,1);
  
  cuda_get_unc_soap<<< nblocks, nthreads,0 , stream[0]>>>(unc_soap_d,multiplicity_array_d, 
                                                           cnk_d, skip_soap_component_flattened_d,
                                                           n_sites, n_soap_uncompressed,
                                                           n_max, l_max);
  
  cuda_get_soap<<< nblocks, nthreads,0 , stream[0]>>>(soap_d, unc_soap_d, compress_soap,
                                                            compress_P_i, compress_P_j, compress_P_el,
                                                            n_sites, n_soap, comp_P_nz, n_soap_uncompressed);
  
  cuda_get_sqrt_dot_p<<< nblocks, nthreads,0 , stream[0]>>>(soap_d,sqrt_dot_d, n_sites, n_soap);                              
  return;
}
