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
#define tpb_pref_force 64
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

extern "C" void create_cublas_handle(hipblasHandle_t *handle,hipStream_t *stream )
{
 	hipblasCreate(handle);
    hipStreamCreate(stream);
    hipblasSetStream(*handle, *stream);
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

extern "C" void cuda_malloc_all(void **a_d, size_t Np, hipStream_t *stream )
{
  gpuErrchk(hipMallocAsync((void **) a_d,  Np ,stream[0]));
  return;
}

extern "C" void cuda_memset_async(void *a_d, int value,  size_t Np, hipStream_t *stream )
{
  hipMemsetAsync( a_d, value , Np ,stream[0]);
}

extern "C" void cuda_free_async(void **a_d, hipStream_t *stream )
{
  gpuErrchk(hipFreeAsync(*a_d, stream[0]));
   return;
}


extern "C" void gpu_device_sync()
{
  gpuErrchk( hipDeviceSynchronize() );
}

__global__ void  cuda_ts_forces_virial(int *i2_index_d, int *j2_index_d, int *i_site_index_d, 
                                        double3 *hirshfeld_v_cart_der_d, double *pref_force1_d, double *pref_force2_d,
                                        double *neighbor_c6_ij_d, double *rjs_d, double3 *xyz_d, double *f_damp_d, double *exp_damp_d,
                                        double *r0_ij_d, double *r6_d, double *r6_der_d, 
                                        double *forces_d, double *virial_d,
                                        double rcut_inner, double rcut, int n_pairs, double d, double sR)
{
   int k= threadIdx.x+blockIdx.x*blockDim.x;
   if(k<n_pairs){
      int i2=i2_index_d[k]-1;
      int j2=j2_index_d[k]-1;
      int i_site=i_site_index_d[k]-1;
      double locrjs=rjs_d[k];
      double locf_damp=f_damp_d[k];
      double locr0_ij=r0_ij_d[k];
      if(locrjs>rcut_inner && rjs_d[k]<rcut){
         double3 lochvc_der=hirshfeld_v_cart_der_d[k];
         double3 tmp_xyz;
         tmp_xyz=xyz_d[k];
         double this_xyz[3],this_force[3];
         this_xyz[0]=tmp_xyz.x;
         this_xyz[1]=tmp_xyz.y;
         this_xyz[2]=tmp_xyz.z;

         this_force[0]=lochvc_der.x*(pref_force1_d[i_site]+pref_force2_d[i_site]);
         this_force[1]=lochvc_der.y*(pref_force1_d[i_site]+pref_force2_d[i_site]);
         this_force[2]=lochvc_der.z*(pref_force1_d[i_site]+pref_force2_d[i_site]);

         atomicAdd(&forces_d[j2*3]  , this_force[0]);
         atomicAdd(&forces_d[j2*3+1], this_force[1]);
         atomicAdd(&forces_d[j2*3+2], this_force[2]);
    
         for(int k1=0;k1<3;k1++){
            for(int k2=0;k2<3;k2++){
               double loc_viri=0.5*(this_force[k1]*this_xyz[k2]+this_force[k2]*this_xyz[k1]); 
               atomicAdd(&virial_d[k2+3*k1], loc_viri);
            }
         }

         if(locr0_ij==0){
            this_force[0]=0.0; this_force[1]=0.0;this_force[2]=0.0;
         }
         else{
            this_force[0]=neighbor_c6_ij_d[k]/locrjs*this_xyz[0]*
                           locf_damp*(-r6_der_d[k]-r6_d[k]*locf_damp*exp_damp_d[k]*d/sR/locr0_ij);
            this_force[1]=neighbor_c6_ij_d[k]/locrjs*this_xyz[1]*
                           locf_damp*(-r6_der_d[k]-r6_d[k]*locf_damp*exp_damp_d[k]*d/sR/locr0_ij);
            this_force[2]=neighbor_c6_ij_d[k]/locrjs*this_xyz[2]*
                           locf_damp*(-r6_der_d[k]-r6_d[k]*locf_damp*exp_damp_d[k]*d/sR/locr0_ij);
         }
         if(j2!=i2){
            atomicAdd(&forces_d[i2*3]  , this_force[0]);
            atomicAdd(&forces_d[i2*3+1], this_force[1]);
            atomicAdd(&forces_d[i2*3+2], this_force[2]);
         }

         for(int k1=0;k1<3;k1++){
            for(int k2=0;k2<3;k2++){
               double loc_viri=-0.25*(this_force[k1]*this_xyz[k2]+this_force[k2]*this_xyz[k1]); 
               atomicAdd(&virial_d[k2+3*k1], loc_viri);
            }
         }
      }
   }
}

extern "C" void gpu_final_ts_forces_virial(int *i2_index_d, int *j2_index_d, int *i_site_index_d, 
                                        double3 *hirshfeld_v_cart_der_d, double *pref_force1_d, double *pref_force2_d,
                                        double *neighbor_c6_ij_d, double *rjs_d, double3 *xyz_d, double *f_damp_d, double *exp_damp_d,
                                        double *r0_ij_d, double *r6_d, double *r6_der_d, 
                                        double *forces_d, double *virial_d,
                                        double rcut_inner, double rcut, int n_pairs, double d, double sR,
                                        hipStream_t *stream ){
     
   int nblocks=(n_pairs+tpb-1)/tpb;

   cuda_ts_forces_virial<<<nblocks, tpb,0, stream[0]>>>(i2_index_d, j2_index_d, i_site_index_d, 
                                           hirshfeld_v_cart_der_d, pref_force1_d, pref_force2_d,
                                           neighbor_c6_ij_d, rjs_d, xyz_d, f_damp_d, exp_damp_d,
                                           r0_ij_d, r6_d, r6_der_d, 
                                           forces_d, virial_d,
                                           rcut_inner, rcut,   n_pairs,  d, sR);
}


__global__ void cuda_compute_pref_forces(double *pref_force1_d, double *pref_force2_d,
                                         double *hirshfeld_v_d,double *r0_ref_d, 
                                         double *r6_d,double *r0_ij_d, double *rjs_d,
                                         double *neighbor_c6_ij_d, double *f_damp_d, double *exp_damp_d,
                                         int *n_neigh_d, int *i2_k_index_d, int *k_start_index_d, 
                                         double rcut_inner, double rcut, double d, double sR,
                                         int n_pairs, int n_sites){
   int i_site=blockIdx.x;
   int tid=threadIdx.x;
   int k_start=k_start_index_d[i_site];
   int my_n_neigh=n_neigh_d[i_site];
   double locrjs,locr0_ij,locr6,locf_damp,locexp_damp, locneighbor_c6_ij;
   int k;

   __shared__ double shpreff1[tpb_pref_force], shpreff2[tpb_pref_force];
   shpreff1[tid]=0.0;
   shpreff2[tid]=0.0;
   __syncthreads();
   for(int j=tid;j<my_n_neigh;j+=tpb_pref_force){
      k=k_start+j;
      locrjs=rjs_d[k]; 
      locr0_ij=r0_ij_d[k];
      locr6=r6_d[k];
      locf_damp=f_damp_d[k]; 
      locexp_damp=exp_damp_d[k];
      locneighbor_c6_ij=neighbor_c6_ij_d[k];
      if(locrjs>rcut_inner && locrjs<rcut && j>0){
         shpreff1[tid]+=locneighbor_c6_ij*locf_damp*locr6;
         shpreff2[tid]+=-locneighbor_c6_ij*locf_damp*locf_damp*locrjs*locr6*locexp_damp*d/sR/pow(locr0_ij,2);
      }
   }
    __syncthreads();
  //reduction
  for (int s=tpb_pref_force/2; s>0; s>>=1) //  s>>=1 <==> s=s/2
  {
    if (tid < s)
    {
      shpreff1[tid]+=shpreff1[tid+s];
      shpreff2[tid]+=shpreff2[tid+s];
    }
    __syncthreads();
  }

   if(threadIdx.x==0){
      double locpref_force1=shpreff1[0];
      double locpref_force2=shpreff2[0];

      int i2=i2_k_index_d[i_site]-1;
      double loch_v=hirshfeld_v_d[i_site];
      if(loch_v==0.0){
         locpref_force1=0.0;
         locpref_force2=0.0;
      }
      else{
         locpref_force1=locpref_force1/ loch_v;
         locpref_force2=locpref_force2*r0_ref_d[i2]/3.0/pow(loch_v,2.0/3.0);
      }
      pref_force1_d[i_site]=locpref_force1;
      pref_force2_d[i_site]=locpref_force2;
   }
}
extern "C" void gpu_compute_pref_forces(double *pref_force1_d, double *pref_force2_d,
                                        double *hirshfeld_v_d,double *r0_ref_d, 
                                        double *r6_d,double *r0_ij_d, double *rjs_d,
                                        double *neighbor_c6_ij_d, double *f_damp_d, double *exp_damp_d,
                                        int *n_neigh_d, int *i2_k_index_d, int *k_start_index_d, 
                                        double rcut_inner, double rcut, double d, double sR,
                                        int n_pairs, int n_sites,
                                        hipStream_t *stream ){
   int nblocks=n_sites;
   cuda_compute_pref_forces<<<nblocks,tpb_pref_force,0,stream[0]>>>(pref_force1_d, pref_force2_d,
                                                                    hirshfeld_v_d,r0_ref_d, 
                                                                    r6_d,r0_ij_d, rjs_d,
                                                                    neighbor_c6_ij_d,f_damp_d, exp_damp_d,
                                                                    n_neigh_d, i2_k_index_d, k_start_index_d, 
                                                                    rcut_inner, rcut, d, sR,
                                                                    n_pairs, n_sites);
}

__global__ void cuda_compute_damp_energy(double *energies_d,double *f_damp_d,double *exp_damp_d,
                                           double *rjs_d, double *r0_ij_d, double *r6_d, double *neighbor_c6_ij_d,
                                           int *n_neigh_d, int *k_start_index_d,
                                           double rcut_inner, double rcut, double d, double sR,
                                           int n_sites)
{
    int i_site = blockIdx.x;
    int tid    = threadIdx.x;

    int k_start     = k_start_index_d[i_site];
    int my_n_neigh  = n_neigh_d[i_site];

    __shared__ double shenergy[tpb_pref_force];
    shenergy[tid] = 0.0;
    __syncthreads();

    for (int j = tid; j < my_n_neigh; j += tpb_pref_force) {
        if (j > 0) {   // j = 2..n_neigh in Fortran
            int k = k_start + j;

            double rjs = rjs_d[k];
            if (rjs > rcut_inner && rjs < rcut) {
                double locexp_damp = exp(-d * (rjs / (sR * r0_ij_d[k]) - 1.0));
                double locf_damp   = 1.0 / (1.0 + locexp_damp);

                exp_damp_d[k] = locexp_damp;
                f_damp_d[k]   = locf_damp;

                shenergy[tid] += neighbor_c6_ij_d[k] * r6_d[k] * locf_damp;
            }
        }
    }

    __syncthreads();

    // reduction
    for (int s = tpb_pref_force / 2; s > 0; s >>= 1) {
        if (tid < s)
            shenergy[tid] += shenergy[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        energies_d[i_site] = -0.5 * shenergy[0];
    }
}
extern "C" void gpu_compute_damp_energy(double *energies_d, double *f_damp_d, double *exp_damp_d,
                                        double *rjs_d, double *r0_ij_d, double *r6_d, double *neighbor_c6_ij_d,
                                        int *n_neigh_d, int *k_start_index_d, 
                                        double rcut_inner, double rcut, double d, double sR,
                                        int n_sites,
                                        hipStream_t *stream){
   
   cuda_compute_damp_energy<<<n_sites, tpb_pref_force, 0, stream[0]>>>(energies_d,f_damp_d, exp_damp_d,
                                                                       rjs_d, r0_ij_d, r6_d, neighbor_c6_ij_d,
                                                                       n_neigh_d, k_start_index_d,
                                                                       rcut_inner, rcut, d, sR,
                                                                       n_sites);
}

