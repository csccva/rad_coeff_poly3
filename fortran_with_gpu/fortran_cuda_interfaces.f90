
! module load gcc/10.3.0 openblas openmpi cuda
! rm cuda_wrappers.o ; nvcc -arch=sm_70 -c src/cuda_wrappers.cu ; make clean; make -B
!  srun  --time=00:10:00 --partition=gputest --account=project_2000634 --nodes=1 --ntasks-per-node=4  --cpus-per-task=1 --gres=gpu:a100:4  ../turbogap_dev/bin/turbogap md
! gnupot
!plot "< paste trajectory_out.xyz_64k_oiriginal trajectory_out.xyz | awk 'NR>2{print $5,$13}'", x
MODULE F_B_C
    INTERFACE
      subroutine gpu_malloc_all(a_d,n,gpu_stream) bind(C,name="cuda_malloc_all")
        use iso_c_binding
        implicit none
        type(c_ptr) :: a_d,gpu_stream
        integer(c_size_t),value :: n
      end subroutine
      
      subroutine gpu_memset_async(a_d,valuetoset,n,gpu_stream) bind(C,name="cuda_memset_async")
        use iso_c_binding
        implicit none
        type(c_ptr),value :: a_d
        type(c_ptr) :: gpu_stream
        integer(c_size_t),value :: n
        integer(c_int),value :: valuetoset
      end subroutine

      subroutine gpu_malloc_all_blocking(a_d,n) bind(C,name="cuda_malloc_all_blocking")
        use iso_c_binding
        implicit none
        type(c_ptr) :: a_d
        integer(c_size_t),value :: n
      end subroutine
      
      subroutine gpu_device_reset() bind(C,name="cuda_device_reset")
        use iso_c_binding
        implicit none
      end subroutine
      subroutine gpu_free(a_d) bind(C,name="cuda_free")
        use iso_c_binding
        implicit none
        type(c_ptr) :: a_d
      end subroutine

      subroutine gpu_free_async(a_d,gpu_stream) bind(C,name="cuda_free_async")
        use iso_c_binding
        implicit none
        type(c_ptr) :: a_d
        type(c_ptr) :: gpu_stream
      end subroutine

      subroutine cpy_htod(a,a_d,n, gpu_stream) bind(C,name="cuda_cpy_htod")
        use iso_c_binding
        implicit none
        type(c_ptr),value :: a_d,a
        type(c_ptr) :: gpu_stream
        integer(c_size_t),value :: n
      end subroutine

      subroutine cpy_htod_blocking(a,a_d,n) bind(C,name="cuda_cpy_htod_blocking")
        use iso_c_binding
        implicit none
        type(c_ptr),value :: a_d,a
        integer(c_size_t),value :: n
      end subroutine

      subroutine cpy_dtod(b_d,a_d,n, gpu_stream) bind(C,name="cuda_cpy_dtod")
        use iso_c_binding
        implicit none
        type(c_ptr),value :: a_d,b_d
        type(c_ptr) :: gpu_stream
        integer(c_size_t),value :: n
      end subroutine

      subroutine cpy_dtoh(a_d,a,n, gpu_stream) bind(C,name="cuda_cpy_dtoh")
        use iso_c_binding
        implicit none
        type(c_ptr),value :: a_d,a
        type(c_ptr) :: gpu_stream
        integer(c_size_t),value :: n
      end subroutine


      subroutine cpy_dtoh_event(a_d,a,n, gpu_stream) bind(C,name="cuda_cpy_dtoh_event")
        use iso_c_binding
        implicit none
        type(c_ptr),value :: a_d,a
        type(c_ptr) :: gpu_stream
        integer(c_size_t),value :: n
      end subroutine
      

      subroutine cpy_dtoh_blocking(a_d,a,n) bind(C,name="cuda_cpy_dtoh_blocking")
        use iso_c_binding
        implicit none
        type(c_ptr),value :: a_d,a
        integer(c_size_t),value :: n
      end subroutine
      subroutine gpu_vector_fill_curand(a_d,n,c) bind(C,name="GPU_fill_rand")
        use iso_c_binding
        implicit none
        type(c_ptr),value :: a_d
        integer(c_int),value :: n,c
      end subroutine

      subroutine create_cublas_handle(cubhandle, gpu_stream)bind(C,name="create_cublas_handle")
        use iso_c_binding
        implicit none
        type(c_ptr) :: cubhandle, gpu_stream
      end subroutine

      subroutine destroy_cublas_handle(cubhandle, gpu_stream)bind(C,name="destroy_cublas_handle")
        use iso_c_binding
        implicit none
        type(c_ptr) :: cubhandle,gpu_stream
      end subroutine

      subroutine gpu_dgemm_n_n(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, cubhandle)  bind(C,name="gpu_dgemm_n_n")
        use iso_c_binding
        implicit none
        type(c_ptr), value :: cubhandle
        integer(c_int),value :: m, n, k, lda, ldb, ldc
        real(c_double),value :: alpha,beta
        type(c_ptr), value   :: A, B, C
        ! real(c_double), device, dimension(lda,*) :: A
        ! real(c_double), device, dimension(ldb,*) :: B
        ! real(c_double), device, dimension(ldc,*) :: C
      end subroutine

      
      subroutine gpu_blas_mmul_t_n(cubhandle, Qs_d, soap_d, kernels_d, &
                         n_sparse, n_soap, n_sites)  bind(C,name="gpu_blas_mmul_t_n")
        use iso_c_binding
        implicit none
        type(c_ptr), value :: cubhandle,kernels_d,Qs_d,soap_d
        integer(c_int),value :: n_sparse, n_soap, n_sites
      end subroutine

      subroutine gpu_blas_mmul_n_t(cubhandle, kernels_der_d, Qs_copy_d, Qss_d, n_sparse, &
                                  n_soap, n_sites, cdelta) bind(C,name="gpu_blas_mmul_n_t")
        use iso_c_binding
        implicit none
        type(c_ptr), value :: cubhandle,kernels_der_d, Qs_copy_d, Qss_d
        integer(c_int),value :: n_sparse,n_soap, n_sites
        real(c_double), value :: cdelta
      end subroutine

      subroutine gpu_blas_mvmul_n(cubhandle, A, B, C, nAx, nAy) bind(C,name="gpu_blas_mvmul_n")
        use iso_c_binding
        implicit none
        type(c_ptr), value :: cubhandle,A,B,C
        integer(c_int),value :: nAx,nAy
      end subroutine

      subroutine gpu_kernels_pow( A,B,  zeta, N,gpu_stream) bind(C,name="gpu_kernels_pow")
        use iso_c_binding
        implicit none
        type(c_ptr), value :: A,B
        type(c_ptr) :: gpu_stream
        integer(c_int),value :: N
        real(c_double), value :: zeta
      end subroutine

      subroutine gpu_axpe( A,  dccc,e0, N, gpu_stream) bind(C,name="gpu_axpc")
        use iso_c_binding
        implicit none
        type(c_ptr), value :: A
        type(c_ptr) :: gpu_stream
        integer(c_int),value :: N
        real(c_double), value :: dccc,e0
      end subroutine

      subroutine gpu_set_device(my_rank) bind(C, name="cuda_set_device")
        use iso_c_binding
        integer(c_int), value :: my_rank
      end subroutine

      subroutine gpu_matvect(kernels_der_d, alphas_d, n_sites, n_sparse,gpu_stream) bind(C,name="cuda_matvect_kernels")
        use iso_c_binding
        type(c_ptr), value :: kernels_der_d, alphas_d
        type(c_ptr) :: gpu_stream
        integer(c_int), value :: n_sites,n_sparse
      end subroutine

      subroutine gpu_final_soap_forces_virial(n_sites, &
                                              Qss_d,n_soap, l_index_d, j2_index_d, &
                                              soap_der_d, &
                                              xyz_d, virial_d, &
                                              n_sites0, &
                                              forces_d, &
                                              n_pairs, gpu_stream) &
                  bind(C,name="gpu_final_soap_forces_virial")
        use iso_c_binding
        type(c_ptr), value :: Qss_d, l_index_d
        type(c_ptr), value :: forces_d, soap_der_d, xyz_d, virial_d
        type(c_ptr), value ::  j2_index_d
        type(c_ptr) :: gpu_stream
        integer(c_int), value :: n_sites,n_soap, n_sites0, n_pairs
      end subroutine

      subroutine gpu_local_property_derivatives(n_sites, &
                                              Qss_d,n_soap, l_index_d, &
                                              soap_der_d, &
                                              local_property_cart_der_d, &
                                              n_pairs, gpu_stream) &
                  bind(C,name="gpu_local_property_derivatives")
        use iso_c_binding
        type(c_ptr), value :: Qss_d, l_index_d
        type(c_ptr), value :: local_property_cart_der_d, soap_der_d
        type(c_ptr) :: gpu_stream
        integer(c_int), value :: n_sites, n_soap, n_pairs
      end subroutine

      
      subroutine gpu_get_sqrt_dot_p(sqrt_dot_d, soap_d, multiplicity_array_d, &
                                    cnk_d, skip_soap_component_d,  &
                                    n_sites, n_soap, n_max,l_max, gpu_stream) &
                  bind(C,name="gpu_get_sqrt_dot_p")
        use iso_c_binding
        type(c_ptr), value :: cnk_d, skip_soap_component_d
        type(c_ptr) :: gpu_stream
        type(c_ptr), value :: sqrt_dot_d, soap_d, multiplicity_array_d
        integer(c_int),value :: n_sites, n_soap, n_max,l_max
      end subroutine

      subroutine gpu_get_soap_der(soap_d, sqrt_dot_d, soap_cart_der_d, &
                                  soap_rad_der_d, soap_azi_der_d, soap_pol_der_d, &
                                  thetas_d, phis_d, rjs_d, &
                                  multiplicity_array_d, &
                                  cnk_d, cnk_rad_der_d, cnk_azi_der_d, cnk_pol_der_d, &
                                  n_neigh_d, i_k2_start_d, k2_i_site_d, k3_index_d, skip_soap_component_d, &
                                  n_sites, n_atom_pairs, n_soap, k_max, n_max, l_max, maxneigh, gpu_stream)  &
                  bind(C,name="gpu_get_soap_der")
        use iso_c_binding
        type(c_ptr), value :: sqrt_dot_d, soap_d, soap_cart_der_d
        type(c_ptr) :: gpu_stream
        type(c_ptr), value :: soap_rad_der_d, soap_azi_der_d, soap_pol_der_d
        type(c_ptr), value :: thetas_d, phis_d, rjs_d
        type(c_ptr), value :: n_neigh_d, i_k2_start_d, k2_i_site_d, k3_index_d, skip_soap_component_d
        type(c_ptr), value :: cnk_d, cnk_rad_der_d, cnk_azi_der_d, cnk_pol_der_d
        type(c_ptr), value :: multiplicity_array_d
        integer(c_int),value :: n_sites, n_atom_pairs, n_soap, k_max, n_max, l_max, maxneigh
      end subroutine

      subroutine gpu_soap_normalize(soap_d, sqrt_dot_d, n_soap, n_sites, gpu_stream)  &
                  bind(C,name="gpu_soap_normalize")
        use iso_c_binding
        type(c_ptr), value :: sqrt_dot_d, soap_d
        type(c_ptr) :: gpu_stream
        integer(c_int),value :: n_sites, n_soap
      end subroutine

      subroutine gpu_get_derivatives(radial_exp_coeff_d, angular_exp_coeff_d, radial_exp_coeff_der_d, &
                                  angular_exp_coeff_rad_der_d, angular_exp_coeff_azi_der_d, angular_exp_coeff_pol_der_d, &
                                  cnk_rad_der_d, cnk_azi_der_d, cnk_pol_der_d, &
                                  rjs_d, &
                                  rcut_max, &
                                  n_atom_pairs, n_sites, n_soap, k_max, n_max, l_max, gpu_stream) &
                                  bind(C,name="gpu_get_derivatives")
        use iso_c_binding
        type(c_ptr), value :: rjs_d
        type(c_ptr) :: gpu_stream
        type(c_ptr), value :: radial_exp_coeff_d, angular_exp_coeff_d, radial_exp_coeff_der_d
        type(c_ptr), value :: angular_exp_coeff_rad_der_d, angular_exp_coeff_azi_der_d, angular_exp_coeff_pol_der_d
        type(c_ptr), value :: cnk_rad_der_d, cnk_azi_der_d, cnk_pol_der_d
        real(c_double), value :: rcut_max
        integer(c_int),value :: n_atom_pairs, n_sites, n_soap, k_max, n_max, l_max
      end subroutine


      subroutine gpu_get_cnk(radial_exp_coeff_d, angular_exp_coeff_d, &
                             cnk_d, &
                             n_neigh_d, k2_start_d, &
                             n_sites,  n_atom_pairs,n_soap, k_max, n_max, l_max, &
                             bintybint, &
                             atom_sigma_r_d, atom_sigma_t_d, rcut_hard_d, &
                             central_weight_d, species_d, i_beg_d, i_end_d, &
                             radial_enhancement, species_multiplicity_d, &
                             W_d, S_d, size_species_1, gpu_stream) &
                             bind(C,name="gpu_get_cnk")
        use iso_c_binding
        type(c_ptr), value :: n_neigh_d, k2_start_d
        type(c_ptr) :: gpu_stream
        type(c_ptr), value :: radial_exp_coeff_d, angular_exp_coeff_d
        type(c_ptr), value :: cnk_d
        integer(c_int),value :: n_sites,  n_atom_pairs,n_soap, k_max, n_max, l_max
        integer(c_int), value :: bintybint, radial_enhancement, size_species_1
        type(c_ptr), value :: rcut_hard_d, species_multiplicity_d
        type(c_ptr), value :: atom_sigma_r_d, atom_sigma_t_d
        type(c_ptr), value :: central_weight_d, species_d, i_beg_d, i_end_d
        type(c_ptr), value :: W_d, S_d
      end subroutine

      subroutine gpu_get_plm_array_global(plm_array_global_d, n_atom_pairs, kmax, &
                                      lmax, thetas_d, gpu_stream) &
                                      bind(C,name="gpu_get_plm_array_global")
      use iso_c_binding
      type(c_ptr), value :: plm_array_global_d, thetas_d
      type(c_ptr) :: gpu_stream
      integer(c_int),value :: n_atom_pairs, kmax, lmax
      end subroutine

      subroutine gpu_get_exp_coeff_array(eimphi_global_d, thetas_d, &
                                      rjs_d, phis_d, &
                                      mask_d, atom_simga_in_d, atom_simga_scaling_d, & 
                                      rcut, n_atom_piars, n_spceies, lmax, kmax, &
                                      prefl_array_global_d, plm_array_global_d, &
                                      plm_array_der_global_d, &
                                      prefl_array_global_der_d, &
                                      preflm_d,exp_coeff_d, &
                                      c_do_derivatives, &
                                      eimphi_rad_der_global_d, eimphi_azi_der_global_d, &
                                      plm_array_div_sin_d, plm_array_der_mul_sin_d, &
                                      exp_coeff_rad_der_d, exp_coeff_azi_der_d, exp_coeff_pol_der_d, &
                                      gpu_stream) & 
                                      bind(C,name="gpu_get_exp_coeff_array")
      use iso_c_binding
      type(c_ptr), value :: eimphi_global_d, rjs_d, phis_d, mask_d, exp_coeff_d
      type(c_ptr) :: gpu_stream
      type(c_ptr), value :: atom_simga_in_d, atom_simga_scaling_d, prefl_array_global_der_d
      integer(c_int), value ::  n_atom_piars, n_spceies, lmax, kmax
      real(c_double), value :: rcut
      type(c_ptr), value :: eimphi_rad_der_global_d, eimphi_azi_der_global_d, thetas_d
      type(c_ptr), value :: preflm_d, prefl_array_global_d, plm_array_global_d, plm_array_der_global_d
      logical(c_bool), value :: c_do_derivatives
      type(c_ptr), value :: exp_coeff_rad_der_d, exp_coeff_azi_der_d, exp_coeff_pol_der_d
      type(c_ptr), value :: plm_array_div_sin_d, plm_array_der_mul_sin_d
      end subroutine

      subroutine gpu_get_radial_exp_coeff_poly3gauss(radial_exp_coeff_d, radial_exp_coeff_der_d, &
                                      i_beg_d, i_end_d,&
                                      global_scaling_d, &
                                      size_radial_exp_coeff_one, size_radial_exp_coeff_two, n_species, &
                                      c_do_derivatives, bintybint, & 
                                      rcut_hard_d, &
                                      k2_i_site_d, k_2start_d,&
                                      gpu_stream) &
                                      bind(C,name="gpu_get_radial_exp_coeff_poly3gauss")
      use iso_c_binding
      type(c_ptr), value :: radial_exp_coeff_d, radial_exp_coeff_der_d, i_beg_d, i_end_d, global_scaling_d
      type(c_ptr), value :: rcut_hard_d, k2_i_site_d, k_2start_d
      type(c_ptr) :: gpu_stream
      integer(c_int), value :: size_radial_exp_coeff_one, size_radial_exp_coeff_two, n_species
      integer(c_int), value :: bintybint
      logical(c_bool), value :: c_do_derivatives
      end subroutine


      subroutine gpu_radial_poly3gauss(n_atom_pairs, n_species, mask_d, rjs_d, rcut_hard_d, n_sites, n_neigh_d, n_max, &
                                       ntemp, do_derivatives, exp_coeff_d, exp_coeff_der_d, rcut_soft_d, atom_sigma_d, &
                                       exp_coeff_temp1_d, exp_coeff_temp2_d, exp_coeff_der_temp_d, i_beg, i_end, &
                                       atom_sigma_scaling_d, mode, radial_enhancement, amplitude_scaling_d, alpha_max_d, &
                                       nf_d, ntemp_der_d, W_d, gpu_stream) &
                                       bind(C,name="gpu_radial_poly3gauss")
      use iso_c_binding
      type(c_ptr), value :: mask_d, rjs_d, rcut_hard_d, n_neigh_d,exp_coeff_d, exp_coeff_der_d
      type(c_ptr), value :: rcut_soft_d, atom_sigma_d, exp_coeff_temp1_d, exp_coeff_temp2_d
      type(c_ptr), value :: exp_coeff_der_temp_d, i_beg, i_end, atom_sigma_scaling_d, amplitude_scaling_d
      type(c_ptr), value :: alpha_max_d, nf_d, W_d
      type(c_ptr) :: gpu_stream
      integer(c_int), value :: n_atom_pairs, n_species, n_sites, ntemp, n_max, mode, radial_enhancement, ntemp_der_d
      logical(c_bool), value :: do_derivatives
      end subroutine

      subroutine gpu_radial_poly3(n_atom_pairs, n_species, mask_d, rjs_d, rcut_hard_d, n_sites, n_neigh_d, n_max, &
                                  ntemp, do_derivatives, exp_coeff_d, exp_coeff_der_d, rcut_soft_d, atom_sigma_d, &
                                  exp_coeff_temp1_d, exp_coeff_temp2_d, exp_coeff_der_temp_d, i_beg, i_end, &
                                  atom_sigma_scaling_d, mode, radial_enhancement, amplitude_scaling_d, alpha_max_d, &
                                  nf_d, ntemp_der_d, W_d, do_central_d, central_weight_d,gpu_stream) &
                                  bind(C,name="gpu_radial_poly3")
      use iso_c_binding
      type(c_ptr), value :: mask_d, rjs_d, rcut_hard_d, n_neigh_d,exp_coeff_d, exp_coeff_der_d
      type(c_ptr), value :: rcut_soft_d, atom_sigma_d, exp_coeff_temp1_d, exp_coeff_temp2_d
      type(c_ptr), value :: exp_coeff_der_temp_d, i_beg, i_end, atom_sigma_scaling_d, amplitude_scaling_d
      type(c_ptr), value :: alpha_max_d, nf_d, W_d,do_central_d, central_weight_d
      type(c_ptr) :: gpu_stream
      integer(c_int), value :: n_atom_pairs, n_species, n_sites, ntemp, n_max, mode, radial_enhancement, ntemp_der_d
      logical(c_bool), value :: do_derivatives
      end subroutine

      subroutine gpu_get_2b_forces_energies(i_beg, i_end, n_sparse, energies_d, e0, n_neigh_d, do_forces, forces_d, virial_d, &
                                            rjs_d, rcut, species_d, neighbor_species_d, sp1, sp2, buffer, delta, cutoff_d,    &
                                            Qs_d, sigma, alphas_d, xyz_d,  gpu_stream )                                       &
                                            bind(C,name="gpu_get_2b_forces_energies")
      use iso_c_binding
      type(c_ptr), value :: energies_d, n_neigh_d, forces_d, virial_d, rjs_d, species_d, neighbor_species_d, cutoff_d
      type(c_ptr), value :: Qs_d, alphas_d, xyz_d
      type(c_ptr) :: gpu_stream
      integer(c_int), value :: i_beg, i_end, n_sparse, sp1, sp2
      logical(c_bool), value :: do_forces
      real(c_double), value :: e0, rcut, buffer, delta, sigma
      end subroutine


      subroutine gpu_get_core_pot_energy_and_forces(i_beg, i_end, do_forces, species_d, sp1, sp2, n_neigh_d, neighbor_species_d,&
                                                    rjs_d, n_sparse, x_d, V_d, dVdx2_d, yp1, ypn, xyz_d, forces_d, virial_d,    &
                                                    energies_d, gpu_stream)                                                     &
                                                    bind(C,name="gpu_get_core_pot_energy_and_forces")
      use iso_c_binding
      type(c_ptr), value :: species_d, n_neigh_d, neighbor_species_d, rjs_d, x_d, V_d, dVdx2_d, xyz_d
      type(c_ptr), value :: forces_d, virial_d, energies_d
      type(c_ptr) :: gpu_stream
      integer(c_int), value :: i_beg, i_end, sp1, sp2, n_sparse
      logical(c_bool), value :: do_forces
      real(c_double), value :: yp1, ypn
      end subroutine

      subroutine get_orthonormalization_matrix_poly3_gpu(alpha_max, S, W, handle, stream) bind(C,name="get_orthonormalization_matrix_poly3")
        use iso_c_binding
        type(c_ptr), value :: S, W
        type(c_ptr)        :: handle, stream
        integer(c_int), value :: alpha_max
      end subroutine

      subroutine get_orthonormalization_matrix_poly3gauss_gpu(alpha_max, atom_sigma_in, rcut_hard_in, S, W,handle,stream) bind(C,name="get_orthonormalization_matrix_poly3gauss")
        use iso_c_binding
        type(c_ptr), value :: S,W
        type(c_ptr)        :: handle, stream
        integer(c_int), value :: alpha_max
        real(c_double), value :: atom_sigma_in, rcut_hard_in
      end subroutine

      subroutine orthonormalization_copy_to_global_matrix(src, dest,src_rowsize, dest_start, dest_rowsize, stream) bind(C,name="copy_to_global_matrix")
        use iso_c_binding
        type(c_ptr), value :: src,dest
        type(c_ptr)        :: stream
        integer(c_int), value :: src_rowsize, dest_start, dest_rowsize
      end subroutine

      subroutine gpu_3b(n_sparse, n_sites, n_atom_pairs, n_sites0, sp0, sp1, sp2, alpha, delta, e0, cutoff, stream, rjs, xyz, n_neigh, species, neighbors_list, neighbor_species, do_forces, rcut,buffer, sigma,qs,str,i_beg, i_end, energy_d, forces_d, virials_d, kappas_array_d) bind(C,name="gpu_3b")
        use iso_c_binding
        implicit none
        type(c_ptr),value :: alpha, cutoff
        type(c_ptr) :: stream
        integer(c_int),value :: n_sparse, n_sites, n_atom_pairs, n_sites0, sp0, sp1, sp2,i_beg,i_end
        real(c_double),value :: delta,e0, rcut,buffer
        type(c_ptr),value :: rjs,xyz,n_neigh,species,neighbors_list,neighbor_species,sigma,qs,energy_d,forces_d,virials_d,kappas_array_d
        logical(c_bool), value :: do_forces
        character(kind=c_char), dimension(*) :: str
      end subroutine
      
      subroutine gpu_create_kappas(kappas_array_d, n_neigh_host,stream,n_sites) bind(C,name="create_kappas_cwrap")
        use iso_c_binding
        implicit none        
        type(c_ptr), value :: kappas_array_d, n_neigh_host
        type(c_ptr)        :: stream
        integer(c_int), value :: n_sites
      end subroutine
      
      subroutine gpu_setup_3bresult_arrays(energy_3b_d, forces_3b_d, virials_3b_d,stream,do_forces, n_sites, n_sites0) bind(C,name="setup_3bresult_arrays_cwrap")
        use iso_c_binding
        implicit none        
        type(c_ptr) :: energy_3b_d, forces_3b_d, virials_3b_d
        type(c_ptr) :: stream
        logical(c_bool), value :: do_forces
	integer(c_int), value :: n_sites, n_sites0
      end subroutine
      
      subroutine gpu_cleanup_3bresult_arrays(energy_3b_d, forces_3b_d, virials_3b_d,energy_3b_h,forces_3b_h,virials_3b_h,stream,do_forces, n_sites, n_sites0) bind(C,name="cleanup_3bresult_arrays_cwrap")
        use iso_c_binding
        implicit none        
        type(c_ptr) :: energy_3b_d, forces_3b_d, virials_3b_d,energy_3b_h,forces_3b_h,virials_3b_h
        type(c_ptr) :: stream
        logical(c_bool), value :: do_forces
	      integer(c_int), value :: n_sites, n_sites0
      end subroutine

      subroutine gpu_device_sync() bind(C,name="gpu_device_sync")
        use iso_c_binding
        implicit none
      end subroutine gpu_device_sync

      subroutine gpu_stream_sync(stream) bind(C,name="gpu_stream_sync")
        use iso_c_binding
        implicit none
        type(c_ptr) :: stream 
      end subroutine gpu_stream_sync

      
      subroutine gpu_meminfo() bind(C,name="gpu_meminfo")
        use iso_c_binding
        implicit none
      end subroutine

      subroutine gpu_get_pair_distribution_nk(i_beg, i_end, n_pairs, n_sites0, neighbors_list_d,&
        n_neigh_d, neighbor_species_d,&
        species_d, rjs_d, xyz_d, r_min, r_max, r_cut,&
        buffer, &
        nk_out_d, nk_flags_d, nk_flags_sum_d, species_1, species_2, stream)  bind(C,name="gpu_get_pair_distribution_nk")
        use iso_c_binding
        implicit none
        integer(c_int), value ::  i_beg, i_end, n_sites0, species_1, species_2, n_pairs
        type(c_ptr), value :: neighbors_list_d, n_neigh_d, neighbor_species_d, species_d, rjs_d, xyz_d, nk_flags_d, nk_flags_sum_d
        real(c_double), value :: r_min, r_max, buffer, r_cut
        type(c_ptr), value ::  nk_out_d
        type(c_ptr) :: stream        
      end subroutine gpu_get_pair_distribution_nk

      subroutine gpu_set_pair_distribution_k_index(i_beg, i_end, n_pairs, n_sites0, neighbors_list_d,&
           rjs_d, xyz_d, k_index_d, j2_index_d, rjs_index_d,  xyz_k_d, nk_flags_d,  nk_sum_flags_d, stream)  bind(C,name="gpu_set_pair_distribution_k_index")
        use iso_c_binding
        implicit none
        integer(c_int), value ::  i_beg, i_end, n_sites0, n_pairs
        type(c_ptr), value :: neighbors_list_d, xyz_d, rjs_d, nk_flags_d
        type(c_ptr), value ::  k_index_d, j2_index_d, rjs_index_d, xyz_k_d, nk_sum_flags_d
        type(c_ptr) :: stream
      end subroutine gpu_set_pair_distribution_k_index


      subroutine gpu_setup_matrix_forces(i_beg, i_end, n_pairs, n_sites0, neighbors_list_d,&
           xyz_all_d, k_index_d, j2_index_d,  xyz_k_d, nk_flags_d, nk_flags_sum_d, stream)  bind(C,name="gpu_setup_matrix_forces")
        use iso_c_binding
        implicit none
        integer(c_int), value ::  i_beg, i_end, n_sites0, n_pairs
        type(c_ptr), value :: neighbors_list_d, xyz_all_d, nk_flags_d, nk_flags_sum_d
        type(c_ptr), value ::  k_index_d, j2_index_d, xyz_k_d
        type(c_ptr) :: stream
      end subroutine gpu_setup_matrix_forces

      
      subroutine gpu_exp_force_virial_collection( n_k, forces0, energy_scale,  fi,&
						 j2_list,  virial,  xyz, stream )bind(C,name="gpu_exp_force_virial_collection")

        use iso_c_binding
        implicit none        
        integer(c_int), value ::  n_k
        real(c_double), value :: energy_scale
        type(c_ptr), value :: forces0, xyz, virial, j2_list, fi
        type(c_ptr) :: stream
      end subroutine gpu_exp_force_virial_collection

      subroutine gpu_get_fi_dgemv( i, n_samples_sf, n_k, dermat_d,&
           & prefactor_d, fi_d,  cublas_handle, stream )bind(C,name&
           &="gpu_get_fi_dgemv")
        use iso_c_binding
        implicit none        
        integer(c_int), value ::  i, n_samples_sf, n_k
        type(c_ptr), value :: dermat_d, prefactor_d, fi_d
        type(c_ptr), value :: cublas_handle, stream
      end subroutine gpu_get_fi_dgemv
        
      subroutine gpu_hadamard_vec_mat_product(n_samples_sf, n_k,&
           & all_scattering_factors_d, dermat_d, stream )bind(C&
           &,name="gpu_hadamard_vec_mat_product")
        use iso_c_binding
        implicit none
        integer(c_int), value ::  n_samples_sf, n_k
        type(c_ptr), value :: dermat_d, all_scattering_factors_d
        type(c_ptr) ::  stream
      end subroutine gpu_hadamard_vec_mat_product

      subroutine gpu_get_Gka(i, n_k, n_samples, Gka_d, Gk_d, xyz_k_d, stream )bind(C&
           &,name="gpu_get_Gka")
        use iso_c_binding
        implicit none
        integer(c_int), value :: i, n_samples, n_k
        type(c_ptr), value :: Gka_d, Gk_d, xyz_k_d
        type(c_ptr) ::  stream
      end subroutine gpu_get_Gka


      subroutine gpu_get_Gka_inplace(i, n_k, n_samples, Gk_d, xyz_k_d, stream )bind(C&
           &,name="gpu_get_Gka_inplace")
        use iso_c_binding
        implicit none
        integer(c_int), value :: i, n_samples, n_k
        type(c_ptr), value :: Gk_d, xyz_k_d
        type(c_ptr) ::  stream
      end subroutine gpu_get_Gka_inplace


      subroutine gpu_print_pointer_int(p)bind(C&
           &,name="gpu_print_pointer_int")
        use iso_c_binding
        implicit none
        type(c_ptr), value :: p
      end subroutine gpu_print_pointer_int
            
      subroutine gpu_print_pointer_double(p)bind(C&
           &,name="gpu_print_pointer_double")
        use iso_c_binding
        implicit none
        type(c_ptr), value :: p
      end subroutine gpu_print_pointer_double
      

      subroutine gpu_get_pair_distribution_and_ders(pair_distribution_d,  pair_distribution_der_d,&
           n_k, n_samples, kde_sigma,  x_d, dV_d, rjs_d, pdf_factor, der_factor, stream) bind(C&
           &,name="gpu_get_pair_distribution_and_ders")
        use iso_c_binding
        implicit none
        integer(c_int), value :: n_k, n_samples
        real(c_double), value :: kde_sigma, pdf_factor, der_factor
        type(c_ptr), value :: pair_distribution_d,  pair_distribution_der_d, x_d, dV_d, rjs_d
        type(c_ptr) :: stream 
      end subroutine gpu_get_pair_distribution_and_ders

      subroutine gpu_get_pair_distribution_only(pair_distribution_d,&
           n_k, n_samples, kde_sigma,  x_d, dV_d, rjs_d, pdf_factor, der_factor, stream) bind(C&
           &,name="gpu_get_pair_distribution_only")
        use iso_c_binding
        implicit none
        integer(c_int), value :: n_k, n_samples
        real(c_double), value :: kde_sigma, pdf_factor, der_factor
        type(c_ptr), value :: pair_distribution_d, x_d, dV_d, rjs_d
        type(c_ptr) :: stream 
      end subroutine gpu_get_pair_distribution_only

      subroutine gpu_get_pair_distribution_only_falloc(pair_distribution_d, pdf_to_reduce,&
           n_k, n_samples, kde_sigma,  x_d, dV_d, rjs_d, pdf_factor, der_factor, stream) bind(C&
           &,name="gpu_get_pair_distribution_only_falloc")
        use iso_c_binding
        implicit none
        integer(c_int), value :: n_k, n_samples
        real(c_double), value :: kde_sigma, pdf_factor, der_factor
        type(c_ptr), value :: pair_distribution_d, x_d, dV_d, rjs_d, pdf_to_reduce
        type(c_ptr) :: stream 
      end subroutine gpu_get_pair_distribution_only_falloc
      

      subroutine gpu_get_pair_distribution_der_only(  pair_distribution_der_d,&
           n_k, n_samples, kde_sigma,  x_d, dV_d, rjs_d, pdf_factor, der_factor, stream) bind(C&
           &,name="gpu_get_pair_distribution_der_only")
        use iso_c_binding
        implicit none
        integer(c_int), value :: n_k, n_samples
        real(c_double), value :: kde_sigma, pdf_factor, der_factor
        type(c_ptr), value :: pair_distribution_der_d, x_d, dV_d, rjs_d
        type(c_ptr) :: stream 
      end subroutine gpu_get_pair_distribution_der_only

      
      

      subroutine gpu_set_Gk( nk, n_samples, k_index_d, Gk_d, pair_distribution_partial_der_d, c_factor, stream)bind(C,name="gpu_set_Gk")
        use iso_c_binding
        implicit none
        integer(c_int), value :: nk, n_samples
        real(c_double), value :: c_factor
        type(c_ptr), value :: k_index_d, Gk_d, pair_distribution_partial_der_d
        type(c_ptr) :: stream 
      end subroutine gpu_set_Gk


      
      subroutine  gpu_set_pair_distribution_k_index_only(n_pairs, k_index_d, nk_sum_flags_d, stream )bind(C,name="gpu_set_pair_distribution_k_index_only")
        use iso_c_binding
        implicit none
        integer(c_int), value :: n_pairs
        type(c_ptr), value :: k_index_d, nk_sum_flags_d
        type(c_ptr) :: stream
      end subroutine gpu_set_pair_distribution_k_index_only
      

      subroutine  gpu_set_pair_distribution_j2_only(n_pairs, n_sites, neighbors_list_d, j2_d, nk_sum_flags_d, stream )bind(C,name="gpu_set_pair_distribution_j2_only")
        use iso_c_binding
        implicit none
        integer(c_int), value :: n_pairs, n_sites
        type(c_ptr), value :: j2_d, nk_sum_flags_d, neighbors_list_d
        type(c_ptr) :: stream
      end subroutine gpu_set_pair_distribution_j2_only



      subroutine  gpu_set_pair_distribution_rjs_only(n_pairs, rjs, rjs_d, nk_sum_flags_d, stream )bind(C,name="gpu_set_pair_distribution_rjs_only")
        use iso_c_binding
        implicit none
        integer(c_int), value :: n_pairs
        type(c_ptr), value :: rjs_d, nk_sum_flags_d, rjs
        type(c_ptr) :: stream
      end subroutine gpu_set_pair_distribution_rjs_only


      subroutine  gpu_set_pair_distribution_xyz_only(n_pairs, xyz, xyz_d, nk_sum_flags_d, stream )bind(C,name="gpu_set_pair_distribution_xyz_only")
        use iso_c_binding
        implicit none
        integer(c_int), value :: n_pairs
        type(c_ptr), value :: xyz_d, nk_sum_flags_d, xyz
        type(c_ptr) :: stream
      end subroutine gpu_set_pair_distribution_xyz_only
            
      

      function host_alloc(s) bind(c, name='host_alloc')
        use iso_c_binding
        implicit none
        type(c_ptr)   :: host_alloc
        integer(c_size_t), value :: s
      end function host_alloc

      subroutine host_free(p) bind(c, name='host_free')
        use, intrinsic :: iso_c_binding, only: c_ptr
        implicit none
        type(c_ptr), intent(in), value :: p
      end subroutine host_free

      subroutine cpy_htoh_pinned(s, d, size) bind(c, name='cpy_htoh_pinned')
        use iso_c_binding
        implicit none
        integer( c_size_t ), value :: size 
        type(c_ptr), value :: s, d 
      end subroutine cpy_htoh_pinned

      subroutine gpu_stream_synchronize(gpu_stream) &
                                      bind(C,name="gpu_stream_synchronize")
      use iso_c_binding
      type(c_ptr) :: gpu_stream
      end subroutine

      subroutine gpu_device_synchronize() &
                                      bind(C,name="gpu_device_synchronize")
      use iso_c_binding
      end subroutine
 
      subroutine gpu_radial_expansion_coefficients_poly3operator(exp_coeff_d, &
                exp_coeff_der_d, n_exp_coeff,n_exp_coeff_der,rcut_hard_in,rcut_soft_in, &
                W_d, k_i_d, alpha_max, n_sites, n_neigh_d, &
                c_do_derivatives, &
                atom_sigma_scaling, atom_sigma_in, atom_sigma, &
                rjs_in_d, mask_d, &
                num_scaling_mode , amplitude_scaling, central_weight, &
                radial_enhancement, c_do_central, &
                rcut_soft, rcut_hard, filter_width, &
                A_d, &
                !global_I_left_array_d, global_I_right_array_d, global_amplitudes_d, global_exp_buffer_d, &
                !global_rjs_idx_d, max_nn, global_nn_d, &
                !global_I0_array_d, global_M_left_array_d, global_M_right_array_d, &
                !global_lim_buffer_array_d, global_B_right_d, global_B_left_d, global_M_rad_mono_d, &
                !global_rjs_d, global_atom_widths_d, &
                !global_g_aux_left_array_d,global_g_aux_right_array_d, &
                stream)  &
                bind(C,name="gpu_radial_expansion_coefficients_poly3operator")
        use iso_c_binding
        implicit none
        type(c_ptr), value :: exp_coeff_d, exp_coeff_der_d
        type(c_ptr), value :: W_d, k_i_d,n_neigh_d
        type(c_ptr), value :: rjs_in_d, mask_d, A_d
        type(c_ptr) :: stream
        ! type(c_ptr), value :: global_I0_array_d, global_M_left_array_d, global_M_right_array_d
        ! type(c_ptr), value :: global_I_left_array_d, global_I_right_array_d, global_amplitudes_d, global_exp_buffer_d
        ! type(c_ptr), value :: global_rjs_idx_d, global_nn_d
        ! type(c_ptr), value :: global_lim_buffer_array_d,  global_M_rad_mono_d
        ! type(c_ptr), value :: global_B_right_d, global_B_left_d
        ! type(c_ptr), value :: global_rjs_d, global_atom_widths_d
        ! type(c_ptr), value :: global_g_aux_left_array_d,global_g_aux_right_array_d
        logical(c_bool), value :: c_do_derivatives, c_do_central
        integer(c_int),value :: radial_enhancement
        ! integer(c_int),value ::  max_nn
        integer(c_int),value :: n_exp_coeff,n_exp_coeff_der
        integer(c_int),value :: n_sites, alpha_max, num_scaling_mode
        real(c_double),value :: rcut_hard,rcut_soft, filter_width
        real(c_double),value :: rcut_hard_in,rcut_soft_in, amplitude_scaling
        real(c_double),value ::  central_weight
        real(c_double),value :: atom_sigma_scaling, atom_sigma_in, atom_sigma
      end subroutine

    END INTERFACE
  END MODULE F_B_C
