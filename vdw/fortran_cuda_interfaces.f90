
MODULE F_B_C
    INTERFACE
      subroutine gpu_set_device(my_rank) bind(C, name="cuda_set_device")
        use iso_c_binding
        integer(c_int), value :: my_rank
      end subroutine

      subroutine create_cublas_handle(cubhandle, gpu_stream)bind(C,name="create_cublas_handle")
        use iso_c_binding
        implicit none
        type(c_ptr) :: cubhandle, gpu_stream
      end subroutine

      subroutine cpy_htod(a,a_d,n, gpu_stream) bind(C,name="cuda_cpy_htod")
        use iso_c_binding
        implicit none
        type(c_ptr),value :: a_d,a
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

      subroutine gpu_malloc_all(a_d,n,gpu_stream) bind(C,name="cuda_malloc_all")
        use iso_c_binding
        implicit none
        type(c_ptr) :: a_d,gpu_stream
        integer(c_size_t),value :: n
      end subroutine

      subroutine gpu_free_async(a_d,gpu_stream) bind(C,name="cuda_free_async")
        use iso_c_binding
        implicit none
        type(c_ptr) :: a_d
        type(c_ptr) :: gpu_stream
      end subroutine

      subroutine gpu_device_sync() bind(C,name="gpu_device_sync")
        use iso_c_binding
        implicit none
      end subroutine gpu_device_sync
      
      subroutine gpu_memset_async(a_d,valuetoset,n,gpu_stream) bind(C,name="cuda_memset_async")
        use iso_c_binding
        implicit none
        type(c_ptr),value :: a_d
        type(c_ptr) :: gpu_stream
        integer(c_size_t),value :: n
        integer(c_int),value :: valuetoset
      end subroutine

      subroutine gpu_final_ts_forces_virial(i2_index_d, j2_index_d, i_site_index_d, &
                                            hirshfeld_v_cart_der_d, pref_force1_d, pref_force2_d, &
                                            neighbor_c6_ij_d, rjs_d, xyz_d, f_damp_d, exp_damp_d, &
                                            r0_ij_d, r6_d, r6_der_d,  &
                                            forces_d,virial_d, &
                                            rcut_inner, rcut, n_pairs,  d,  sR, &
                                            gpu_stream ) &
                                            bind(C,name="gpu_final_ts_forces_virial")
        use iso_c_binding
        implicit none
        real(c_double),value :: rcut_inner, rcut, d, sR
        type(c_ptr),value :: forces_d,virial_d
        type(c_ptr),value :: r0_ij_d, r6_d, r6_der_d
        type(c_ptr),value :: neighbor_c6_ij_d, rjs_d, xyz_d, f_damp_d, exp_damp_d
        type(c_ptr),value :: hirshfeld_v_cart_der_d, pref_force1_d, pref_force2_d
        type(c_ptr),value :: i2_index_d, j2_index_d, i_site_index_d
        integer(c_int),value :: n_pairs
        type(c_ptr) :: gpu_stream
      end subroutine



      subroutine gpu_compute_pref_forces(pref_force1_d, pref_force2_d, &
                                         hirshfeld_v_d, r0_ref_d, & 
                                         r6_d, r0_ij_d, rjs_d, &
                                         neighbor_c6_ij_d, f_damp_d, exp_damp_d, &
                                         n_neigh_d, i2_k_index_d, k_start_index_d, &
                                         rcut_inner, rcut, d, sR, &
                                         n_pairs, n_sites, &
                                         gpu_stream ) &
                                         bind(C,name="gpu_compute_pref_forces")
        use iso_c_binding
        implicit none
        real(c_double),value :: rcut_inner, rcut, d, sR
        type(c_ptr),value :: pref_force1_d, pref_force2_d
        type(c_ptr),value :: hirshfeld_v_d, r0_ref_d
        type(c_ptr),value :: n_neigh_d, i2_k_index_d, k_start_index_d
        type(c_ptr),value :: r6_d, r0_ij_d, rjs_d
        type(c_ptr),value :: neighbor_c6_ij_d, f_damp_d, exp_damp_d

        integer(c_int),value :: n_pairs,n_sites
        type(c_ptr) :: gpu_stream
      end subroutine

      subroutine gpu_compute_damp_energy(energies_d, f_damp_d, exp_damp_d, &
                                         rjs_d, r0_ij_d, r6_d, neighbor_c6_ij_d, &
                                         n_neigh_d, k_start_index_d, &
                                         rcut_inner, rcut, d, sR, &
                                         n_sites, gpu_stream ) &
                                         bind(C,name="gpu_compute_damp_energy")

        use iso_c_binding
        implicit none
        real(c_double), value :: rcut_inner, rcut, d, sR
        integer(c_int),  value :: n_sites
        type(c_ptr), value :: energies_d
        type(c_ptr), value :: f_damp_d, exp_damp_d
        type(c_ptr), value :: rjs_d, r0_ij_d, r6_d
        type(c_ptr), value :: neighbor_c6_ij_d
        type(c_ptr), value :: n_neigh_d, k_start_index_d
        type(c_ptr) :: gpu_stream
      end subroutine

      subroutine gpu_compute_pair_params( &
        neighbor_c6_ii_d, r0_ii_d, neighbor_alpha0_d, &
        neighbor_c6_ij_d, r0_ij_d, &
        n_neigh_d, k_start_index_d, &
        n_sites, gpu_stream ) &
        bind(C,name="gpu_compute_pair_params")

  use iso_c_binding
  implicit none

  integer(c_int), value :: n_sites

  type(c_ptr), value :: neighbor_c6_ii_d
  type(c_ptr), value :: r0_ii_d
  type(c_ptr), value :: neighbor_alpha0_d
  type(c_ptr), value :: neighbor_c6_ij_d
  type(c_ptr), value :: r0_ij_d
  type(c_ptr), value :: n_neigh_d
  type(c_ptr), value :: k_start_index_d

  type(c_ptr) :: gpu_stream

end subroutine


    END INTERFACE
  END MODULE F_B_C
