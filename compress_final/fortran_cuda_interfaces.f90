
MODULE F_B_C
    INTERFACE
      subroutine gpu_set_device(my_rank) bind(C, name="cuda_set_device")
        use iso_c_binding
        integer(c_int), value :: my_rank
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
      
      subroutine gpu_memset_async(a_d,valuetoset,n,gpu_stream) bind(C,name="cuda_memset_async")
        use iso_c_binding
        implicit none
        type(c_ptr),value :: a_d
        type(c_ptr) :: gpu_stream
        integer(c_size_t),value :: n
        integer(c_int),value :: valuetoset
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


      subroutine gpu_check_error() bind(C,name="gpu_check_error")
        use iso_c_binding
        implicit none
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

      subroutine gpu_soap_normalize(soap_d, sqrt_dot_d, n_soap, n_sites, gpu_stream)  &
                  bind(C,name="gpu_soap_normalize")
        use iso_c_binding
        type(c_ptr), value :: sqrt_dot_d, soap_d
        type(c_ptr) :: gpu_stream
        integer(c_int),value :: n_sites, n_soap
      end subroutine



      subroutine gpu_get_soap_der(soap_d, sqrt_dot_d, soap_cart_der_d, &
                                  soap_rad_der_d, soap_azi_der_d, soap_pol_der_d, &
                                  g_this_soap_rad_der_d, g_this_soap_azi_der_d, g_this_soap_pol_der_d, &
                                  gdorpro_azi_d, gdorpro_rad_d, gdorpro_pol_d, &
                                  transp_azi_d, transp_rad_d, transp_pol_d, &
                                  thetas_d, phis_d, rjs_d, &
                                  multiplicity_array_d, &
                                  cnk_d, cnk_rad_der_d, cnk_azi_der_d, cnk_pol_der_d, &
                                  n_neigh_d, i_k2_start_d, k2_i_site_d, k3_index_d, skip_soap_component_d, &
                                  c_compress_soap, &
                                  compress_P_i, compress_P_j, compress_P_el, &
                                  n_sites, n_atom_pairs, n_soap,comp_P_nz, n_soap_uncompressed, &
                                  k_max, n_max, l_max, maxneigh, gpu_stream)  &
                  bind(C,name="gpu_get_soap_der")
        use iso_c_binding
        type(c_ptr), value :: sqrt_dot_d, soap_d, soap_cart_der_d
        type(c_ptr) :: gpu_stream
        type(c_ptr), value :: soap_rad_der_d, soap_azi_der_d, soap_pol_der_d
        type(c_ptr), value :: compress_P_i, compress_P_j, compress_P_el
        type(c_ptr), value :: g_this_soap_rad_der_d, g_this_soap_azi_der_d, g_this_soap_pol_der_d
        type(c_ptr), value :: gdorpro_azi_d, gdorpro_rad_d, gdorpro_pol_d
        type(c_ptr), value :: transp_azi_d, transp_rad_d, transp_pol_d
        type(c_ptr), value :: thetas_d, phis_d, rjs_d
        type(c_ptr), value :: n_neigh_d, i_k2_start_d, k2_i_site_d, k3_index_d, skip_soap_component_d
        type(c_ptr), value :: cnk_d, cnk_rad_der_d, cnk_azi_der_d, cnk_pol_der_d
        type(c_ptr), value :: multiplicity_array_d
        integer(c_int),value :: n_sites, n_atom_pairs, n_soap, k_max, n_max, l_max, maxneigh
        integer(c_int),value :: comp_P_nz, n_soap_uncompressed
        logical(c_bool), value :: c_compress_soap
      end subroutine

    END INTERFACE
  END MODULE F_B_C