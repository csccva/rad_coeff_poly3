program turbotest
 
  use vdw
  implicit none
  
  integer :: n_sites, n_atom_pairs, i_beg, i_end, j_beg, j_end
  i_beg = 1
  i_end = n_sites
  j_beg = 1
  j_end = n_atom_pairs
  
!   call get_ts_energy_and_forces( local_properties(i_beg:i_end, vdw_lp_index), &
!                 & local_properties_cart_der(1:3, j_beg:j_end, vdw_lp_index), &
!                 n_neigh(i_beg:i_end), neighbors_list(j_beg:j_end), &
!                 neighbor_species(j_beg:j_end), &
!                 params%vdw_rcut, params%vdw_buffer, &
!                 params%vdw_rcut_inner, params%vdw_buffer_inner, &
!                 rjs(j_beg:j_end), xyz(1:3, j_beg:j_end), v_neigh_vdw, &
!                 params%vdw_sr, params%vdw_d, params%vdw_c6_ref, params%vdw_r0_ref, &
!                 params%vdw_alpha0_ref, params%do_forces, &
! #ifdef _MPIF90
!                 this_energies_vdw(i_beg:i_end), this_forces_vdw, this_virial_vdw)
! #else
!                 energies_vdw(i_beg:i_end), forces_vdw, virial_vdw )
! #endif

end program