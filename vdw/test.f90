program turbotest
 
  use vdw
  implicit none
  
  integer :: n_sites, n_atom_pairs, i_beg, i_end, j_beg, j_end
  real*8, allocatable :: local_properties:(:,:), local_properties_cart_der(:,:,:)
  integer :: n_neigh(:), neighbors_list(:), neighbor_species(:)
  real*8 ::vdw_rcut, vdw_buffer, vdw_rcut_inner, vdw_buffer_inner,  sR, d,
  real*8, allocatable :: rjs(:), xyz(:,:), v_neigh_vdw(:)
  real*8, allocatable :: vdw_c6_ref(:), vdw_r0_ref(:), vdw_alpha0_ref(:)
  logical :: do_forces
  real*8 :: this_virial(1:3, 1:3)
  real*8, allocatable :: this_energies(:), this_forces(:,:)
  integer :: n_vdw_c6_ref, n_vdw_r0_ref, n_vdw_r0_ref
  integer :: n_n_neigh, n_n_neighbors_list, n_neighbor_species
  integer :: n_v_neigh_vdw, n_rjs, n_xyz, n_energies, n_forces
  integer :: itmp, jtmp
  real*8 :: lptmp, lpocdtmpx,lpcdtmpy,lpcdtmpz


!  subroutine get_ts_energy_and_forces( hirshfeld_v, hirshfeld_v_cart_der, &
!                                        n_neigh, neighbors_list, neighbor_species, &
!                                        rcut, buffer, rcut_inner, buffer_inner, rjs, xyz, hirshfeld_v_neigh, &
!                                        sR, d, c6_ref, r0_ref, alpha0_ref, do_forces, &
!                                        energies, forces0, virial )
  n_sites=27000
  n_atom_pairs = 28728796

  i_beg = 1
  i_end = n_sites
  j_beg = 1
  j_end = n_atom_pairs
  vdw_rcut =    15.000000000000000 
  vdw_buffer=    1.0000000000000000     
  vdw_rcut_inner =   0.50000000000000000     
  vdw_buffer_inner =   0.50000000000000000
  vdw_sr =   0.93999999999999995     
  vdw_d =    20.000000000000000  
  n_vdw_c6_ref =            1
  n_vdw_r0_ref =            1 
  n_vdw_r0_ref =            1

  allocate(vdw_c6_ref(1:n_vdw_c6_ref))
  allocate(vdw_r0_ref(1:n_vdw_r0_ref))
  allocate(vdw_alpha0_ref(1:n_vdw_alpha0_ref))

  vdw_c6_ref(1) =    27.844753405694011     
  vdw_r0_ref(1) =    1.8999999999999999     
  vdw_r0_ref(1) =    1.8999999999999999
  
  n_n_neigh =        n_sites
  n_n_neighbors_list =     n_atom_pairs
  n_neighbor_species =     n_atom_pairs

  allocate(n_neigh(i_beg:i_end))
  allocate(neighbors_list(j_beg:j_end))
  allocate(neighbor_species(j_beg:j_end))
  n_v_neigh_vdw =     n_atom_pairs
  allocate(v_neigh_vdw(1:n_v_neigh_vdw) )

  
  n_rjs =     n_atom_pairs
  n_xyz =     n_atom_pairs
  allocate(rjs(1:n_rjs) )
  allocate(xyz(1:3,1:n_xyz) )


  n_energies =     n_sites
  n_forces =    n_sitees
  allocate(this_energies(1:n_rjs) )
  allocate(this_forces(1:3,1:n_forces) )
  
  allocate(local_properties(i_beg:i_end,1))
  allocate(local_properties(1:3, j_beg:j_end,1))
  
!   call get_ts_energy_and_forces( local_properties(i_beg:i_end, vdw_lp_index), &
!                 & local_properties_cart_der(1:3, j_beg:j_end, vdw_lp_index), &
!                 n_neigh(i_beg:i_end), neighbors_list(j_beg:j_end), &
!                 neighbor_species(j_beg:j_end), &
!                 params%vdw_rcut, params%vdw_buffer, &
!                 params%vdw_rcut_inner, params%vdw_buffer_inner, &
!                 rjs(j_beg:j_end), xyz(1:3, j_beg:j_end), v_neigh_vdw, &
!                 params%vdw_sr, params%vdw_d, params%vdw_c6_ref, params%vdw_r0_ref, &
!                 params%vdw_alpha0_ref, params%do_forces, &
!                 this_energies_vdw(i_beg:i_end), this_forces_vdw, this_virial_vdw)


end program