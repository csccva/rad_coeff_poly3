program turbotest
 
  use vdw
  implicit none
  
  integer :: n_sites, n_atom_pairs, i_beg, i_end, j_beg, j_end
  real*8, allocatable :: local_properties(:,:), local_properties_cart_der(:,:,:)
  real*8 ::vdw_rcut, vdw_buffer, vdw_rcut_inner, vdw_buffer_inner,  vdw_sr, vdw_d
  real*8, allocatable :: rjs(:), xyz(:,:), v_neigh_vdw(:)
  real*8, allocatable :: vdw_c6_ref(:), vdw_r0_ref(:), vdw_alpha0_ref(:)
  real*8, allocatable :: this_energies_vdw(:), this_forces_vdw(:,:)
  integer, allocatable :: n_neigh(:), neighbors_list(:), neighbor_species(:)
  integer :: n_vdw_c6_ref, n_vdw_r0_ref, n_vdw_alpha0_ref
  logical :: do_forces
  real*8 :: this_virial_vdw(1:3, 1:3)
  integer, allocatable :: n_n_neigh, n_n_neighbors_list, n_neighbor_species
  integer :: n_v_neigh_vdw, n_rjs, n_xyz, n_energies, n_forces
  integer :: itmp, jtmp, i,j
  real*8 :: lptmp, lpcdtmpx,lpcdtmpy,lpcdtmpz
  
  call init_problem()

  call get_ts_energy_and_forces(local_properties(i_beg:i_end,1), &
                & local_properties_cart_der(1:3, j_beg:j_end, 1), &
                n_neigh(i_beg:i_end), neighbors_list(j_beg:j_end), &
                neighbor_species(j_beg:j_end), &
                vdw_rcut, vdw_buffer, &
                vdw_rcut_inner, vdw_buffer_inner, &
                rjs(j_beg:j_end), xyz(1:3, j_beg:j_end), v_neigh_vdw, &
                vdw_sr, vdw_d, vdw_c6_ref, vdw_r0_ref, &
                vdw_alpha0_ref, do_forces, &
                this_energies_vdw(i_beg:i_end), this_forces_vdw, this_virial_vdw)
  

        open(unit=41, file="cpu_energies.output", status="unknown")
        do i=1,size(this_energies_vdw,1)
         write(41,*) i, this_energies_vdw(i)
        enddo
        close(41)
        
        write(*,*) sum(this_energies_vdw)
        
        open(unit=43,file="cpu_forces.output", status="unknown")
        do j=1,size(this_forces_vdw,2)
         write(43,*) j, this_forces_vdw(1,j),this_forces_vdw(2,j), this_forces_vdw(3,j) 
        enddo
        close(43)

        open(unit=47,file="cpu_virial.output", status="unknown")
        write(47,*) this_virial_vdw(1,1),this_virial_vdw(1,2),this_virial_vdw(1,3)
        write(47,*) this_virial_vdw(2,1),this_virial_vdw(2,2),this_virial_vdw(2,3)
        write(47,*) this_virial_vdw(3,1),this_virial_vdw(3,2),this_virial_vdw(3,3)
        close (47)
  
  contains

  subroutine init_problem()

    n_sites=27000
    n_atom_pairs = 28728796

    i_beg = 1
    i_end = n_sites
    j_beg = 1
    j_end = n_atom_pairs
    vdw_rcut =    15.000000000000000d0
    vdw_buffer=    1.0000000000000000d0    
    vdw_rcut_inner =   0.50000000000000000d0     
    vdw_buffer_inner =   0.50000000000000000d0
    vdw_sr =   0.93999999999999995d0
    do_forces = .true.
    vdw_d =    20.000000000000000d0  
    n_vdw_c6_ref =            1
    n_vdw_r0_ref =            1 
    n_vdw_alpha0_ref =            1

    allocate(vdw_c6_ref(1:n_vdw_c6_ref))
    allocate(vdw_r0_ref(1:n_vdw_r0_ref))
    allocate(vdw_alpha0_ref(1:n_vdw_alpha0_ref))

    vdw_c6_ref(1) =    27.844753405694011d0    
    vdw_r0_ref(1) =    1.899999999999999d0    
    vdw_alpha0_ref(1) =    1.7782165342468843d0
  
    n_n_neigh =        n_sites
    allocate(n_neigh(i_beg:i_end))
    open(unit=17,file="n_neigh.input",status="old")
    do i=i_beg,i_end
      read(17,*) itmp,jtmp
      n_neigh(itmp) = jtmp
    enddo 
    close(17)

  
    n_n_neighbors_list =     n_atom_pairs
    allocate(neighbors_list(j_beg:j_end))
    open(unit=19,file="neighbors_list.input",status="old")
    do j=j_beg,j_end
      read(19,*) jtmp,itmp
      neighbors_list(jtmp)=itmp
    enddo
    close(19)
  
  
    n_neighbor_species =     n_atom_pairs
    allocate(neighbor_species(j_beg:j_end))
    open(unit=23,file="neighbor_species.input",status="old")
    do j=j_beg,j_end
      read(23,*) jtmp,itmp
      neighbor_species(jtmp)=itmp
    enddo
    close(23)


    n_v_neigh_vdw =     n_atom_pairs
    allocate(v_neigh_vdw(1:n_v_neigh_vdw) )
    open(unit=29,file="v_neigh_vdw.input",status="old")
    do j=1,size(v_neigh_vdw,1)
      read(29,*) jtmp,lptmp
      v_neigh_vdw(jtmp)=lptmp
    enddo
    close(29)

  
    n_rjs =     n_atom_pairs
    allocate(rjs(1:n_rjs) )
    open(unit=31,file="rjs.input",status="old")
    do j=1,size(rjs,1)
      read(31,*) jtmp,lptmp
      rjs(jtmp)=lptmp
    enddo
    close(31)

  
    n_xyz =     n_atom_pairs
    allocate(xyz(1:3,1:n_xyz) )
    open(unit=37,file="xyz.input",status="old")
    do j=1,size(xyz,2)
      read(37,*) jtmp,lpcdtmpx, lpcdtmpy, lpcdtmpz
      xyz(1,jtmp)=lpcdtmpx
      xyz(2,jtmp)=lpcdtmpy
      xyz(3,jtmp)=lpcdtmpz
    enddo
    close(37)
  
    allocate(local_properties_cart_der(1:3, j_beg:j_end,1))
    open(unit=13, file="local_properties_cart_der.input", status="old")
    do j=j_beg,j_end
      read(13,*) jtmp, lpcdtmpx, lpcdtmpy, lpcdtmpz
      local_properties_cart_der(1,jtmp,1)=lpcdtmpx
      local_properties_cart_der(2,jtmp,1)=lpcdtmpy
      local_properties_cart_der(3,jtmp,1)=lpcdtmpz
    enddo
    close(13)


    allocate(local_properties(i_beg:i_end,1))
    open(unit=11, file="local_properties.input", status="old")
    do i=i_beg,i_end
      read(11,*) itmp, lptmp
      local_properties(itmp,1) = lptmp
    enddo
    close(11)

  
    n_energies =     n_sites
    n_forces =    n_sites
    allocate(this_energies_vdw(1:n_energies) )
    this_energies_vdw=0.0d0
    allocate(this_forces_vdw(1:3,1:n_forces) )
    this_forces_vdw=0.0d0
    this_virial_vdw=0.0d0
    write(*,*) "READING INPUT DATA and CPU  ALLOCATIONS DONE!"


           
           write(*,*) 
           write(*,*) "n_sites = ", n_sites, "n_atom_pairs = ", n_atom_pairs 
           write(*,*) "i_beg = ", i_beg, "i_end = ", i_end
           write(*,*) "j_beg = ", j_beg, "j_end = ", j_end
           write(*,*) "vdw_rcut = ", vdw_rcut
           write(*,*) "vdw_buffer= ", vdw_buffer    
           write(*,*) "vdw_rcut_inner = ", vdw_rcut_inner
           write(*,*) "vdw_buffer_inner = ", vdw_buffer_inner
           write(*,*)  "vdw_sr = ", vdw_sr
           write(*,*)  "vdw_d = ", vdw_d
           write(*,*)  "do_forces = ", do_forces
           write(*,*)  "n_vdw_c6_ref = ", size(vdw_c6_ref,1)
           write(*,*)  "n_vdw_r0_ref = ", size(vdw_r0_ref,1)
           write(*,*)  "n_vdw__alpha0_ref = ", size(vdw_alpha0_ref,1)


           write(*,*)  "vdw_c6_ref(1) = ", vdw_c6_ref(1)
           write(*,*)  "vdw_r0_ref(1) = ", vdw_r0_ref(1)
           write(*,*)  "vdw_alpha0_ref(1) = ", vdw_alpha0_ref(1)
           

           write(*,*) "n_n_neigh = ", size(n_neigh,1)
           write(*,*) "n_n_neighbors_list = ", size(neighbors_list,1)
           write(*,*) "n_neighbor_species = ", size(neighbor_species,1)

           write(*,*) "n_v_neigh_vdw = ", size(v_neigh_vdw,1)
           write(*,*) "n_rjs = ", size(rjs,1)
           write(*,*) "n_xyz = ", size(xyz,2)

           write(*,*) "n_energies = ", size(this_energies_vdw,1)
           write(*,*) "n_forces = ", size(this_forces_vdw,2)
  end subroutine

end program