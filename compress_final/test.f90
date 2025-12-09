! rm *.o *.mod; ftn  -fPIC -O3 -h flex_mp=intolerant  soap_turbo_functions.f90 soap_turbo_compress.f90 soap_turbo_angular.f90 soap_turbo_radial.f90 soap_turbo.f90  test.f90 
!  srun --nodes=1 --account=project_462000007 --partition=standard-g --time=00:15:00 --gpus-per-node=8 --ntasks=1 ../turbogap_cpu/bin/turbogap predict 
program test_compress
   use soap_turbo_desc
   use F_B_C
   implicit none
   integer :: n_sites, n_species,max_species_multiplicity, n_atom_pairs,n_soap
   integer :: n_neigh_len, alpha_max_len
   integer, allocatable :: n_neigh(:), compress_P_i(:), compress_P_j(:)
   integer, allocatable :: species(:, :), species_multiplicity(:), alpha_max(:)
   real*8, allocatable :: rcut_hard(:), rcut_soft(:),global_scaling(:),nf(:),compress_P_el(:)
   real*8, allocatable :: atom_sigma_r(:), atom_sigma_r_scaling(:), central_weight(:)
   real*8, allocatable :: atom_sigma_t(:), atom_sigma_t_scaling(:), amplitude_scaling(:)
   logical, allocatable :: mask(:, :)
   logical :: do_timing, do_derivatives, compress_soap
   real*8, allocatable :: rjs(:), thetas(:),phis(:)
   integer :: l_max, radial_enhancement, compress_P_nonzero
   character*64 :: basis
   character*32 :: scaling_mode
   real*8, allocatable :: soap(:,:), soap_cart_der(:,:,:)
   integer :: i,j,k
   integer :: i_site_one, i_site_der

   
   call allocate_all()

   call initialize()

   call get_soap(n_sites, n_neigh, n_species, species, species_multiplicity, n_atom_pairs, mask, rjs, &
                          thetas, phis, alpha_max, l_max, rcut_hard, rcut_soft, nf, global_scaling, &
                          atom_sigma_r, atom_sigma_r_scaling, atom_sigma_t, atom_sigma_t_scaling, &
                          amplitude_scaling, radial_enhancement, central_weight, basis, scaling_mode, do_timing, &
                          do_derivatives, compress_soap, compress_P_nonzero, compress_P_i, compress_P_j, &
                          compress_P_el, soap, soap_cart_der)

   
   open(unit=5,file="soap_cpu.output",status="unknown")
   do i_site_one=1,n_sites
      do i=1,n_soap
         write(5, *) i_site_one, soap(i, i_site_one), &
                  soap_cart_der(1, i, i_site_one),soap_cart_der(2, i, i_site_one),soap_cart_der(3, i, i_site_one)
      enddo
   enddo
   close(5)
   
   call initialize()

   call get_soap_gpu(n_sites, n_neigh, n_species, species, species_multiplicity, n_atom_pairs, mask, rjs, &
                          thetas, phis, alpha_max, l_max, rcut_hard, rcut_soft, nf, global_scaling, &
                          atom_sigma_r, atom_sigma_r_scaling, atom_sigma_t, atom_sigma_t_scaling, &
                          amplitude_scaling, radial_enhancement, central_weight, basis, scaling_mode, do_timing, &
                          do_derivatives, compress_soap, compress_P_nonzero, compress_P_i, compress_P_j, &
                          compress_P_el, soap, soap_cart_der)

   open(unit=5,file="soap_gpu.output",status="unknown")
   do i_site_one=1,n_sites
      do i=1,n_soap
         write(5, *) i_site_one, soap(i, i_site_one), &
                  soap_cart_der(1, i, i_site_one),soap_cart_der(2, i, i_site_one),soap_cart_der(3, i, i_site_one)
      enddo
   enddo
   close(5)

   contains

   subroutine initialize()
      soap=0.0
      soap_cart_der=0.0
      alpha_max=8
      rcut_hard=4.5
      rcut_soft=4.0
      nf=4.0
      global_scaling=1.0
      atom_sigma_r=0.5
      atom_sigma_r_scaling=0.0
      atom_sigma_t=0.5
      atom_sigma_t_scaling=0.0
      amplitude_scaling=1.0
      central_weight=1.0
      open(unit=7, file="n_neigh.input", status="old" )
      do i=1,size(n_neigh,1)
         read(7,*) n_neigh(i)
      end do
      close(7)

      open(unit=9, file="species_multiplicity.input", status="old" )
      do i=1,size(species_multiplicity,1)
         read(9,*) species_multiplicity(i)
      end do
      close(9)

      open(unit=8,file="species.input", status="old")
      do j=1,size(species,2)
         do i=1,size(species,1)
            read(8,*) species(i,j)
         end do
      end do
      close(8)

      open(unit=11,file="mask.input", status="old")
      do j=1,size(mask,2)
         do i=1,size(mask,1)
            read(11,*) mask(i,j)
         enddo
      enddo
      close(11)

      open(unit=10,file="rjs.input", status="old")
      open(unit=11,file="thetas.input", status="old")
      open(unit=12,file="phis.input", status="old")
      do i=1,size(rjs,1)
         read(10, *) rjs(i)
         read(11, *) thetas(i)
         read(12, *) phis(i)
      enddo
      close(12)
      close(10)
      close(11)

      open(unit=19,file="compress_P_i.input", status="old")
      do i=1,size(compress_P_i,1)
         read(19,*) compress_P_i(i)
      enddo
      close(19)


      open(unit=19,file="compress_P_j.input", status="old")
      do i=1,size(compress_P_j,1)
         read(19,*) compress_P_j(i)
      enddo
      close(19)

      open(unit=19,file="compress_P_el.input", status="old")
      do i=1,size(compress_P_el,1)
         read(19,*) compress_P_el(i)
      enddo
      close(19)
   
   end subroutine

   subroutine allocate_all()
      i_site_one=67
      i_site_der=71
      basis="poly3gauss"
      scaling_mode= "polynomial"
      do_timing=  .False.
      do_derivatives = .True.  
      compress_soap= .True.
      n_sites=2451 ! 2698 !2707 !27000
      n_species=1
      n_neigh_len=n_sites
      alpha_max_len=1
      l_max=8
      compress_P_nonzero= 72
      max_species_multiplicity=1
      n_atom_pairs=175176 ! 192654 !1926616
      radial_enhancement=1
      n_soap=72
      allocate(n_neigh(n_neigh_len))
      allocate (species(1:max_species_multiplicity, 1:n_sites))
      allocate (species_multiplicity(1:n_sites))
      allocate (mask(1:n_atom_pairs, 1:n_species))
      allocate (rjs(1:n_atom_pairs))
      allocate (thetas(1:n_atom_pairs))
      allocate (phis(1:n_atom_pairs))
      allocate (alpha_max(1:alpha_max_len))
      allocate (rcut_hard(1:alpha_max_len), rcut_soft(1:alpha_max_len))
      allocate (global_scaling(1:alpha_max_len))
      allocate (nf(1:alpha_max_len))
      allocate (atom_sigma_r(1:alpha_max_len))
      allocate (atom_sigma_r_scaling(1:alpha_max_len))
      allocate (atom_sigma_t(1:alpha_max_len))
      allocate (atom_sigma_t_scaling(1:alpha_max_len))
      allocate (amplitude_scaling(1:alpha_max_len))
      allocate (central_weight(1:alpha_max_len))
      allocate (compress_P_i(1:compress_P_nonzero))
      allocate (compress_P_j(1:compress_P_nonzero))
      allocate (compress_P_el(1:compress_P_nonzero))

      allocate (soap_cart_der(1:3, 1:n_soap, 1:n_atom_pairs))
      allocate (soap(1:n_soap, 1:n_sites))
   
   endsubroutine
end program