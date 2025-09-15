program soap_cpu_main
    implicit none
    integer, parameter :: n_sites = 3
    integer, parameter :: n_species = 2
    integer, parameter :: n_atom_pairs = 6
    integer, parameter :: l_max = 2
  
    ! Arrays
    integer :: alpha_max(n_species), species(n_species, n_sites), species_multiplicity(n_sites)
    integer :: n_neigh(n_sites)
    integer :: compress_P_i(1), compress_P_j(1), compress_P_nonzero, radial_enhancement
    real(8) :: rjs(n_atom_pairs), thetas(n_atom_pairs), phis(n_atom_pairs)
    real(8) :: amplitude_scaling(n_species), atom_sigma_r(n_species)
    real(8) :: atom_sigma_r_scaling(n_species), atom_sigma_t(n_species), atom_sigma_t_scaling(n_species)
    real(8) :: central_weight(n_species), global_scaling(n_species), nf(n_species)
    real(8) :: rcut_hard(n_species), rcut_soft(n_species), compress_P_el(1)
    real(8) :: soap(20, n_sites), soap_cart_der(3,20,n_atom_pairs)
    logical :: mask(n_sites,n_species), do_derivatives, do_timing, compress_soap
    character(len=10) :: basis, scaling_mode
    integer :: i
  
    ! Initialize inputs
    alpha_max = (/3,3/)
    species = reshape([1,2,1,2,1,2], shape(species))
    species_multiplicity = (/1,1,1/)
    n_neigh = (/2,2,2/)
    rjs = (/1.0d0,1.2d0,0.9d0,1.1d0,1.0d0,1.3d0/)
    thetas = (/0.1d0,0.5d0,1.0d0,0.7d0,0.2d0,0.3d0/)
    phis = (/0.0d0,1.0d0,0.5d0,0.2d0,1.5d0,0.7d0/)
    amplitude_scaling = 1.0d0
    atom_sigma_r = 0.5d0
    atom_sigma_r_scaling = 1.0d0
    atom_sigma_t = 0.5d0
    atom_sigma_t_scaling = 1.0d0
    central_weight = 1.0d0
    global_scaling = 1.0d0
    nf = 1.0d0
    rcut_hard = 3.0d0
    rcut_soft = 2.5d0
    compress_P_el = 1.0d0
    compress_P_i = 1
    compress_P_j = 1
    compress_P_nonzero = 1
    mask = .true.
    do_derivatives = .false.
    do_timing = .false.
    compress_soap = .false.
    basis = 'poly3'
    scaling_mode = 'none'
    soap = 0.d0
    soap_cart_der = 0.d0
  
    ! Call subroutine
    call get_soap(n_sites, n_neigh, n_species, species, species_multiplicity, n_atom_pairs, mask, rjs, &
                  thetas, phis, alpha_max, l_max, rcut_hard, rcut_soft, nf, global_scaling, atom_sigma_r, &
                  atom_sigma_r_scaling, atom_sigma_t, atom_sigma_t_scaling, &
                  amplitude_scaling, radial_enhancement, central_weight, basis, scaling_mode, do_timing, &
                  do_derivatives, compress_soap, compress_P_nonzero, compress_P_i, compress_P_j, &
                  compress_P_el, soap, soap_cart_der)
  
    ! Print output
    do i = 1, n_sites
       print *, 'SOAP vector for site', i, ':', soap(:,i)
    end do
  
  
  contains
  
    subroutine get_soap(n_sites, n_neigh, n_species, species, species_multiplicity, n_atom_pairs, mask, rjs, &
                        thetas, phis, alpha_max, l_max, rcut_hard, rcut_soft, nf, global_scaling, atom_sigma_r, &
                        atom_sigma_r_scaling, atom_sigma_t, atom_sigma_t_scaling, &
                        amplitude_scaling, radial_enhancement, central_weight, basis, scaling_mode, do_timing, &
                        do_derivatives, compress_soap, compress_P_nonzero, compress_P_i, compress_P_j, &
                        compress_P_el, soap, soap_cart_der)
      implicit none
      integer, intent(in) :: n_sites, n_species, n_atom_pairs, l_max, radial_enhancement
      integer, intent(in) :: alpha_max(n_species), species(n_species,n_sites), species_multiplicity(n_sites)
      integer, intent(in) :: n_neigh(n_sites), compress_P_nonzero, compress_P_i(:), compress_P_j(:)
      real(8), intent(in) :: rjs(:), thetas(:), phis(:)
      real(8), intent(in) :: amplitude_scaling(:), atom_sigma_r(:), atom_sigma_r_scaling(:)
      real(8), intent(in) :: atom_sigma_t(:), atom_sigma_t_scaling(:), central_weight(:)
      real(8), intent(in) :: global_scaling(:), nf(:), rcut_hard(:), rcut_soft(:), compress_P_el(:)
      logical, intent(in) :: mask(:,:), do_derivatives, do_timing, compress_soap
      character(*), intent(in) :: basis, scaling_mode
      real(8), intent(inout) :: soap(:,:), soap_cart_der(:,:,:)
  
      integer :: i,j,n
      integer :: n_soap_uncompressed
      real(8), allocatable :: this_soap(:)
      real(8) :: radial, angular
  
      n_soap_uncompressed = size(soap,1)
      allocate(this_soap(n_soap_uncompressed))
  
      ! Simplified CPU-only SOAP computation
      do i = 1, n_sites
         this_soap = 0.d0
         do j = 1, n_neigh(i)
            do n = 1, n_soap_uncompressed
               ! radial contribution (decaying Gaussian)
               radial = exp(- (rjs(j)**2)/(2.0d0*atom_sigma_r(species(1,i))**2) )
               ! angular contribution (simplified as Legendre P_l(cos(theta)))
               angular = 1.0d0
               if (l_max >= 1) angular = angular + cos(thetas(j))
               if (l_max >= 2) angular = angular + 0.5d0*(3.0d0*cos(thetas(j))**2-1.0d0)
               this_soap(n) = this_soap(n) + radial * angular
            end do
         end do
  
         ! Apply global scaling
         this_soap = this_soap * global_scaling(1)
  
         ! Store in output
         soap(:,i) = this_soap(:)
      end do
  
      deallocate(this_soap)
    end subroutine get_soap
  
  end program soap_cpu_main
  