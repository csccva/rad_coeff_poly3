!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
! Test of GPU implementation of the compression subroutine 
! compared to CPU implementation.
! rm *.o *.mod; ftn  -fPIC -O3 -h flex_mp=intolerant  soap_turbo_functions.f90 soap_turbo_radial.f90  soap_turbo_compress.f90  test_compress.f90 
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

program test_compress

  use soap_turbo_compress_module
  use soap_turbo_radial
  use soap_turbo_angular


  implicit none

  integer :: l_max=5
  !integer :: alpha_max(2)=[5, 5]          ! one value per chemical species
  integer :: alpha_max(1)=[7], int_alpha_max=7       ! one value per chemical species
  character*16 :: compress_mode="trivial" ! see compress_options
  logical :: do_all_modes=.false.          ! .true. runs all compress_options,
                                          ! .false. only runs compress_mode
  character*16 :: compress_options(10)=[character*10 :: "trivial","0_0","0_1", &
                                        "0_2","1_0","1_1","1_2","2_0","2_1","2_2"]

  integer :: i, dim, P_nonzero, P_nonzero_1
  integer, allocatable :: P_i(:), P_j(:), P_i_1(:), P_j_1(:)
  real*8 :: t1, t2
  real*8, allocatable :: P_el(:), P_el_1(:)
  
  integer :: n_sites, radial_enhancement, n_max
  integer, allocatable :: n_neigh(:)
  real*8 :: rcut_soft_in, rcut_hard_in, atom_sigma_in, atom_sigma_scaling, &
            amplitude_scaling, central_weight, nf = 4.d0
  real*8, allocatable :: rjs_in(:), W(:,:), S(:,:), exp_coeff(:,:), exp_coeff_der(:,:),  &
                         exp_coeff_cp(:,:), exp_coeff_der_cp(:,:)
  logical :: do_derivatives, do_central
  logical, allocatable :: mask(:)
  character*16 :: scaling_mode = "polynomial"

!***************************************
! SOAP parameters
  central_weight = 1.d0
  rcut_soft_in = 3.65d0
  rcut_hard_in = 4.d0
  atom_sigma_in = 0.3d0
  atom_sigma_scaling = 0.2d0
  atom_sigma_scaling = 0.d0
  amplitude_scaling = 1.d0
  radial_enhancement = 2

  do_central = .true.
  do_derivatives = .true.
!  do_derivatives = .false.
!**************************************
  n_sites = 1000
  allocate( n_neigh(1:n_sites) )
  n_neigh = 50
  allocate( rjs_in(1:n_sites*n_neigh(1)) )
  allocate( mask(1:n_sites*n_neigh(1)) )
  allocate( exp_coeff(1:int_alpha_max,1:n_sites*n_neigh(1)) )
  allocate( exp_coeff_der(1:int_alpha_max,1:n_sites*n_neigh(1)) )

  exp_coeff = 0.d0
  exp_coeff_der = 0.d0

  call RANDOM_NUMBER(rjs_in)
  rjs_in = 1.d0 + rjs_in*3.d0
  rjs_in(1) = 0.d0 
!  write(*,*) rjs_in
  mask = .true.
  
  ! This is to build the radial basis
  n_max = 0
  n_species=size(alpha_max,1)
  do i = 1, n_species
    n_max = n_max + alpha_max(i)
  end do
  l_max= ????? 

! Uncompressed SOAP dimensions
  k_max = 1 + l_max*(l_max+1)/2 + l_max
  n_soap_uncompressed = n_max*(n_max+1)/2 * (l_max+1)

! This is for the expansion coefficients and the soap vectors
  allocate( radial_exp_coeff(1:n_max, 1:n_atom_pairs) )
  radial_exp_coeff = 0.d0
  allocate( angular_exp_coeff(1:k_max, 1:n_atom_pairs) )
  angular_exp_coeff = 0.d0
  allocate( cnk( 1:k_max, 1:n_max, 1:n_sites) )
  cnk = 0.d0


  allocate( W(1:int_alpha_max, 1:int_alpha_max) )
  allocate( S(1:int_alpha_max, 1:int_alpha_max) )
  W = 0.d0
  S = 0.d0

  call get_orthonormalization_matrix_poly3_tabulated(int_alpha_max, S, W)

  call p_i_p_j()


! Using poly3
  call cpu_time(t1)
  call get_radial_expansion_coefficients_poly3(n_sites, n_neigh, rjs_in, int_alpha_max, rcut_soft_in, &
                                               rcut_hard_in, atom_sigma_in, atom_sigma_scaling, &
                                               amplitude_scaling, nf, W, scaling_mode, mask, &
                                               radial_enhancement, do_derivatives, do_central, &
                                               central_weight, exp_coeff, exp_coeff_der)
  ! write(*,*) exp_coeff, exp_coeff_der

! For the angular expansion the masking works differently, since we do not have a species-augmented basis as in the
! radial expansion part.
  call get_angular_expansion_coefficients(n_sites, n_neigh, thetas, phis, rjs, atom_sigma_t, atom_sigma_t_scaling, &
                                          rcut_max, l_max, eimphi, preflm, plm_array, prefl, prefm, &
                                          fact_array, mask, n_species, eimphi_rad_der, &
                                          do_derivatives, prefl_rad_der, angular_exp_coeff, angular_exp_coeff_rad_der, &
                                          angular_exp_coeff_azi_der, angular_exp_coeff_pol_der )
                                          
  

  call cpu_time(t2)  
! Just to make sure the calculation is done with -O3
!  write(*,*) exp_coeff(1:alpha_max, 1)
  write(*,*) t2-t1, "seconds for poly3"
  
  contains

  subroutine p_i_p_j()
    !***************************************

  ! write(*,*) size(compress_options)
  ! stop
  
  do i=1, size(compress_options)
    if( .not. do_all_modes .and. compress_options(i) /= compress_mode )then
      cycle
    else
      write(*,*) compress_options(i)
!     Using poly3
!     1) Get the dimension of the compressed vector, needed for allocation later
      call cpu_time(t1)
      call get_compress_indices( compress_options(i), alpha_max, l_max, dim, &
                                 P_nonzero, P_i, P_j, P_el, "get_dim" )
      call cpu_time(t2)
      write(*,*) t2-t1, "seconds for CPU version (dim)"
      write(*,*) dim, P_nonzero
!     2) Get the set of indices
      allocate( P_i(1:P_nonzero) )
      allocate( P_j(1:P_nonzero) )
      allocate( P_el(1:P_nonzero) )
      call cpu_time(t1)
      call get_compress_indices( compress_options(i), alpha_max, l_max, dim, &
                                 P_nonzero, P_i, P_j, P_el, "set_indices" )
      call cpu_time(t2)
      write(*,*) "P arrays done ",t2-t1, " seconds for CPU version (indices)"
! !     GPU implementation
! !     change with whatever gpu subroutine!!!
! !     1) Get the dimension of the compressed vector, needed for allocation later
!       call cpu_time(t1)
!       ! call get_compress_indices_gpu( compress_options(i), alpha_max, l_max, dim, &
!       !                                P_nonzero_1, P_i_1, P_j_1, P_el_1, "get_dim" )
!       call cpu_time(t2)
!       P_nonzero_1=P_nonzero
!       write(*,*) t2-t1, "seconds for GPU version (dim)"
!       write(*,*) dim, P_nonzero, P_nonzero_1
      
! !     2) Get the set of indices
!       allocate( P_i_1(1:P_nonzero_1) )
!       allocate( P_j_1(1:P_nonzero_1) )
!       allocate( P_el_1(1:P_nonzero_1) )
      
!       P_i_1=-10000
!       P_j_1=-10000
!       P_el_1=-1000

!       call cpu_time(t1)
!       ! call get_compress_indices_gpu( compress_options(i), alpha_max, l_max, dim, &
!       !                                P_nonzero_1, P_i_1, P_j_1, P_el_1, "set_indices" )
!       call cpu_time(t2)
!       write(*,*) t2-t1, "seconds for GPU version (indices)"

! !     Check results
!       write(*,*) "P_nonzero (CPU | GPU):", P_nonzero, P_nonzero_1
!       if( .not. all(P_i == P_i_1) )then
!         write(*,*) "P_i (CPU): ", P_i
!         write(*,*) "P_i (GPU): ", P_i_1
!       else
!         write(*,*) "P_i okay!"
!       end if
!       if( .not. all(P_j == P_j_1) )then
!         write(*,*) "P_j (CPU): ", P_j
!         write(*,*) "P_j (GPU): ", P_j_1
!       else
!         write(*,*) "P_j okay!"
!       end if
!       if( .not. all(P_el == P_el_1) )then
!         write(*,*) "P_el (CPU): ", P_el
!         write(*,*) "P_el (GPU): ", P_el_1
!       else
!         write(*,*) "P_el okay!"
!       end if
     end if
    write(*,*) "--------------------------------------------------------"
     write(*,*) "P_i (CPU): ", P_i
    ! write(*,*) "P_i (GPU): ", P_i_1
     write(*,*) "P_el (CPU): ", P_el
    ! write(*,*) "P_el (GPU): ", P_el_1
    deallocate( P_i, P_j, P_el)
    ! deallocate( P_i, P_i_1, P_j, P_j_1, P_el, P_el_1 )
  end do

  end subroutine


end program
