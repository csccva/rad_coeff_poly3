!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
! Test of timings and errors of different subroutines computing
! SOAP radial expansion coefficients.
! It uses random generated positions, given a n. of atoms (n_sites)
! and a n. of neighbours per atom (n_neigh)
!
! Be sure to have OMP_NUM_THREADS=1 when using optimization -O3
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

program test_radial

  use soap_turbo_radial
  use soap_turbo_radial_op

  implicit none

  integer :: n_sites, alpha_max, radial_enhancement
  integer, allocatable :: n_neigh(:)
  real*8 :: rcut_soft_in, rcut_hard_in, atom_sigma_in, atom_sigma_scaling, &
            amplitude_scaling, central_weight, nf = 4.d0
  real*8, allocatable :: rjs_in(:), W(:,:), S(:,:), exp_coeff(:,:), exp_coeff_der(:,:),  &
                         exp_coeff_cp(:,:), exp_coeff_der_cp(:,:)
  logical :: do_derivatives, do_central
  logical, allocatable :: mask(:)
  character*16 :: scaling_mode = "polynomial"

  real*8 :: t1, t2
  integer :: i, j, k

!***************************************
! SOAP parameters
  alpha_max = 7
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
  allocate( exp_coeff(1:alpha_max,1:n_sites*n_neigh(1)) )
  allocate( exp_coeff_der(1:alpha_max,1:n_sites*n_neigh(1)) )
  allocate( exp_coeff_cp(1:alpha_max,1:n_sites*n_neigh(1)) )
  allocate( exp_coeff_der_cp(1:alpha_max,1:n_sites*n_neigh(1)) )

  exp_coeff = 0.d0
  exp_coeff_der = 0.d0

  call RANDOM_NUMBER(rjs_in)
  rjs_in = 1.d0 + rjs_in*3.d0
  rjs_in(1) = 0.d0 
!  write(*,*) rjs_in
  mask = .true.

  allocate( W(1:alpha_max, 1:alpha_max) )
  allocate( S(1:alpha_max, 1:alpha_max) )

  call get_orthonormalization_matrix_poly3_tabulated(alpha_max, S, W)



! Using poly3
  call cpu_time(t1)
  call get_radial_expansion_coefficients_poly3(n_sites, n_neigh, rjs_in, alpha_max, rcut_soft_in, &
                                               rcut_hard_in, atom_sigma_in, atom_sigma_scaling, &
                                               amplitude_scaling, nf, W, scaling_mode, mask, &
                                               radial_enhancement, do_derivatives, do_central, &
                                               central_weight, exp_coeff, exp_coeff_der)
  call cpu_time(t2)
! Just to make sure the calculation is done with -O3
!  write(*,*) exp_coeff(1:alpha_max, 1)
  write(*,*) t2-t1, "seconds for poly3"


! Using operator 
  call cpu_time(t1)
  call get_radial_expansion_coefficients_poly3operator(n_sites, n_neigh, rjs_in, alpha_max, &
                                                       rcut_soft_in, rcut_hard_in, atom_sigma_in, &
                                                       atom_sigma_scaling, amplitude_scaling, W, &
                                                       scaling_mode, mask, radial_enhancement, &
                                                       do_derivatives, do_central, central_weight, &
                                                       exp_coeff, exp_coeff_der)
  call cpu_time(t2)
! Just to make sure the calculation is done with -O3
!  write(*,*) exp_coeff(1:alpha_max, 1)
  write(*,*) t2-t1, "seconds for operator"
  exp_coeff_cp = exp_coeff
  exp_coeff_der_cp = exp_coeff_der


! Using any other version of the radial coeff. subroutine
!  call cpu_time(t1)
!  call get_radial_expansion_coefficients_poly3operator_fast(n_sites, n_neigh, rjs_in, alpha_max, &
!                                                       rcut_soft_in, rcut_hard_in, atom_sigma_in, &
!                                                       atom_sigma_scaling, amplitude_scaling, W, &
!                                                       scaling_mode, mask, radial_enhancement, &
!                                                       do_derivatives, do_central, central_weight, &
!                                                       exp_coeff, exp_coeff_der)
!  call cpu_time(t2)
! Just to make sure the calculation is done with -O3
!  write(*,*) exp_coeff(1:alpha_max, 1)
!  write(*,*) t2-t1, "seconds for operator_fast", sum(abs(exp_coeff_cp-exp_coeff))


end program
