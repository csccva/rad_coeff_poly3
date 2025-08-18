!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
! Test of GPU implementation of the compression subroutine 
! compared to CPU implementation.
! It uses synthetic data.
!
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

program test_compress

  use soap_turbo_compress_module

  implicit none

  integer :: l_max, dim, P_nonzero
  integer, allocatable :: alpha_max(:), P_i(:), P_j(:)
  real*8, allocatable :: P_el(:)
  character*16 :: compress_mode, what_to_do

  real*8 :: t1, t2

!***************************************
! subroutine parameters
  compress_mode = "trivial" 
  alpha_max = 7
  l_max = 7
  what_to_do = "get_dim" ! "set_indices"

  compress_options = ["trivial","0_0","0_1","0_2","1_0", &
                      "1_1","1_2","2_0","2_1","2_2"]
!**************************************




! Using poly3
  call cpu_time(t1)
  call get_compress_indices( compress_mode, alpha_max, l_max, dim,
                             P_nonzero, P_i, P_j, P_el, what_to_do )
  call cpu_time(t2)
  write(*,*) t2-t1, "seconds for CPU version"


! GPU implementation
! change with whatever gpu subroutine!!!
  call cpu_time(t1)
  call get_compress_indices_gpu( compress_mode, alpha_max, l_max, dim,
                                 P_nonzero, P_i, P_j, P_el, what_to_do )
  call cpu_time(t2)
  write(*,*) t2-t1, "seconds for GPU version"



end program
