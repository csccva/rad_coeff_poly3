!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
! Test of GPU implementation of the compression subroutine 
! compared to CPU implementation.
! 
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

program test_compress

  use soap_turbo_compress_module

  implicit none

  integer :: l_max=5
  integer :: alpha_max(2)=[5, 5]          ! one value per chemical species
  character*16 :: compress_mode="trivial" ! see compress_options
  logical :: do_all_modes=.false.          ! .true. runs all compress_options,
                                          ! .false. only runs compress_mode
  character*16 :: compress_options(10)=[character*10 :: "trivial","0_0","0_1", &
                                        "0_2","1_0","1_1","1_2","2_0","2_1","2_2"]

  integer :: i, dim, P_nonzero, P_nonzero_1
  integer, allocatable :: P_i(:), P_j(:), P_i_1(:), P_j_1(:)
  real*8 :: t1, t2
  real*8, allocatable :: P_el(:), P_el_1(:)
  

  call p_i_p_j()
  contains

  subroutine p_i_p_j()
    !***************************************

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
    ! write(*,*) "P_i (CPU): ", P_i
    ! write(*,*) "P_i (GPU): ", P_i_1
    ! write(*,*) "P_el (CPU): ", P_el
    ! write(*,*) "P_el (GPU): ", P_el_1
    deallocate( P_i, P_j, P_el)
    ! deallocate( P_i, P_i_1, P_j, P_j_1, P_el, P_el_1 )
  end do

  end subroutine


end program
