program matmul_test
    use iso_fortran_env, only : REAL64
    implicit none
  
    integer, parameter :: n = 50
    integer :: i, j, k
  
    real(REAL64), dimension(n,n) :: a, b, c1, c2, c3
    real(REAL64) :: alpha, beta
  
    call random_init(.true., .true.)
    call random_number(a)
    call random_number(b)
    write(*,*) sum(a), sum(b)
  
    c1 = 0.0
    c2 = 0.0
  
    do i=1, n
      do j=1, n
        do k=1, n
          c1(i,j) = c1(i,j) + a(i,k)*b(k,j)
        end do
      end do
    end do
  
    c3 = matmul(a, b)
  
    alpha = 1.0
    beta = 0.0
    call dgemm('N', 'N', n, n, n, alpha, a, n, b, n, beta, c2, n)
  
    write(*,*) maxval(abs(c1 - c2)), maxval(abs(c1 - c2))/maxval(c2)
  
    write(*,*) maxval(abs(c2 - c3)), maxval(abs(c2 - c3))/maxval(c2)
  
  end program