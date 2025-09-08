program matmul_comparison
    implicit none
  
    ! Declare variables
    real, allocatable :: A(:), B(:), C1(:,:), C2(:)
    real :: temp
    integer :: m, n, p
    integer :: i, j, k
  
    ! Input dimensions of the matrices
    print *, "Enter dimensions of A (m x n):"
    read(*,*) m, n
    print *, "Enter dimensions of B (n x p):"
    read(*,*) n, p
  
    ! Allocate matrices
    allocate(A(m* n), B(n *p), C1(m,p), C2(m*p))
  
    ! Initialize matrices A and B with some values
    print *, "Initializing matrix A:"
    call initialize_matrix(A, m, n)
  
    print *, "Initializing matrix B:"
    call initialize_matrix(B, n, p)
  
    ! Matrix multiplication using MATMUL
    C1 = matmul(reshape(A,[ m, n ]), reshape(B,[ n,p ]))
  
    ! Matrix multiplication using explicit DO loops
    C2 = 0.0  ! Initialize C2 to zero
    do i = 1, m
       do j = 1, p
        temp=0
        do k = 1, n
            temp = temp + A(i+ (k-1)*m) * B(k+(j-1)*n)
            ! temp = temp + A(i, k) * B(k, j)
         end do
         C2(i+(j-1)*m) = temp
        !  C2(i, j) = temp
       end do
    end do
  
    ! Compare results of MATMUL and explicit DO loops
    if (all(C1 == reshape(C2,[m,p]))) then
       print *, "The results are identical!"
    else
       print *, "The results are different!"
    end if
  
    ! Display results
  
  
    do i = 1, m 
        do j = 1, p
            write(*,*) i, j, C1(i,j), C2(i+m*(j-1))
     end do
    enddo
  
    ! Deallocate matrices
    deallocate(A, B, C1, C2)
  
  contains
  
    ! Subroutine to initialize a matrix with sample values
    subroutine initialize_matrix(matrix, rows, cols)
      real, intent(out) :: matrix(:)
      integer, intent(in) :: rows, cols
      integer :: i, j
  
      do i = 1, rows
         do j = 1, cols
            matrix(i+rows*(j-1)) = real(i + j)  ! Example initialization
         end do
      end do
    end subroutine initialize_matrix
  
  end program matmul_comparison
  