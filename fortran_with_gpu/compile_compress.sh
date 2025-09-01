rm *.o *.mod test_c.exe;
ftn -fPIC -h flex_mp=intolerant -c soap_turbo_compress.f90
ftn -fPIC -h flex_mp=intolerant -c test_compress.f90
ftn -fPIC -h flex_mp=intolerant *.o -o  test_c.exe
#gfortran -c -fcheck=bounds -g -fcheck=all -Wall soap_turbo_compress.f90
#gfortran -o test_c.exe soap_turbo_compress.o test_compress.f90 -fcheck=bounds -g -fcheck=all -Wall 
