rm *.o *.mod test.exe;
ftn -fPIC -O3  -h flex_mp=intolerant -c soap_turbo_functions.f90
ftn -fPIC -O3  -h flex_mp=intolerant -c soap_turbo_radial.f90
ftn -fPIC -O3  -h flex_mp=intolerant -c soap_turbo_radial_operator.f90
ftn -fPIC -O3  -h flex_mp=intolerant -c test_radial_coeff.f90
ftn -fPIC -O3  -h flex_mp=intolerant *.o -o  test.exe