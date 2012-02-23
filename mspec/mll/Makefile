getmll.so:
	$(F90C) -fPIC -c *.f
	f2py $(LAPACKL) -c mll.f90 *.o -m mll only: getmll

clean:
	-rm *.o *.so
