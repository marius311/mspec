l3l4sum.so:
	f2py --f90flags='-cpp -ffree-line-length-none -O3' -c l3l4sum.f90 -m l3l4sum
	-rm -f l3l4sum.mod l3l4sum.o

clean:
	-rm -f l3l4sum.mod l3l4sum.o l3l4sum.so
