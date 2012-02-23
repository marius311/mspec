module mll

contains

   function getmll(wl,lmaxin,lmax) result(kern)

        implicit none

        integer, parameter :: I4B = 4
        integer, parameter :: DP = 8
        real(DP), parameter :: pi = 3.141592
        real(DP), parameter :: twopi = 2*pi
        real(DP), parameter :: fourpi = 4*pi

        integer(I4B),   intent(in)              :: lmaxin, lmax
        real(DP), intent(in)               :: wl(0:lmaxin-1)
        real(DP)              :: kern(0:lmax-2,0:lmax-2)

        integer(I4B)            :: Ndim, nbins
        integer(I4B)            :: l_int, lp_int, lpp_int_f, lpp_int_t
        real(DP)                :: l_dp,  lp_dp,  lpp_dp_f,  lpp_dp_t
        real(DP)                :: ierr, tmp_dp

        real(DP),       allocatable     :: lpp(:), wig3j(:)
        real(DP),       allocatable     :: kern_tmp(:,:)

!        print *, lmaxin, lmax
!        print *, lbound(wl), ubound(wl)
!        print *, lbound(kern,dim=1), ubound(kern,dim=1), lbound(kern,dim=2), ubound(kern,dim=2)
!         200          50
!           0         199
!           0          48           0          48

        if (size(wl(:)) - 1 .lt. lmax) then
            print *, size(wl(:)), lmax
            write(*,*) "Too few wl multipoles. STOP!"
            stop
        else if (size(wl(:)) - 1 .lt. 2*lmax) then
            write(*,*) "wl does not contain enough multipoles. STOP!"
            stop
        endif

        nbins = lmax-1

        if (size(kern(0,:)) .ne. nbins .or. size(kern(:,0)) .ne. nbins) then
            print *, size(kern(0,:)), size(kern(:,0)), nbins
            write(*,*) "Invalid kern structure. STOP!"
            stop
        endif

        ALLOCATE(wig3j(0:2*lmax))
        ALLOCATE(lpp(0:2*lmax))
        ALLOCATE(kern_tmp(0:lmax,0:lmax))
        wig3j = 0.0_dp
        kern_tmp = 0.0_dp
        ierr = 0.0_dp

        do l_int=0, 2*lmax
            lpp(l_int) = real(l_int, kind=dp)
        enddo

        do l_int=0, lmax
            l_dp = real(l_int, kind=dp)
            do lp_int=l_int, lmax
                lp_dp = real(lp_int, kind=dp)

                lpp_int_f = abs(l_int - lp_int)
                lpp_int_t = l_int + lp_int

                lpp_dp_f = real(lpp_int_f, kind=DP)
                lpp_dp_t = real(lpp_int_t, kind=DP)

                Ndim = lpp_int_t - lpp_int_f + 1

                call DRC3JJ(l_dp, lp_dp, 0.0_dp, 0.0_dp, lpp_dp_f, lpp_dp_t, &
                & wig3j(lpp_int_f:lpp_int_t), Ndim, ierr)

                if (ierr .ne. 0.0_dp) then
                    write(*,*) 'ERROR', l_int, lp_int
                    stop
                endif

                tmp_dp = sum(wl(lpp_int_f:lpp_int_t) * (2.d0*lpp(lpp_int_f:lpp_int_t)+1.d0) * &
                 & wig3j(lpp_int_f:lpp_int_t)**2)

                kern_tmp(l_int,lp_int) = (2.d0*lp_dp+1.d0)/fourpi * tmp_dp
                kern_tmp(lp_int,l_int) = (2.d0*l_dp+1.d0)/fourpi * tmp_dp

            enddo
        enddo

        kern = kern_tmp

        DEALLOCATE(wig3j, lpp, kern_tmp)

    end function

   function getkll(wl,lmaxin,lmax) result(kern)

        implicit none

        integer, parameter :: I4B = 4
        integer, parameter :: DP = 8
        real(DP), parameter :: pi = 3.141592
        real(DP), parameter :: twopi = 2*pi
        real(DP), parameter :: fourpi = 4*pi

        integer(I4B),   intent(in)              :: lmaxin, lmax
        real(DP), intent(in)               :: wl(0:lmaxin-1)
        real(DP)              :: kern(0:lmax-2,0:lmax-2)

        integer(I4B)            :: i, j!, lwork, info
        integer(I4B)            :: Ndim, nbins
        integer(I4B)            :: l_int, lp_int, lpp_int_f, lpp_int_t
        real(DP)                :: l_dp,  lp_dp,  lpp_dp_f,  lpp_dp_t
        real(DP)                :: ierr, tmp_dp

!        integer(I4B),   allocatable     :: ipiv(:)
!        real(DP),       allocatable     :: work(:)

        integer(I4B),   allocatable     :: binvec(:)
        real(DP),       allocatable     :: lpp(:), wig3j(:)
        real(DP),       allocatable     :: pbl(:,:), qlb(:,:)
        real(DP),       allocatable     :: kern_tmp(:,:), temp(:)

!        print *, lmaxin, lmax
!        print *, lbound(wl), ubound(wl)
!        print *, lbound(kern,dim=1), ubound(kern,dim=1), lbound(kern,dim=2), ubound(kern,dim=2)
!         200          50
!           0         199
!           0          48           0          48

        if (size(wl(:)) - 1 .lt. lmax) then
            print *, size(wl(:)), lmax
            write(*,*) "Too few wl multipoles. STOP!"
            stop
        else if (size(wl(:)) - 1 .lt. 2*lmax) then
            write(*,*) "wl does not contain enough multipoles. STOP!"
            stop
        endif

        nbins = lmax-1

        if (size(kern(0,:)) .ne. nbins .or. size(kern(:,0)) .ne. nbins) then
            print *, size(kern(0,:)), size(kern(:,0)), nbins
            write(*,*) "Invalid kern structure. STOP!"
            stop
        endif

        ALLOCATE(binvec(0:nbins-1))
        do i=2, lmax
            binvec(i-2) = i
        enddo

        ALLOCATE(pbl(0:nbins-1,0:lmax), qlb(0:lmax,0:nbins-1))
        pbl = 0.0_dp
        qlb = 0.0_dp
        do j=0,nbins-2
            do i=binvec(j), binvec(j+1)-1
                pbl(j,i) = 1.0_dp/twopi * real(i*(i+1), kind=DP) / &
                & real(binvec(j+1)-binvec(j),kind=DP)
                if (i .ne. 0) qlb(i,j)=twopi / real(i*(i+1), kind=DP)
            enddo
        enddo

        j=nbins-1
        do i=binvec(j), lmax
            pbl(j,i) = 1.0_dp/twopi * real(i*(i+1), kind=DP) / &
            & real(lmax+1-binvec(j), kind=DP)
            if (i .ne. 0) qlb(i,j) = twopi / real(i*(i+1), kind=DP)
        enddo

        ALLOCATE(wig3j(0:2*lmax))
        ALLOCATE(lpp(0:2*lmax))
        ALLOCATE(kern_tmp(0:lmax,0:lmax))
        wig3j = 0.0_dp
        kern_tmp = 0.0_dp
        ierr = 0.0_dp

        do l_int=0, 2*lmax
            lpp(l_int) = real(l_int, kind=dp)
        enddo

        do l_int=0, lmax
            l_dp = real(l_int, kind=dp)
            do lp_int=l_int, lmax
                lp_dp = real(lp_int, kind=dp)

                lpp_int_f = abs(l_int - lp_int)
                lpp_int_t = l_int + lp_int

                lpp_dp_f = real(lpp_int_f, kind=DP)
                lpp_dp_t = real(lpp_int_t, kind=DP)

                Ndim = lpp_int_t - lpp_int_f + 1

                call DRC3JJ(l_dp, lp_dp, 0.0_dp, 0.0_dp, lpp_dp_f, lpp_dp_t, &
                & wig3j(lpp_int_f:lpp_int_t), Ndim, ierr)

                if (ierr .ne. 0.0_dp) then
                    write(*,*) 'ERROR', l_int, lp_int
                    stop
                endif

                tmp_dp = sum(wl(lpp_int_f:lpp_int_t) * (2.d0*lpp(lpp_int_f:lpp_int_t)+1.d0) * &
                 & wig3j(lpp_int_f:lpp_int_t)**2)

                kern_tmp(l_int,lp_int) = (2.d0*lp_dp+1.d0)/fourpi * tmp_dp
                kern_tmp(lp_int,l_int) = (2.d0*l_dp+1.d0)/fourpi * tmp_dp

            enddo
        enddo

        ALLOCATE(temp(0:lmax))
        do i=0, nbins-1
            temp = matmul(pbl(i,0:lmax), kern_tmp(0:lmax,0:lmax))
            do j=0, nbins-1
                kern(i,j) = sum(temp(0:lmax)*qlb(0:lmax,j))
            enddo
        enddo

        DEALLOCATE(binvec, pbl, qlb)
        DEALLOCATE(wig3j, lpp, kern_tmp, temp)

    end function

end module

