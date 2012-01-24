module l3l4sum

contains

function l3l4_sum_double(imll,Cac,Cbd,Cad,Cbc,gll2,dlmode,lmax) result(ans)

    real(8), dimension(lmax,lmax) :: imll, gll2, ans
    real(8), dimension(lmax) :: Cac, Cbd, Cad, Cbc
    real(8) :: a
    integer dlmode
    integer lmax
    integer :: l1, l2, l3, l4

    ans = 0

    do l1 = 1, lmax
        do l2 = max(1,l1-dlmode), min(lmax,l1+dlmode)
            a=0
            do l3 = max(1,l1-dlmode), min(lmax,l1+dlmode)
                do l4 = max(1,l2-dlmode,l3-dlmode), min(lmax,l2+dlmode,l3+dlmode)
                    a = a + imll(l1,l3)*imll(l2,l4)*(Cac(l3)*Cbd(l4)+Cad(l3)*Cbc(l4)+Cac(l4)*Cbd(l3)+Cad(l4)*Cbc(l3))*gll2(l3,l4)
                end do
            end do
            ans(l1,l2) = a/2
        end do
    end do

end function

function l3l4_sum_single(imll,Cac,Cbd,Cad,Cbc,gll2,dlmode,lmax) result(ans)

    real(4), dimension(lmax,lmax) :: imll, gll2, ans
    real(4), dimension(lmax) :: Cac, Cbd, Cad, Cbc
    real(4) :: a
    integer dlmode
    integer lmax
    integer :: l1, l2, l3, l4

    ans = 0

    do l1 = 1, lmax
        do l2 = max(1,l1-dlmode), min(lmax,l1+dlmode)
            a=0
            do l3 = max(1,l1-dlmode), min(lmax,l1+dlmode)
                do l4 = max(1,l2-dlmode,l3-dlmode), min(lmax,l2+dlmode,l3+dlmode)
                    a = a + imll(l1,l3)*imll(l2,l4)*(Cac(l3)*Cbd(l4)+Cad(l3)*Cbc(l4)+Cac(l4)*Cbd(l3)+Cad(l4)*Cbc(l3))*gll2(l3,l4)
                end do
            end do
            ans(l1,l2) = a/2
        end do
    end do

end function

end module

