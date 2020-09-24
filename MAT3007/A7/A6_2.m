cvx_begin
    variable x(4)
    minimize( -x(1) - x(2) + max(x(3),x(4)) )
    subject to
        (x(1)-x(2))^2 + (x(3)+2*x(4))^4 <= 5
        [1 2 1 2] * x <= 6
        x >= zeros(4,1)
cvx_end        