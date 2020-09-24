v = [2;1;3;2;1;4;2];
a = [2;0.5;0.5;0.1;0.5;1;1.5];
C = [3;2];

cvx_begin
variable X(7,2) nonnegative
maximize( v' * X * ones(2,1) )
subject to
    X  * ones(2,1) <= ones(7,1);
    X' * a <= C;
    X <= ones(7,2);
cvx_end