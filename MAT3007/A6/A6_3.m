%h 	= @(x) 2*(x-1)/(x+1);
%f	= @(x) ( h(x) - log(x) ) / (x-1)^2;
%sol = ausection(f, 1.5, 4.5, 1e-5); 

g   = @(x) exp(-x) - cos(x);
gp  = @(x) -exp(-x) + sin(x);
[au_x, au_fx, au_iter] = ausection(g, 0, 1, 1e-5);
[bi_x, bi_fx, bi_iter] = bisection(g, gp, 0, 1, 1e-5);
fprintf("Golden Section: Minimum %f found at x = %f after %d iterations.\n", au_fx, au_x, au_iter);
fprintf("Bisection: Minimum %f found at x = %f after %d iterations.\n", bi_fx, bi_x, bi_iter);