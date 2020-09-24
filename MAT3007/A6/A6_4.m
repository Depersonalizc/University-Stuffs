g   = @(x) x(2) - x(1)^2;
h   = @(x) 1 - x(1);

f   = @(x) 100 * g(x)^2 + h(x)^2;
df  = @(x) [-400 * x(1) * g(x) - 2 * h(x); 200 * g(x)];


%[ex, efx, eiter, xs, nrms] = gd(f, df, [0;0], 1e-5, "exact", 1/2, 1/2);
[bx, bfx, biter, xs, nrms] = gd(f, df, [2;5], 1e-6, "backtrack", 1/2, 0.1);  
%fprintf("Exact Line Search: Minimum %f found at x = (%f,%f) after %d iterations.\n", efx, ex', eiter);
fprintf("Backtracking: Minimum %f found at x = (%f,%f) after %d iterations.\n", bfx, bx', biter);
diffs = xs - repmat(bx', size(xs,1), 1);
dists = norms(diffs, 2, 2);
plot(dists);
%plot(nrms);

function [x, fx, iter, xs, nrms] = gd(f, df, init, tol, line_search, sigma, gamma)

    % minimize function f using gradient descent.
    % args: f: function handle
    %       df: function gradient
    %       init: initial point
    %       tol: stopping tolerance
    %       line_search: "exact" or "backtrack"
    %       sigma, gamma: backtracking parameters
    
    iter    = 1;
    x       = init;
    fx      = f(x);
    dfx     = df(x);
    nrm     = norm(dfx);
    xs      = zeros(48, 2);
    xs(1,:) = x';
    nrms    = zeros(48, 1);
    nrms(1) = nrm;
    
    while (nrm > tol)
        fprintf("iter:%02d      x:(%f,%f)     norm:%f   optval:%f\n", ...
                iter, x(1), x(2), nrm, fx);
        
        if (line_search == "exact")
            step = ausection(@(a) f(x-a*dfx), 0, 10, 1e-5);

        elseif (line_search == "backtrack")
            step = 1;
            while ( f(x-step*dfx) - f(x) > -gamma*step*nrm^2 )
                step = step * sigma;
            end

        else
            fprintf("invalid line search method");
            return
        end
        
        iter        = iter + 1;
        x           = x - step * dfx;
        fx          = f(x);
        dfx         = df(x);
        nrm         = norm(dfx);
        xs(iter,:)  = x';
        nrms(iter)  = nrm;
    end
    
    iter = iter - 1;
end