g   = @(x) x(2) - x(1)^2;
h   = @(x) 1 - x(1);

f   = @(x) 100 * g(x)^2 + h(x)^2;
df  = @(x) [-400 * x(1) * g(x) - 2 * h(x); 200 * g(x)];
hf  = @(x) [-400 * (x(2)-3*x(1)^2) + 2, -400 * x(1); -400 * x(1), 200];

[x, fx, iter, xs] = global_newton(f, df, hf, [2;5], 1e-6, 1/2, 0.1, 1e-6, 0.1);  
fprintf("Global Newton: Minimum %f found at x = (%f,%f) after %d iterations.\n", fx, x', iter);
diffs = xs - repmat(x', size(xs,1), 1);
dists = norms(diffs, 2, 2);
plot(dists);

function [x, fx, iter, xs] = global_newton(f, df, hf, init, tol, sigma, gamma, g1, g2)

    % minimize function f using gradient the globalized newton method.
    % args: f: function handle
    %       df: function gradient
    %       hf: function hessian
    %       init: initial point
    %       tol: stopping tolerance
    %       sigma, gamma: backtracking parameters
    %       g1, g2: parameters for descent condition
    
    iter    = 1;
    x       = init;
    fx      = f(x);     % function val
    dfx     = df(x);    % gradient val
    hfx     = hf(x);    % hessian  val
    s       = -hfx\dfx; % newton direction
    nrm     = norm(dfx);
    xs      = zeros(15, 2);
    xs(1,:) = x';
    
    while (nrm > tol)
        fprintf("iter:%02d      x:(%f,%f)     norm:%f   optval:%f", ...
                 iter, x(1), x(2), nrm, fx);
             
        % test descent condition
        if (-dfx' * s >= g1 * min([1, norm(s)^g2]) * norm(s)^2)
            dir = s;
            fprintf("...Newton\n");
        else
            dir = -dfx;
            fprintf("...Gradient\n");
        end
            
        % backtrack to decide stepsize
        step = 1;
        while ( f(x+step*dir) - f(x) > -gamma*step*dfx'*dir )
            step = step * sigma;
        end
        
        % update everything
        iter        = iter + 1;
        x           = x + step * dir;
        fx          = f(x);
        dfx         = df(x);
        hfx         = hf(x);
        s           = -hfx\dfx;
        nrm         = norm(dfx);
        xs(iter,:)  = x';

    end
    
    iter = iter - 1;
    xs = xs(1:iter,:);
    
end